"""Compare functional replay with unmodified official token-by-token autograd."""

import argparse
import copy
import hashlib
import importlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/iquest-contract/61e8589747f6987ec7725e4ffe205f7a84561bd2"
sys.path.insert(0, str(ROOT / "evidence-archive-worktree"))
from iquest_replay_eager import replay


def official_forward(model, tokens: torch.Tensor) -> torch.Tensor:
    cache, logits = None, []
    for position in range(tokens.shape[1]):
        mask = torch.zeros((1, 1, 1, position + 1))
        result = model(
            input_ids=tokens[:, position : position + 1],
            past_key_values=cache,
            attention_mask={"full_attention": mask},
            use_cache=True,
            cache_position=torch.tensor([position]),
            position_ids=torch.tensor([[position]]),
        )
        cache = result.past_key_values
        logits.append(result.logits)
    return torch.cat(logits, dim=1)


def build_model(seed: int):
    package = types.ModuleType("pinned_iquest")
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    config_module = importlib.import_module(
        "pinned_iquest.configuration_iquestloopcoder"
    )
    model_module = importlib.import_module("pinned_iquest.modeling_iquestloopcoder")
    torch.manual_seed(seed)
    config = config_module.IQuestLoopCoderConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=131072,
        loop_num=2,
        loop_window_size=64,
    )
    config._attn_implementation = "eager"
    model = model_module.IQuestLoopCoderForCausalLM(config).float()
    with torch.no_grad():
        for gate in model.model.gate_projections:
            gate.weight.normal_(std=0.1)
            gate.bias.normal_(std=0.1)
    return model


def main(math_backend: bool) -> None:
    torch.set_num_threads(1)
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((SOURCE / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    filename = (
        "functional-replay-eager-math-audit.json"
        if math_backend
        else "functional-replay-eager-audit.json"
    )
    output = ROOT / "evidence/iquest-contract" / filename
    assert not output.exists()
    receipt = {
        "state": "RUNNING",
        "revision": SOURCE.name,
        "torch": torch.__version__,
        "official_weights": False,
        "device": "cpu",
        "sdpa_backend": "NONE_EXPLICIT_EAGER",
        "candidate_sha256": hashlib.sha256((ROOT / "bench/iquest_replay_eager.py").read_bytes()).hexdigest(),
        "gradient_relative_l2_gate": 2e-6,
        "logits_atol": 1e-5,
        "logits_rtol": 1e-5,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "cases": [],
        "scope": "Tiny supervised numerical/Adam equivalence; not RL reward convergence or 40B execution.",
    }
    for seed in (42, 43):
        for length in (1, 4, 8, 65):
            for recompute in (False, True):
                reference = build_model(seed)
                candidate = copy.deepcopy(reference)
                tokens = torch.randint(3, 97, (1, length))
                targets = torch.randint(0, 97, (length,))
                params = [dict(m.named_parameters()) for m in (reference, candidate)]
                assert params[0].keys() == params[1].keys()
                optimizers = [
                    torch.optim.AdamW(
                        m.parameters(),
                        lr=1e-4,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.1,
                    )
                    for m in (reference, candidate)
                ]
                case = {
                    "seed": seed,
                    "length": length,
                    "recompute": recompute,
                    "tokens": tokens.tolist(),
                    "targets": targets.tolist(),
                    "steps": [],
                }
                receipt["cases"].append(case)
                for step in (1, 2):
                    before = [
                        {k: p.detach().clone() for k, p in row.items()}
                        for row in params
                    ]
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    expected = official_forward(reference, tokens)
                    actual = replay(candidate, tokens, recompute=recompute)
                    record = {
                        "step": step,
                        "logits_max_abs": (expected - actual).abs().max().item(),
                        "failures": [],
                    }
                    case["steps"].append(record)
                    output.write_text(json.dumps(receipt, indent=2) + "\n")
                    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                        record["failures"].append({"field": "logits"})
                    losses = [
                        F.cross_entropy(logits.reshape(-1, 97), targets)
                        for logits in (expected, actual)
                    ]
                    for loss in losses:
                        loss.backward()
                    for row in params:
                        assert all(
                            p.grad is not None and torch.isfinite(p.grad).all()
                            for p in row.values()
                        )
                    norm = sum(
                        p.grad.double().square().sum().item()
                        for p in params[0].values()
                    )
                    error = sum(
                        (params[0][k].grad.double() - p.grad.double())
                        .square()
                        .sum()
                        .item()
                        for k, p in params[1].items()
                    )
                    record["gradient_relative_l2"] = (error / norm) ** 0.5
                    assert norm > 0
                    if record["gradient_relative_l2"] > 2e-6:
                        record["failures"].append({"field": "gradient_relative_l2"})
                    for optimizer in optimizers:
                        optimizer.step()
                    delta_max = 0.0
                    for name in params[0]:
                        p, q = params[0][name], params[1][name]
                        dp, dq = p - before[0][name], q - before[1][name]
                        delta_max = max(delta_max, (dp - dq).abs().max().item())
                        a, b = optimizers[0].state[p], optimizers[1].state[q]
                        assert a["step"].item() == b["step"].item() == step
                        for field, left, right, atol, rtol in (
                            ("parameter_delta", dq, dp, 1e-7, 1e-5),
                            ("parameter", q, p, 1e-7, 1e-5),
                            ("exp_avg", b["exp_avg"], a["exp_avg"], 1e-8, 2e-6),
                            (
                                "exp_avg_sq",
                                b["exp_avg_sq"],
                                a["exp_avg_sq"],
                                1e-10,
                                2e-6,
                            ),
                        ):
                            close = torch.isclose(left, right, atol=atol, rtol=rtol)
                            if not bool(close.all()):
                                record["failures"].append(
                                    {
                                        "tensor": name,
                                        "field": field,
                                        "mismatched_elements": int((~close).sum()),
                                        "max_abs": (left - right).abs().max().item(),
                                    }
                                )
                    record.update(
                        loss_reference=losses[0].item(),
                        loss_candidate=losses[1].item(),
                        parameter_delta_max_abs=delta_max,
                        all_parameters_and_moments_checked=True,
                        nonzero_applied_update=all(
                            any(bool((row[k] - initial[k]).any()) for k in row)
                            for row, initial in zip(params, before, strict=True)
                        ),
                        passed=not record["failures"],
                    )
                    assert record["nonzero_applied_update"]
                    output.write_text(json.dumps(receipt, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "length": length,
                            "recompute": recompute,
                            "steps": 2,
                        }
                    ),
                    flush=True,
                )
    receipt["state"] = (
        "PASS"
        if all(s["passed"] for c in receipt["cases"] for s in c["steps"])
        else "FAILED_NUMERICAL_GATE"
    )
    receipt["case_count"] = 16
    receipt["paired_steps"] = 32
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"state": receipt["state"], "cases": 16, "paired_steps": 32}))
    raise SystemExit(0 if receipt["state"] == "PASS" else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdpa-math", action="store_true")
    args = parser.parse_args()
    with sdpa_kernel(SDPBackend.MATH) if args.sdpa_math else nullcontext():
        main(args.sdpa_math)
