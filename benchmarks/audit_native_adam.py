"""Check stored FP32 AdamW updates against a zero-new-gradient counterfactual."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.audit_native_learning_signal import audit as audit_signal
from benchmarks.native_checkpoint import Checkpoint, compare_checkpoints, fingerprint


def new_gradient_effect(before, after, old_m, old_v, new_m, *, step, lr, betas, eps, weight_decay) -> dict:
    """Allow FP32 rounding and both standard MCore decay/no-decay parameter groups."""
    tensors = (before, after, old_m, old_v, new_m)
    assert all(t.dtype == torch.float32 and t.shape == before.shape and bool(torch.isfinite(t).all()) for t in tensors)
    assert bool((old_v >= 0).all()) and step > 0
    beta1, beta2 = betas
    null_m, null_v = old_m * beta1, old_v * beta2
    tolerance = 4 * torch.finfo(torch.float32).eps
    moment_scale = torch.maximum(null_m.abs(), new_m.abs()).clamp_min(torch.finfo(torch.float32).tiny)
    new_signal = bool(((new_m - null_m).abs() > tolerance * moment_scale).any())
    update = lr * (null_m / (1 - beta1**step)) / ((null_v / (1 - beta2**step)).sqrt() + eps)
    residuals, changed = [], []
    for decay in (0.0, weight_decay):
        null_parameter = before * (1 - lr * decay) - update
        difference = (after - null_parameter).abs()
        scale = torch.maximum(after.abs(), null_parameter.abs()).clamp_min(torch.finfo(torch.float32).tiny)
        residuals.append(float(difference.max()))
        changed.append(bool((difference > tolerance * scale).any()))
    return {
        "fresh_moment_signal": new_signal,
        "beyond_zero_gradient_update": all(changed),
        "minimum_max_parameter_residual": min(residuals),
        "useful_parameter_update": new_signal and all(changed),
    }


def audit_adam(checkpoints: list[Checkpoint], initial_weights: Path, optimizer: dict) -> dict:
    prefix = "optimizer.state.exp_avg."
    moments = sorted({key[0][len(prefix) :] for key in checkpoints[0].index if key[0].startswith(prefix)})
    assert moments, "No saved Adam moments; an SGD checkpoint cannot qualify Adam"
    rounds = []
    with safe_open(initial_weights, framework="pt", device="cpu") as initial:
        for iteration, checkpoint in enumerate(checkpoints):
            assert checkpoint.common["iteration"] == iteration
            state = checkpoint.common["optimizer"]
            assert float(state["state"]["common_step"]) == iteration + 1
            for group in state["param_groups"]:
                assert group.get("decoupled_weight_decay") is True, (
                    "Saved optimizer state must explicitly identify AdamW"
                )
                assert tuple(group["betas"]) == tuple(optimizer["betas"])
                assert group["eps"] == optimizer["eps"] and group["lr"] == optimizer["lr"]
                assert group["weight_decay"] in (0, optimizer["weight_decay"])
                assert not group.get("amsgrad", False) and not group.get("maximize", False)
                assert group.get("bias_correction", True) and group.get("adam_w_mode", True)
            actual = sorted({key[0][len(prefix) :] for key in checkpoint.index if key[0].startswith(prefix)})
            assert actual == moments
            updates = []
            for name in moments:
                before = (
                    initial.get_tensor(name).float()
                    if iteration == 0
                    else checkpoints[iteration - 1].full_tensor(name)
                )
                old_m = (
                    torch.zeros_like(before)
                    if iteration == 0
                    else checkpoints[iteration - 1].full_tensor(prefix + name)
                )
                old_v = (
                    torch.zeros_like(before)
                    if iteration == 0
                    else checkpoints[iteration - 1].full_tensor("optimizer.state.exp_avg_sq." + name)
                )
                new_v = checkpoint.full_tensor("optimizer.state.exp_avg_sq." + name)
                assert new_v.shape == before.shape and bool((new_v >= 0).all())
                result = new_gradient_effect(
                    before,
                    checkpoint.full_tensor(name),
                    old_m,
                    old_v,
                    checkpoint.full_tensor(prefix + name),
                    step=iteration + 1,
                    lr=optimizer["lr"],
                    betas=optimizer["betas"],
                    eps=optimizer["eps"],
                    weight_decay=optimizer["weight_decay"],
                )
                updates.append({"name": name, **result})
            rounds.append(
                {
                    "rollout_id": iteration,
                    "parameters": updates,
                    "useful_parameter_update": any(row["useful_parameter_update"] for row in updates),
                }
            )
    return {
        "rounds": rounds,
        "useful_updates": [row["rollout_id"] for row in rounds if row["useful_parameter_update"]],
        "scope": "stored unsharded FP32 AdamW state; excludes argument/source and fresh-worker identity checks",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("grpo", "rltt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.initial_weights.open("rb") as stream:
        initial_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        assert initial_sha == args.initial_sha256
    profile = json.loads((args.packet / "qualification.json").read_text())
    assert profile["precision"] == "fp32" and profile["updates"] == 3 and profile["optimizer"]["name"] == "adam"
    roots = [run / "checkpoints/actor" for run in (args.continuous, args.resumed)]
    for root in roots:
        assert (root / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
    checkpoints = [[Checkpoint(root / f"iter_{i:07d}") for i in range(3)] for root in roots]
    comparisons = [compare_checkpoints(a, b) for a, b in zip(*checkpoints, strict=True)]
    for i in range(3):
        counters = [
            torch.load(root / f"rollout/global_dataset_state_dict_{i}.pt", weights_only=False) for root in roots
        ]
        assert fingerprint(counters[0]) == fingerprint(counters[1]), f"Rollout counters differ at update {i}"
        assert counters[0]["sample_index"] == (i + 1) * 32 and counters[0]["sample_group_index"] == (i + 1) * 4
        assert counters[0]["sample_offset"] == (i + 1) * 4 and counters[0]["epoch_id"] == 0
    adam = audit_adam(checkpoints[0], args.initial_weights, profile["optimizer"])
    signals = [
        audit_signal(run, args.packet, args.input_audit, args.algorithm) for run in (args.continuous, args.resumed)
    ]
    joint = sorted(set(adam["useful_updates"]).intersection(*(set(signal["useful_updates"]) for signal in signals)))
    report = {
        "inputs": {
            "initial_weights_sha256": initial_sha,
            "qualification_sha256": hashlib.sha256((args.packet / "qualification.json").read_bytes()).hexdigest(),
            "token_audit_sha256": hashlib.sha256(args.input_audit.read_bytes()).hexdigest(),
            "continuous": str(args.continuous.resolve()),
            "resumed": str(args.resumed.resolve()),
            "algorithm": args.algorithm,
        },
        "checkpoint_comparisons": comparisons,
        "adam": adam,
        "joint_useful_updates": joint,
        "signals": signals,
        "stored_state_and_update_pass": bool(joint),
        "heldout_and_worker_identity": "REQUIRES_SEPARATE_AUDIT",
        "reward_convergence": "NOT_ESTABLISHED",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if joint else 1)


if __name__ == "__main__":
    main()
