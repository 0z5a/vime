"""Synthetic evidence fixtures exercise rejection gates; these are not RL runs."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks.audit_native_learning_signal import audit
from vime.utils.types import RecurrentTrace, Sample


def evidence(tmp_path: Path, *, mixed=True):
    packet, run = tmp_path / "packet", tmp_path / "run"
    packet.mkdir()
    (run / "details/rollout_data").mkdir(parents=True)
    (run / "details/train_data").mkdir()
    profile = {
        "updates": 3,
        "prompts_per_rollout": 4,
        "completions_per_prompt": 8,
        "response_limit": 2048,
        "seed": 42,
        "sampling": {"temperature": 0.9},
        "model_revision": "model-pin",
        "engine_revision": "engine-pin",
    }
    manifest = packet / "qualification.json"
    manifest.write_text(json.dumps(profile))
    rows = [{"metadata": {"id": str(i)}, "label": "1"} for i in range(12)]
    (packet / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    profile["files"] = {"train.jsonl": {"sha256": hashlib.sha256((packet / "train.jsonl").read_bytes()).hexdigest()}}
    manifest.write_text(json.dumps(profile))
    prefix = [2, 3]
    inputs = {
        "qualification_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rows": [
            {"id": str(i), "token_ids_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
            for i in range(12)
        ],
    }
    input_audit = packet / "inputs.json"
    input_audit.write_text(json.dumps(inputs))
    for step in range(3):
        samples, trained = [], []
        for i in range(32):
            reward = int(mixed and i % 8 == 0)
            index, group = step * 32 + i, step * 4 + i // 8
            trace = RecurrentTrace(
                1,
                "ouro",
                "model-pin",
                "engine-pin",
                "epoch",
                step + 1,
                str(step + 1) * 64,
                str(index),
                42 + step * 1_000_003 + index,
                4,
                [4],
                0.9,
                "stop",
            )
            sample = Sample(
                index=index,
                group_index=group,
                tokens=prefix + [5],
                label="1",
                reward=reward,
                response=r"\boxed{1}" if reward else r"\boxed{0}",
                response_length=1,
                status=Sample.Status.COMPLETED,
                metadata={"id": str(group)},
                recurrent_trace=trace,
                weight_versions=[str(step + 1)],
                rollout_log_probs=[-1.0],
            )
            samples.append(sample.to_dict())
            trained.append(
                {
                    "rollout_position": i,
                    "sample_index": index,
                    "tokens": torch.tensor(sample.tokens),
                    "response_lengths": 1,
                    "local_raw_reward": reward,
                    "recurrent_inputs": trace,
                    "rollout_log_probs": torch.tensor([-1.0]),
                    "log_probs": torch.tensor([-1.0]),
                    "ref_log_probs": torch.tensor([-1.0]),
                    "advantages": torch.tensor([float(reward) - 0.125 if mixed else 0]),
                    "returns": torch.tensor([reward]),
                    "kl": torch.tensor([0.0]),
                    "loss_masks": [1],
                }
            )
        torch.save({"rollout_id": step, "samples": samples}, run / f"details/rollout_data/{step}.pt")
        torch.save({"rollout_id": step, "samples": trained}, run / f"details/train_data/{step}.pt")
        torch.save(1.0, run / f"grad-actor-{step}-0.pt")
    return run, packet, input_audit


@pytest.mark.parametrize("algorithm", ["grpo", "rltt"])
def test_signal_gate_never_claims_optimizer_resume_or_convergence(tmp_path, algorithm):
    report = audit(*evidence(tmp_path), algorithm)
    assert report["learning_signal_pass"] and report["useful_updates"] == [0, 1, 2]
    assert all(row["mixed_groups"] == 4 for row in report["rounds"])
    assert report["optimizer_parameter_delta"] == "REQUIRES_SEPARATE_CHECKPOINT_AUDIT"
    assert report["heldout_and_fresh_worker_resume"] == "REQUIRES_SEPARATE_COMPARISON"
    assert report["reward_convergence"] == "NOT_ESTABLISHED"


def test_nonzero_gradient_and_new_versions_alone_do_not_pass(tmp_path):
    report = audit(*evidence(tmp_path, mixed=False), "rltt")
    assert not report["learning_signal_pass"] and not report["useful_updates"]
    assert all(row["gradient_norm"] == 1.0 for row in report["rounds"])


@pytest.mark.parametrize("corruption", ["reward", "prefix", "index", "gradient", "reference"])
def test_rejects_mismatched_evidence(tmp_path, corruption):
    run, packet, inputs = evidence(tmp_path)
    path = run / "details/rollout_data/0.pt"
    record = torch.load(path, weights_only=False)
    if corruption == "reward":
        record["samples"][0]["response"] = r"\boxed{0}"
    elif corruption == "prefix":
        record["samples"][0]["tokens"][0] += 1
    elif corruption == "index":
        record["samples"][0]["index"] += 1
    elif corruption == "gradient":
        torch.save(float("nan"), run / "grad-actor-0-0.pt")
    else:
        train_path = run / "details/train_data/0.pt"
        train = torch.load(train_path, weights_only=False)
        train["samples"][0]["ref_log_probs"].add_(0.01)
        torch.save(train, train_path)
    torch.save(record, path)
    with pytest.raises(AssertionError):
        audit(run, packet, inputs, "rltt")
