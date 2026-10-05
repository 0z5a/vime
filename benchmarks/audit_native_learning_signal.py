"""Check actual saved native rollouts and gradients without claiming convergence."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from vime.rollout.rm_hub.math_utils import grade_answer_verl


def audit(run: Path, packet: Path, input_audit: Path, algorithm: str) -> dict:
    profile = json.loads((packet / "qualification.json").read_text())
    assert algorithm in ("grpo", "rltt")
    assert (
        hashlib.sha256((packet / "train.jsonl").read_bytes()).hexdigest() == profile["files"]["train.jsonl"]["sha256"]
    )
    inputs = json.loads(input_audit.read_text())
    assert hashlib.sha256((packet / "qualification.json").read_bytes()).hexdigest() == inputs["qualification_sha256"]
    token_hashes = {row["id"]: row["token_ids_sha256"] for row in inputs["rows"]}
    source = [json.loads(line) for line in (packet / "train.jsonl").read_text().splitlines()]
    group_size, prompts = profile["completions_per_prompt"], profile["prompts_per_rollout"]
    count = group_size * prompts
    rounds, useful_updates = [], []
    for rollout_id in range(profile["updates"]):
        rollout = torch.load(run / f"details/rollout_data/{rollout_id}.pt", map_location="cpu", weights_only=False)
        training = torch.load(run / f"details/train_data/{rollout_id}.pt", map_location="cpu", weights_only=False)
        assert rollout["rollout_id"] == training["rollout_id"] == rollout_id
        samples, trained = rollout["samples"], training["samples"]
        assert len(samples) == len(trained) == count
        rewards, cohort, nonzero_advantages, response_tokens = [], set(), 0, 0
        for i, (sample, batch) in enumerate(zip(samples, trained, strict=True)):
            row = source[rollout_id * prompts + i // group_size]
            identity, length = row["metadata"]["id"], sample["response_length"]
            assert sample["metadata"]["id"] == identity and sample["label"] == row["label"]
            assert sample["index"] == rollout_id * count + i
            assert sample["group_index"] == rollout_id * prompts + i // group_size
            assert 0 < length <= profile["response_limit"] and len(sample["tokens"]) > length
            prefix_hash = hashlib.sha256(json.dumps(sample["tokens"][:-length]).encode()).hexdigest()
            assert prefix_hash == token_hashes[identity]
            assert sample["status"] in ("completed", "truncated") and sample["reward"] in (0, 1)
            assert int(grade_answer_verl(sample["response"], row["label"])) == sample["reward"]
            assert batch["rollout_position"] == i and batch["sample_index"] == sample["index"]
            assert batch["tokens"].tolist() == sample["tokens"] and batch["response_lengths"] == length
            assert batch["local_raw_reward"] == sample["reward"]
            trace = sample["recurrent_trace"]
            assert asdict(batch["recurrent_inputs"]) == trace
            assert trace["model_revision"] == profile["model_revision"]
            assert trace["engine_revision"] == profile["engine_revision"]
            assert trace["model_family"] == "ouro" and trace["prefill_depth"] == 4
            assert trace["decode_depths"] == [4] * length
            assert trace["policy_version"] == rollout_id + 1
            assert sample["weight_versions"] == [str(rollout_id + 1)]
            assert trace["temperature"] == profile["sampling"]["temperature"]
            assert trace["seed"] == (profile["seed"] + rollout_id * 1_000_003 + sample["index"]) % (2**63)
            assert trace["finish_reason"] in ("length", "stop") and len(trace["publication_digest"]) == 64
            cohort.add((trace["runtime_epoch"], trace["policy_version"], trace["publication_digest"]))
            old = torch.tensor(sample["rollout_log_probs"])
            torch.testing.assert_close(old, batch["rollout_log_probs"], atol=0, rtol=0)
            assert old.shape == (length,) and bool(torch.isfinite(old).all())
            for field in ("advantages", "returns", "kl"):
                assert batch[field].shape == (length,) and bool(torch.isfinite(batch[field]).all())
            mask = torch.as_tensor(batch["loss_masks"])
            assert mask.shape == (length,) and bool(((mask == 0) | (mask == 1)).all())
            nonzero_advantages += int(torch.count_nonzero(batch["advantages"] * mask))
            if algorithm == "rltt":
                reference = batch["ref_log_probs"]
                assert reference.shape == (length,) and bool(torch.isfinite(reference).all())
                if rollout_id == 0:
                    torch.testing.assert_close(reference.float(), old, atol=1e-4, rtol=3e-5)
            else:
                torch.testing.assert_close(batch["log_probs"].float(), old, atol=1e-4, rtol=3e-5)
            response_tokens += length
            rewards.append(float(sample["reward"]))
        assert len(cohort) == 1, "One rollout crossed policy cohorts"
        mixed = sum(len(set(rewards[i : i + group_size])) > 1 for i in range(0, count, group_size))
        norm = torch.as_tensor(
            torch.load(run / f"grad-actor-{rollout_id}-0.pt", map_location="cpu", weights_only=False)
        )
        assert norm.numel() == 1 and bool(torch.isfinite(norm)) and float(norm) >= 0
        useful = mixed > 0 and nonzero_advantages > 0 and float(norm) > 0
        if useful:
            useful_updates.append(rollout_id)
        epoch, version, digest = cohort.pop()
        rounds.append(
            {
                "rollout_id": rollout_id,
                "mixed_groups": mixed,
                "total_groups": prompts,
                "reward_mean": sum(rewards) / count,
                "nonzero_masked_advantage_tokens": nonzero_advantages,
                "gradient_norm": float(norm),
                "response_tokens": response_tokens,
                "runtime_epoch": epoch,
                "policy_version": version,
                "publication_digest": digest,
                "useful_signal": useful,
            }
        )
    return {
        "algorithm": algorithm,
        "rounds": rounds,
        "useful_updates": useful_updates,
        "learning_signal_pass": bool(useful_updates),
        "scope": "saved rollout/advantage/gradient signal only",
        "optimizer_parameter_delta": "REQUIRES_SEPARATE_CHECKPOINT_AUDIT",
        "heldout_and_fresh_worker_resume": "REQUIRES_SEPARATE_COMPARISON",
        "reward_convergence": "NOT_ESTABLISHED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("grpo", "rltt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = audit(options.run, options.packet, options.input_audit, options.algorithm)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["learning_signal_pass"] else 1)


if __name__ == "__main__":
    main()
