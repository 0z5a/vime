"""Compare complete native training and held-out outputs across fresh recovery."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.audit_native_learning_signal import audit as audit_signal
from benchmarks.audit_native_runtime import audit as audit_runtime
from vime.rollout.native_eval import evaluation_seed
from vime.rollout.rm_hub.math_utils import grade_answer_verl

SCORE_ATOL, SCORE_RTOL = 1e-5, 3e-5
DATASET = "qualification-development"


def compare_samples(left: dict, right: dict) -> float:
    for key in (
        "index",
        "group_index",
        "prompt",
        "tokens",
        "response_length",
        "response",
        "label",
        "reward",
        "status",
        "metadata",
        "weight_versions",
    ):
        assert left[key] == right[key], f"Changed sample {left['index']}: {key}"
    traces = [
        {key: value for key, value in sample["recurrent_trace"].items() if key not in {"runtime_epoch", "request_id"}}
        for sample in (left, right)
    ]
    assert traces[0] == traces[1], "Changed semantic trace"
    return compare_scores(left["rollout_log_probs"], right["rollout_log_probs"])


def compare_scores(left, right) -> float:
    a, b = torch.as_tensor(left), torch.as_tensor(right)
    assert a.shape == b.shape and a.numel() > 0 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    torch.testing.assert_close(a, b, atol=SCORE_ATOL, rtol=SCORE_RTOL)
    return float((a - b).abs().max())


def validate_response(sample: dict, limit: int, tokenizer) -> None:
    length, trace = sample["response_length"], sample["recurrent_trace"]
    assert 0 < length <= limit and len(sample["tokens"]) > length
    assert trace["finish_reason"] in ("stop", "length")
    assert sample["status"] == ("completed" if trace["finish_reason"] == "stop" else "truncated")
    if trace["finish_reason"] == "length":
        assert length == limit
    else:
        assert sample["tokens"][-1] == tokenizer.eos_token_id
    assert sample["response"] == tokenizer.decode(sample["tokens"][-length:], skip_special_tokens=True), (
        "Saved text does not decode from its tokens"
    )
    assert (
        sample["reward"] in (0, 1) and int(grade_answer_verl(sample["response"], sample["label"])) == sample["reward"]
    )
    scores = torch.as_tensor(sample["rollout_log_probs"])
    assert scores.shape == (length,) and bool(torch.isfinite(scores).all())


def audit(continuous: Path, resumed: Path, packet: Path, input_audit: Path, algorithm: str, tokenizer) -> dict:
    runtime = audit_runtime(continuous, resumed, packet)
    signals = [audit_signal(run, packet, input_audit, algorithm) for run in (continuous, resumed)]
    consumed = dict(runtime["consumed_sha256"])

    def consume(path):
        consumed[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def load(path):
        return torch.load(consume(path), map_location="cpu", weights_only=False)

    profile = json.loads(consume(packet / "qualification.json").read_text())
    assert profile["eval_completions_per_prompt"] == 1
    assert profile["prompts_per_rollout"] == 4 and profile["completions_per_prompt"] == 8
    train_rows = [json.loads(line) for line in consume(packet / "train.jsonl").read_text().splitlines()]
    train_ids = [row["metadata"]["id"] for row in train_rows]
    assert len(train_ids) == len(set(train_ids)) == 12 and train_ids == profile["selected_ids"]["train"]
    inputs = json.loads(consume(input_audit).read_text())
    development = packet / "development.jsonl"
    assert (
        hashlib.sha256(consume(development).read_bytes()).hexdigest() == profile["files"][development.name]["sha256"]
    )
    rows = [json.loads(line) for line in development.read_text().splitlines()]
    ids = [row["metadata"]["id"] for row in rows]
    assert len(ids) == len(set(ids)) == profile["files"][development.name]["rows"] == 8
    assert ids == profile["selected_ids"]["development"] and not set(ids) & set(profile["selected_ids"]["train"])
    assert len(inputs["rows"]) == 20 and {row["id"] for row in inputs["rows"]} == set(train_ids + ids)
    token_rows = [row for row in inputs["rows"] if row["split"] == "development"]
    assert len(token_rows) == 8 and [row["id"] for row in token_rows] == ids
    token_hashes = {row["id"]: row["token_ids_sha256"] for row in token_rows}
    training, evaluation, summaries = [], [], []
    for run_index, run in enumerate((continuous, resumed)):
        train_rounds, eval_rounds, curve = [], [], []
        for step in range(3):
            data = load(run / f"details/rollout_data/{step}.pt")
            trained = load(run / f"details/train_data/{step}.pt")
            for sample in data["samples"]:
                validate_response(sample, profile["response_limit"], tokenizer)
            train_rounds.append((data["samples"], trained["samples"]))
            consume(run / f"grad-actor-{step}-0.pt")
        for step in range(4):
            data = load(run / f"details/rollout_data/eval_{step}.pt")
            samples = data["samples"]
            assert data["rollout_id"] == step and len(samples) == len(rows)
            phase = "continuous" if run_index == 0 else ("split" if step <= 2 else "resume")
            epoch = runtime["phases"][phase]["runtime_epoch"]
            digests, requests = set(), set()
            for index, (sample, row) in enumerate(zip(samples, rows, strict=True)):
                assert sample["index"] == sample["group_index"] == index
                assert sample["metadata"]["id"] == row["metadata"]["id"] and sample["label"] == row["label"]
                assert sample["metadata"]["native_eval"] == {
                    "dataset": DATASET,
                    "prompt_index": index,
                    "completion": 0,
                    "seed_profile": "native-heldout-v1",
                    "top_p": 1,
                    "top_k": -1,
                    "max_tokens": profile["response_limit"],
                    "stop_token_ids": [],
                }
                length, trace = sample["response_length"], sample["recurrent_trace"]
                validate_response(sample, profile["response_limit"], tokenizer)
                prefix = hashlib.sha256(json.dumps(sample["tokens"][:-length]).encode()).hexdigest()
                assert prefix == token_hashes[row["metadata"]["id"]]
                assert trace["runtime_epoch"] == epoch and trace["policy_version"] == step + 1
                assert sample["weight_versions"] == [str(step + 1)]
                assert trace["model_family"] == "ouro" and trace["schema_version"] == 1
                assert (
                    trace["model_revision"] == profile["model_revision"]
                    and trace["engine_revision"] == profile["engine_revision"]
                )
                assert trace["prefill_depth"] == 4 and trace["decode_depths"] == [4] * length
                assert trace["temperature"] == 0 and trace["seed"] == evaluation_seed(
                    profile["seed"], DATASET, index, 0
                )
                assert trace["latent_seed"] is None and trace["latent_profile"] is None
                assert (
                    isinstance(trace["request_id"], str)
                    and trace["request_id"]
                    and trace["request_id"] not in requests
                )
                requests.add(trace["request_id"])
                assert len(trace["publication_digest"]) == 64
                digests.add(trace["publication_digest"])
            assert len(digests) == 1, "Mixed held-out policy publications"
            digest = digests.pop()
            if step < 3:
                assert digest == signals[run_index]["rounds"][step]["publication_digest"], (
                    "Held-out and training policy differ"
                )
            curve.append(
                dict(
                    completed_updates=step,
                    reward_mean=sum(sample["reward"] for sample in samples) / len(samples),
                    response_tokens=sum(sample["response_length"] for sample in samples),
                    truncated=sum(sample["status"] == "truncated" for sample in samples),
                    publication_digest=digest,
                )
            )
            eval_rounds.append(samples)
        assert len({row["publication_digest"] for row in curve}) == 4, "An update reused an old publication"
        training.append(train_rounds)
        evaluation.append(eval_rounds)
        summaries.append(curve)
    errors = {"training_scores": 0.0, "heldout_scores": 0.0, "training_tensors": 0.0, "gradient_norm": 0.0}
    for left, right in zip(signals[0]["rounds"], signals[1]["rounds"], strict=True):
        errors["gradient_norm"] = max(
            errors["gradient_norm"], compare_scores(left["gradient_norm"], right["gradient_norm"])
        )
    for left, right in zip(*training, strict=True):
        for a, b in zip(left[0], right[0], strict=True):
            errors["training_scores"] = max(errors["training_scores"], compare_samples(a, b))
        for a, b in zip(left[1], right[1], strict=True):
            assert torch.equal(torch.as_tensor(a["loss_masks"]), torch.as_tensor(b["loss_masks"]))
            for key in ("advantages", "returns", "kl", "ref_log_probs" if algorithm == "rltt" else "log_probs"):
                errors["training_tensors"] = max(errors["training_tensors"], compare_scores(a[key], b[key]))
    for left, right in zip(*evaluation, strict=True):
        for a, b in zip(left, right, strict=True):
            errors["heldout_scores"] = max(errors["heldout_scores"], compare_samples(a, b))
    useful = sorted(set(signals[0]["useful_updates"]) & set(signals[1]["useful_updates"]))
    return {
        "output_equivalence_pass": True,
        "joint_signal_updates": useful,
        "output_and_signal_pass": bool(useful),
        "training_samples_per_run": 96,
        "heldout_samples_per_run": 32,
        "numeric_tolerance": {"atol": SCORE_ATOL, "rtol": SCORE_RTOL},
        "max_absolute_errors": errors,
        "continuous_curve": summaries[0],
        "resumed_curve": summaries[1],
        "runtime": runtime,
        "signals": signals,
        "consumed_sha256": consumed,
        "scope": "complete three-update native output/held-out recovery qualification, not a convergence experiment",
        "weights_optimizer_frozen_sources": "REQUIRES_SEPARATE_AUDITS",
        "reward_convergence": "NOT_ESTABLISHED",
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("continuous", "resumed", "packet", "input-audit", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--algorithm", choices=("grpo", "rltt"), required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    report = audit(args.continuous, args.resumed, args.packet, args.input_audit, args.algorithm, tokenizer)
    report["tokenizer_sha256"] = {
        name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
        if (args.tokenizer / name).is_file()
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["output_and_signal_pass"] else 1)


if __name__ == "__main__":
    main()
