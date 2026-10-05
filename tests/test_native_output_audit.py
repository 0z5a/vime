"""Synthetic saved runs exercise complete-output gates, with a real local tokenizer."""

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from benchmarks.audit_native_outputs import audit, main
from tests.test_native_learning_signal_audit import evidence
from vime.rollout.native_eval import evaluation_seed
from vime.utils.types import RecurrentTrace

pytest_plugins = ["tests.test_native_runtime_audit"]


@pytest.fixture
def runs(tmp_path, records):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[BOS]": 1, "p": 2, "q": 3, "other": 4, r"\boxed{0}": 5, r"\boxed{1}": 6, "[EOS]": 7},
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    built = []
    for index, original in enumerate(records[:2]):
        directory = tmp_path / f"signals-{index}"
        directory.mkdir()
        run, packet, inputs = evidence(directory)
        shutil.copytree(original / "runtime", run / "runtime")
        for path in original.glob("*-process.json"):
            shutil.copyfile(path, run / path.name)
        for step in range(3):
            rollout_path, train_path = run / f"details/rollout_data/{step}.pt", run / f"details/train_data/{step}.pt"
            rollout, trained = (torch.load(path, weights_only=False) for path in (rollout_path, train_path))
            phase = "continuous" if index == 0 else ("split" if step < 2 else "resume")
            for sample, batch in zip(rollout["samples"], trained["samples"], strict=True):
                sample["tokens"] = [2, 3, 6 if sample["reward"] else 5, 7]
                sample["response_length"], sample["rollout_log_probs"] = 2, [-1.0, -1.0]
                sample["recurrent_trace"].update(runtime_epoch=phase, decode_depths=[4, 4])
                batch["tokens"], batch["response_lengths"] = torch.tensor(sample["tokens"]), 2
                batch["recurrent_inputs"] = RecurrentTrace(**sample["recurrent_trace"])
                batch["loss_masks"] = [1, 1]
                for key in ("rollout_log_probs", "log_probs", "ref_log_probs", "advantages", "returns", "kl"):
                    batch[key] = batch[key].repeat(2)
            torch.save(rollout, rollout_path)
            torch.save(trained, train_path)
        built.append(run)
    development = [{"prompt": "p q", "label": "1", "metadata": {"id": f"dev-{i}"}} for i in range(8)]
    data_path = packet / "development.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in development))
    profile_path = packet / "qualification.json"
    profile = json.loads(profile_path.read_text())
    profile.update(
        response_limit=2,
        eval_completions_per_prompt=1,
        split_stop_after=2,
        precision="fp32",
        selected_ids={
            "train": [str(i) for i in range(12)],
            "development": [row["metadata"]["id"] for row in development],
        },
    )
    profile["files"]["development.jsonl"] = {"rows": 8, "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
    profile_path.write_text(json.dumps(profile))
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    input_rows = [{**row, "split": "train"} for row in json.loads(inputs.read_text())["rows"]]
    input_rows.extend(
        {
            "split": "development",
            "id": row["metadata"]["id"],
            "token_ids_sha256": hashlib.sha256(json.dumps([2, 3]).encode()).hexdigest(),
        }
        for row in development
    )
    inputs.write_text(json.dumps({"qualification_sha256": profile_sha, "rows": input_rows}))
    for index, run in enumerate(built):
        for path in run.glob("*-process.json"):
            receipt = json.loads(path.read_text())
            receipt["qualification_sha256"] = profile_sha
            path.write_text(json.dumps(receipt))
        template = torch.load(run / "details/rollout_data/0.pt", weights_only=False)["samples"][0]
        for step in range(4):
            samples = []
            for i, row in enumerate(development):
                sample = copy.deepcopy(template)
                reward = int(i % 2 == 0)
                sample.update(
                    index=i,
                    group_index=i,
                    prompt="p q",
                    tokens=[2, 3, 6 if reward else 5, 7],
                    response=tokenizer.decode([6 if reward else 5, 7], skip_special_tokens=True),
                    reward=reward,
                    metadata={
                        **row["metadata"],
                        "native_eval": {
                            "dataset": "qualification-development",
                            "prompt_index": i,
                            "completion": 0,
                            "seed_profile": "native-heldout-v1",
                            "top_p": 1,
                            "top_k": -1,
                            "max_tokens": 2,
                            "stop_token_ids": [],
                        },
                    },
                    weight_versions=[str(step + 1)],
                )
                phase = "continuous" if index == 0 else ("split" if step <= 2 else "resume")
                sample["recurrent_trace"].update(
                    runtime_epoch=phase,
                    policy_version=step + 1,
                    publication_digest=str(step + 1) * 64,
                    request_id=f"{phase}-{step}-{i}",
                    temperature=0,
                    seed=evaluation_seed(42, "qualification-development", i, 0),
                )
                samples.append(sample)
            torch.save({"rollout_id": step, "samples": samples}, run / f"details/rollout_data/eval_{step}.pt")
    return *built, packet, inputs, tokenizer_path, tokenizer


@pytest.mark.parametrize("algorithm", ["grpo", "rltt"])
def test_full_cli_reads_all_saved_outputs_with_real_tokenizer(runs, tmp_path, monkeypatch, algorithm):
    continuous, resumed, packet, inputs, tokenizer_path, _ = runs
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--continuous",
            str(continuous),
            "--resumed",
            str(resumed),
            "--packet",
            str(packet),
            "--input-audit",
            str(inputs),
            "--tokenizer",
            str(tokenizer_path),
            "--algorithm",
            algorithm,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    report = json.loads(output.read_text())
    assert report["output_and_signal_pass"] and report["joint_signal_updates"] == [0, 1, 2]
    assert report["training_samples_per_run"] == 96 and report["heldout_samples_per_run"] == 32
    assert report["continuous_curve"] == report["resumed_curve"]
    assert all(value == 0 for value in report["max_absolute_errors"].values())
    assert report["reward_convergence"] == "NOT_ESTABLISHED" and len(report["consumed_sha256"]) == 39
    for path, digest in report["consumed_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for name, digest in report["tokenizer_sha256"].items():
        assert hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_round",
        "missing_row",
        "reordered_id",
        "prefix",
        "decoded_text",
        "seed",
        "version",
        "epoch",
        "eos",
        "nonfinite",
        "publication",
        "stale_final",
        "changed_output",
        "training_tensor",
        "gradient",
        "training_text",
    ],
)
def test_rejects_incomplete_or_changed_outputs(runs, corruption):
    continuous, resumed, packet, inputs, _, tokenizer = runs
    path = resumed / "details/rollout_data/eval_3.pt"
    data = torch.load(path, weights_only=False)
    sample = data["samples"][1]
    if corruption == "missing_round":
        path.unlink()
    elif corruption == "missing_row":
        data["samples"].pop()
    elif corruption == "reordered_id":
        sample["metadata"]["id"] = "dev-0"
    elif corruption == "prefix":
        sample["tokens"][0] = 4
    elif corruption == "decoded_text":
        sample.update(response=r"\boxed{1}", reward=1)
    elif corruption == "seed":
        sample["recurrent_trace"]["seed"] += 1
    elif corruption == "version":
        sample["recurrent_trace"]["policy_version"] = 3
    elif corruption == "epoch":
        sample["recurrent_trace"]["runtime_epoch"] = "split"
    elif corruption == "eos":
        sample["tokens"][-1] = 4
    elif corruption == "nonfinite":
        sample["rollout_log_probs"][0] = float("nan")
    elif corruption in ("publication", "stale_final"):
        if corruption == "publication":
            path = resumed / "details/rollout_data/eval_1.pt"
            data = torch.load(path, weights_only=False)
        for row in data["samples"]:
            row["recurrent_trace"]["publication_digest"] = ("f" if corruption == "publication" else "3") * 64
    elif corruption == "changed_output":
        sample["tokens"][-2] = 6
        sample.update(response=r"\boxed{1}", reward=1)
    elif corruption == "training_tensor":
        target = resumed / "details/train_data/2.pt"
        training = torch.load(target, weights_only=False)
        training["samples"][0]["kl"].add_(0.1)
        torch.save(training, target)
    elif corruption == "gradient":
        torch.save(2.0, resumed / "grad-actor-2-0.pt")
    elif corruption == "training_text":
        target = resumed / "details/rollout_data/2.pt"
        rollout = torch.load(target, weights_only=False)
        rollout["samples"][1].update(response=r"\boxed{1}", reward=1)
        torch.save(rollout, target)
        target = resumed / "details/train_data/2.pt"
        training = torch.load(target, weights_only=False)
        training["samples"][1]["local_raw_reward"] = 1
        torch.save(training, target)
    if corruption != "missing_round":
        torch.save(data, path)
    with pytest.raises((AssertionError, FileNotFoundError)):
        audit(continuous, resumed, packet, inputs, "rltt", tokenizer)


def test_identical_outputs_with_zero_learning_signal_do_not_qualify(runs):
    continuous, resumed, packet, inputs, _, tokenizer = runs
    for run in (continuous, resumed):
        for step in range(3):
            rollout_path, train_path = run / f"details/rollout_data/{step}.pt", run / f"details/train_data/{step}.pt"
            rollout, trained = (torch.load(path, weights_only=False) for path in (rollout_path, train_path))
            for sample, batch in zip(rollout["samples"], trained["samples"], strict=True):
                sample.update(reward=0, response=r"\boxed{0}")
                sample["tokens"][-2] = 5
                batch["tokens"] = torch.tensor(sample["tokens"])
                batch["local_raw_reward"] = 0
                batch["advantages"].zero_()
                batch["returns"].zero_()
            torch.save(rollout, rollout_path)
            torch.save(trained, train_path)
            torch.save(0.0, run / f"grad-actor-{step}-0.pt")
    report = audit(continuous, resumed, packet, inputs, "rltt", tokenizer)
    assert report["output_equivalence_pass"] and not report["output_and_signal_pass"]
    assert report["joint_signal_updates"] == [] and report["reward_convergence"] == "NOT_ESTABLISHED"


def test_predeclared_score_tolerance_reports_small_finite_difference(runs):
    continuous, resumed, packet, inputs, _, tokenizer = runs
    path = resumed / "details/rollout_data/eval_3.pt"
    data = torch.load(path, weights_only=False)
    data["samples"][0]["rollout_log_probs"][0] += 1e-6
    torch.save(data, path)
    report = audit(continuous, resumed, packet, inputs, "rltt", tokenizer)
    assert report["output_and_signal_pass"]
    assert 0 < report["max_absolute_errors"]["heldout_scores"] < 2e-6
    assert report["numeric_tolerance"] == {"atol": 1e-5, "rtol": 3e-5}
