"""Join synthetic raw runs with real CPU Adam/DCP and publication files."""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from benchmarks.audit_native_qualification import MODEL_FILES, audit, main
from benchmarks.native_publications import publication_index
from benchmarks.native_sources import module_index, sha, verify_local
from tests.test_native_checkpoint_audit import OPTIMIZER, trajectory
from vime.backends.vllm_rlt_utils.engine import weight_digest

pytest_plugins = ["tests.test_native_output_audit", "tests.test_native_source_membership"]


def replace_digest(run, step, digest):
    paths = [run / f"details/rollout_data/eval_{step}.pt"]
    if step < 3:
        paths.extend([run / f"details/rollout_data/{step}.pt", run / f"details/train_data/{step}.pt"])
    for path in paths:
        data = torch.load(path, weights_only=False)
        for sample in data["samples"]:
            if "recurrent_trace" in sample:
                sample["recurrent_trace"]["publication_digest"] = digest
            else:
                sample["recurrent_inputs"] = replace(sample["recurrent_inputs"], publication_digest=digest)
        torch.save(data, path)


@pytest.fixture(params=["learning"])
def chain(runs, frozen, tmp_path, request):
    continuous, resumed, packet, inputs, model, _ = runs
    manifest, roots, sources = frozen
    profile_path = packet / "qualification.json"
    profile = json.loads(profile_path.read_text())
    profile["optimizer"] = {"name": "adam", **OPTIMIZER}
    profile_path.write_text(json.dumps(profile))
    input_record = json.loads(inputs.read_text())
    input_record["qualification_sha256"] = sha(profile_path.read_bytes())
    inputs.write_text(json.dumps(input_record))
    (model / "config.json").write_text(json.dumps({"model_type": "ouro"}))
    # WordLevel controls use tokenizer.json; these two unused files stand in for
    # the official BPE asset names so the manifest has the full required set.
    (model / "merges.txt").write_text("# fixture\n")
    (model / "vocab.json").write_text("{}\n")
    modules = module_index(manifest)
    for run in (continuous, resumed):
        actor = run / "checkpoints/actor"
        actor.parent.mkdir()
        checkpoints, initial = trajectory(actor, request.param)
        (actor / "latest_checkpointed_iteration.txt").write_text("2")
        (actor / "rollout").mkdir()
        for step in range(3):
            torch.save(
                {
                    "sample_index": (step + 1) * 32,
                    "sample_group_index": (step + 1) * 4,
                    "sample_offset": (step + 1) * 4,
                    "epoch_id": 0,
                    "metadata": {},
                },
                actor / f"rollout/global_dataset_state_dict_{step}.pt",
            )
        shutil.copyfile(initial, model / "model.safetensors")
        for step in range(4):
            folder = run / "publications" / f"weight_v{step + 1:06d}"
            folder.mkdir(parents=True)
            with safe_open(initial, framework="pt", device="cpu") as weights:
                value = (
                    weights.get_tensor("model.weight")
                    if step == 0
                    else checkpoints[step - 1].full_tensor("model.weight")
                )
            save_file({"model.weight": value}, folder / "model.safetensors")
            digest, _ = publication_index(folder, {})
            assert digest == weight_digest(folder)[0]
            replace_digest(run, step, digest)
        for receipt_path in run.glob("*-process.json"):
            receipt = json.loads(receipt_path.read_text())
            receipt.update(
                qualification_sha256=sha(profile_path.read_bytes()),
                source_manifest_sha256=sha(sources.read_bytes()),
                source_preflight=verify_local(manifest, roots),
            )
            receipt_path.write_text(json.dumps(receipt))
        for record in (run / "runtime").glob("*/*.json"):
            row = json.loads(record.read_text())
            required = (
                ("vime_plugins.ouro.model", "megatron.core.optimizer.optimizer")
                if row["role"] == "actor"
                else ("vllm_rlt.models.ouro",)
            )
            for name in required:
                row["loaded_modules"][name] = "/" + modules[name]["path"]
            row["source_sha256"] = {name: modules[module]["sha256"] for module, name in row["loaded_modules"].items()}
            record.write_text(json.dumps(row))
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "revision": profile["model_revision"],
                "files": [
                    {"path": name, "bytes": (model / name).stat().st_size, "sha256": sha((model / name).read_bytes())}
                    for name in sorted(MODEL_FILES)
                ],
            }
        )
    )
    for run in (continuous, resumed):
        for path in run.glob("*-process.json"):
            receipt = json.loads(path.read_text())
            receipt.update(
                model_manifest_sha256=sha(model_manifest.read_bytes()),
                model_preflight={
                    "model": "/weights",
                    "files": {name: sha((model / name).read_bytes()) for name in MODEL_FILES},
                },
            )
            path.write_text(json.dumps(receipt))
    return continuous, resumed, packet, inputs, model, model_manifest, sources


@pytest.mark.parametrize("algorithm", ["grpo", "rltt"])
def test_complete_cli_reopens_all_evidence(chain, algorithm, tmp_path, monkeypatch):
    output = tmp_path / "qualification.json"
    argv = ["audit", "--algorithm", algorithm, "--output", str(output)]
    for name, path in zip(
        ("continuous", "resumed", "packet", "input-audit", "model", "model-manifest", "source-manifest"),
        chain,
        strict=True,
    ):
        argv.extend(["--" + name, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as finished:
        main()
    assert finished.value.code == 0
    report = json.loads(output.read_text())
    assert report["evidence_chain_pass"] and report["joint_useful_updates"] == [0, 1, 2]
    assert report["reward_convergence"] == "NOT_ESTABLISHED" and report["performance"] == "NOT_MEASURED"
    assert report["resource_handback"] == "REQUIRES_SEPARATE_PROCESS_AND_READER_AUDIT"
    assert all(len(row["rounds"]) == 4 for row in report["publications"])
    for name, digest in report["consumed_sha256"].items():
        assert sha(Path(name).read_bytes()) == digest


@pytest.mark.parametrize(
    "change",
    [
        "model_revision",
        "model_bytes",
        "tokenizer_bytes",
        "publication_bytes",
        "rehashed_wrong_update",
        "missing_tensor",
        "duplicate_tensor",
        "missing_version",
    ],
)
def test_rejects_unbound_model_and_publications(chain, change):
    continuous, resumed, packet, inputs, model, manifest, sources = chain
    if change == "model_revision":
        record = json.loads(manifest.read_text())
        record["revision"] = "unrelated"
        manifest.write_text(json.dumps(record))
    elif change in {"model_bytes", "tokenizer_bytes"}:
        path = model / ("model.safetensors" if change == "model_bytes" else "tokenizer.json")
        path.write_bytes(path.read_bytes() + b" ")
    else:
        for run in (continuous, resumed):
            folder = run / "publications/weight_v000003"
            path = folder / "model.safetensors"
            with safe_open(path, framework="pt", device="cpu") as weights:
                value = weights.get_tensor("model.weight").clone()
            if change == "missing_version":
                path.unlink()
            elif change == "missing_tensor":
                save_file({"unrelated.weight": value}, path)
            elif change == "duplicate_tensor":
                save_file({"model.weight": value}, folder / "duplicate.safetensors")
            else:
                save_file({"model.weight": value + 0.1}, path)
            if change in {"rehashed_wrong_update", "missing_tensor"}:
                replace_digest(run, 2, weight_digest(folder)[0])
    with pytest.raises(AssertionError):
        audit(*chain, "rltt")


@pytest.mark.parametrize("change", ["manifest", "preflight", "worker_path"])
def test_rejects_worker_model_binding(chain, change):
    run = chain[0]
    path = run / "continuous-process.json"
    if change == "worker_path":
        path = next((run / "runtime/continuous").glob("*.json"))
    record = json.loads(path.read_text())
    if change == "manifest":
        record["model_manifest_sha256"] = "0" * 64
    elif change == "preflight":
        record["model_preflight"]["files"]["model.safetensors"] = "0" * 64
    else:
        record["details"]["model_path"] = "/unverified-model"
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        audit(*chain, "rltt")


@pytest.mark.parametrize("chain", ["decay", "momentum"], indirect=True)
def test_no_shared_useful_update_retains_failed_report(chain, request):
    # Decay has no fresh Adam signal. Momentum only has a fresh signal at step0;
    # remove the raw RL signal there, leaving two individually positive gates
    # with no shared useful update.
    if request.node.callspec.params["chain"] == "momentum":
        for run in chain[:2]:
            torch.save(0.0, run / "grad-actor-0-0.pt")
    report = audit(*chain, "rltt")
    assert not report["evidence_chain_pass"] and report["joint_useful_updates"] == []
    assert report["reward_convergence"] == "NOT_ESTABLISHED"


def test_failed_chain_cli_returns_one_and_writes_report(chain, tmp_path, monkeypatch):
    for run in chain[:2]:
        for step in range(3):
            torch.save(0.0, run / f"grad-actor-{step}-0.pt")
    output = tmp_path / "failed.json"
    argv = ["audit", "--algorithm", "rltt", "--output", str(output)]
    for name, path in zip(
        ("continuous", "resumed", "packet", "input-audit", "model", "model-manifest", "source-manifest"),
        chain,
        strict=True,
    ):
        argv.extend(["--" + name, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as finished:
        main()
    assert finished.value.code == 1 and not json.loads(output.read_text())["evidence_chain_pass"]
