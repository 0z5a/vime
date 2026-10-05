"""Synthetic startup/process records test audit rejection; no Ray worker is launched."""

import hashlib
import json
import sys

import pytest
import torch

from benchmarks.audit_native_runtime import audit, main
from benchmarks.prepare_native_qualification import EXECUTION_VARIANTS


@pytest.fixture
def records(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    profile = dict(
        updates=3,
        split_stop_after=2,
        precision="fp32",
        prompts_per_rollout=4,
        completions_per_prompt=8,
        model_revision="model-pin",
        engine_revision="engine-pin",
    )
    manifest = packet / "qualification.json"
    manifest.write_text(json.dumps(profile))
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    for phase, run, indices in (
        ("continuous", continuous, (0, 1, 2)),
        ("split", resumed, (0, 1)),
        ("resume", resumed, (2,)),
    ):
        directory = run / "runtime" / phase
        directory.mkdir(parents=True)
        (run / f"{phase}-process.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "returncode": 0,
                    "qualification_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                }
            )
        )
        for role in ("actor", "rollout"):
            modules = {
                "vime.observability.native_runtime": "/native_runtime.py",
                "vime.backends.megatron_utils.model"
                if role == "actor"
                else "vime.backends.vllm_rlt_utils.engine": f"/{role}.py",
            }
            details = dict(
                declared_model_revision="model-pin",
                declared_engine_revision="engine-pin",
                parameters_dtype="torch.float32",
                model_path="/weights",
            )
            if role == "actor":
                details.update(
                    loaded_checkpoint_iteration=1 if phase == "resume" else 0,
                    load_path="/resume" if phase == "resume" else "/weights",
                    parallelism={"tp": 1, "pp": 1, "cp": 1},
                    optimizers=[{"implementation": {"class": "control.AdamW"}}],
                    models=[{"implementation": {"class": "control.Model"}}],
                )
            else:
                details.update(runtime_epoch=phase, model={"implementation": {"class": "control.Model"}}, depth=4)
            row = dict(
                schema="native-runtime-identity-v1",
                role=role,
                rank=0,
                ray=dict(job=phase, node="test-node", worker=f"{phase}-{role}", actor=f"{phase}-{role}"),
                pid=10,
                process_start_ticks=100 + len(list(tmp_path.rglob("*actor.json"))) * 2 + (role == "rollout"),
                python="/test/python",
                python_version="test",
                torch_version="test",
                torch_cuda="test",
                ray_version="test",
                loaded_modules=modules,
                source_sha256={path: "a" * 64 for path in modules.values()},
                details=details,
            )
            (directory / f"{role}.json").write_text(json.dumps(row))
        (run / "details/rollout_data").mkdir(parents=True, exist_ok=True)
        for index in indices:
            samples = [
                {
                    "recurrent_trace": dict(
                        runtime_epoch=phase,
                        policy_version=index + 1,
                        model_revision="model-pin",
                        engine_revision="engine-pin",
                    )
                }
                for _ in range(32)
            ]
            torch.save({"rollout_id": index, "samples": samples}, run / f"details/rollout_data/{index}.pt")
    return continuous, resumed, packet


def test_cli_binds_all_six_synthetic_reports_and_records_input_hashes(records, tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--continuous",
            str(records[0]),
            "--resumed",
            str(records[1]),
            "--packet",
            str(records[2]),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["fresh_worker_evidence_pass"] and len(report["consumed_sha256"]) == 16
    assert report["phases"]["resume"]["rollouts"] == [2]
    assert report["reward_convergence"] == "NOT_ESTABLISHED"
    for path, digest in report["consumed_sha256"].items():
        from pathlib import Path

        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("variant", ["b-baseline", "b-prefix"])
@pytest.mark.parametrize("corrupt", [None, "missing", "dispatch", "pid", "generation", "samples", "remat", "nan"])
def test_b_contract_requires_matching_completed_dispatches(records, variant, corrupt):
    continuous, resumed, packet = records
    manifest = packet / "qualification.json"
    profile = json.loads(manifest.read_text())
    execution = dict(EXECUTION_VARIANTS[variant])
    profile.update(execution_variant=variant, learner_execution=execution)
    manifest.write_text(json.dumps(profile))
    paths = []
    for phase, run, indices in (
        ("continuous", continuous, (0, 1, 2)),
        ("split", resumed, (0, 1)),
        ("resume", resumed, (2,)),
    ):
        path = run / f"{phase}-process.json"
        receipt = json.loads(path.read_text())
        receipt["qualification_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        path.write_text(json.dumps(receipt))
        directory = run / "runtime" / phase
        path = directory / "actor.json"
        actor = json.loads(path.read_text())
        actor["details"]["learner_execution"] = execution
        path.write_text(json.dumps(actor))
        (directory / "steps").mkdir()
        for index in indices:
            step = {key: actor[key] for key in ("pid", "process_start_ticks", "rank")}
            step.update(
                rollout_id=index,
                step_id=0,
                schedule_completed=execution["actor_schedule"],
                actor_generation_before=index * 32,
                actor_generation_after=(index + 1) * 32,
                logical_samples=32,
                gradient_norm=0.1,
                learner_execution=execution,
            )
            path = directory / "steps" / f"{index}.json"
            path.write_text(json.dumps(step))
            paths.append(path)
    if corrupt == "missing":
        paths[-1].unlink()
    elif corrupt is not None:
        step = json.loads(paths[-1].read_text())
        if corrupt == "dispatch":
            step["schedule_completed"] = "mcore" if variant == "b-prefix" else "prefix"
        elif corrupt == "pid":
            step["pid"] += 1
        elif corrupt == "generation":
            step["actor_generation_after"] -= 1
        elif corrupt == "samples":
            step["logical_samples"] -= 1
        elif corrupt == "remat":
            step["learner_execution"]["recompute"] = True
        else:
            step["gradient_norm"] = float("nan")
        paths[-1].write_text(json.dumps(step))
    if corrupt is None:
        result = audit(*records)
        assert len(result["consumed_sha256"]) == 22
    else:
        with pytest.raises(AssertionError):
            audit(*records)


@pytest.mark.parametrize(
    "corruption",
    [
        "actor",
        "worker",
        "process",
        "source",
        "optimizer",
        "iteration",
        "revision",
        "environment",
        "job",
        "duplicate",
        "epoch",
        "unfinished",
    ],
)
def test_rejects_stale_or_mismatched_records(records, corruption):
    continuous, resumed, packet = records
    path = resumed / "runtime/resume/actor.json"
    row = json.loads(path.read_text())
    previous = json.loads((resumed / "runtime/split/actor.json").read_text())
    if corruption in ("actor", "worker"):
        row["ray"][corruption] = previous["ray"][corruption]
    elif corruption == "process":
        row["process_start_ticks"] = previous["process_start_ticks"]
    elif corruption == "source":
        row["source_sha256"]["/actor.py"] = "b" * 64
    elif corruption == "optimizer":
        row["details"]["optimizers"][0]["implementation"]["class"] = "control.SGD"
    elif corruption == "iteration":
        row["details"]["loaded_checkpoint_iteration"] = 0
    elif corruption == "revision":
        row["details"]["declared_model_revision"] = "wrong"
    elif corruption == "environment":
        row["torch_version"] = "changed"
    elif corruption == "job":
        row["ray"]["job"] = "unrelated-job"
    elif corruption == "duplicate":
        (path.parent / "duplicate.json").write_text(path.read_text())
    elif corruption == "epoch":
        rollout_path = resumed / "details/rollout_data/2.pt"
        data = torch.load(rollout_path, weights_only=False)
        data["samples"][0]["recurrent_trace"]["runtime_epoch"] = "split"
        torch.save(data, rollout_path)
    elif corruption == "unfinished":
        receipt = resumed / "resume-process.json"
        data = json.loads(receipt.read_text())
        data["returncode"] = None
        receipt.write_text(json.dumps(data))
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        audit(continuous, resumed, packet)
