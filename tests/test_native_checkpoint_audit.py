"""Actual CPU AdamW/DCP serialization controls; no model or RL qualification."""

import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from benchmarks.audit_native_adam import audit_adam, main
from benchmarks.native_checkpoint import Checkpoint, compare_checkpoints, fingerprint
from tests.test_native_learning_signal_audit import evidence

OPTIMIZER = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.1}


def trajectory(root: Path, mode="learning", *, lr=1e-3):
    root.mkdir()
    torch.manual_seed(81)
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    initial = root / "initial.safetensors"
    save_file({"model.weight": parameter.detach().clone()}, initial)
    optimizer = torch.optim.AdamW([parameter], **{**OPTIMIZER, "lr": lr})
    checkpoints = []
    for i in range(3):
        parameter.grad = (
            torch.zeros_like(parameter)
            if mode == "decay" or (mode == "momentum" and i > 0)
            else torch.randn_like(parameter)
        )
        optimizer.step()
        state = optimizer.state[parameter]
        folder = root / f"iter_{i:07d}"
        dcp.save(
            {
                "model.weight": parameter.detach(),
                "optimizer.state.exp_avg.model.weight": state["exp_avg"],
                "optimizer.state.exp_avg_sq.model.weight": state["exp_avg_sq"],
                "rng_state/shard_0.0_1.1": [
                    {
                        "torch_rng_state": torch.get_rng_state(),
                        "random_rng_state": random.getstate(),
                        "np_rng_state": np.random.get_state(),
                    }
                ],
            },
            checkpoint_id=folder,
            no_dist=True,
            planner=dcp.DefaultSavePlanner(flatten_state_dict=False),
        )
        groups = [{key: value for key, value in optimizer.param_groups[0].items() if key != "params"}]
        torch.save(
            {
                "args": Namespace(decoupled_weight_decay=True),
                "iteration": i,
                "checkpoint_version": 3.0,
                "optimizer": {"state": {"common_step": state["step"]}, "param_groups": groups},
                "opt_param_scheduler": {"num_steps": (i + 1) * 32, "lr_decay_style": "constant"},
                "num_floating_point_operations_so_far": 0,
            },
            folder / "common.pt",
        )
        checkpoints.append(Checkpoint(folder))
    return checkpoints, initial


@pytest.mark.parametrize("mode,expected", [("learning", [0, 1, 2]), ("decay", []), ("momentum", [0])])
@pytest.mark.parametrize("lr", [1e-3, 1e-6])
def test_actual_adamw_distinguishes_new_signal_from_decay_and_old_momentum(tmp_path, mode, expected, lr):
    checkpoints, initial = trajectory(tmp_path / mode, mode, lr=lr)
    result = audit_adam(checkpoints, initial, {**OPTIMIZER, "lr": lr})
    assert result["useful_updates"] == expected
    # All modes move parameters, including the two negative controls.
    assert not torch.equal(checkpoints[0].full_tensor("model.weight"), checkpoints[2].full_tensor("model.weight"))
    report = compare_checkpoints(checkpoints[2], Checkpoint(checkpoints[2].folder))
    assert len(report["chunks"]) == 4 and report["stored_state_exact"]
    assert not report["fresh_worker_execution_proven"] and report["excluded"] == ["common.args"]


@pytest.mark.parametrize("field", ["model", "moment", "rng", "scheduler", "optimizer_step", "iteration"])
def test_rejects_corrupted_state_even_with_reusable_fingerprints(tmp_path, field):
    left, _ = trajectory(tmp_path / "left")
    right, _ = trajectory(tmp_path / "right")
    source, target = left[-1], right[-1]
    if field in ("model", "moment", "rng"):
        values = {key[0]: target.read(key) for key in target.index}
        if field == "rng":
            values["rng_state/shard_0.0_1.1"][0]["torch_rng_state"][0] ^= 1
        else:
            key = "model.weight" if field == "model" else "optimizer.state.exp_avg.model.weight"
            values[key] = values[key] + 0.1
        dcp.save(
            values, checkpoint_id=target.folder, no_dist=True, planner=dcp.DefaultSavePlanner(flatten_state_dict=False)
        )
    else:
        common = target.common
        if field == "scheduler":
            common["opt_param_scheduler"]["num_steps"] += 1
        elif field == "optimizer_step":
            common["optimizer"]["state"]["common_step"] += 1
        else:
            common["iteration"] += 1
        torch.save(common, target.folder / "common.pt")
    target = Checkpoint(target.folder)
    forged = {
        "schema_version": 1,
        "metadata_sha256": hashlib.sha256((source.folder / ".metadata").read_bytes()).hexdigest(),
        "common_sha256": hashlib.sha256((source.folder / "common.pt").read_bytes()).hexdigest(),
        "chunks": [
            {"name": key[0], "offsets": list(key[1]), "sha256": fingerprint(target.read(key))} for key in target.index
        ],
    }
    (source.folder / "fingerprints.json").write_text(json.dumps(forged))
    with pytest.raises(AssertionError):
        compare_checkpoints(source, target)


def test_missing_adam_and_rng_do_not_qualify(tmp_path):
    checkpoints, initial = trajectory(tmp_path / "checkpoints")
    checkpoint = checkpoints[0]
    checkpoint.index = {key: value for key, value in checkpoint.index.items() if not key[0].startswith("optimizer.")}
    with pytest.raises(AssertionError, match="No saved Adam"):
        audit_adam(checkpoints, initial, OPTIMIZER)
    checkpoint.index = {key: value for key, value in checkpoint.index.items() if not key[0].startswith("rng_state/")}
    with pytest.raises(AssertionError, match="RNG checkpoint is missing"):
        compare_checkpoints(checkpoint, checkpoint)


@pytest.mark.parametrize("value", [torch.tensor(float("nan")), np.array([float("inf")]), float("nan")])
def test_nonfinite_state_is_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        fingerprint(value)


@pytest.mark.parametrize("corruption", [None, "counters", "optimizer_mode"])
def test_combined_cli_reads_actual_state_and_both_raw_signal_runs(tmp_path, monkeypatch, corruption):
    runs = []
    for name in ("continuous", "resumed"):
        fixture = tmp_path / name
        fixture.mkdir()
        run, packet, inputs = evidence(fixture)
        actor = run / "checkpoints/actor"
        actor.parent.mkdir()
        checkpoints, initial = trajectory(actor)
        (actor / "latest_checkpointed_iteration.txt").write_text("2")
        (actor / "rollout").mkdir()
        for i in range(3):
            torch.save(
                {
                    "sample_index": (i + 1) * 32,
                    "sample_group_index": (i + 1) * 4,
                    "sample_offset": (i + 1) * 4,
                    "epoch_id": 0,
                    "metadata": {},
                },
                actor / f"rollout/global_dataset_state_dict_{i}.pt",
            )
        runs.append(run)
    profile = json.loads((packet / "qualification.json").read_text())
    profile.update(precision="fp32", optimizer={"name": "adam", **OPTIMIZER})
    (packet / "qualification.json").write_text(json.dumps(profile))
    input_record = json.loads(inputs.read_text())
    input_record["qualification_sha256"] = hashlib.sha256((packet / "qualification.json").read_bytes()).hexdigest()
    inputs.write_text(json.dumps(input_record))
    if corruption == "counters":
        path = runs[1] / "checkpoints/actor/rollout/global_dataset_state_dict_2.pt"
        state = torch.load(path, weights_only=False)
        state["sample_index"] -= 1
        torch.save(state, path)
    elif corruption == "optimizer_mode":
        for run in runs:
            path = run / "checkpoints/actor/iter_0000000/common.pt"
            state = torch.load(path, weights_only=False)
            del state["optimizer"]["param_groups"][0]["decoupled_weight_decay"]
            torch.save(state, path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_native_adam",
            "--continuous",
            str(runs[0]),
            "--resumed",
            str(runs[1]),
            "--packet",
            str(packet),
            "--initial-weights",
            str(initial),
            "--initial-sha256",
            hashlib.sha256(initial.read_bytes()).hexdigest(),
            "--input-audit",
            str(inputs),
            "--algorithm",
            "rltt",
            "--output",
            str(output),
        ],
    )
    if corruption:
        with pytest.raises(AssertionError, match="Rollout counters differ|explicitly identify AdamW"):
            main()
    else:
        with pytest.raises(SystemExit) as exit_code:
            main()
        assert exit_code.value.code == 0
        report = json.loads(output.read_text())
        assert report["joint_useful_updates"] == [0, 1, 2]
        assert report["reward_convergence"] == "NOT_ESTABLISHED"
        assert report["heldout_and_worker_identity"] == "REQUIRES_SEPARATE_AUDIT"
