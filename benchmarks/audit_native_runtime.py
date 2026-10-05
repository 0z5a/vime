"""Bind native startup records to raw rollout epochs across a three-update resume."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from benchmarks.prepare_native_qualification import EXECUTION_VARIANTS


def audit(continuous: Path, resumed: Path, packet: Path) -> dict:
    consumed = {}

    def consume(path):
        consumed[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    profile = json.loads(consume(packet / "qualification.json").read_text())
    assert profile["updates"] == 3 and profile["split_stop_after"] == 2 and profile["precision"] == "fp32"
    roles, phases, epochs = {}, {}, set()
    actors, workers, processes, sources = set(), set(), set(), {}
    environment = None
    for phase, run, indices in (
        ("continuous", continuous, (0, 1, 2)),
        ("split", resumed, (0, 1)),
        ("resume", resumed, (2,)),
    ):
        receipt = json.loads(consume(run / f"{phase}-process.json").read_text())
        assert receipt["phase"] == phase and receipt["returncode"] == 0, "Phase did not finish naturally"
        assert receipt["qualification_sha256"] == consumed[str((packet / "qualification.json").resolve())]
        paths = sorted((run / "runtime" / phase).glob("*.json"))
        assert len(paths) == 2, "Require exactly one learner and one rollout startup per phase"
        records = [json.loads(consume(path).read_text()) for path in paths]
        assert sorted(row["role"] for row in records) == ["actor", "rollout"]
        phase_roles = {row["role"]: row for row in records}
        assert len({row["ray"]["job"] for row in records}) == 1, "Mixed Ray jobs within a phase"
        for row in records:
            assert row["schema"] == "native-runtime-identity-v1" and row["rank"] == 0
            assert all(row["ray"].values()) and row["pid"] > 0
            assert isinstance(row["process_start_ticks"], int) and row["process_start_ticks"] > 0
            actor, worker = row["ray"]["actor"], row["ray"]["worker"]
            process = (row["ray"]["node"], row["pid"], row["process_start_ticks"])
            assert actor not in actors and worker not in workers and process not in processes, "Reused worker identity"
            actors.add(actor)
            workers.add(worker)
            processes.add(process)
            current = {
                key: row[key] for key in ("python", "python_version", "torch_version", "torch_cuda", "ray_version")
            }
            if environment is None:
                environment = current
            assert current == environment, "Runtime environment changed"
            for module, path in row["loaded_modules"].items():
                digest = row["source_sha256"][path]
                assert len(digest) == 64
                if module in sources:
                    assert sources[module] == digest, f"Loaded source changed: {module}"
                sources[module] = digest
            required = (
                "vime.backends.megatron_utils.model"
                if row["role"] == "actor"
                else "vime.backends.vllm_rlt_utils.engine"
            )
            assert required in row["loaded_modules"] and "vime.observability.native_runtime" in row["loaded_modules"]
            details = row["details"]
            assert details["declared_model_revision"] == profile["model_revision"]
            assert details["declared_engine_revision"] == profile["engine_revision"]
            assert details["parameters_dtype"] == "torch.float32"
            stable = {
                key: value
                for key, value in details.items()
                if key not in {"loaded_checkpoint_iteration", "load_path", "runtime_epoch"}
            }
            if row["role"] in roles:
                assert stable == roles[row["role"]], "Model or effective optimizer contract changed"
            roles[row["role"]] = stable
        learner, rollout = phase_roles["actor"]["details"], phase_roles["rollout"]["details"]
        if "execution_variant" in profile:
            execution = EXECUTION_VARIANTS[profile["execution_variant"]]
            assert profile["learner_execution"] == learner["learner_execution"] == execution
            steps = [
                json.loads(consume(path).read_text())
                for path in sorted((run / "runtime" / phase / "steps").glob("*.json"))
            ]
            assert len(steps) == len(indices), "Missing or extra completed learner dispatches"
            assert {(step["rollout_id"], step["step_id"]) for step in steps} == {(i, 0) for i in indices}
            samples = profile["prompts_per_rollout"] * profile["completions_per_prompt"]
            for step in steps:
                assert all(step[key] == phase_roles["actor"][key] for key in ("pid", "process_start_ticks", "rank"))
                assert (
                    step["learner_execution"] == execution
                    and step["schedule_completed"] == execution["actor_schedule"]
                )
                assert step["logical_samples"] == samples
                assert step["actor_generation_before"] == step["rollout_id"] * samples
                assert step["actor_generation_after"] == (step["rollout_id"] + 1) * samples
                assert math.isfinite(step["gradient_norm"]) and step["gradient_norm"] >= 0
        assert learner["loaded_checkpoint_iteration"] == (1 if phase == "resume" else 0)
        assert learner["parallelism"] == {"tp": 1, "pp": 1, "cp": 1} and len(learner["optimizers"]) == 1
        epoch = rollout["runtime_epoch"]
        assert epoch and epoch not in epochs, "Reused native runtime epoch"
        epochs.add(epoch)
        for index in indices:
            path = consume(run / f"details/rollout_data/{index}.pt")
            data = torch.load(path, map_location="cpu", weights_only=False)
            assert data["rollout_id"] == index
            assert len(data["samples"]) == profile["prompts_per_rollout"] * profile["completions_per_prompt"]
            for sample in data["samples"]:
                trace = sample["recurrent_trace"]
                assert trace["runtime_epoch"] == epoch, "Rollout does not belong to its reported worker"
                assert trace["policy_version"] == index + 1
                assert (
                    trace["model_revision"] == profile["model_revision"]
                    and trace["engine_revision"] == profile["engine_revision"]
                )
        phases[phase] = {
            "runtime_epoch": epoch,
            "ray": {role: row["ray"] for role, row in phase_roles.items()},
            "rollouts": list(indices),
        }
    return {
        "fresh_worker_evidence_pass": True,
        "phases": phases,
        "environment": environment,
        "reported_module_sha256": sources,
        "optimizer_contract": roles["actor"]["optimizers"],
        "consumed_sha256": consumed,
        "scope": "consistency of trusted task-owned startup/process records and raw rollout epochs; not remote attestation",
        "weights_checkpoint_heldout": "REQUIRES_SEPARATE_AUDITS",
        "reward_convergence": "NOT_ESTABLISHED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.continuous, args.resumed, args.packet), indent=2) + "\n")


if __name__ == "__main__":
    main()
