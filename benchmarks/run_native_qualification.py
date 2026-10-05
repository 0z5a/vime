"""Launch one finite qualification phase on an already reserved Ray cluster."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def command(packet: Path, model: Path, output: Path, address: str, algorithm: str, phase: str) -> list[str]:
    if not address or address == "local":
        raise ValueError("Use an existing reserved Ray cluster; this launcher does not start one")
    profile = json.loads((packet / "qualification.json").read_text())
    if profile["schema"] != "native-reward-qualification-v1" or algorithm not in profile["algorithms"]:
        raise ValueError("Unknown qualification profile or algorithm")
    for name, record in profile["files"].items():
        if hashlib.sha256((packet / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed qualification data: {name}")
    run = output / algorithm / ("continuous" if phase == "continuous" else "resumed")
    optimizer = profile["optimizer"]
    values = {
        "model": model,
        "data": packet / "train.jsonl",
        "output": run,
        "model-revision": profile["model_revision"],
        "engine-revision": profile["engine_revision"],
        "ray-address": address,
        "algorithm": algorithm,
        "precision": profile["precision"],
        "updates": profile["updates"],
        "rollout-batch-size": profile["prompts_per_rollout"],
        "n-samples-per-prompt": profile["completions_per_prompt"],
        "rollout-max-prompt-len": profile["prompt_limit"],
        "rollout-max-response-len": profile["response_limit"],
        "seq-length": profile["prompt_limit"] + profile["response_limit"],
        "rollout-temperature": profile["sampling"]["temperature"],
        "rollout-top-p": profile["sampling"]["top_p"],
        "rollout-top-k": profile["sampling"]["top_k"],
        "rlt-kv-blocks": 1536,
        "rlt-max-num-seqs": 2,
        "optimizer": optimizer["name"],
        "adam-beta1": optimizer["betas"][0],
        "adam-beta2": optimizer["betas"][1],
        "adam-eps": optimizer["eps"],
        "lr": optimizer["lr"],
        "weight-decay": optimizer["weight_decay"],
        "clip-grad": optimizer["clip_grad"],
        "lr-decay-iters": profile["updates"],
        "seed": profile["seed"],
        "rollout-seed": profile["seed"],
        "save-interval": 1,
        "eval-interval": 1,
        "n-samples-per-eval-prompt": profile["eval_completions_per_prompt"],
        "eval-temperature": 0,
        "eval-max-prompt-len": profile["prompt_limit"],
        "eval-max-response-len": profile["response_limit"],
        "eval-max-context-len": profile["prompt_limit"] + profile["response_limit"],
        "dump-details": run / "details",
        "rlt-runtime-report-dir": run / "runtime" / phase,
        "ci-save-grad-norm": run / "grad-{role}-{rollout_id}-{step_id}.pt",
    }
    if algorithm == "rltt":
        values.update(
            {
                "rltt-progressive-alpha": profile["rltt"]["alpha"],
                "rltt-reduction": profile["rltt"]["reduction"],
                "kl-loss-coef": profile["rltt"]["kl_coefficient"],
                "rltt-loop-checkpoint": 1,
                "rltt-layer-checkpoint": 1,
                "rltt-token-chunk": 256,
            }
        )
    if phase == "split":
        values["stop-after"] = profile["split_stop_after"]
    argv = [sys.executable, str(Path(__file__).resolve().parents[1] / "examples/looped_ppo/run.py")]
    for name, value in values.items():
        argv.extend(("--" + name, str(value)))
    argv.extend(
        (
            "--colocate-resident",
            "--recompute",
            "--apply-chat-template",
            "--eval-prompt-data",
            "qualification-development",
            str(packet / "development.jsonl"),
        )
    )
    if phase == "resume":
        argv.append("--resume")
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--algorithm", choices=("grpo", "rltt"), required=True)
    parser.add_argument("--phase", choices=("continuous", "split", "resume"), required=True)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    packet, model, output = (path.resolve() for path in (options.packet, options.model, options.output))
    argv = command(packet, model, output, options.ray_address, options.algorithm, options.phase)
    if not options.execute:
        print(json.dumps({"executed": False, "argv": argv}, indent=2))
        return
    run = output / options.algorithm / ("continuous" if options.phase == "continuous" else "resumed")
    profile_sha = hashlib.sha256((packet / "qualification.json").read_bytes()).hexdigest()
    if options.phase == "resume":
        split = json.loads((run / "split-process.json").read_text())
        if split["returncode"] != 0 or split["qualification_sha256"] != profile_sha:
            raise ValueError("The split phase must have completed naturally before resume")
        if (run / "checkpoints/actor/latest_checkpointed_iteration.txt").read_text().strip() != "1":
            raise ValueError("Resume requires the completed two-update split checkpoint")
    else:
        run.mkdir(parents=True, exist_ok=False)
    receipt = {"argv": argv, "phase": options.phase, "started_ns": time.time_ns(), "qualification_sha256": profile_sha}
    receipt_path = run / f"{options.phase}-process.json"
    with (run / f"{options.phase}.log").open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        receipt["pid"] = process.pid
        receipt["returncode"] = None
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["returncode"] = process.wait()
    receipt["finished_ns"] = time.time_ns()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(receipt["returncode"])


if __name__ == "__main__":
    main()
