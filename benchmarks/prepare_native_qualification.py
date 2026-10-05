"""Freeze a small, reward-independent MATH subset for actual native RL startup."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REVISION = "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"
ENGINE_REVISION = "fd993ec5512904b68e16f4d541682c076c487b5d"
SOURCE_MANIFEST = "a272932944329fa41b4da023b2d115606a1dbf3a364a1d1d300df6f251233b53"
TRAIN_SHA = "f3e52c440246c455b7a7fa089c63ab08f69fdac4a29bc9c828fdd567b0a72a90"
EXECUTION_VARIANTS = {
    name: {
        "actor_schedule": schedule,
        "prefix_wave_size": wave,
        "recompute": False,
        "loop_checkpoint": 0,
        "layer_checkpoint": 0,
        "token_chunk": 0,
    }
    for name, schedule, wave in (("b-baseline", "mcore", 0), ("b-prefix", "prefix", 2))
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path, variant: str = "legacy-remat") -> dict:
    if variant not in {"legacy-remat", *EXECUTION_VARIANTS}:
        raise ValueError("Unknown qualification execution variant")
    if sha(source / "manifest.json") != SOURCE_MANIFEST or sha(source / "train-p1024.jsonl") != TRAIN_SHA:
        raise ValueError("Qualification requires the frozen RLTT-source public input profile")
    manifest = json.loads((source / "manifest.json").read_text())
    original_dev = source / "development.jsonl"
    declared_dev = manifest["artifacts"]["development"]
    if sha(original_dev) != declared_dev["sha256"]:
        raise ValueError("Development split differs from the frozen manifest")
    output.mkdir(parents=True, exist_ok=False)
    files, ids = {}, {}
    for split, name, count in (("train", "train-p1024.jsonl", 12), ("development", "development.jsonl", 8)):
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        # Selection uses identity only, never model output, reward or an answer.
        rows.sort(
            key=lambda row: hashlib.sha256(("native-qualification-v1:" + row["metadata"]["id"]).encode()).digest()
        )
        selected = rows[:count]
        if len(selected) != count:
            raise ValueError("Insufficient rows for the frozen qualification")
        ids[split] = [row["metadata"]["id"] for row in selected]
        target = output / f"{split}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
        )
        files[target.name] = {"sha256": sha(target), "rows": count}
    assert not set(ids["train"]) & set(ids["development"])
    packet = {
        "schema": "native-reward-qualification-v1",
        "scope": "three-update startup and fresh-process recovery, not convergence",
        "source_manifest_sha256": SOURCE_MANIFEST,
        "source_train_sha256": TRAIN_SHA,
        "model_revision": MODEL_REVISION,
        "engine_revision": ENGINE_REVISION,
        "files": files,
        "selected_ids": ids,
        "updates": 3,
        "split_stop_after": 2,
        "prompts_per_rollout": 4,
        "completions_per_prompt": 8,
        "prompt_limit": 1024,
        "response_limit": 2048,
        "eval_completions_per_prompt": 1,
        "seed": 42,
        "algorithms": ["grpo", "rltt"],
        "precision": "fp32",
        "optimizer": {
            "name": "adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "lr": 1e-6,
            "weight_decay": 0.1,
            "clip_grad": 0.1,
        },
        "sampling": {"temperature": 0.9, "top_p": 1, "top_k": -1, "natural_eos": True},
        "rltt": {"alpha": 0, "reduction": "response_mean", "kl_coefficient": 0.001},
        "qualification_gates": [
            "actual native Ray/CUDA workers and declared sources",
            "all 96 new training completions and complete held-out rounds",
            "at least one mixed-reward group with nonzero advantages and finite nonzero gradient",
            "nonzero model update beyond weight decay and advanced Adam moments",
            "new committed policy after each update",
            "fresh worker restart with matching final model, optimizer, scheduler and RNG",
            "matching continued samples, rewards, traces and held-out results",
            "natural process completion and complete resource handback",
        ],
    }
    if variant != "legacy-remat":
        packet.update(
            algorithms=["rltt"],
            execution_variant=variant,
            learner_execution=dict(EXECUTION_VARIANTS[variant]),
        )
    (output / "qualification.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("legacy-remat", *EXECUTION_VARIANTS), default="legacy-remat")
    options = parser.parse_args()
    packet = prepare(options.source, options.output, options.variant)
    print(json.dumps({"files": packet["files"], "manifest_sha256": sha(options.output / "qualification.json")}))


if __name__ == "__main__":
    main()
