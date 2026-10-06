"""Freeze ordinary Ouro GRPO inputs for RFC465's separate-resource milestone."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.prepare_native_qualification import ENGINE_REVISION

SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SOURCE_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


def prepare(source: Path, output: Path) -> dict:
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Use the pinned official GSM8K training file")
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 7473
    indexed = [(f"gsm8k/train/{index:05d}", row) for index, row in enumerate(rows)]
    indexed.sort(key=lambda item: hashlib.sha256(("rfc465-ordinary-v1:" + item[0]).encode()).digest())
    selected = indexed[:20]
    assert len({row["question"] for _, row in selected}) == 20
    output.mkdir(parents=True, exist_ok=False)
    files, ids = {}, {}
    for split, items in (("train", selected[:12]), ("development", selected[12:])):
        records = []
        for identity, row in items:
            reasoning, separator, label = row["answer"].rpartition("#### ")
            assert reasoning and separator and label.strip()
            records.append(
                {
                    "prompt": row["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    "label": label.strip(),
                    "metadata": {"id": identity, "source_revision": SOURCE_REVISION},
                }
            )
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(records)}
        ids[split] = [row["metadata"]["id"] for row in records]
    profile = {
        "schema": "native-reward-qualification-v1",
        "acceptance_profile": "rfc465-ouro-grpo-separate-v1",
        "scope": "standard-entry three-update startup and fresh-process recovery; remaining RFC465 gates are separate",
        "source_repository": "https://github.com/openai/grade-school-math",
        "source_revision": SOURCE_REVISION,
        "source_train_sha256": SOURCE_SHA256,
        "official_test_split_consumed": False,
        "selection": "Identity-only hash order; no model outputs, rewards or answers used",
        "model_revision": MODEL_REVISION,
        "engine_revision": ENGINE_REVISION,
        "files": files,
        "selected_ids": ids,
        "resource_layout": "separate",
        "physical_gpus_required": 2,
        "attention_backend": "torch",
        "updates": 3,
        "split_stop_after": 2,
        "prompts_per_rollout": 4,
        "completions_per_prompt": 8,
        "prompt_limit": 1024,
        "response_limit": 2048,
        "eval_completions_per_prompt": 1,
        "seed": 42,
        "algorithms": ["grpo"],
        "precision": "fp32",
        "optimizer": {"name": "adam", "betas": [0.9, 0.999], "eps": 1e-8, "lr": 1e-6, "weight_decay": 0.1, "clip_grad": 0.1},
        "sampling": {"temperature": 0.9, "top_p": 1, "top_k": -1, "natural_eos": True},
        "reference": False,
        "critic": False,
        "kl_coefficient": 0,
        "fixed_depth": 4,
        "reward_convergence": "NOT_RUN",
    }
    (output / "qualification.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
