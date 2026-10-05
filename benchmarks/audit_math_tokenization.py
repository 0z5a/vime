"""Measure every prepared prompt with the pinned official Ouro tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.download_math_data import SOURCES, Source, verify
from vime.rollout.vllm_rlt_rollout import _tokenizer
from vime.utils.data import Dataset


def audit(prepared: Path, tokenizer_root: Path, output: Path) -> dict:
    lock = SOURCES.with_name("ouro_tokenizer.json")
    source: Source = json.loads(lock.read_text())[0]
    for item in source["files"]:
        verify((tokenizer_root / item["path"]).read_bytes(), item)
    tokenizer = _tokenizer(str(tokenizer_root))
    manifest = json.loads((prepared / "manifest.json").read_text())
    modes = {"plain": None, "chat": {"enable_thinking": False}, "thinking": {"enable_thinking": True}}
    summaries = {}
    records = []
    for split, artifact in manifest["artifacts"].items():
        path = prepared / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Prepared data changed: {split}")
        for mode, kwargs in modes.items():
            dataset = Dataset(
                str(path),
                tokenizer,
                None,
                None,
                prompt_key="prompt",
                label_key="label",
                apply_chat_template=kwargs is not None,
                apply_chat_template_kwargs=kwargs,
            )
            lengths, over_cap = [], []
            for sample in dataset.samples:
                tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
                count = len(tokens)
                lengths.append(count)
                record = {
                    "split": split,
                    "mode": mode,
                    "id": sample.metadata["id"],
                    "tokens": count,
                    "token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                }
                records.append(record)
                if count > 1024:
                    over_cap.append({"id": sample.metadata["id"], "tokens": count})
            ordered = sorted(lengths)
            summaries[f"{split}/{mode}"] = {
                "rows": len(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "p50": ordered[(len(ordered) - 1) // 2],
                "p95": ordered[(len(ordered) - 1) * 95 // 100],
                "p99": ordered[(len(ordered) - 1) * 99 // 100],
                "over_1024": over_cap,
            }
            print(f"{split}/{mode}: rows={len(lengths)}, max={max(lengths)}, over_1024={len(over_cap)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    excluded = {row["id"] for row in summaries["train/thinking"]["over_1024"]}
    training_lines = (prepared / "train.jsonl").read_text().splitlines()
    admitted = [line for line in training_lines if json.loads(line)["metadata"]["id"] not in excluded]
    training = "\n".join(admitted) + "\n"
    training_name = "train-ouro-thinking-p1024.jsonl"
    (output / training_name).write_text(training)
    rows = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    (output / "token-lengths.jsonl").write_text(rows)
    report = {
        "tokenizer_repository": source["repository"],
        "tokenizer_revision": source["revision"],
        "tokenizer_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "prepared_manifest_sha256": hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest(),
        "token_lengths_sha256": hashlib.sha256(rows.encode()).hexdigest(),
        "quantile_rule": "sorted[floor((n-1)*q)]",
        "add_special_tokens": False,
        "summaries": summaries,
        "training_profile": {
            "file": training_name,
            "rows": len(admitted),
            "sha256": hashlib.sha256(training.encode()).hexdigest(),
            "max_prompt_tokens": 1024,
            "apply_chat_template": True,
            "apply_chat_template_kwargs": {"enable_thinking": True},
            "excluded": summaries["train/thinking"]["over_1024"],
            "note": "Only this explicit training profile omits over-cap rows; full input artifacts remain intact",
        },
        "scope": "Actual tokenizer and Dataset, no model or rollout execution; no rows filtered",
    }
    (output / "tokenization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
