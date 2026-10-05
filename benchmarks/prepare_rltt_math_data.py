"""Prepare a separate public MATH profile with the pinned RLTT source prompt.

The original converter function bodies run with real pandas/PyArrow. Selecting
functions avoids the package's reward import and its unrelated exit-time writer.
This does not execute verl, the source grader, or any model.
"""

import __future__
import argparse
import ast
import hashlib
import json
import logging
import os
import re
import runpy
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.download_math_data import SOURCES, verify
from benchmarks.prepare_math_data import load_rows, split_rows
from vime.rollout.vllm_rlt_rollout import _tokenizer
from vime.utils.data import Dataset

LOCK = SOURCES.with_name("rltt_source.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(root: Path) -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text())
    for item in lock["files"]:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"RLTT source changed: {item['path']}")
    namespace = runpy.run_path(str(root / "math_utils/prompting.py"))
    namespace.update(json=json, os=os, re=re, pd=pd, pa=pa, pq=pq, logger=logging.getLogger(__name__))
    for relative, names in [
        ("math_utils/answer_parsing.py", {"extract_boxed_answer", "get_gold_answer"}),
        ("rltt_experiments/data_utils.py", {"load_jsonl", "convert_math_to_parquet"}),
    ]:
        path = root / relative
        nodes = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if len(nodes) != len(names):
            raise ValueError(f"Expected source functions absent: {relative}")
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
            ),
            namespace,
        )
    return namespace, lock


def write_records(path: Path, records: list[dict]) -> dict:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))
    return {"file": path.name, "rows": len(records), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare(raw: Path, source: Path, tokenizer_root: Path, output: Path, evidence: Path) -> dict:
    upstream, source_lock = load_source(source)
    # Validate exactly the tokenizer inputs already frozen for the v1 profile.
    tokenizer_lock = SOURCES.with_name("ouro_tokenizer.json")
    tokenizer_source = json.loads(tokenizer_lock.read_text())[0]
    for item in tokenizer_source["files"]:
        verify((tokenizer_root / item["path"]).read_bytes(), item)
    tokenizer = _tokenizer(str(tokenizer_root))
    groups, receipts = load_rows(raw)
    splits, excluded = split_rows(groups, 256)
    if {name: len(rows) for name, rows in splits.items()} != {
        "train": 7498,
        "development": 256,
        "math500": 500,
        "reserve": 4244,
    }:
        raise ValueError("Pinned split counts changed")
    output.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    artifacts, token_records, over_cap = {}, [], {}
    for split, rows in splits.items():
        records = []
        for row in rows:
            record = row.record()
            record.update(
                prompt=upstream["build_chat_messages"](row.problem, use_few_shot=False),
                problem=row.problem,
                answer=row.label,
                subject=row.subject,
                level=row.level,
            )
            records.append(record)
        path = output / f"{split}.jsonl"
        artifacts[split] = write_records(path, records)
        parquet = output / f"{split}.parquet"
        upstream["convert_math_to_parquet"](str(path), str(parquet), use_few_shot=False)
        converted = pq.ParquetFile(parquet).read().to_pylist()
        dataset = Dataset(
            str(path),
            tokenizer,
            None,
            None,
            prompt_key="prompt",
            label_key="label",
            apply_chat_template=True,
            apply_chat_template_kwargs={},
        )
        over_cap[split] = []
        for row, original, sample in zip(rows, converted, dataset.samples, strict=True):
            expected = upstream["format_math_prompt"](row.problem, tokenizer, use_few_shot=False)
            rendered = tokenizer.apply_chat_template(original["prompt"], tokenize=False, add_generation_prompt=True)
            if rendered != expected or sample.prompt != expected:
                raise ValueError(f"Prompt mismatch: {row.id}")
            if (
                original["extra_info"]["problem"] != row.problem
                or original["reward_model"]["ground_truth"] != row.label
            ):
                raise ValueError(f"Converter question or label mismatch: {row.id}")
            if sample.label != row.label or sample.metadata["id"] != row.id:
                raise ValueError(f"VIME label or identity mismatch: {row.id}")
            tokens = tokenizer.encode(expected, add_special_tokens=False)
            if tokenizer.encode(sample.prompt, add_special_tokens=False) != tokens:
                raise ValueError(f"Token mismatch: {row.id}")
            token_records.append(
                {
                    "split": split,
                    "id": row.id,
                    "tokens": len(tokens),
                    "token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                    "label_sha256": hashlib.sha256(row.label.encode()).hexdigest(),
                    "problem_sha256": row.record()["metadata"]["problem_sha256"],
                }
            )
            if len(tokens) > 1024:
                over_cap[split].append({"id": row.id, "tokens": len(tokens)})
        artifacts[split]["parquet"] = {
            "file": parquet.name,
            "sha256": sha256(parquet),
            "bytes": parquet.stat().st_size,
        }
        if split == "train":
            omitted = {row["id"] for row in over_cap[split]}
            profile = [record for record in records if record["metadata"]["id"] not in omitted]
            artifacts["train_p1024"] = write_records(output / "train-p1024.jsonl", profile)
            profile_parquet = output / "train-p1024.parquet"
            upstream["convert_math_to_parquet"](str(output / "train-p1024.jsonl"), str(profile_parquet))
            selected = pq.ParquetFile(profile_parquet).read().to_pylist()
            expected_selected = [item for row, item in zip(rows, converted, strict=True) if row.id not in omitted]
            for index, item in enumerate(expected_selected):
                item["extra_info"]["index"] = index
            if selected != expected_selected:
                raise ValueError("Filtered source Parquet changed prompts or labels")
            artifacts["train_p1024"]["parquet"] = {
                "file": profile_parquet.name,
                "sha256": sha256(profile_parquet),
                "bytes": profile_parquet.stat().st_size,
            }
        print(
            f"{split}: {len(rows)} source/VIME prompt and label pairs match; over1024={len(over_cap[split])}",
            flush=True,
        )
    tokens_receipt = write_records(evidence / "token-identity.jsonl", token_records)
    manifest = {
        "schema": "flashrlt-rltt-source-prompt-v1",
        "rltt_revision": source_lock["revision"],
        "rltt_source_lock_sha256": sha256(LOCK),
        "dataset_source_lock_sha256": sha256(SOURCES),
        "label_repairs_sha256": sha256(SOURCES.with_name("label_repairs.json")),
        "sources": receipts,
        "tokenizer_revision": tokenizer_source["revision"],
        "tokenizer_lock_sha256": sha256(tokenizer_lock),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_eos_id": tokenizer.eos_token_id,
        "prompt_instruction": upstream["INSTRUCTION"],
        "use_few_shot": False,
        "apply_chat_template": True,
        "apply_chat_template_kwargs": {},
        "add_special_tokens": False,
        "tokenization": tokens_receipt,
        "over_prompt_cap_1024": over_cap,
        "excluded_training_rows": excluded,
        "repaired_training_rows": sorted(json.loads(SOURCES.with_name("label_repairs.json").read_text())),
        "artifacts": artifacts,
        "scope": "All 12498 public rows; original conversion functions and actual VIME Dataset/tokenizer; no model, grader, verl or RL execution",
        "data_identity": "Same public rows, repairs, order and split as math-v1; source instruction and default chat mode differ",
        "cap_policy": "Only train-p1024 omits listed over-cap rows. Full splits and all held-out rows retained. No prompt truncation.",
    }
    content = json.dumps(manifest, indent=2) + "\n"
    (output / "manifest.json").write_text(content)
    (evidence / "manifest.json").write_text(content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "source", "tokenizer", "output", "evidence"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.source, args.tokenizer, args.output, args.evidence)


if __name__ == "__main__":
    main()
