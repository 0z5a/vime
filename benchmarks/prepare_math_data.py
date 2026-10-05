"""Prepare audited MATH training, development and MATH-500 held-out JSONL."""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as pq

from benchmarks.download_math_data import SOURCES, Source, verify
from vime.rollout.rm_hub.math_utils import extract_boxed_answer

REPAIRS = SOURCES.with_name("label_repairs.json")
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DEV_NAMESPACE = "flashrlt-math-development-v1"


class Repair(TypedDict):
    solution_sha256: str
    label: str
    reason: str


@dataclass(frozen=True)
class Row:
    id: str
    problem: str
    solution: str
    label: str
    subject: str
    level: int | None
    source_file: str
    source_row: int
    original_id: str
    repaired: bool

    @property
    def key(self) -> str:
        return problem_key(self.problem)

    def record(self) -> dict:
        return {
            "prompt": self.problem + INSTRUCTION,
            "label": self.label,
            "metadata": {
                "id": self.id,
                "rm_type": "math",
                "subject": self.subject,
                "level": self.level,
                "source_file": self.source_file,
                "source_row_zero_based": self.source_row,
                "original_id": self.original_id,
                "problem_sha256": digest(self.problem),
                "normalized_problem_sha256": self.key,
                "solution_sha256": digest(self.solution),
                "label_repaired": self.repaired,
            },
        }


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def problem_key(problem: str) -> str:
    # NFC preserves mathematical compatibility characters, such as superscripts.
    return digest(" ".join(unicodedata.normalize("NFC", problem).split()))


def source_level(value: str) -> int | None:
    return None if value == "Level ?" else int(value.removeprefix("Level "))


def source_label(identity: str, solution: str, repairs: dict[str, Repair]) -> str:
    if identity in repairs:
        repair = repairs[identity]
        if digest(solution) != repair["solution_sha256"]:
            raise ValueError(f"Label repair source changed: {identity}")
        label = repair["label"]
    else:
        label = extract_boxed_answer(solution)
    if not label or not label.strip():
        raise ValueError(f"Missing label: {identity}; add an explicit, source-bound repair")
    return label


def load_rows(root: Path) -> tuple[dict[str, list[Row]], list[dict]]:
    sources: list[Source] = json.loads(SOURCES.read_text())
    repairs: dict[str, Repair] = json.loads(REPAIRS.read_text())
    groups: dict[str, list[Row]] = {name: [] for name in ("train", "test", "math500")}
    receipts = []
    for source in sources:
        for item in source["files"]:
            relative = f"{source['repository']}/{source['revision']}/{item['path']}"
            path = root / relative
            receipts.append({"path": relative, "sha256": verify(path.read_bytes(), item), "bytes": item["size"]})
            if item["path"].endswith(".parquet"):
                subject = path.parent.name
                split = path.name.split("-")[0]
                for index, raw in enumerate(pq.ParquetFile(path).read().to_pylist()):
                    identity = f"{subject}/{split}/{index}"
                    groups[split].append(
                        Row(
                            identity,
                            raw["problem"],
                            raw["solution"],
                            source_label(identity, raw["solution"], repairs),
                            raw["type"],
                            source_level(raw["level"]),
                            relative,
                            index,
                            "",
                            identity in repairs,
                        )
                    )
            elif item["path"] == "test.jsonl":
                for index, line in enumerate(path.read_text().splitlines()):
                    raw = json.loads(line)
                    if extract_boxed_answer(raw["solution"]) != raw["answer"]:
                        raise ValueError(f"MATH-500 label disagrees with its solution: {index}")
                    groups["math500"].append(
                        Row(
                            f"math500/{index}",
                            raw["problem"],
                            raw["solution"],
                            raw["answer"],
                            raw["subject"],
                            raw["level"],
                            relative,
                            index,
                            raw["unique_id"],
                            False,
                        )
                    )
    used = {row.id for rows in groups.values() for row in rows if row.repaired}
    if used != repairs.keys():
        raise ValueError(f"Repair identities do not match sources: {used ^ repairs.keys()}")
    return groups, receipts


def split_rows(groups: dict[str, list[Row]], development_size: int) -> tuple[dict[str, list[Row]], list[dict]]:
    test = {row.key: row for row in groups["test"]}
    heldout = {row.key: row for row in groups["math500"]}
    if len(test) != len(groups["test"]) or len(heldout) != len(groups["math500"]):
        raise ValueError("Held-out source contains duplicate normalized problems")
    if not heldout.keys() <= test.keys():
        raise ValueError("MATH-500 contains a question outside the pinned MATH test split")
    train: dict[str, Row] = {}
    excluded = []
    for row in groups["train"]:
        if row.key in test:
            excluded.append({"id": row.id, "reason": "test_overlap", "matches": test[row.key].id})
        elif row.key in train:
            prior = train[row.key]
            if row.label != prior.label:
                raise ValueError(f"Duplicate training question has conflicting labels: {row.id}, {prior.id}")
            excluded.append({"id": row.id, "reason": "train_duplicate", "matches": prior.id})
        else:
            train[row.key] = row
    candidates = [row for row in groups["test"] if row.key not in heldout]
    ordered = sorted(candidates, key=lambda row: (digest(f"{DEV_NAMESPACE}:{row.key}"), row.id))
    if not 0 < development_size <= len(ordered):
        raise ValueError("Development size must fit the non-MATH-500 test pool")
    return {
        "train": list(train.values()),
        "development": ordered[:development_size],
        "math500": groups["math500"],
        "reserve": ordered[development_size:],
    }, excluded


def prepare(root: Path, output: Path, development_size: int = 256) -> dict:
    groups, sources = load_rows(root)
    if {name: len(rows) for name, rows in groups.items()} != {"train": 7500, "test": 5000, "math500": 500}:
        raise ValueError("Pinned dataset row counts changed")
    splits, excluded = split_rows(groups, development_size)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in splits.items():
        content = "".join(json.dumps(row.record(), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        (output / f"{name}.jsonl").write_text(content)
        artifacts[name] = {
            "rows": len(rows),
            "sha256": digest(content),
            "bytes": len(content.encode()),
            "ids": [row.id for row in rows],
            "subjects": dict(sorted(Counter(row.subject for row in rows).items())),
            "levels": dict(
                sorted(Counter(str(row.level) if row.level is not None else "unknown" for row in rows).items())
            ),
        }
    manifest = {
        "schema": "flashrlt-math-inputs-v1",
        "source_lock_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
        "label_repairs_sha256": hashlib.sha256(REPAIRS.read_bytes()).hexdigest(),
        "sources": sources,
        "source_rows": {name: len(rows) for name, rows in groups.items()},
        "prompt_instruction": INSTRUCTION,
        "deduplication": "NFC then Unicode whitespace collapse, case-sensitive",
        "development_selection": f"lowest SHA256({DEV_NAMESPACE}:normalized_problem_sha256), then source id",
        "excluded_training_rows": excluded,
        "repaired_training_rows": sorted(json.loads(REPAIRS.read_text())),
        "math500_source_labels_exact": True,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.raw, args.output)
    print(
        json.dumps(
            {
                "counts": {k: v["rows"] for k, v in manifest["artifacts"].items()},
                "excluded": manifest["excluded_training_rows"],
                "repairs": manifest["repaired_training_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
