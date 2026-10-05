"""Check the complete prepared corpus through VIME's actual data and reward paths."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from benchmarks.prepare_math_data import REPAIRS, load_rows, prepare
from vime.rollout.rm_hub import async_rm
from vime.rollout.rm_hub.math_utils import grade_answer_verl
from vime.utils.data import Dataset


async def audit(raw: Path, prepared: Path) -> dict:
    manifest = json.loads((prepared / "manifest.json").read_text())
    groups, _ = load_rows(raw)
    source_rejections = [
        row.id for rows in groups.values() for row in rows if not grade_answer_verl(row.solution, row.label)
    ]
    if set(source_rejections) != json.loads(REPAIRS.read_text()).keys():
        raise ValueError(f"Unexpected source solution grading failures: {source_rejections}")
    args = Namespace(custom_rm_path=None, rm_type="math")
    accepted = {}
    for name, artifact in manifest["artifacts"].items():
        path = prepared / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Prepared artifact changed: {name}")
        dataset = Dataset(str(path), None, None, None, prompt_key="prompt", label_key="label")
        if [sample.metadata["id"] for sample in dataset.samples] != artifact["ids"]:
            raise ValueError(f"Data loader changed the row count or order: {name}")
        for sample in dataset.samples:
            sample.response = rf"\boxed{{{sample.label}}}"
            if await async_rm(args, sample) != 1:
                raise ValueError(f"Canonical answer rejected: {sample.metadata['id']}")
            sample.response = "No final boxed answer."
            if await async_rm(args, sample) != 0:
                raise ValueError(f"Missing answer accepted: {sample.metadata['id']}")
        accepted[name] = len(dataset)
    with tempfile.TemporaryDirectory(prefix="vime-math-repro-") as temporary:
        regenerated = Path(temporary)
        prepare(raw, regenerated, manifest["artifacts"]["development"]["rows"])
        for name in ["manifest.json", *(f"{name}.jsonl" for name in manifest["artifacts"])]:
            if (prepared / name).read_bytes() != (regenerated / name).read_bytes():
                raise ValueError(f"Regeneration is not byte-identical: {name}")
    return {
        "prepared_manifest_sha256": hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest(),
        "source_solutions_checked": sum(len(rows) for rows in groups.values()),
        "source_solution_rejections": source_rejections,
        "actual_dataset_rows": accepted,
        "canonical_answer_reward_one": sum(accepted.values()),
        "missing_box_reward_zero": sum(accepted.values()),
        "regeneration_byte_identical": True,
        "tokenizer_or_model_execution": False,
        "scope": "Known-label data/grader plumbing; not model accuracy or reward convergence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit(args.raw, args.prepared))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
