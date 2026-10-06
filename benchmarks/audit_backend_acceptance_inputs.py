"""Audit all ordinary-Ouro startup prompts through the real Dataset/tokenizer."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from vime.utils.data import Dataset


def audit(packet: Path, metadata: Path) -> dict:
    profile_data = (packet / "qualification.json").read_bytes()
    profile = json.loads(profile_data)
    assert profile["acceptance_profile"] == "rfc465-ouro-grpo-separate-v1"
    source = json.loads((packet / "model-metadata-source.json").read_bytes())
    assert source["model_revision"] == profile["model_revision"]
    for row in source["files"]:
        data = (metadata / row["name"]).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    config = json.loads((metadata / "config.json").read_bytes())
    assert config["model_type"] == "ouro" and config["total_ut_steps"] == profile["fixed_depth"] == 4
    tokenizer = AutoTokenizer.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    records = []
    for split in ("train", "development"):
        path = packet / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == profile["files"][path.name]["sha256"]
        dataset = Dataset(
            str(path), tokenizer, None, None,
            prompt_key="prompt", label_key="label", apply_chat_template=True,
            apply_chat_template_kwargs={},
        )
        assert len(dataset) == profile["files"][path.name]["rows"]
        assert [sample.metadata["id"] for sample in dataset.samples] == profile["selected_ids"][split]
        for sample in dataset.samples:
            tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
            assert 0 < len(tokens) <= profile["prompt_limit"]
            records.append({
                "id": sample.metadata["id"], "split": split, "prompt_tokens": len(tokens),
                "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            })
    assert len(records) == len({row["id"] for row in records}) == 20
    assert not torch.cuda.is_initialized()
    return {
        "qualification_sha256": hashlib.sha256(profile_data).hexdigest(), "rows": records,
        "tokenizer_class": type(tokenizer).__name__, "python": sys.executable,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "all_inputs_preserved": True, "model_weights_opened": False, "cuda_initialized": False,
        "scope": "CPU input/tokenizer boundaries only; model generation, optimizer and reward convergence NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.packet, args.metadata)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
