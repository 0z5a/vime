"""Verify the retained Ouro model and tokenizer bytes without importing a runtime."""

import hashlib
import json
import sys
from pathlib import Path

from benchmarks.native_sources import read_resident

MODEL_FILES = {
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def verify_model(model: Path, manifest_path: Path, revision: str) -> dict[str, str]:
    manifest_data = read_resident(manifest_path)
    manifest = json.loads(manifest_data)
    assert manifest["revision"] == revision, "Model manifest uses a different revision"
    rows = manifest["files"]
    assert len(rows) == len(MODEL_FILES) and {row["path"] for row in rows} == MODEL_FILES
    consumed = {str(manifest_path.resolve()): hashlib.sha256(manifest_data).hexdigest()}
    for row in rows:
        path = model / row["path"]
        if sys.platform == "darwin" and path.stat().st_flags & 0x40000000:
            raise OSError(f"Model file is not locally resident: {path}")
        assert path.stat().st_size == row["bytes"], f"Changed model file size: {path.name}"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == row["sha256"], f"Changed model file: {path.name}"
        consumed[str(path.resolve())] = digest
    return consumed
