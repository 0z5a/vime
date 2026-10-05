"""Fetch immutable public MATH sources and verify their published content IDs."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NotRequired, TypedDict


class SourceFile(TypedDict):
    path: str
    size: int
    sha256: NotRequired[str]
    git_blob_sha1: NotRequired[str]


class Source(TypedDict):
    repository: str
    revision: str
    files: list[SourceFile]
    kind: NotRequired[Literal["dataset", "model"]]


SOURCES = Path(__file__).parent / "datasets/math/sources.json"


def verify(data: bytes, item: SourceFile) -> str:
    if len(data) != item["size"]:
        raise ValueError(f"Incorrect size for {item['path']}: {len(data)} != {item['size']}")
    sha256 = hashlib.sha256(data).hexdigest()
    if "sha256" in item:
        actual, expected = sha256, item["sha256"]
    else:
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        expected = item["git_blob_sha1"]
    if actual != expected:
        raise ValueError(f"Incorrect content hash for {item['path']}: {actual} != {expected}")
    return sha256


def download(root: Path, source: Source, item: SourceFile) -> dict[str, str | int]:
    relative = Path(source["repository"]) / source["revision"] / item["path"]
    target = root / relative
    prefix = "datasets/" if source.get("kind", "dataset") == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{source['repository']}/resolve/{source['revision']}/{item['path']}"
    if target.exists():
        data = target.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        verify(data, item)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    digest = verify(data, item)
    print(f"verified {relative} ({len(data)} bytes)", flush=True)
    return {"path": relative.as_posix(), "bytes": len(data), "sha256": digest, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    args = parser.parse_args()
    sources: list[Source] = json.loads(args.sources.read_text())
    jobs = [(source, item) for source in sources for item in source["files"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, args.output, source, item) for source, item in jobs]
        receipts = [future.result() for future in futures]
    (args.output / "download-manifest.json").write_text(
        json.dumps(
            {"source_lock_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(), "files": receipts}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
