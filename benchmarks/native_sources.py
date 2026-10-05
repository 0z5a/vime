"""Freeze native runtime files from Git objects and the retained MCore archive."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOTS = {
    "vime": ("vime", "vime_plugins", "train.py", "examples/looped_ppo/run.py"),
    "rlt": ("vllm_rlt",),
    "megatron": ("megatron",),
}
SCHEMA = "native-source-membership-v1"


def selected(source: str, path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in ROOTS[source])


def read_resident(path: Path) -> bytes:
    if sys.platform == "darwin" and path.stat().st_flags & 0x40000000:
        raise OSError(f"Source is not locally resident: {path}")
    return path.read_bytes()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(source: str, path: str, data: bytes) -> dict:
    return {"source": source, "path": path, "bytes": len(data), "sha256": sha(data)}


def git_source(repo: Path, revision: str, source: str) -> tuple[dict, list[dict]]:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = git("rev-parse", revision + "^{commit}").decode().strip()
    tree = git("rev-parse", commit + "^{tree}").decode().strip()
    files = []
    for entry in git("ls-tree", "-rz", commit, "--", *ROOTS[source]).split(b"\0"):
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"Expected a regular source file: {entry!r}")
        files.append(file_record(source, path_bytes.decode(), git("cat-file", "blob", blob)))
    return {"kind": "git", "commit": commit, "tree": tree}, files


def archive_source(archive: Path, receipt: Path) -> tuple[dict, list[dict]]:
    receipt_bytes = read_resident(receipt)
    recorded = json.loads(receipt_bytes)
    digest = sha(read_resident(archive))
    if digest != recorded["sha256"]:
        raise ValueError("MCore archive differs from the retained copy receipt")
    files = []
    with tarfile.open(archive) as stream:
        for member in stream:
            path = member.name.removeprefix("./")
            if not selected("megatron", path) or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Expected a regular MCore source file: {path}")
            with stream.extractfile(member) as contents:
                files.append(file_record("megatron", path, contents.read()))
    return {
        "kind": "archive",
        "sha256": digest,
        "receipt_sha256": sha(receipt_bytes),
        "recorded_base": recorded["base"],
        "recorded_patch": recorded["patch"],
    }, files


def module_index(manifest: dict) -> dict[str, dict]:
    if manifest["schema"] != SCHEMA or set(manifest["sources"]) != set(ROOTS):
        raise ValueError("Unknown native source manifest")
    paths, modules, present = set(), {}, set()
    for row in manifest["files"]:
        source, path = row["source"], row["path"]
        parts = PurePosixPath(path)
        if source not in ROOTS or parts.is_absolute() or ".." in parts.parts or str(parts) != path:
            raise ValueError(f"Invalid source path: {source}:{path}")
        if not selected(source, path) or (source, path) in paths:
            raise ValueError(f"Duplicate or unselected source: {source}:{path}")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]) or row["bytes"] < 0:
            raise ValueError(f"Invalid source digest or size: {source}:{path}")
        paths.add((source, path))
        present.add(source)
        if path.endswith(".py") and parts.parts[0] in {"vime", "vime_plugins", "vllm_rlt", "megatron"}:
            module = path[:-3].replace("/", ".").removesuffix(".__init__")
            if module in modules:
                raise ValueError(f"Duplicate module: {module}")
            modules[module] = row
    if present != set(ROOTS):
        raise ValueError("Each native source must contain files")
    for path in ("train.py", "examples/looped_ppo/run.py"):
        if ("vime", path) not in paths:
            raise ValueError(f"Missing qualification entrypoint: {path}")
    return modules


def verify_local(manifest: dict, roots: dict[str, Path]) -> dict:
    module_index(manifest)
    if set(roots) != set(ROOTS):
        raise ValueError("Supply exactly the vime, rlt and megatron source roots")
    counts = dict.fromkeys(ROOTS, 0)
    for row in manifest["files"]:
        root = roots[row["source"]].resolve()
        path = root / row["path"]
        if path.resolve() != path:
            raise ValueError(f"Source path contains a symlink: {path}")
        data = read_resident(path)
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise ValueError(f"Changed source: {row['source']}:{row['path']}")
        counts[row["source"]] += 1
    return {"files_checked": counts, "roots": {name: str(path.resolve()) for name, path in roots.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vime", type=Path, required=True)
    parser.add_argument("--vime-revision", required=True)
    parser.add_argument("--rlt", type=Path, required=True)
    parser.add_argument("--rlt-revision", required=True)
    parser.add_argument("--mcore-archive", type=Path, required=True)
    parser.add_argument("--mcore-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources, files = {}, []
    for name, repo, revision in (("vime", args.vime, args.vime_revision), ("rlt", args.rlt, args.rlt_revision)):
        sources[name], records = git_source(repo, revision, name)
        files.extend(records)
    sources["megatron"], records = archive_source(args.mcore_archive, args.mcore_receipt)
    files.extend(records)
    manifest = {
        "schema": SCHEMA,
        "sources": sources,
        "files": sorted(files, key=lambda row: (row["source"], row["path"])),
    }
    module_index(manifest)
    with args.output.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
