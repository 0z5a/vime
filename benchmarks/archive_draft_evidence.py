"""Stream immutable evidence into verified draft assets with bounded local disk."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

BLOCK = 8 * 1024**2
PART = 512 * 1024**2
REPO = "repos/0z5a/vime"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(BLOCK), b""):
            digest.update(data)
    return digest.hexdigest()


def upload(path: Path, name: str, release: dict, evidence: Path) -> dict:
    url = release["upload_url"].split("{", 1)[0] + "?" + urlencode({"name": name})
    result = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            url,
            "-H",
            "Content-Type: application/octet-stream",
            "--input",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    (evidence / (name + ".upload.json")).write_bytes(result.stdout)
    (evidence / (name + ".upload.log")).write_bytes(result.stderr)
    if result.returncode:
        return {"name": name, "upload_exit": result.returncode, "verified": False}
    asset = json.loads(result.stdout)
    expected = file_sha(path)
    with (evidence / (name + ".download.log")).open("wb") as log:
        process = subprocess.Popen(
            [
                "gh",
                "api",
                f"{REPO}/releases/assets/{asset['id']}",
                "-H",
                "Accept: application/octet-stream",
            ],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        downloaded = hashlib.sha256()
        size = 0
        for data in iter(lambda: process.stdout.read(BLOCK), b""):
            downloaded.update(data)
            size += len(data)
        process.stdout.close()
        code = process.wait()
    verified = code == 0 and size == asset["size"] == path.stat().st_size and asset["state"] == "uploaded" and asset["digest"] == "sha256:" + expected and downloaded.hexdigest() == expected
    return {
        "name": name,
        "asset_id": asset["id"],
        "bytes": asset["size"],
        "sha256": expected,
        "server_digest": asset["digest"],
        "download_sha256": downloaded.hexdigest(),
        "upload_exit": result.returncode,
        "download_exit": code,
        "verified": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--format", choices=("tar.gz", "tar"), default="tar.gz")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    observed = json.loads(subprocess.check_output(["gh", "api", f"{REPO}/releases/{release['id']}"], text=True))
    assert observed["draft"] and observed["tag_name"] == release["tag_name"]
    assert args.name.replace("-", "").isalnum()
    previous = json.loads(args.index.read_text()) if args.index.exists() else {"parts": []}
    if previous["parts"]:
        assert previous["release_id"] == release["id"] and previous["name"] == args.name
        assert previous.get("stream_format", "tar.gz") == args.format
    index = {
        "state": "STREAMING",
        "release_id": release["id"],
        "name": args.name,
        "stream_format": args.format,
        "part_bytes": PART,
        "parts": [],
    }
    stream = args.input.open("rb") if args.input else sys.stdin.buffer
    total = hashlib.sha256()
    ordinal, size, failed = 0, 0, False
    while True:
        path = args.index.parent / (args.name + f".pending-{ordinal:05d}")
        count = 0
        with path.open("wb") as part:
            while count < PART:
                data = stream.read(min(BLOCK, PART - count))
                if not data:
                    break
                total.update(data)
                count += len(data)
                size += len(data)
                if not failed:
                    part.write(data)
        if not count:
            path.unlink()
            break
        if failed:
            path.unlink()
            continue  # Drain the producer to its natural EOF after an upload failure.
        if ordinal < len(previous["parts"]):
            receipt = previous["parts"][ordinal]
            assert receipt["verified"] and receipt["bytes"] == count and receipt["sha256"] == file_sha(path)
        else:
            receipt = upload(
                path,
                args.name + f".{args.format}.part-{ordinal:05d}",
                release,
                args.index.parent,
            )
        failed = not receipt["verified"]
        index["parts"].append(receipt)
        args.index.write_text(json.dumps(index, indent=2) + "\n")
        print(
            json.dumps({"part": ordinal, "bytes": count, "verified": receipt["verified"]}),
            flush=True,
        )
        if not failed:
            path.unlink()
        ordinal += 1
    if args.input:
        stream.close()
    assert failed or ordinal >= len(previous["parts"]), "Input ended before the retained part index"
    index.update(
        state="UPLOAD_FAILED_SOURCE_RETAINED" if failed else "PARTS_OFFBOX_VERIFIED",
        stream_sha256=total.hexdigest(),
        stream_bytes=size,
        stream_eof=True,
        producer_exit="EXTERNAL_PRODUCER_RECEIPT_REQUIRED" if not args.input else "IMMUTABLE_LOCAL_INPUT",
    )
    args.index.write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "state": index["state"],
                "parts": len(index["parts"]),
                "stream_bytes": size,
            }
        ),
        flush=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
