"""Recover exact test input text into task storage without changing an environment."""

import base64
import csv
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVIDENCE = Path(__file__).parent.parent / "evidence"
TARGET = EVIDENCE / "resident-dependencies"
ROWS = json.loads((EVIDENCE / "huginn-resident-source-remaining.json").read_text())


def store(row, data, source):
    digest = hashlib.sha256(data).digest()
    assert "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=") == row["record"], row["path"]
    assert len(data) == row["bytes"], row["path"]
    target = TARGET / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return dict(row, source=source, sha256=digest.hex())


def torch_source(row):
    target = TARGET / row["path"]
    if target.is_file():
        return [store(row, target.read_bytes(), "https://github.com/pytorch/pytorch/blob/v2.13.0/" + row["path"])]
    info = json.loads(subprocess.check_output(
        ["gh", "api", "--method", "GET", "repos/pytorch/pytorch/contents/" + row["path"], "-f", "ref=v2.13.0"],
        text=True,
    ))
    return [store(row, base64.b64decode(info["content"]), info["html_url"])]


def wheel_sources(distribution):
    package, version = distribution.removesuffix(".dist-info").rsplit("-", 1)
    metadata = TARGET / (package + "-pypi.json")
    subprocess.run(["curl", "--fail", "--max-time", "30", "-sS", f"https://pypi.org/pypi/{package}/{version}/json", "-o", str(metadata)], check=True)
    wheels = [w for w in json.loads(metadata.read_text())["urls"] if w["packagetype"] == "bdist_wheel"]
    installed = Path("/Users/0z5a/Documents/infra/looped-grpo-integration-20261003/.venv/lib/python3.12/site-packages")
    tags = [line.removeprefix("Tag: ") for line in (installed / distribution / "WHEEL").read_text().splitlines() if line.startswith("Tag: ")]
    wheels = [w for w in wheels if "-".join(w["filename"].removesuffix(".whl").rsplit("-", 3)[1:]) in tags]
    wheel = min(wheels, key=lambda w: w["size"])
    archive = TARGET / wheel["filename"]
    subprocess.run(["curl", "--fail", "--max-time", "60", "-sS", wheel["url"], "-o", str(archive)], check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == wheel["digests"]["sha256"]
    with (installed / distribution / "RECORD").open() as stream:
        rows = [{"path": r[0], "record": r[1], "bytes": int(r[2]), "distribution": distribution}
                for r in csv.reader(stream) if r[0].endswith(".py") and r[1]]
    with zipfile.ZipFile(archive) as stream:
        return [store(row, stream.read(row["path"]), wheel["url"]) for row in rows]


def main():
    receipts = []
    distributions = {r["distribution"] for r in ROWS} - {"torch-2.13.0.dist-info", "ray-2.58.0.dist-info"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(wheel_sources, name) for name in sorted(distributions)]
        futures += [pool.submit(torch_source, row) for row in ROWS if row["distribution"] == "torch-2.13.0.dist-info"]
        errors = []
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                errors.append(str(error))
                print(f"Source verification failed: {error}", flush=True)
                continue
            receipts.extend(future.result())
            (EVIDENCE / "huginn-official-test-source-restoration.json").write_text(json.dumps(receipts, indent=2) + "\n")
            print(f"Verified {len(receipts)} exact source files; original environment unchanged", flush=True)
        assert not errors, errors


if __name__ == "__main__":
    main()
