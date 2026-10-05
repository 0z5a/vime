"""Run existing local dependencies, failing promptly on iCloud-evicted source.

No package is installed, replaced or stubbed. Evicted bytecode falls back to
its original source through Python's normal SourceFileLoader behavior.
"""

import base64
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import linecache
import os
import runpy
import sys
from pathlib import Path

original_get_data = importlib.machinery.SourceFileLoader.get_data
original_getline = linecache.getline
original_getlines = linecache.getlines
site = "/Users/0z5a/Documents/infra/looped-grpo-integration-20261003/.venv/lib/python3.12/site-packages/"
mirror = "/Users/0z5a/Documents/infra/vime-sm120-initial-20260912/.venv/lib/python3.12/site-packages/"
recorded = {}
for record in Path(site).glob("*.dist-info/RECORD"):
    if not record.stat().st_flags & 0x40000000:
        with record.open() as stream:
            recorded.update({row[0]: row[1] for row in csv.reader(stream) if row[1]})

cached_distributions = {}
cache_roots = []
for root in Path("/Users/0z5a/.cache/uv/archive-v0").iterdir():
    if root.is_dir():
        cache_roots.append(str(root) + "/")
        for folder in root.iterdir():
            if folder.name.endswith(".dist-info"):
                cached_distributions.setdefault(folder.name, []).append(folder)
for folder in (Path(__file__).parent.parent / "evidence/resident-dependencies").glob("*.dist-info"):
    cached_distributions.setdefault(folder.name, []).append(folder)
original_metadata_read = importlib.metadata.PathDistribution.read_text


def resident_metadata(distribution, filename):
    path = distribution._path / filename
    if path.exists() and path.stat().st_flags & 0x40000000:
        original_metadata = distribution._path / "METADATA"
        for folder in cached_distributions.get(distribution._path.name, []):
            candidates = (original_metadata, folder / "METADATA", folder / "RECORD", folder / filename)
            if all(p.is_file() and not p.stat().st_flags & 0x40000000 for p in candidates):
                if original_metadata.read_bytes() != (folder / "METADATA").read_bytes():
                    continue
                with (folder / "RECORD").open() as stream:
                    cache_record = {row[0]: row[1] for row in csv.reader(stream)}
                relative = folder.name + "/" + filename
                data = (folder / filename).read_bytes()
                digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                assert digest == cache_record[relative], relative
                print(f"resident matching-distribution metadata: {relative}", flush=True)
                return data.decode("utf-8")
        raise OSError(61, "iCloud metadata is not resident", str(path))
    return original_metadata_read(distribution, filename)


def concise_exception(kind, error, trace):
    while trace is not None:
        print(f"{trace.tb_frame.f_code.co_filename}:{trace.tb_lineno}", file=sys.stderr)
        trace = trace.tb_next
    print(f"{kind.__name__}: {error}", file=sys.stderr)


def resident_line(filename, lineno, module_globals=None):
    if os.path.exists(filename) and os.stat(filename).st_flags & 0x40000000:
        return ""
    return original_getline(filename, lineno, module_globals)


def resident_lines(filename, module_globals=None):
    if os.path.exists(filename) and os.stat(filename).st_flags & 0x40000000:
        return resident_data(None, filename).decode("utf-8").splitlines(keepends=True)
    return original_getlines(filename, module_globals)


def resident_data(loader, path):
    if os.path.exists(path) and os.stat(path).st_flags & 0x40000000:
        relative = path.removeprefix(site)
        for root in (mirror, str(Path(__file__).parent.parent / "evidence/resident-dependencies") + "/", *cache_roots):
            other = root + relative
            if relative in recorded and path.endswith(".py") and os.path.isfile(other) and not os.stat(other).st_flags & 0x40000000:
                data = original_get_data(loader, other)
                digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                if digest == recorded[relative]:
                    print(f"resident same-RECORD source: {other}", flush=True)
                    return data
        raise OSError(61, "iCloud file is not resident", path)
    return original_get_data(loader, path)


importlib.machinery.SourceFileLoader.get_data = resident_data
importlib.metadata.PathDistribution.read_text = resident_metadata
linecache.getline = resident_line
linecache.getlines = resident_lines
sys.excepthook = concise_exception
import pytest
import torch
import setuptools
print(f"torch={torch.__version__} {torch.__file__}", flush=True)
print(f"setuptools={setuptools.__version__} {setuptools.__file__}", flush=True)
if sys.argv[1:2] == ["--script"]:
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    raise SystemExit(pytest.main(sys.argv[1:]))
