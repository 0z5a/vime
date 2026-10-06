"""Reuse same-RECORD cached source for observed iCloud dependency evictions."""

import base64
import csv
import hashlib
import importlib.machinery
import linecache
import os
import runpy
import site
import sys
from pathlib import Path

origin = Path("/Users/0z5a/Documents/infra/looped-grpo-integration-20261003/.venv/lib/python3.12/site-packages")
mirror = Path("/Users/0z5a/Documents/infra/vime-sm120-initial-20260912/.venv/lib/python3.12/site-packages")
bundled = Path("/Users/0z5a/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages")
record = origin / "setuptools-84.0.0.dist-info/RECORD"
assert not record.stat().st_flags & 0x40000000
data = record.read_bytes()
assert hashlib.sha256(data).hexdigest() == "b236c7a32c86e59de7a293a435085617f920e6b5fffa4edbb7ac69c0bdcf9ca5"
recorded = {row[0]: row[1] for row in csv.reader(data.decode().splitlines())}
for name in ("pytest-9.1.1.dist-info/RECORD", "numpy-2.5.3.dist-info/RECORD", "torch-2.13.0.dist-info/RECORD"):
    path = origin / name
    assert not path.stat().st_flags & 0x40000000
    recorded.update({row[0]: row[1] for row in csv.reader(path.read_text().splitlines())})
original_data = importlib.machinery.SourceFileLoader.get_data
original_line = linecache.getline


def resident_data(loader, filename):
    path = Path(filename)
    if path.exists() and path.stat().st_flags & 0x40000000:
        relative = filename.removeprefix(str(origin) + os.sep)
        if relative in recorded and path.suffix == ".py":
            for cached in (mirror / relative, bundled / relative):
                if cached.is_file() and not cached.stat().st_flags & 0x40000000:
                    value = original_data(loader, str(cached))
                    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
                    if digest == recorded[relative]:
                        print(f"same-RECORD resident source: {relative}", flush=True)
                        return value
        raise OSError(61, "Source is not locally resident", filename)
    return original_data(loader, filename)


def resident_line(filename, number, module_globals=None):
    path = Path(filename)
    if path.exists() and path.stat().st_flags & 0x40000000:
        return ""
    return original_line(filename, number, module_globals)


def concise_error(kind, error, trace):
    while trace is not None:
        print(f"{trace.tb_frame.f_code.co_filename}:{trace.tb_lineno}", file=sys.stderr)
        trace = trace.tb_next
    print(f"{kind.__name__}: {error}", file=sys.stderr)


importlib.machinery.SourceFileLoader.get_data = resident_data
linecache.getline = resident_line
sys.excepthook = concise_error
site.main()
print(f"read-only runtime: {sys.executable}; no package update", flush=True)
if sys.argv[1] == "--script":
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
