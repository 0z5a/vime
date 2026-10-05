"""Run existing local dependencies, failing promptly on iCloud-evicted source.

No package is installed, replaced or stubbed. Evicted bytecode falls back to
its original source through Python's normal SourceFileLoader behavior.
"""

import base64
import csv
import hashlib
import importlib.machinery
import linecache
import os
import runpy
import sys

import pytest
import torch

original_get_data = importlib.machinery.SourceFileLoader.get_data
original_getline = linecache.getline
site = "/Users/0z5a/Documents/infra/looped-grpo-integration-20261003/.venv/lib/python3.12/site-packages/"
mirror = "/Users/0z5a/Documents/infra/vime-sm120-initial-20260912/.venv/lib/python3.12/site-packages/"
with open(site + "setuptools-84.0.0.dist-info/RECORD") as stream:
    recorded = {row[0]: row[1] for row in csv.reader(stream)}


def concise_exception(kind, error, trace):
    while trace is not None:
        print(f"{trace.tb_frame.f_code.co_filename}:{trace.tb_lineno}", file=sys.stderr)
        trace = trace.tb_next
    print(f"{kind.__name__}: {error}", file=sys.stderr)


def resident_line(filename, lineno, module_globals=None):
    if os.path.exists(filename) and os.stat(filename).st_flags & 0x40000000:
        return ""
    return original_getline(filename, lineno, module_globals)


def resident_data(loader, path):
    if os.path.exists(path) and os.stat(path).st_flags & 0x40000000:
        relative = path.removeprefix(site)
        other = mirror + relative
        if relative in recorded and path.endswith(".py") and os.path.isfile(other) and not os.stat(other).st_flags & 0x40000000:
            data = original_get_data(loader, other)
            digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            assert digest == recorded[relative], relative
            print(f"resident same-RECORD source: {other}", flush=True)
            return data
        raise OSError(61, "iCloud file is not resident", path)
    return original_get_data(loader, path)


importlib.machinery.SourceFileLoader.get_data = resident_data
linecache.getline = resident_line
sys.excepthook = concise_exception
import setuptools
print(f"torch={torch.__version__} {torch.__file__}", flush=True)
print(f"setuptools={setuptools.__version__} {setuptools.__file__}", flush=True)
if sys.argv[1:2] == ["--script"]:
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    raise SystemExit(pytest.main(sys.argv[1:]))
