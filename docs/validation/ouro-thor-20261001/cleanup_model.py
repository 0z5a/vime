"""Remove this experiment's completed model/checkpoint payloads after archiving evidence."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("size", choices=("tiny", "full"))
args = parser.parse_args()
root = Path("/home/jwipc/experiments/ouro-vime-20260930").resolve()
names = ("eager", "graph", "early", "async", "spec")
for name in names:
    marker = root / f"shared-{args.size}-{name}" / "complete-rank0.json"
    assert json.loads(marker.read_text())["completed"]
removed = []
for name in names:
    run = root / f"shared-{args.size}-{name}"
    targets = [*run.glob("checkpoint/iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]"), run / "hf-export"]
    for target in targets:
        if target.exists():
            assert not target.is_symlink() and target.resolve().is_relative_to(root)
            size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
            shutil.rmtree(target)
            removed.append({"path": str(target), "bytes": size})
model = root / "models" / ("Ouro-tiny" if args.size == "tiny" else "Ouro-1.4B-parallel")
assert not model.is_symlink() and model.resolve().is_relative_to(root)
size = sum(path.stat().st_size for path in model.rglob("*") if path.is_file())
shutil.rmtree(model)
removed.append({"path": str(model), "bytes": size})
if args.size == "tiny":
    for target in (root / "tiny-conversion" / "checkpoint" / "iter_0000002", root / "tiny-conversion" / "hf-export"):
        if target.exists():
            assert not target.is_symlink() and target.resolve().is_relative_to(root)
            size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
            shutil.rmtree(target)
            removed.append({"path": str(target), "bytes": size})
(root / f"shared-{args.size}-cleanup.json").write_text(json.dumps({"removed": removed}, indent=2))
print("REMOVED", sum(item["bytes"] for item in removed), "BYTES")
