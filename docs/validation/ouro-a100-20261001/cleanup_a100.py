import argparse
import json
import shutil
from pathlib import Path

root = Path("/workspace/0z5a/work/ouro-vime-contract-20261001")
parser = argparse.ArgumentParser()
parser.add_argument("size", choices=("tiny", "full"))
args = parser.parse_args()
if args.size == "tiny":
    assert json.loads((root / "tiny-pd-matched-checkpoint.json").read_text())["equal"]
    assert json.loads((root / "tiny-pd-matched-hf.json").read_text())["equal"]
    names = [
        "train-tiny-local",
        "train-tiny-pd-graph",
        "train-tiny-pd-continuous",
        "train-tiny-pd-matched-eager",
        "train-tiny-pd-matched-graph",
    ]
else:
    assert json.loads((root / "full-pd-checkpoint.json").read_text())["equal"]
    assert json.loads((root / "full-pd-hf.json").read_text())["equal"]
    names = ["train-full-pd-eager", "train-full-pd-graph"]
folders = [root / "models" / ("Ouro-tiny" if args.size == "tiny" else "Ouro-1.4B")]
for name in names:
    run = root / name
    assert json.loads((run / "complete-rank0.json").read_text())["completed"]
    folders.extend((run / "checkpoint").glob("iter_[0-9]*"))
    if (run / "export").exists():
        folders.append(run / "export")
removed, size = [], 0
for folder in folders:
    assert not folder.is_symlink() and folder.resolve().is_relative_to(root)
    size += sum(p.stat().st_size for p in folder.rglob("*") if not p.is_symlink() and p.is_file())
    shutil.rmtree(folder)
    removed.append(str(folder.relative_to(root)))
(root / f"{args.size}-cleanup.json").write_text(json.dumps(dict(removed=removed, bytes=size), indent=2))
print("REMOVED", args.size, size, flush=True)
