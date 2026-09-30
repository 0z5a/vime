import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def hashes(folder):
    metadata = dcp.FileSystemReader(folder).read_metadata()
    state = {name: torch.empty(value.size, dtype=value.properties.dtype) if isinstance(value, TensorStorageMetadata) else None for name, value in metadata.state_dict_metadata.items()}
    planner = DefaultLoadPlanner(flatten_state_dict=False)
    dcp.load(state, checkpoint_id=folder, planner=planner)
    state = planner.state_dict
    result = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
    print("READ", folder, len(result), flush=True)
    optimizer = {name: value for name, value in state.items() if "optimizer/shard_" in name}
    assert optimizer
    return result, optimizer


parser = argparse.ArgumentParser()
parser.add_argument("left", type=Path)
parser.add_argument("right", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
(left, left_optimizer), (right, right_optimizer) = hashes(args.left), hashes(args.right)
assert left and left.keys() == right.keys()
different = [name for name in left if left[name] != right[name]]
args.output.write_text(
    json.dumps(
        {
            "equal": not different,
            "tensor_count": len(left),
            "different": different,
            "hashes": left,
            "optimizer_metadata_equal": left_optimizer == right_optimizer,
            "optimizer_metadata": left_optimizer,
        },
        indent=2,
    )
)
assert not different, different
assert left_optimizer == right_optimizer
print("EXACT", len(left), flush=True)
