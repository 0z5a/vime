"""Compare a distributed training checkpoint with a converted HF export."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from vllm_rlt.models.ouro import OuroForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("export", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
model = OuroForCausalLM.from_pretrained(args.export, dtype=torch.bfloat16)
parameters = dict(model.named_parameters())
metadata = dcp.FileSystemReader(args.checkpoint).read_metadata()
assert parameters.keys() <= metadata.state_dict_metadata.keys()
state = {name: torch.empty_like(parameter) for name, parameter in parameters.items()}
dcp.load(state, checkpoint_id=args.checkpoint, planner=DefaultLoadPlanner(flatten_state_dict=False))
for name, parameter in parameters.items():
    torch.testing.assert_close(state[name], parameter, rtol=0, atol=0)
args.output.write_text(json.dumps({"equal": True, "tensors": len(parameters)}, indent=2))
print("EXACT HF EXPORT", len(parameters))
