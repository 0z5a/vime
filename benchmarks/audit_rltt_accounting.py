"""Execute unchanged RLTT config functions and report conditional source accounting.

This does not import or emulate the verl dataloader, DataProto, FSDP or optimizer.
"""

import __future__
import argparse
import ast
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
lock = json.loads((Path(__file__).parent / "datasets/math/rltt_source.json").read_text())
source = options.source / "rltt_experiments/rltt_train.py"
expected = next(item for item in lock["files"] if item["path"] == "rltt_experiments/rltt_train.py")
if hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
    raise ValueError("Pinned RLTT training source changed")
tree = ast.parse(source.read_text())
names = {"parse_args", "get_lora_target_modules_for_verl", "build_verl_config"}
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(definitions) == 3
namespace = {"__file__": str(source), "argparse": argparse, "os": os, "logger": logging.getLogger("source-audit")}
module = ast.Module(body=definitions, type_ignores=[])
exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)

records = []
for prompts, generations, accumulation in [(32, 8, 1), (32, 8, 2), (4, 8, 1)]:
    sys.argv = [
        str(source),
        "--n_gpus",
        "4",
        "--num_prompts_per_batch",
        str(prompts),
        "--num_generations",
        str(generations),
        "--gradient_accumulation_steps",
        str(accumulation),
        "--max_steps",
        "140",
        "--no_lora",
        "--no_sft_checkpoint",
        "--no_wandb",
    ]
    args = namespace["parse_args"]()
    config = namespace["build_verl_config"](args)
    data = config["data"]
    actor = config["actor_rollout_ref"]["actor"]
    group = config["actor_rollout_ref"]["rollout"]["n"]
    before = data.get("gen_batch_size", data["train_batch_size"])
    after = before * group
    per_rank_minibatch = actor["ppo_mini_batch_size"] * group // 4
    actor_calls = (140 + accumulation - 1) // accumulation
    steps_per_full_call = (after * accumulation // 4 + per_rank_minibatch - 1) // per_rank_minibatch
    records.append(
        {
            "cli_prompts": prompts,
            "generations": generations,
            "accumulation_rollouts": accumulation,
            "executed_config": {
                "data": data,
                "actor_minibatch": actor["ppo_mini_batch_size"],
                "ppo_epochs": actor["ppo_epochs"],
                "optimizer": actor["optim"],
            },
            "conditional_counts": {
                "loader_prompt_rows": before,
                "new_completions_per_rollout": after,
                "per_rank_minibatch_completions": per_rank_minibatch,
                "completions_per_full_actor_call": after * accumulation,
                "optimizer_steps_per_full_actor_call": steps_per_full_call,
                "rollout_batches": 140,
                "actor_calls": actor_calls,
                "total_new_completions": 140 * after,
                "optimizer_steps_for_140_full_rollouts": 140 * after // (4 * per_rank_minibatch),
            },
        }
    )
result = {
    "rltt_revision": "06189850bb23a1b1b715ad29768e68f36b1c4a19",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "executed": "Three unchanged source AST function definitions; explicit n_gpus avoids any Torch/CUDA call",
    "not_executed": ["dataloader", "DataProto", "FSDP", "model", "optimizer"],
    "assumptions": [
        "Published verl 0.6.1 semantics",
        "4 equal DP ranks, sequence parallel size 1",
        "No gen_batch_size override",
        "140 complete rollout batches are available",
        "No hidden private source or configuration modifications",
    ],
    "records": records,
}
options.output.write_text(json.dumps(result, indent=2) + "\n")
for record in records:
    print(json.dumps({k: v for k, v in record.items() if k != "executed_config"}), flush=True)
