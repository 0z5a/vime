import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm_rlt.models.config import OuroConfig
from vllm_rlt.models.ouro import OuroForCausalLM

root = Path("/home/jwipc/experiments/ouro-vime-20260930")
source = root / "models/Ouro-1.4B-mirror"
target = root / "models/Ouro-tiny"
target.mkdir(exist_ok=True)
config = OuroConfig.tiny(
    vocab_size=49152,
    hidden_size=256,
    intermediate_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=2048,
)
values = json.loads((source / "config.json").read_text())
values.update(asdict(config))
(target / "config.json").write_text(json.dumps(values))
for name in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "configuration_ouro.py",
):
    shutil.copyfile(source / name, target / name)
torch.manual_seed(42)
model = OuroForCausalLM(config)
save_file(model.state_dict(), str(target / "model.safetensors"))
records = [
    {"prompt": f"What is {i}+{i}?", "label": str(i + i), "metadata": {"problem_id": str(i)}} for i in range(1, 9)
]
(root / "train.jsonl").write_text("\n".join(json.dumps(row) for row in records[:4]) + "\n")
(root / "eval.jsonl").write_text("\n".join(json.dumps(row) for row in records[4:]) + "\n")
(root / "smoke_reward.py").write_text("def score(response, label):\n    return sum(response.encode()) % 2\n")
print("PREPARED", target, flush=True)
