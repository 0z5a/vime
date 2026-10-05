import hashlib
import inspect
import json
from pathlib import Path

import transformers
from transformers import AutoConfig

root = Path(__file__).resolve().parents[1]
model = root / "data/math-v1/model-config/ouro-thinking"
config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
source = Path(inspect.getfile(type(config)))
result = {
    "transformers_version": transformers.__version__,
    "transformers_source": transformers.__file__,
    "class": type(config).__name__,
    "loaded_configuration_source": str(source),
    "loaded_configuration_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "model_type": config.model_type,
    "hidden_size": config.hidden_size,
    "num_hidden_layers": config.num_hidden_layers,
    "num_attention_heads": config.num_attention_heads,
    "vocab_size": config.vocab_size,
    "scope": "Actual official AutoConfig only; no model, weight, GPU or full VIME parser execution",
}
(root / "evidence/ouro-autoconfig.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
