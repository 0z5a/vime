"""Read existing MCore optimizer classes; no worker, CUDA or environment mutation."""

import json

import torch
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig

config = OptimizerConfig(optimizer="adam", lr=1e-6, weight_decay=0.1)
print(json.dumps({
    "chained_class": f"{ChainedOptimizer.__module__}.{ChainedOptimizer.__qualname__}",
    "decoupled_weight_decay": config.decoupled_weight_decay,
    "cuda_initialized": torch.cuda.is_initialized(),
}))
