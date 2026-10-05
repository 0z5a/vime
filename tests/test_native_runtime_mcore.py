"""Use existing MCore config/chain classes with CPU optimizer/model wrapper doubles."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from vime.observability import native_runtime as runtime

optimizer_module = pytest.importorskip("megatron.core.optimizer.optimizer")
config_module = pytest.importorskip("megatron.core.optimizer.optimizer_config")


@pytest.mark.parametrize("kind", ["none", "single", "chained"])
def test_learner_capture_preserves_mcore_defaults_and_loaded_iteration(tmp_path, monkeypatch, kind):
    model = torch.nn.Linear(3, 2)
    config = config_module.OptimizerConfig(optimizer="adam", lr=1e-6, weight_decay=0.1)
    component = SimpleNamespace(config=config, optimizer=torch.optim.AdamW(model.parameters(), lr=config.lr))
    optimizer = {"none": None, "single": component, "chained": optimizer_module.ChainedOptimizer([component])}[kind]
    args = Namespace(
        rlt_runtime_report_dir=str(tmp_path),
        rank=0,
        rlt_model_revision="model-pin",
        rlt_engine_revision="engine-pin",
        hf_checkpoint=str(tmp_path / "model"),
        load=str(tmp_path / "checkpoint"),
        params_dtype=torch.float32,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        loss_type="rltt_loss",
        rltt_prefix_wave_size=2,
        recompute_granularity=None,
        rltt_loop_checkpoint=0,
        rltt_layer_checkpoint=0,
        rltt_token_chunk=0,
    )
    captured = {}

    def writer(directory, role, rank, details):
        captured.update(directory=directory, role=role, rank=rank, details=details)
        return tmp_path / "not-written.json"

    monkeypatch.setattr(runtime, "write_worker_report", writer)
    runtime.record_learner(args, "actor", [SimpleNamespace(module=model)], optimizer, 1)
    details = captured["details"]
    assert captured["rank"] == 0 and captured["role"] == "actor"
    assert details["loaded_checkpoint_iteration"] == 1 and details["parallelism"] == {"tp": 1, "pp": 1, "cp": 1}
    assert details["models"][0]["state"]["weight"]["shape"] == [2, 3]
    assert details["learner_execution"] == {
        "actor_schedule": "prefix",
        "prefix_wave_size": 2,
        "recompute": False,
        "loop_checkpoint": 0,
        "layer_checkpoint": 0,
        "token_chunk": 0,
    }
    if kind == "none":
        assert details["optimizers"] == []
    else:
        actual = details["optimizers"][0]
        assert actual["effective_config"]["decoupled_weight_decay"] is True
        assert actual["implementation"]["class"] == "torch.optim.adamw.AdamW"
    assert not component.optimizer.state and not torch.cuda.is_initialized()
    assert not list(tmp_path.iterdir())
