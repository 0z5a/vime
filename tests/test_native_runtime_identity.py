"""Real CPU implementations and process metadata, with explicit Ray/transport doubles."""

import ast
import hashlib
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vime.backends.vllm_rlt_utils.arguments import add_arguments
from vime.backends.vllm_rlt_utils.engine import NativeEngine
from vime.observability import native_runtime as runtime


@pytest.fixture
def context(monkeypatch):
    identity = dict(job="test-job", node="test-node", worker="test-worker", actor="test-actor")
    context = SimpleNamespace(
        get_job_id=lambda: identity["job"],
        get_node_id=lambda: identity["node"],
        get_worker_id=lambda: identity["worker"],
        get_actor_id=lambda: identity["actor"],
    )
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(__version__="context-double", get_runtime_context=lambda: context)
    )
    return identity


def test_reports_process_and_loaded_bytes_without_copying_model_state(tmp_path, monkeypatch, context):
    model = torch.nn.Linear(3, 2)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    module = ModuleType("vime.runtime_source_control")
    source = tmp_path / "source.py"
    source.write_text("CONTROL = 1\n")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    details = {"model": runtime.model_identity(model)}
    first = runtime.write_worker_report(str(tmp_path / "reports"), "actor", 0, details)
    second = runtime.write_worker_report(str(tmp_path / "reports"), "actor", 0, details)
    assert first != second and len(list(first.parent.glob("*.json"))) == 2
    report = json.loads(first.read_text())
    assert report["ray"] == context and report["pid"] == os.getpid()
    assert report["python"] == sys.executable and report["ray_version"] == "context-double"
    assert report["loaded_modules"][module.__name__] == str(source.resolve())
    assert report["source_sha256"][str(source.resolve())] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert details["model"]["state"]["weight"] == {"shape": [2, 3], "dtype": "torch.float32"}
    for path, digest in details["model"]["implementation"]["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["job", "node", "worker", "actor"])
def test_missing_actor_identity_writes_no_report(tmp_path, context, field):
    context[field] = None
    directory = tmp_path / "reports"
    with pytest.raises(RuntimeError, match="inside an actual Ray actor"):
        runtime.write_worker_report(str(directory), "actor", 0, {})
    assert not directory.exists()


@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
def test_captures_actual_optimizer_class_and_group_overrides(optimizer_cls):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = optimizer_cls([{"params": [parameter], "lr": 0.003, "weight_decay": 0.0}], lr=1e-6)
    config = SimpleNamespace(
        optimizer="adam",
        decoupled_weight_decay=optimizer_cls is torch.optim.AdamW,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        lr=1e-6,
        weight_decay=0.1,
        clip_grad=0.1,
        fp16=False,
        bf16=False,
        use_distributed_optimizer=False,
    )
    row = runtime.optimizer_identity([SimpleNamespace(config=config, optimizer=optimizer)])[0]
    assert row["implementation"]["class"] == f"{optimizer_cls.__module__}.{optimizer_cls.__qualname__}"
    assert row["effective_config"]["lr"] == 1e-6 and row["groups"][0]["settings"]["lr"] == 0.003
    assert row["groups"][0]["settings"]["weight_decay"] == 0
    assert row["groups"][0]["parameters"] == 1 and row["groups"][0]["numel"] == 3
    assert not optimizer.state and parameter.grad is None
    assert len(row["implementation"]["code_sha256"]) == 64


@pytest.mark.parametrize("enabled", [False, True])
def test_native_init_records_real_model_through_explicit_cpu_transport(tmp_path, monkeypatch, context, enabled):
    import vllm_rlt

    args = add_arguments(ArgumentParser()).parse_args(
        ["--rlt-model-revision", "model-pin", "--rlt-engine-revision", "engine-pin"]
    )
    assert args.rlt_runtime_report_dir is None
    args.hf_checkpoint, args.params_dtype, args.rlt_depth = str(tmp_path), torch.float32, 4
    directory = tmp_path / "reports"
    if enabled:
        args.rlt_runtime_report_dir = str(directory)
    model = torch.nn.Linear(3, 2)
    launches = []

    def cpu_transport(*positional, **kwargs):
        launches.append((positional, kwargs))
        return SimpleNamespace(engine=SimpleNamespace(model=model))

    monkeypatch.setattr(vllm_rlt, "LLM", cpu_transport)
    adapter = NativeEngine()
    adapter.init(args)
    assert launches[0][1]["device"] == "cuda"  # The double replaces construction, not the production device.
    assert not torch.cuda.is_initialized()
    if enabled:
        report = json.loads(next(directory.glob("*.json")).read_text())
        assert report["details"]["runtime_epoch"] == adapter.epoch
        assert report["details"]["declared_model_revision"] == "model-pin"
        assert report["details"]["model"]["implementation"]["class"] == "torch.nn.modules.linear.Linear"
    else:
        assert not directory.exists()


@pytest.mark.parametrize("backend,enabled", [("vllm", True), ("vllm-rlt", False), ("vllm-rlt", True)])
def test_actual_learner_function_records_after_load_without_importing_full_framework(monkeypatch, backend, enabled):
    source = Path(__file__).resolve().parents[1] / "vime/backends/megatron_utils/model.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_model_and_optimizer"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    events, models, optimizer, scheduler = [], [SimpleNamespace()], object(), object()

    def load(*args, **kwargs):
        events.append("load")
        return 1, 0

    def record(args, role, actual_models, actual_optimizer, iteration):
        assert actual_models is models and actual_optimizer is optimizer and iteration == 1
        events.append("record")

    monkeypatch.setattr(runtime, "record_learner", record)
    namespace = dict(
        torch=torch,
        setup_model_and_optimizer=lambda *args: (models, optimizer, scheduler),
        _critic_output_layer_needs_reinit=lambda *args: False,
        clear_memory=lambda: events.append("clear"),
        load_checkpoint=load,
    )
    exec(compile(module, str(source), "exec"), namespace)
    args = Namespace(rollout_backend=backend, rlt_runtime_report_dir="reports" if enabled else None)
    result = namespace["initialize_model_and_optimizer"](args)
    assert result == (models, optimizer, scheduler, 1)
    assert events == ["clear", "load", "clear"] + (["record"] if backend == "vllm-rlt" and enabled else [])
