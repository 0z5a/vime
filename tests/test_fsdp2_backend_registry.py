"""CPU tests for training-backend selection and the fsdp2 backend's own flags."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

_tests_root = Path(__file__).resolve().parent
for path in (_tests_root, _tests_root.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _unit_stubs

_unit_stubs.install_vllm_cli_stubs()

from vime.backends import registry
from vime.backends.fsdp_utils.arguments import validate_fsdp_args

NUM_GPUS = 0

TRAIN_ACTOR_METHODS = (
    "init",
    "train",
    "save_model",
    "update_weights",
    "sleep",
    "wake_up",
    "clear_memory",
    "set_rollout_manager",
    "compute_log_prob",
    "load_other_checkpoint",
    "get_master_addr_and_port",
)


def _pre_parse(monkeypatch, *argv):
    from vime.utils.arguments import _pre_parse_mode

    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return _pre_parse_mode()


def test_registered_backends():
    assert registry.train_backend_names()[:2] == ["megatron", "fsdp2"]
    with pytest.raises(ValueError, match="Unknown training backend"):
        registry.get_train_actor_cls("fsdp")


def test_fsdp2_actor_resolves_lazily_and_matches_the_train_group_contract():
    module = "vime.backends.fsdp_utils.actor"
    sys.modules.pop(module, None)
    registry.train_backend_names()
    registry.parse_train_backend_args("fsdp2", [])
    assert module not in sys.modules, "listing backends or parsing flags must not import the actor"

    cls = registry.get_train_actor_cls("fsdp2")
    assert cls.__name__ == "FSDPTrainRayActor"
    missing = [name for name in TRAIN_ACTOR_METHODS if not callable(getattr(cls, name, None))]
    assert not missing, missing


def test_pre_parse_merges_fsdp2_flags(monkeypatch):
    args = _pre_parse(
        monkeypatch,
        "--train-backend",
        "fsdp2",
        "--lr",
        "1e-6",
        "--fsdp-attn-implementation",
        "sdpa",
        "--fsdp-cpu-offload",
        "--no-fsdp-reshard-after-forward",
        "--fsdp-param-dtype",
        "fp32",
    )
    assert args.train_backend == "fsdp2"
    assert args.fsdp_attn_implementation == "sdpa"
    assert args.fsdp_cpu_offload is True
    assert args.fsdp_reshard_after_forward is False
    assert args.fsdp_param_dtype == "fp32"
    assert args.fsdp_gradient_checkpointing is False
    assert not hasattr(args, "lr"), "shared flags are left to the main parser"


def test_megatron_runs_do_not_get_fsdp2_flags(monkeypatch):
    args = _pre_parse(monkeypatch, "--lr", "1e-6")
    assert args.train_backend == "megatron"
    assert not any(name.startswith("fsdp_") for name in vars(args))


def test_unknown_backend_is_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        _pre_parse(monkeypatch, "--train-backend", "fsdp")


def test_train_group_selects_the_registered_actor(monkeypatch):
    import ray

    from vime.ray.actor_group import RayTrainGroup

    selected = []

    class _Stop(Exception):
        pass

    def fake_remote(**options):
        def wrap(actor_impl):
            selected.append(actor_impl)
            raise _Stop

        return wrap

    monkeypatch.setattr(ray, "remote", fake_remote)
    args = Namespace(train_backend="fsdp2", train_env_vars={}, offload_train=False, use_routing_replay=False)
    group = RayTrainGroup(args, num_nodes=1, num_gpus_per_node=1, pg=(object(), [0], [0]))
    with pytest.raises(_Stop):
        group._allocate_gpus_for_actor(group._pg, 1)
    assert selected == [registry.get_train_actor_cls("fsdp2")]


def _valid_args(**overrides):
    values = dict(
        use_critic=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        use_opd=False,
        opd_type=None,
        dspark_enabled=False,
        enable_mtp_training=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        update_weight_transport="nccl",
        update_weight_mode="full",
        async_save=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_validate_accepts_the_supported_configuration():
    validate_fsdp_args(_valid_args())
    validate_fsdp_args(_valid_args(use_opd=True, opd_type="vllm"))


@pytest.mark.parametrize(
    ("overrides", "role", "message"),
    [
        ({}, "critic", "critic"),
        ({"use_critic": True}, "actor", "critic"),
        ({"tensor_model_parallel_size": 2}, "actor", "--tensor-model-parallel-size"),
        ({"context_parallel_size": 2}, "actor", "--context-parallel-size"),
        ({"use_opd": True, "opd_type": "megatron"}, "actor", "--opd-type megatron"),
        ({"dspark_enabled": True}, "actor", "draft/MTP"),
        ({"use_rollout_routing_replay": True}, "actor", "routing replay"),
        ({"update_weight_transport": "disk"}, "actor", "disk/delta"),
        ({"async_save": True}, "actor", "--async-save"),
    ],
)
def test_validate_rejects_unsupported_features(overrides, role, message):
    with pytest.raises(ValueError, match=message):
        validate_fsdp_args(_valid_args(**overrides), role)


def test_validate_rejects_multimodal_and_quantized_checkpoints():
    with pytest.raises(ValueError, match="multimodal"):
        validate_fsdp_args(_valid_args(), hf_config=Namespace(vision_config={}))
    with pytest.raises(ValueError, match="quantized"):
        validate_fsdp_args(_valid_args(), hf_config=Namespace(quantization_config={"quant_method": "fp8"}))


def test_gpu_alignment_script_replays_megatron_rollout_with_fsdp2(monkeypatch, tmp_path):
    """Check the (GPU-only) alignment test's commands and comparison without running them."""
    import runpy
    import shlex
    import types

    import torch

    from vime.utils import external_utils

    commands = []
    launcher = types.ModuleType("vime.utils.external_utils.command_utils")
    launcher.execute_train = lambda **kwargs: commands.append(kwargs)
    launcher.get_default_wandb_args = lambda _: ""
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    monkeypatch.setattr(external_utils, "command_utils", launcher, raising=False)
    script = runpy.run_path(str(_tests_root / "test_qwen3_0.6B_fsdp2_megatron_align.py"))
    monkeypatch.setitem(script["execute"].__globals__, "_log_prob_diffs", lambda *_: [0.0])
    script["execute"]()

    megatron, fsdp = (shlex.split(command["train_args"]) for command in commands)
    assert [command["megatron_model_type"] for command in commands] == ["qwen3-0.6B"] * 2

    def value(argv, flag):
        return argv[argv.index(flag) + 1]

    assert "--train-backend" not in megatron
    assert value(fsdp, "--train-backend") == "fsdp2"
    assert value(megatron, "--save-debug-rollout-data") == value(fsdp, "--load-debug-rollout-data")
    assert value(megatron, "--ci-save-grad-norm") == value(fsdp, "--ci-load-grad-norm")
    assert value(fsdp, "--ref-load").rstrip("/").endswith("Qwen3-0.6B")
    assert registry.parse_train_backend_args("fsdp2", fsdp).fsdp_attn_implementation == "flash_attention_2"

    def dump(path, offset):
        samples = [
            {
                "rollout_position": i,
                "loss_masks": torch.tensor([1, 1, 0]),
                "log_probs": torch.tensor([0.0, offset, 9.0]),
            }
            for i in range(2)
        ]
        torch.save({"samples": samples}, path)

    dump(tmp_path / "megatron.pt", 0.0)
    dump(tmp_path / "fsdp.pt", 0.1)
    real_diffs = runpy.run_path(str(_tests_root / "test_qwen3_0.6B_fsdp2_megatron_align.py"))["_log_prob_diffs"]
    assert real_diffs(str(tmp_path / "megatron.pt"), str(tmp_path / "fsdp.pt")) == pytest.approx([0.05, 0.05])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
