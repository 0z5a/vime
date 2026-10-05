"""Validate native residency and actual placement options without starting Ray."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
from test_megatron_argument_validation import make_vime_validate_args

from vime.backends.vllm_rlt_utils import deployment
from vime.backends.vllm_rlt_utils.arguments import validate_args
from vime.ray import placement_group
from vime.utils.arguments import vime_validate_args


def arguments(tmp_path, **overrides):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ouro", "total_ut_steps": 4}))
    values = vars(make_vime_validate_args()) | {
        "hf_checkpoint": str(tmp_path),
        "rlt_depth": None,
        "rlt_kv_blocks": 128,
        "rlt_max_num_seqs": 2,
        "rlt_start_version": 0,
        "rltt_prefix_wave_size": 0,
        "rollout_num_gpus": 1,
        "rollout_num_gpus_per_engine": 1,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "num_gpus_per_node": 1,
        "colocate": True,
        "offload_train": False,
        "offload_rollout": False,
        "use_fault_tolerance": False,
        "use_critic": False,
        "use_kl_loss": False,
        "check_weight_update_equal": False,
        "update_weight_mode": "full",
        "update_weight_transport": "disk",
        "update_weight_disk_dir": str(tmp_path / "publications"),
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_backend": "vllm-rlt",
        "rollout_external": False,
        "rollout_function_path": "vime.rollout.vllm_rlt_rollout.generate_rollout",
        "advantage_estimator": "grpo",
        "n_samples_per_prompt": 2,
        "eval_interval": None,
    }
    return Namespace(**(values | overrides))


def test_real_common_validation_preserves_resident_profile(tmp_path):
    args = arguments(tmp_path)
    vime_validate_args(args)
    validate_args(args)
    assert args.colocate and not args.offload_train and not args.offload_rollout
    assert placement_group._get_placement_group_layout(args) == (1, 0)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"offload_rollout": True}, "rollout resident"),
        ({"offload_train": True}, "no training offload"),
        ({"use_critic": True}, "no critic"),
        ({"release_train": True}, "release-train"),
        ({"use_fault_tolerance": True}, "recovery"),
        ({"actor_num_nodes": 2}, "one learner"),
        ({"actor_num_gpus_per_node": 2}, "one learner"),
    ],
)
def test_colocation_rejects_unimplemented_residency_contracts(tmp_path, overrides, error):
    with pytest.raises(ValueError, match=error):
        validate_args(arguments(tmp_path, **overrides))


def test_bare_colocate_does_not_silently_enable_native_offload(tmp_path):
    args = arguments(tmp_path, offload_train=None, offload_rollout=None)
    vime_validate_args(args)
    assert args.offload_train and args.offload_rollout
    with pytest.raises(ValueError, match="rollout resident"):
        validate_args(args)


@pytest.mark.parametrize("colocate,debug", [(True, False), (False, False), (False, True)])
def test_native_deployment_and_learner_fit_the_actual_bundle(tmp_path, monkeypatch, colocate, debug):
    args = arguments(tmp_path, colocate=colocate, debug_rollout_only=debug)
    options, train_options = {}, {}
    engine = SimpleNamespace(init=SimpleNamespace(remote=lambda config: ("init", config)))

    class Remote:
        def options(self, **values):
            options.update(values)
            return self

        def remote(self):
            return engine

    def remote_type(implementation):
        assert implementation is deployment.NativeEngine
        return Remote()

    monkeypatch.setattr(deployment.ray, "remote", remote_type)
    monkeypatch.setattr(placement_group, "RayTrainGroup", lambda **values: train_options.update(values))
    placement = SimpleNamespace(id="existing-placement")
    pg = (placement, [0], [6])
    servers, pending = deployment.start_rollout_servers(args, pg)
    assert pending == [("init", args)] and args.rlt_engine is engine
    server = servers["default"]
    assert server.engine_gpu_offsets == ([0] if colocate or debug else [1])
    assert server.engine_gpu_counts == (1,)
    assert options["scheduling_strategy"].placement_group is placement
    assert options["scheduling_strategy"].placement_group_bundle_index == 0
    assert options["num_cpus"] == options["num_gpus"] == (0.5 if colocate else 1)
    if colocate:
        placement_group.allocate_train_group(args, 1, 1, pg)
        assert train_options["pg"] is pg
        # The production allocator reserves this fraction for both CPU and GPU.
        assert train_options["num_gpus_per_actor"] == 0.4
        assert options["num_gpus"] + train_options["num_gpus_per_actor"] <= 1
        assert options["num_cpus"] + train_options["num_gpus_per_actor"] <= 1
        assert placement_group._get_placement_group_layout(args) == (1, 0)
