import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vime.ray.placement_group import _create_placement_group, _get_placement_group_layout

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (16, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


@pytest.mark.parametrize("loss_type,with_ref", [("rltt_loss", True), ("policy_loss", False)])
def test_zero_kl_reference_allocation(monkeypatch, loss_type, with_ref):
    from vime.ray import placement_group

    allocated = {}

    class Group:
        def create(self, *, rollout_manager):
            assert rollout_manager == "rollouts"
            return [0]

    group = Group()

    def allocate(**kwargs):
        allocated.update(kwargs)
        return group

    monkeypatch.setattr(placement_group, "allocate_train_group", allocate)
    args = _args(megatron_config_path=None, kl_coef=0, use_kl_loss=False, loss_type=loss_type, use_opd=False)
    assert placement_group.create_actor_model(args, {"actor": "pg"}, "rollouts") == (group, [0])
    assert allocated["with_ref"] is with_ref


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
