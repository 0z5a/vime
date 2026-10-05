"""Execute the real train loop with transport doubles and actual evaluation dumps."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import train
from vime.observability.rollout_data_utils import save_debug_rollout_data
from vime.observability import rollout_metrics
from vime.utils.types import Sample


@pytest.mark.parametrize(
    "backend,start,updates,expected",
    [
        ("vllm-rlt", 0, 3, [0, 1, 2, 3]),
        ("vllm-rlt", 2, 3, [3]),
        ("vllm-rlt", 2, 0, [2]),
        ("vllm", 0, 2, [0, 0, 1]),
    ],
)
def test_completed_update_indices_preserve_initial_and_resumed_eval(
    monkeypatch, tmp_path, backend, start, updates, expected
):
    evaluations = []
    policy = start - 1

    def publish():
        nonlocal policy
        policy += 1

    def evaluate(rollout_id):
        evaluations.append(rollout_id)
        sample = Sample(tokens=[policy + 1], response_length=1, reward=float(policy))
        save_debug_rollout_data(
            str(tmp_path / "{rollout_id}.pt"),
            {"heldout": {"samples": [sample]}},
            rollout_id=rollout_id,
            evaluation=True,
        )

    def remote(function):
        return SimpleNamespace(remote=function)

    manager = SimpleNamespace(
        eval=remote(evaluate),
        generate=remote(lambda step: step),
        save=remote(lambda step: None),
        dispose=remote(lambda: None),
        shutdown=remote(lambda: None),
    )
    actor = SimpleNamespace(
        update_weights=publish,
        async_train=lambda *args: None,
        save_model=lambda *args, **kwargs: None,
        clear_memory=lambda: None,
        close=lambda: None,
    )
    args = Namespace(
        release_train=False,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        num_rollout=updates,
        eval_interval=1,
        start_rollout_id=start,
        stop_after_rollout=None,
        skip_eval_before_train=False,
        use_critic=False,
        save_interval=None,
        rollout_global_dataset=True,
        rollout_backend=backend,
    )
    monkeypatch.setattr(train, "configure_logger", lambda: None)
    monkeypatch.setattr(train, "init_tracking", lambda args: None)
    monkeypatch.setattr(train, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(train, "create_rollout_manager", lambda *args: (manager, 1))
    monkeypatch.setattr(train, "create_training_models", lambda *args: (actor, None))
    monkeypatch.setattr(train.ray, "get", lambda value: value)
    train.train(args)
    assert evaluations == expected
    if backend == "vllm-rlt":
        assert len(list(tmp_path.glob("eval_*.pt"))) == len(expected)
        for step in expected:
            saved = torch.load(tmp_path / f"eval_{step}.pt", weights_only=False)
            assert saved["rollout_id"] == step
            assert saved["samples"][0]["reward"] == step


def test_pass_rate_uses_each_eval_datasets_completion_count(monkeypatch):
    group_sizes = []

    def pass_rate(flat_rewards, group_size):
        group_sizes.append(group_size)
        return {"pass@1": sum(flat_rewards) / len(flat_rewards)}

    monkeypatch.setattr(rollout_metrics, "compute_pass_rate", pass_rate)
    monkeypatch.setattr(rollout_metrics, "compute_rollout_step", lambda args, step: step)
    monkeypatch.setattr(rollout_metrics.logging_utils, "log", lambda *args, **kwargs: None)
    args = Namespace(custom_eval_rollout_log_function_path=None, log_passrate=True, n_samples_per_eval_prompt=7)
    result = rollout_metrics.log_eval_rollout_data(
        2,
        args,
        {
            "first": {"rewards": [1, 0, 0, 1], "n_samples_per_prompt": 2},
            "second": {"rewards": [1, 0], "n_samples_per_prompt": 1},
            "legacy": {"rewards": [1] * 7},
        },
    )
    assert group_sizes == [2, 1, 7]
    assert result["eval/first"] == result["eval/second"] == 0.5
