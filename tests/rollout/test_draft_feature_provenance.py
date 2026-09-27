from types import SimpleNamespace

from megatron.core import mpu

from vime.observability import train_metric_utils
from vime.ray.rollout import RolloutManager
from vime.utils.types import Sample


def test_collect_only_preserves_rollout_versions_through_dp_split(monkeypatch):
    manager = object.__new__(RolloutManager.__ray_metadata__.modified_class)
    manager.args = SimpleNamespace(draft_feature_mode="collect-only", rollout_top_p=1.0, global_batch_size=2)
    manager.custom_convert_samples_to_train_data_func = None
    manager._post_process_rewards = lambda samples: ([0.0, 1.0], [0.0, 1.0])
    samples = [
        Sample(index=11, group_index=5, rollout_id=7, tokens=[1, 2, 3], response_length=1, weight_versions=["v2"]),
        Sample(
            index=22, group_index=5, rollout_id=8, tokens=[4, 5, 6], response_length=1, weight_versions=["v1", "v2"]
        ),
    ]
    data = manager._convert_samples_to_train_data(samples)
    assert data["weight_versions"] == [("v2",), ("v1", "v2")]
    assert data["group_indices"] == [5, 5]

    manager.train_parallel_config = {"dp_size": 1}
    monkeypatch.setattr("vime.ray.rollout.build_dp_schedule", lambda *args, **kwargs: ([[0, 1]], [[[1, 0]]], [1], [2]))
    monkeypatch.setattr("vime.ray.rollout.tensorize_rollout_data_for_training", lambda shard: None)
    monkeypatch.setattr("vime.ray.rollout.ray.put", lambda shard: shard)
    shard = manager._split_train_data_by_dp(data)[0].inner
    assert shard["weight_versions"] == [("v2",), ("v1", "v2")]
    assert shard["group_indices"] == [5, 5]
    assert shard["micro_batch_indices"] == [[1, 0]]


def test_rollout_metrics_ignore_draft_identity_and_version(monkeypatch):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)
    captured = {}
    monkeypatch.setattr(
        train_metric_utils,
        "gather_log_data",
        lambda name, args, rollout_id, values: captured.update(values),
    )
    train_metric_utils.log_rollout_data(
        0,
        SimpleNamespace(ci_test=False, log_multi_turn=False, log_passrate=False, log_correct_samples=False),
        {
            "response_lengths": [2],
            "loss_masks": [[1, 1]],
            "total_lengths": [3],
            "rollout_mask_sums": [2],
            "global_batch_sizes": [1],
            "group_indices": [7],
            "weight_versions": [("v1",)],
        },
    )
    assert captured["response_lengths"] == (2, 1)
    assert "group_indices" not in captured
    assert "weight_versions" not in captured
