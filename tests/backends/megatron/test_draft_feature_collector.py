import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import load_file

from vime.backends.megatron_utils.draft_feature_collector import DraftFeatureCollector


class TinyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.share_embeddings_and_output_weights = False
        self.embedding = torch.nn.Embedding(32, 4)
        self.output_layer = torch.nn.Linear(4, 7, bias=False)

    def forward(self, input_ids):
        return self.output_layer(self.embedding(input_ids).transpose(0, 1))


class TiedOutput(torch.nn.Module):
    bias = None
    weight = None

    def forward(self, hidden, weight):
        return torch.nn.functional.linear(hidden, weight)


class TinyTiedTarget(torch.nn.Module):
    share_embeddings_and_output_weights = True

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.output_layer = TiedOutput()

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        return self.output_layer(self.embedding(input_ids).transpose(0, 1), self.embedding.weight)


def _args(tmp_path, max_bytes=4096):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "tokenizer.json").write_text("{}")
    return Namespace(
        hf_checkpoint=str(checkpoint),
        draft_feature_output_dir=str(tmp_path / "features"),
        draft_feature_run_id="run_a",
        draft_feature_max_tokens=4,
        draft_feature_max_batches=2,
        draft_feature_max_bytes=max_bytes,
    )


def _batch():
    tokens = torch.tensor([1, 2, 3, 4])
    return {
        "unconcat_tokens": [tokens],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1])],
    }


def _rollout_data():
    return {"sample_indices": [7], "group_indices": [2], "rollout_ids": [3], "weight_versions": [("v1",)]}


def test_collect_only_rebuilds_logits_from_owned_head_input(tmp_path):
    torch.manual_seed(7)
    model = TinyTarget()
    args = _args(tmp_path)
    collector = DraftFeatureCollector(args, model, round_id=3, source_version="target-v3")
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = model(tokens).detach()
    actual = collector.forward(model, {"input_ids": tokens}, _batch(), _rollout_data(), [0])
    collector.finish()

    root = tmp_path / "features" / "run_a" / "round-3"
    features = load_file(root / "batch-0000.safetensors")["features"]
    head = load_file(root / "head.safetensors")["weight"]
    manifest = json.loads((root / "batch-0000.json").read_text())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.nn.functional.linear(features, head), expected[[1, 2], 0])
    assert [
        (token["sample_id"], token["token_position"], token["target_token_id"])
        for token in manifest["token_map"]["selected_tokens"]
    ] == [(7, 1, 3), (7, 2, 4)]
    assert manifest["token_map"]["sequences"][0]["weight_versions"] == ["v1"]
    assert manifest["policy_source_version"] == manifest["head_source_version"] == "target-v3"
    assert not model.output_layer._forward_pre_hooks
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.output_layer.weight.zero_()
    torch.testing.assert_close(load_file(root / "batch-0000.safetensors")["features"], features)
    torch.testing.assert_close(load_file(root / "head.safetensors")["weight"], head)


def test_byte_budget_stops_collection_without_skipping_target_forward(tmp_path):
    model = TinyTarget()
    collector = DraftFeatureCollector(_args(tmp_path, max_bytes=1), model, 0, "target-v0")
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(
        collector.forward(model, {"input_ids": tokens}, _batch(), _rollout_data(), [0]), model(tokens)
    )
    assert collector.stop_reason == "byte budget"
    assert not model.output_layer._forward_pre_hooks
    with pytest.raises(ValueError, match="no selected"):
        collector.finish()


def test_shared_embedding_head_snapshot_rebuilds_logits(tmp_path):
    model = TinyTiedTarget()
    collector = DraftFeatureCollector(_args(tmp_path), model, 0, "target-v0")
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = collector.forward(model, {"input_ids": tokens}, _batch(), _rollout_data(), [0])
    collector.finish()
    root = tmp_path / "features" / "run_a" / "round-0"
    features = load_file(root / "batch-0000.safetensors")["features"]
    head = load_file(root / "head.safetensors")["weight"]
    torch.testing.assert_close(torch.nn.functional.linear(features, head), logits[[1, 2], 0])


def test_cuda_export_matches_forward_after_device_copy(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model = TinyTarget().cuda()
    collector = DraftFeatureCollector(_args(tmp_path), model, 0, "target-v0")
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    logits = collector.forward(model, {"input_ids": tokens}, _batch(), _rollout_data(), [0])
    collector.finish()
    root = tmp_path / "features" / "run_a" / "round-0"
    features = load_file(root / "batch-0000.safetensors")["features"]
    head = load_file(root / "head.safetensors")["weight"]
    torch.testing.assert_close(torch.nn.functional.linear(features, head), logits[[1, 2], 0].cpu())
    with torch.no_grad():
        model.embedding.weight.zero_()
    torch.testing.assert_close(load_file(root / "batch-0000.safetensors")["features"], features)
