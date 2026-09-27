import pytest

from vime.utils.draft_feature_contract import DraftFeatureManifest, DraftSequence, DraftToken, DraftTokenMap, build_token_map


def _sequences():
    # The reordered second sample has a masked prompt and a distinct rollout version.
    return (
        DraftSequence(22, 4, 8, (10, 11, 12, 13), (0, 1, 1, 0), ("rollout-v2",)),
        DraftSequence(11, 4, 7, (20, 21, 22), (1, 0, 0), ("rollout-v1",)),
    )


def _manifest(**changes):
    values = dict(
        schema_version=1,
        feature_batch_id="run-a/round-3/attempt-0/batch-0",
        run_id="run-a",
        round_id=3,
        policy_source_version="target-v3",
        head_source_version="target-v3",
        model_tag="old_actor",
        model_config_digest="model-sha256",
        tokenizer_digest="tokenizer-sha256",
        capture_point="lm_head_input",
        dtype="bfloat16",
        token_map=build_token_map(_sequences(), 3),
        head_snapshot_ref="head-v3.safetensors",
        payload_ref="features-v3.safetensors",
        byte_count=24,
        ready=True,
    )
    values.update(changes)
    return DraftFeatureManifest(**values)


def test_reordered_packed_token_map_preserves_targets_and_versions():
    token_map = build_token_map(_sequences(), 3)
    assert token_map.sequence_offsets == (0, 4, 7)
    assert [(token.sample_id, token.token_position, token.packed_position, token.target_token_id) for token in token_map.selected_tokens] == [
        (22, 1, 1, 12),
        (22, 2, 2, 13),
        (11, 0, 4, 21),
    ]
    manifest = _manifest(token_map=token_map)
    assert manifest.policy_source_version == "target-v3"
    assert [sequence.weight_versions for sequence in manifest.token_map.sequences] == [("rollout-v2",), ("rollout-v1",)]


def test_empty_response_and_masked_tail_cannot_cross_sequence_boundary():
    sequence = DraftSequence(3, 1, 2, (10, 11), (0, 0), ())
    with pytest.raises(ValueError, match="no selected"):
        build_token_map((sequence,), 5)
    with pytest.raises(ValueError, match="final token"):
        DraftSequence(3, 1, 2, (10, 11), (0, 1), ())


def test_budget_rejects_entire_batch_instead_of_publishing_partial_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        build_token_map(_sequences(), 2)
    with pytest.raises(ValueError, match="unique"):
        build_token_map((_sequences()[0], _sequences()[0]), 6)


def test_token_map_rejects_wrong_packed_offset_and_cross_sample_target():
    token_map = build_token_map(_sequences(), 3)
    with pytest.raises(ValueError, match="sequence offsets"):
        DraftTokenMap(token_map.sequences, (0, 3, 7), token_map.selected_tokens)
    wrong_target = DraftToken(22, 2, 2, 21)
    with pytest.raises(ValueError, match="source sequence"):
        DraftTokenMap(token_map.sequences, token_map.sequence_offsets, (wrong_target,))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"head_source_version": "target-v4"}, "same source version"),
        ({"model_tag": "teacher"}, "actor or old_actor"),
        ({"tp": 2}, "TP=PP=CP=DP=1"),
        ({"capture_point": "last_layer"}, "capture_point"),
        ({"layer_ids": (3,)}, "does not use decoder layer"),
        ({"payload_ref": ""}, "requires payload"),
        ({"head_snapshot_ref": ""}, "requires payload"),
    ],
)
def test_invalid_feature_manifests_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        _manifest(**changes)
