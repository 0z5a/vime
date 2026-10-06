"""Freeze meaningful online-RL settings at the actual recipe command boundary."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.prepare_native_qualification import EXECUTION_VARIANTS, prepare
from benchmarks.run_native_qualification import command


@pytest.fixture
def packet(tmp_path, monkeypatch):
    import benchmarks.prepare_native_qualification as preparation

    source = tmp_path / "source"
    source.mkdir()
    for name, count in (("train-p1024", 15), ("development", 10)):
        (source / f"{name}.jsonl").write_text(
            "".join(
                json.dumps({"prompt": f"question {i}", "label": str(i), "metadata": {"id": f"{name}/{i}"}}) + "\n"
                for i in range(count)
            )
        )
    (source / "manifest.json").write_text(
        json.dumps({"artifacts": {"development": {"sha256": preparation.sha(source / "development.jsonl")}}})
    )
    monkeypatch.setattr(preparation, "SOURCE_MANIFEST", preparation.sha(source / "manifest.json"))
    monkeypatch.setattr(preparation, "TRAIN_SHA", preparation.sha(source / "train-p1024.jsonl"))
    first, second = tmp_path / "packet", tmp_path / "repeat"
    prepare(source, first)
    prepare(source, second)
    for name in ("qualification.json", "train.jsonl", "development.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    return first


@pytest.mark.parametrize(
    "variant,algorithm",
    [("legacy-remat", "grpo"), ("legacy-remat", "rltt"), ("b-baseline", "rltt"), ("b-prefix", "rltt")],
)
@pytest.mark.parametrize("phase", ["continuous", "split", "resume"])
@pytest.mark.parametrize("layout", ["colocate-resident", "separate"])
def test_actual_recipe_receives_optimizer_data_eval_and_resume(
    tmp_path, monkeypatch, packet, algorithm, phase, variant, layout
):
    if variant != "legacy-remat":
        original = json.loads((packet / "qualification.json").read_text())
        packet = tmp_path / variant
        selected = prepare(tmp_path / "source", packet, variant)
        assert selected["files"] == original["files"] and selected["selected_ids"] == original["selected_ids"]
    profile = json.loads((packet / "qualification.json").read_text())
    profile["resource_layout"] = layout
    (packet / "qualification.json").write_text(json.dumps(profile))
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("native_qualification_recipe", source / "examples/looped_ppo/run.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    model, output = tmp_path / "model", tmp_path / "runs"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ouro",
                "total_ut_steps": 4,
                "num_hidden_layers": 24,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 128,
                "vocab_size": 49152,
                "intermediate_size": 5632,
                "max_position_embeddings": 65536,
                "rope_theta": 1000000,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            }
        )
    )
    if phase == "resume":
        checkpoint = output / algorithm / "resumed/checkpoints/actor"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("1")
    argv = command(packet, model, output, "reserved-cluster:6379", algorithm, phase)
    monkeypatch.setattr(sys, "argv", argv[1:])
    captured = {}
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(init=lambda **kwargs: captured.update(ray=kwargs)))
    monkeypatch.setattr(recipe.runpy, "run_path", lambda *args, **kwargs: captured.update(argv=sys.argv[1:]))
    recipe.main()
    actual = captured["argv"]

    def last(name):
        # The ordinary launcher appends explicit overrides after its probe defaults.
        return actual[max(i for i, value in enumerate(actual) if value == "--" + name) + 1]

    assert last("optimizer") == "adam" and last("adam-beta2") == "0.999"
    assert last("lr") == "1e-06" and last("weight-decay") == "0.1"
    assert last("rollout-max-prompt-len") == "1024" and last("rollout-max-response-len") == "2048"
    assert last("seq-length") == "3072" and last("rlt-kv-blocks") == "1536"
    assert last("global-batch-size") == "32" and last("n-samples-per-prompt") == "8"
    assert last("num-rollout") == "3" and last("num-steps-per-rollout") == "1"
    assert last("save-interval") == last("eval-interval") == "1"
    assert last("lr-decay-iters") == "3" and last("rollout-temperature") == "0.9"
    assert last("eval-temperature") == "0" and last("n-samples-per-eval-prompt") == "1"
    assert last("eval-prompt-data") == "qualification-development"
    assert str(packet / "development.jsonl") in actual
    assert "--apply-chat-template" in actual and "--recurrent-fp32" in actual
    assert last("num-gpus-per-node") == ("1" if layout == "colocate-resident" else "2")
    if layout == "colocate-resident":
        assert "--colocate" in actual and "--no-offload-train" in actual and "--no-offload-rollout" in actual
    else:
        assert "--colocate" not in actual and "--offload-train" in actual
    if algorithm == "grpo":
        assert "--ref-load" not in actual and "--use-critic" not in actual and last("kl-coef") == "0"
    assert last("rlt-start-version") == ("2" if phase == "resume" else "0")
    run = output / algorithm / ("continuous" if phase == "continuous" else "resumed")
    assert last("rlt-runtime-report-dir") == str(run / "runtime" / phase)
    if phase == "split":
        assert last("stop-after-rollout") == "2"
    else:
        assert "--stop-after-rollout" not in actual
    if algorithm == "rltt":
        assert last("ref-load") == str(model) and last("loss-type") == "rltt_loss"
        assert last("rltt-progressive-alpha") == "0"
        if variant == "legacy-remat":
            assert last("rltt-token-chunk") == "256" and last("recompute-granularity") == "full"
        else:
            assert last("rltt-prefix-wave-size") == str(EXECUTION_VARIANTS[variant]["prefix_wave_size"])
            assert last("rltt-token-chunk") == last("rltt-loop-checkpoint") == last("rltt-layer-checkpoint") == "0"
            assert "--recompute-granularity" not in actual
    assert captured["ray"]["address"] == "reserved-cluster:6379"


def test_changed_data_and_new_cluster_are_rejected(packet, tmp_path):
    with pytest.raises(ValueError, match="existing reserved Ray"):
        command(packet, tmp_path, tmp_path, "local", "rltt", "continuous")
    (packet / "train.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="Changed qualification data"):
        command(packet, tmp_path, tmp_path, "reserved:6379", "rltt", "continuous")


def test_unknown_layout_is_rejected_before_recipe(packet, tmp_path):
    profile = json.loads((packet / "qualification.json").read_text())
    profile["resource_layout"] = "logical-ranks-on-one-card"
    (packet / "qualification.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="resource layout"):
        command(packet, tmp_path, tmp_path, "reserved:6379", "grpo", "continuous")


@pytest.mark.parametrize("change", ["wave", "remat", "algorithm", "variant"])
def test_b_execution_is_frozen_before_launch(packet, tmp_path, change):
    target = tmp_path / "b-prefix"
    profile = prepare(tmp_path / "source", target, "b-prefix")
    if change == "wave":
        profile["learner_execution"]["prefix_wave_size"] = 1
    elif change == "remat":
        profile["learner_execution"]["recompute"] = True
    elif change == "algorithm":
        profile["algorithms"] = ["grpo", "rltt"]
    else:
        profile["execution_variant"] = "unqualified"
    (target / "qualification.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="execution contract"):
        command(target, tmp_path, tmp_path, "reserved:6379", "rltt", "continuous")
