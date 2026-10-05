"""Real tiny native generation/math rewards; Ray transport and tokenizer are test seams."""

import json
import random
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_native_rlt_contract import engine as engine
from test_native_rlt_contract import publish

from vime.rollout import native_eval, vllm_rlt_rollout
from vime.utils.eval_config import EvalDatasetConfig
from vime.utils.types import Sample


class Tokenizer:
    def encode(self, prompt, **kwargs):
        return [int(token) for token in prompt.split()]

    def decode(self, tokens, **kwargs):
        return r"\boxed{" + str(tokens[-1]) + "}"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + " 3"


def setup_eval(engine, tmp_path, monkeypatch):
    path = tmp_path / "heldout.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt": prompt, "label": "2", "metadata": {"source": "heldout"}})
            for prompt in ("1 2", "3 4")
        )
        + "\n"
    )
    config = EvalDatasetConfig(
        "heldout",
        str(path),
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        n_samples_per_eval_prompt=2,
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        max_response_len=2,
    )
    args = Namespace(
        **(
            vars(engine.args)
            | {
                "hf_checkpoint": "tiny-fixture",
                "eval_datasets": [config],
                "group_rm": False,
                "multimodal_keys": None,
                "apply_chat_template": False,
                "apply_chat_template_kwargs": {},
                "eval_max_prompt_len": 8,
                "eval_max_context_len": 16,
                "eval_reward_key": None,
                "reward_key": None,
                "custom_rm_path": None,
                "rm_type": "math",
                "rlt_max_num_seqs": 2,
                "rlt_engine": SimpleNamespace(evaluate=SimpleNamespace(remote=engine.evaluate)),
            }
        )
    )
    engine.args = args
    monkeypatch.setattr(native_eval, "_tokenizer", lambda checkpoint: Tokenizer())
    monkeypatch.setattr(native_eval.ray, "get", lambda result: result)
    publish(engine, tmp_path / "weights", 1)
    engine.continue_generation()
    return args, config


def observed(output):
    return {
        name: [
            (
                s.tokens,
                s.reward,
                s.rollout_log_probs,
                s.status,
                s.recurrent_trace.seed,
                s.recurrent_trace.policy_version,
            )
            for s in data["samples"]
        ]
        for name, data in output.data.items()
    }


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
def test_complete_datasets_fixed_seeds_and_unchanged_training_state(engine, tmp_path, monkeypatch, family):
    if family != "ouro":
        from vllm_rlt import LLM, CacheConfig
        from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM
        from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM

        engine.llm.close()
        if family == "nanbeige":
            native = NanbeigeForCausalLM(
                NanbeigeConfig(
                    vocab_size=11,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    num_loops=2,
                    max_position_embeddings=32,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=None,
                )
            )
        else:
            native = HuginnForCausalLM(
                HuginnConfig(
                    vocab_size=11,
                    padded_vocab_size=11,
                    n_embd=8,
                    n_heads=2,
                    n_layers=3,
                    n_layers_in_prelude=1,
                    n_layers_in_recurrent_block=1,
                    n_layers_in_coda=1,
                    intermediate_size=16,
                    mean_recurrence=2,
                    block_size=32,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=None,
                )
            )
        engine.llm = LLM(native, cache_config=CacheConfig(num_blocks=16))
        engine.args.rlt_model_family = "huginn_raven" if family == "huginn" else family
        engine.args.rlt_depth = 2
    args, config = setup_eval(engine, tmp_path, monkeypatch)
    args.eval_datasets.append(replace(config, name="second", n_samples_per_eval_prompt=1))
    before = engine.generate([Sample(tokens=[1, 2], index=7, group_index=2)], 9)[0]
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    train_data = SimpleNamespace(get_samples=lambda count: pytest.fail("Evaluation consumed training data"))
    first = vllm_rlt_rollout.generate_rollout(args, 0, train_data, evaluation=True)
    second = vllm_rlt_rollout.generate_rollout(args, 15, train_data, evaluation=True)
    after = engine.generate([Sample(tokens=[1, 2], index=7, group_index=2)], 9)[0]
    assert observed(first) == observed(second)
    assert (before.tokens, before.rollout_log_probs, before.recurrent_trace.seed) == (
        after.tokens,
        after.rollout_log_probs,
        after.recurrent_trace.seed,
    )
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert first.metrics["eval/native_samples"] == 6
    assert first.metrics["eval/native_policy_version"] == 1
    assert first.data["heldout"]["n_samples_per_prompt"] == 2
    assert first.data["second"]["n_samples_per_prompt"] == 1
    seeds = [s.recurrent_trace.seed for data in first.data.values() for s in data["samples"]]
    assert len(set(seeds)) == 6
    for data in first.data.values():
        for sample, reward in zip(data["samples"], data["rewards"], strict=True):
            assert reward == float(sample.tokens[-1] == 2)
            assert sample.metadata["source"] == "heldout"
            assert sample.recurrent_trace.publication_digest == engine.committed_digest
            if family == "huginn":
                assert sample.recurrent_trace.latent_seed == sample.recurrent_trace.seed
                assert sample.recurrent_trace.latent_profile == "like-init-cpu-f32-v1"


def test_sampling_and_chat_overrides_are_used(engine, tmp_path, monkeypatch):
    args, config = setup_eval(engine, tmp_path, monkeypatch)
    args.eval_datasets = [replace(config, top_k=1, temperature=0, max_response_len=1, apply_chat_template=True)]
    output = native_eval.evaluate(args)
    for sample in output.data["heldout"]["samples"]:
        assert sample.response_length == 1 and sample.tokens[2] == 3
        assert sample.recurrent_trace.temperature == 0
        assert sample.metadata["native_eval"]["top_k"] == 1
    assert engine.args.rollout_temperature == 0.8 and engine.args.rollout_max_response_len == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("stop", ["END"]),
        ("min_new_tokens", 1),
        ("repetition_penalty", 1.2),
        ("no_stop_trim", False),
        ("custom_generate_function_path", "custom.generate"),
        ("app_service", "http://example.invalid"),
        ("eval_task_timeout", 1),
        ("n_samples_per_eval_prompt", 0),
    ],
)
def test_unsupported_eval_settings_reject_before_generation(engine, tmp_path, monkeypatch, field, value):
    args, config = setup_eval(engine, tmp_path, monkeypatch)
    args.eval_datasets = [replace(config, **{field: value})]
    args.rlt_engine.evaluate.remote = lambda *args: pytest.fail("Invalid evaluation generated tokens")
    with pytest.raises(ValueError):
        native_eval.evaluate(args)


@pytest.mark.parametrize("limit", ["prompt", "context"])
def test_long_heldout_prompts_are_not_silently_dropped(engine, tmp_path, monkeypatch, limit):
    args, _ = setup_eval(engine, tmp_path, monkeypatch)
    if limit == "prompt":
        args.eval_max_prompt_len = 1
    else:
        args.eval_max_context_len = 3
    args.rlt_engine.evaluate.remote = lambda *args: pytest.fail("Invalid evaluation generated tokens")
    with pytest.raises(ValueError, match="limit"):
        native_eval.evaluate(args)


def test_policy_publication_between_batches_invalidates_evaluation(engine, tmp_path, monkeypatch):
    args, _ = setup_eval(engine, tmp_path, monkeypatch)
    calls = 0

    def evaluate(samples, sampling, seeds):
        nonlocal calls
        if calls:
            engine.pause_generation()
            publish(engine, tmp_path / "next-policy", 2)
            engine.continue_generation()
        calls += 1
        return engine.evaluate(samples, sampling, seeds)

    args.rlt_engine.evaluate.remote = evaluate
    with pytest.raises(RuntimeError, match="policy cohorts"):
        native_eval.evaluate(args)


def test_reordered_or_missing_heldout_samples_are_rejected(engine, tmp_path, monkeypatch):
    args, _ = setup_eval(engine, tmp_path, monkeypatch)

    def evaluate(samples, sampling, seeds):
        return list(reversed(engine.evaluate(samples, sampling, seeds)))

    args.rlt_engine.evaluate.remote = evaluate
    with pytest.raises(ValueError, match="every held-out sample"):
        native_eval.evaluate(args)


def test_reward_key_and_metadata_override(engine, tmp_path, monkeypatch):
    args, config = setup_eval(engine, tmp_path, monkeypatch)
    args.eval_reward_key = "accuracy"
    args.eval_datasets = [replace(config, metadata_overrides={"rm_type": "custom"})]

    async def reward(args, sample):
        assert sample.metadata["rm_type"] == "custom"
        return {"accuracy": int(sample.tokens[-1] == 2), "other": -1}

    monkeypatch.setattr(vllm_rlt_rollout, "async_rm", reward)
    output = native_eval.evaluate(args)
    assert all(value in (0.0, 1.0) for value in output.data["heldout"]["rewards"])
