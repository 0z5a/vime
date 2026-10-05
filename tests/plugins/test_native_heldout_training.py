"""Held-out calls preserve real tiny RLTT updates and fresh Adam recovery.

This is CPU component integration, with token-parity training rewards and test
tokenization. It does not qualify Ray/CUDA training or reasoning convergence.
"""

import copy
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from test_looped_grpo import cpu_engine, publish
from test_rltt_online_cycle import pair, update

from benchmarks.audit_native_outputs import compare_samples
from vime.rollout import native_eval
from vime.utils.eval_config import EvalDatasetConfig


def evaluate(engine, data, monkeypatch, capture=None):
    tokenizer = SimpleNamespace(
        encode=lambda prompt, **kwargs: [int(value) for value in prompt.split()],
        decode=lambda tokens, **kwargs: r"\boxed{" + str(tokens[-1]) + "}",
    )
    monkeypatch.setattr(native_eval, "_tokenizer", lambda checkpoint: tokenizer)
    monkeypatch.setattr(native_eval.ray, "get", lambda result: result)
    config = EvalDatasetConfig(
        "heldout",
        str(data),
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
    output = native_eval.evaluate(args)
    if capture is not None:
        capture.append([sample.to_dict() for sample in output.data["heldout"]["samples"]])
    return [
        {
            "tokens": sample.tokens,
            "reward": sample.reward,
            "seed": sample.recurrent_trace.seed,
            "version": sample.recurrent_trace.policy_version,
            "digest": sample.recurrent_trace.publication_digest,
        }
        for sample in output.data["heldout"]["samples"]
    ]


def assert_optimizer_equal(left, right):
    assert left["param_groups"] == right["param_groups"]
    for index, state in left["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["state"][index][key], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn_raven"])
def test_evaluation_preserves_updates_and_fresh_resume(family, tmp_path, monkeypatch, record_property):
    data = tmp_path / "heldout.jsonl"
    data.write_text('{"prompt":"4 5","label":"2"}\n{"prompt":"7 8","label":"3"}\n')
    evaluations, raw_evaluations, reference_runs, final_states = [], [], [], []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        directory.mkdir()
        native, actor = pair(family)
        reference = copy.deepcopy(actor).requires_grad_(False)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0)
        engine = cpu_engine(native, family)
        records = []
        for step in range(3):
            records.append(update(actor, reference, optimizer, engine, family, directory, step, "sdpa-reference"))
            if enabled:
                evaluations.append(evaluate(engine, data, monkeypatch, raw_evaluations))
            if enabled and step == 1:
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "reference": reference.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                    },
                    tmp_path / "resume.pt",
                )
        reference_runs.append(records)
        final_states.append((copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())))
        engine.close()
    assert reference_runs[0] == reference_runs[1]
    for name, expected in final_states[0][0].items():
        assert torch.equal(expected, final_states[1][0][name]), name
    assert_optimizer_equal(final_states[0][1], final_states[1][1])

    checkpoint = torch.load(tmp_path / "resume.pt", weights_only=True)
    native, resumed = pair(family)
    resumed.load_state_dict(checkpoint["actor"])
    reference = copy.deepcopy(resumed).requires_grad_(False)
    reference.load_state_dict(checkpoint["reference"])
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0)
    optimizer.load_state_dict(checkpoint["optimizer"])
    engine = cpu_engine(native, family)
    publish(resumed, engine, family, tmp_path / "restored-policy", 3)
    torch.set_rng_state(checkpoint["rng"])
    resumed_outputs = []
    assert evaluate(engine, data, monkeypatch, resumed_outputs) == evaluations[1]
    directory = tmp_path / "fresh"
    directory.mkdir()
    actual = update(resumed, reference, optimizer, engine, family, directory, 2, "sdpa-reference")
    assert actual == reference_runs[1][2]
    assert evaluate(engine, data, monkeypatch, resumed_outputs) == evaluations[2]
    errors = [
        compare_samples(left, right)
        for expected, actual in zip(raw_evaluations[1:], resumed_outputs, strict=True)
        for left, right in zip(expected, actual, strict=True)
    ]
    assert len(errors) == 8
    for name, expected in final_states[1][0].items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    assert_optimizer_equal(final_states[1][1], optimizer.state_dict())
    engine.close()
    record_property("scope", "tiny CPU native/RLTT/Adam recovery; transport and tokenizer doubles; no quality claim")
    record_property("training", json.dumps(reference_runs[1]))
    record_property("heldout", json.dumps(evaluations))
    record_property("paired_raw_heldout_samples", len(errors))
    record_property("paired_raw_score_max_error", max(errors))
