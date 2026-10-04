"""Tiny native sampling -> output reward -> RLTT -> publication -> Adam resume.

This exercises CPU components with a token-parity reward, not a mathematical
reasoning benchmark or the Ray/MCore distributed training lifecycle.
"""

import copy
import json
from argparse import Namespace
from functools import partial

import pytest
import torch
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from megatron.core.transformer.transformer_config import TransformerConfig
from test_looped_grpo import cpu_engine, model_pair, publish
from test_rltt_training import packed
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime.utils.reward_normalization import normalize_rewards
from vime.utils.types import Sample
from vime_plugins.looped.training import collect_log_probs, masked_token_weights, megatron_loss
from vime_plugins.ouro.model import OuroMegatronModel


def pair(family):
    torch.manual_seed(42)
    if family != "ouro":
        return model_pair(family, True)
    native = OuroForCausalLM(
        OuroConfig(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            total_ut_steps=2,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
    )
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2, kv_channels=4, ffn_hidden_size=16)
    return native, OuroMegatronModel(config, copy.deepcopy(native), recompute=True)


def update(actor, reference, optimizer, engine, family, path, step):
    publish(actor, engine, family, path / f"before-{step}", step + 1)
    prompts = ([1, 2], [2, 2])
    requests = [
        Sample(tokens=list(prompt), group_index=2 * step + group, index=16 * step + 8 * group + sibling)
        for group, prompt in enumerate(prompts)
        for sibling in range(8)
    ]
    samples = engine.generate(requests, step)
    assert len({sample.recurrent_trace.seed for sample in samples}) == 16
    items = [(torch.tensor(s.tokens), s.response_length, s.recurrent_trace) for s in samples]
    # Reward is computed from the sampled answer and an immutable prompt label.
    rewards = [float(s.tokens[-s.response_length] % 2 == sum(prompts[i // 8]) % 2) for i, s in enumerate(samples)]
    mixed = sum(len(set(rewards[start : start + 8])) > 1 for start in (0, 8))
    assert mixed > 0
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=8,
        rollout_batch_size=2,
        rltt_progressive_alpha=1,
        kl_loss_coef=0.01,
        entropy_coef=0,
    )
    advantages = normalize_rewards(args, rewards)
    lengths = [s.response_length for s in samples]
    with torch.no_grad():
        # Engine sampling and reference/actor replay use the same temperature.
        inputs = packed(items, all_loops=False, entropy=False, temperature=0.8)
        _, ref = collect_log_probs(reference(**inputs), response_lengths=lengths, with_entropy=False)
        _, current = collect_log_probs(actor(**inputs), response_lengths=lengths, with_entropy=False)
    error = max(
        (score - torch.tensor(s.rollout_log_probs)).abs().max().item()
        for score, s in zip(current["log_probs"], samples, strict=True)
    )
    assert error < 4e-6
    weights = masked_token_weights([torch.ones(length) for length in lengths], "response_mean")
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in actor.parameters()]
    losses = []
    for start in range(0, 16, 4):
        subset = items[start : start + 4]
        inputs = packed(subset, all_loops=True, entropy=False, temperature=0.8)
        batch = {
            "advantages": [
                torch.full((length,), advantage)
                for length, advantage in zip(lengths[start : start + 4], advantages[start : start + 4], strict=True)
            ],
            "ref_log_probs": ref["log_probs"][start : start + 4],
        }
        loss_fn = partial(megatron_loss, args, batch, weights[start : start + 4], 4, 16)
        loss, _ = forward_step_calc_loss(
            actor, actor(**inputs), loss_fn, actor.config, None, False, 4, [], cp_group_size=1, is_last_stage=True
        )
        loss.backward()
        losses.append(loss.item())
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    grad_norm = sum(parameter.grad.double().square().sum() for parameter in trainable).sqrt().item()
    assert grad_norm > 0
    optimizer.step()
    delta = (
        sum(
            (parameter - old).double().square().sum()
            for parameter, old in zip(actor.parameters(), before, strict=True)
        )
        .sqrt()
        .item()
    )
    assert delta > 0 and all(parameter.grad is None for parameter in reference.parameters())
    publish(actor, engine, family, path / f"after-{step}", step + 2)
    return dict(
        step=step,
        tokens=[s.tokens for s in samples],
        rewards=rewards,
        mixed_groups=mixed,
        seeds=[s.recurrent_trace.seed for s in samples],
        loss=sum(losses),
        grad_norm=grad_norm,
        parameter_delta=delta,
        score_error=error,
        publication_digest=engine.committed_digest,
    )


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn_raven"])
def test_fresh_reward_updates_and_reference_adam_resume(family, tmp_path, record_property):
    native, actor = pair(family)
    reference = copy.deepcopy(actor).requires_grad_(False)
    frozen = copy.deepcopy(reference.state_dict())
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0)
    engine = cpu_engine(native, family)
    records = [update(actor, reference, optimizer, engine, family, tmp_path, step) for step in range(2)]
    torch.save(
        dict(
            actor=actor.state_dict(),
            reference=reference.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(),
            next_step=2,
        ),
        tmp_path / "resume.pt",
    )
    expected = update(actor, reference, optimizer, engine, family, tmp_path, 2)
    checkpoint = torch.load(tmp_path / "resume.pt", weights_only=True)
    _, resumed = pair(family)
    resumed.load_state_dict(checkpoint["actor"])
    resumed_ref = copy.deepcopy(resumed).requires_grad_(False)
    resumed_ref.load_state_dict(checkpoint["reference"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_engine = cpu_engine(native, family)
    resume_path = tmp_path / "fresh"
    resume_path.mkdir()
    torch.set_rng_state(checkpoint["rng"])
    actual = update(
        resumed, resumed_ref, resumed_optimizer, resumed_engine, family, resume_path, checkpoint["next_step"]
    )
    assert actual == expected
    for name, value in actor.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    for name, value in frozen.items():
        assert torch.equal(reference.state_dict()[name], value) and torch.equal(resumed_ref.state_dict()[name], value)
    left, right = optimizer.state_dict(), resumed_optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    for index, state in left["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["state"][index][key], rtol=0, atol=0)
    record_property("scope", "tiny CPU components; native sampling; token-parity reward; no Ray or CUDA lifecycle")
    record_property("updates", json.dumps([*records, expected]))
    record_property("resume", "exact actor/reference/Adam moments/tokens/rewards/scores/publication")
    engine.close()
    resumed_engine.close()
