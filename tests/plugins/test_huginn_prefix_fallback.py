"""Actual latent grouping and the production group's complete-replay fallback."""

import copy
import json
from argparse import Namespace
from dataclasses import asdict, replace

import pytest
import torch
from test_differentiable_prefix import objective
from test_looped_logical_step import rollout
from test_looped_response_readout import make_actor
from test_rltt_training import packed

from vime.backends.megatron_utils.data import DataIterator
from vime.backends.megatron_utils.looped_prefix_schedule import backward_group
from vime_plugins.looped.logical_step import plan_step
from vime_plugins.looped.training import logical_rltt_loss


@pytest.mark.parametrize("mode,shared", [("independent", 0), ("mixed", 2), ("shared", 5)])
@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("microbatches,wave", [([[0, 1, 2, 3, 4]], 2), ([[4, 1], [3, 0, 2]], 1)])
@pytest.mark.parametrize("scale", [1.0, 8.0])
def test_latent_grouping_matches_full_replay_two_updates(
    mode, shared, reduction, microbatches, wave, scale, record_property
):
    baseline, depth = make_actor("huginn", False)
    actor = copy.deepcopy(baseline)
    reference = copy.deepcopy(baseline).requires_grad_(False)
    with torch.no_grad():
        reference.lm_head.weight.mul_(0.9)
    data = rollout(depth, "huginn")
    # Explicit recorded-trajectory fixtures; production never changes latent or sampling seeds.
    if mode != "independent":
        data["recurrent_inputs"][2] = replace(data["recurrent_inputs"][2], latent_seed=0)
    if mode == "shared":
        data["recurrent_inputs"][4] = replace(data["recurrent_inputs"][4], latent_seed=0)
        data["recurrent_inputs"][3] = replace(data["recurrent_inputs"][3], latent_seed=1)
    traces_before = [asdict(trace) for trace in data["recurrent_inputs"]]
    behavior = [torch.full((n,), -2.5) for n in data["response_lengths"]]
    data["rollout_log_probs"] = behavior
    items = list(zip(data["tokens"], data["response_lengths"], data["recurrent_inputs"], strict=True))
    inputs = packed(items, all_loops=True)
    with torch.no_grad():
        ref = reference(**inputs)[:, -2]
    refs = list(ref.split(data["response_lengths"]))
    args = Namespace(
        rltt_vocab_tile=5,
        rollout_temperature=0.7,
        entropy_coef=0.03,
        rltt_loop_checkpoint=0,
        rltt_progressive_alpha=1.5,
        kl_loss_coef=0.1,
    )
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.1) for m in (baseline, actor)]
    calls = []
    hook = actor.register_forward_pre_hook(
        lambda module, inputs, kwargs: calls.append(bool(kwargs.get("prefix_only"))), with_kwargs=True
    )
    rows = []
    for update in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        before = [torch.cat([p.detach().flatten().clone() for p in m.parameters()]) for m in (baseline, actor)]
        step = plan_step(
            DataIterator(data, microbatches),
            len(microbatches),
            actor_generation=update,
            model_revision="tiny-cpu",
            loop_depth=depth,
            model_family="huginn_raven",
            suffix_wave_size=wave,
            reduction=reduction,
        )
        assert sum(len(g.indices) for g in step.groups if len(g.indices) > 1) == shared
        assert step.sample_ids == tuple(data["sample_indices"][i] for i in step.indices)
        expected = objective(
            baseline(**inputs), ref, torch.cat(data["advantages"]), torch.cat([step.weights[i] for i in range(5)])
        )
        (expected * scale).backward()
        actor.config.grad_scale_func = lambda loss: loss * scale
        losses, calls[:] = [], []
        for group in step.groups:

            def loss(output, positions, group=group, step=step, losses=losses):
                indices = [group.indices[p] for p in positions]
                value, _ = logical_rltt_loss(
                    args,
                    {
                        "advantages": [data["advantages"][i] for i in indices],
                        "ref_log_probs": [refs[i] for i in indices],
                    },
                    [step.weights[i] for i in indices],
                    output,
                )
                losses.append(float(value.detach()))
                return value

            backward_group(actor, actor, group, data["tokens"], data["recurrent_inputs"], args, loss)
        assert sum(calls) == sum(len(g.indices) > 1 for g in step.groups)
        assert calls.count(False) == 5 - shared
        assert sum(losses) == pytest.approx(float(expected.detach()), abs=2e-6, rel=2e-6)
        gradients = []
        for model in (baseline, actor):
            for parameter in model.parameters():
                assert parameter.grad is not None
                parameter.grad.div_(scale)
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        torch.testing.assert_close(gradients[1], gradients[0], atol=1e-5, rtol=8e-5)
        relative = float((gradients[1] - gradients[0]).norm() / gradients[0].norm())
        assert relative < 2e-6
        for optimizer in optimizers:
            optimizer.step()
        deltas = [
            torch.cat([p.detach().flatten() for p in model.parameters()]) - old
            for model, old in zip((baseline, actor), before, strict=True)
        ]
        assert all(bool(delta.norm() > 0) for delta in deltas)
        torch.testing.assert_close(deltas[1], deltas[0], atol=2e-6, rtol=8e-5)
        for left, right in zip(baseline.parameters(), actor.parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=8e-5)
            for field in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    optimizers[0].state[left][field], optimizers[1].state[right][field], atol=1e-6, rtol=8e-5
                )
        rows.append(
            {
                "update": update,
                "shared_fraction": shared / 5,
                "full_replay_samples": 5 - shared,
                "gradient_relative_l2": relative,
                "delta_max_error": float((deltas[1] - deltas[0]).abs().max()),
            }
        )
    hook.remove()
    assert all(p.grad is None for p in reference.parameters())
    assert traces_before == [asdict(trace) for trace in data["recurrent_inputs"]]
    assert all(bool((scores == -2.5).all()) for scores in behavior)
    record_property("huginn_two_update", json.dumps(rows))


@pytest.mark.parametrize("change", ["missing_seed", "wrong_profile", "boolean_seed"])
def test_huginn_plan_requires_recorded_latent(change):
    data = rollout(3, "huginn")
    values = {
        "missing_seed": {"latent_seed": None},
        "wrong_profile": {"latent_profile": "other"},
        "boolean_seed": {"latent_seed": True},
    }
    data["recurrent_inputs"][0] = replace(data["recurrent_inputs"][0], **values[change])
    with pytest.raises(ValueError, match="recorded latent"):
        plan_step(
            DataIterator(data, [[0, 1, 2, 3, 4]]),
            1,
            actor_generation=0,
            model_revision="tiny-cpu",
            loop_depth=3,
            model_family="huginn_raven",
            suffix_wave_size=2,
            reduction="response_mean",
        )


def test_huginn_prefix_entry_requires_latent_and_rejects_mixed_contract():
    actor, _ = make_actor("huginn", False)
    with pytest.raises(ValueError, match="recorded latent"):
        actor(input_ids=torch.tensor([[1, 3]]), prefix_only=True)
    with pytest.raises(ValueError, match="recorded latent"):
        actor(input_ids=torch.tensor([[1, 3]]), prefix_only=True, prefix_latent_seed=3, recurrent_inputs=[])
