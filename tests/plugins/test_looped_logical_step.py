"""Logical sample planning and independent two-update scaling/Adam oracles."""

import copy
import json
from argparse import Namespace
from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from megatron.core.pipeline_parallel.schedules import backward_step, forward_step_calc_loss
from test_looped_response_readout import make_actor
from test_rltt_training import packed, trace

from vime.backends.megatron_utils.data import DataIterator
from vime_plugins.looped.logical_step import plan_step
from vime_plugins.looped.prefix import PrefixReplay
from vime_plugins.looped.training import logical_rltt_loss, masked_token_weights, megatron_loss

LAYOUTS = [([[0, 1, 2, 3, 4]], 1), ([[0], [1], [2], [3], [4]], 2), ([[4, 1], [3, 0, 2]], 3)]


class PrecisionMismatch(AssertionError):
    """The recorded two-update trajectory exceeds the unchanged numeric gate."""


def rollout(depth):
    prompts = ([1, 3, 5], [2, 4], [1, 3, 5], [2, 4], [1, 3, 5])
    responses = ([7], [], [4, 6, 8], [9, 11], [3, 5])
    masks = ([1.0], [], [1.0, 0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    return {
        "tokens": [
            torch.tensor([*prompt, *response], dtype=torch.long)
            for prompt, response in zip(prompts, responses, strict=True)
        ],
        "response_lengths": [len(response) for response in responses],
        "loss_masks": [torch.tensor(mask) for mask in masks],
        "sample_indices": [10, 22, 14, 23, 18],
        "group_indices": [5, 7, 5, 7, 9],
        "recurrent_inputs": [trace("ouro", depth, i, len(response)) for i, response in enumerate(responses)],
        "advantages": [torch.tensor(value) for value in ([0.7], [], [-0.3, 0.2, 0.4], [0.8, -0.6], [-0.2, 0.5])],
    }


def plan(data, groups, wave, reduction, generation=10):
    iterator = DataIterator(data, [[1], *groups, [0]])
    iterator.offset = 1
    return iterator, plan_step(
        iterator,
        len(groups),
        actor_generation=generation,
        model_revision="tiny-cpu",
        loop_depth=4,
        suffix_wave_size=wave,
        reduction=reduction,
    )


def dense_scores(actor, items):
    """Independent depth replays with the readout's specified FP32 products."""
    depth, scores, entropies = actor.readout_depth, [], []
    for tokens, length, _ in items:
        begin = len(tokens) - length - 1
        loops = []
        for loop in range(1, depth + 1):
            actor.loop_budget = loop
            hidden = next(actor.iter_readout_states(tokens[: max(1, len(tokens) - 1)]))
            logits = F.linear(hidden[begin : begin + length].float(), actor.lm_head.weight.float())
            log_p = F.log_softmax(logits / 0.7, dim=-1)
            loops.append(log_p.gather(1, tokens[begin + 1 :, None])[:, 0])
        scores.append(torch.stack(loops, -1))
        entropies.append(-(log_p.exp() * log_p).sum(-1))
    actor.loop_budget = depth
    return torch.cat(scores), torch.cat(entropies)


@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("groups,wave", LAYOUTS)
def test_plan_preserves_logical_weights_and_original_identities(groups, wave, reduction):
    data = rollout(4)
    iterator, step = plan(data, groups, wave, reduction)
    assert iterator.offset == step.iterator_offset == 1
    assert step.microbatches == tuple(map(tuple, groups))
    assert step.indices == tuple(i for group in groups for i in group)
    assert step.sample_ids == tuple(data["sample_indices"][i] for i in step.indices)
    assert step.group_ids == tuple(data["group_indices"][i] for i in step.indices)
    expected = masked_token_weights(data["loss_masks"], reduction)
    assert sorted(i for group in step.groups for i in group.indices) == list(range(5))
    assert {group.identity.prompt_tokens for group in step.groups} == {(1, 3, 5), (2, 4)}
    for group in step.groups:
        assert tuple(i for wave in group.waves for i in wave) == group.indices
        assert group.identity.policy_version == 10  # Behavior version is 3.
        assert all(len(indices) <= wave for indices in group.waves)
    for index in step.indices:
        torch.testing.assert_close(step.weights[index], expected[index], atol=0, rtol=0)
    for generation, offset in ((11, 1), (10, 2)):
        with pytest.raises(ValueError, match="another actor update"):
            step.check_current(generation, offset)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "short_step",
        "sample_id",
        "group_id",
        "zero",
        "mask",
        "depth",
        "revision",
        "latent",
        "position",
        "attention",
    ],
)
def test_plan_rejects_incompatible_or_incomplete_step(change):
    data, groups = rollout(4), [[0, 1, 2], [3, 4]]
    if change == "duplicate":
        groups = [[0, 1, 2], [2, 4]]
    elif change == "sample_id":
        data["sample_indices"][1] = 10
    elif change == "group_id":
        data["group_indices"][1] = None
    elif change == "zero":
        data["loss_masks"] = [torch.zeros_like(mask) for mask in data["loss_masks"]]
    elif change == "mask":
        data["loss_masks"][0] = torch.ones(2)
    elif change == "depth":
        data["recurrent_inputs"][0] = replace(data["recurrent_inputs"][0], decode_depths=[2])
    elif change == "revision":
        data["recurrent_inputs"][0] = replace(data["recurrent_inputs"][0], model_revision="other")
    elif change == "latent":
        data["recurrent_inputs"][0] = replace(data["recurrent_inputs"][0], latent_seed=42)
    elif change == "position":
        data["position_ids"] = [torch.arange(len(t)) for t in data["tokens"]]
    elif change == "attention":
        data["attention_mask"] = [torch.ones(1)]
    iterator = DataIterator(data, groups)
    with pytest.raises(ValueError):
        plan_step(
            iterator,
            len(groups) + int(change == "short_step"),
            actor_generation=0,
            model_revision="tiny-cpu",
            loop_depth=4,
            suffix_wave_size=2,
            reduction="response_mean",
        )


@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("groups,wave", LAYOUTS)
@pytest.mark.parametrize("scale", [1.0, 8.0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                strict=True,
                raises=PrecisionMismatch,
                reason="CPU BF16 Adam trajectories exceed the frozen gate; full wrapper/CUDA unqualified",
            ),
        ),
    ],
)
def test_two_adam_updates_match_dense_and_mcore_scaling(groups, wave, reduction, scale, dtype, record_property):
    oracle, depth = make_actor("ouro", False)
    oracle = oracle.to(dtype)
    legacy, shared = copy.deepcopy(oracle), copy.deepcopy(oracle)
    reference = copy.deepcopy(oracle).requires_grad_(False)
    with torch.no_grad():
        reference.lm_head.weight.mul_(0.9)
    models = (oracle, legacy, shared)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.1) for model in models]
    data = rollout(depth)
    lengths = data["response_lengths"]
    items = list(zip(data["tokens"], lengths, data["recurrent_inputs"], strict=True))
    with torch.no_grad():
        ref = reference(**packed(items, all_loops=False))[:, -2]
    data["ref_log_probs"] = list(ref.split(lengths))
    args = Namespace(rltt_progressive_alpha=1.5, kl_loss_coef=0.1, entropy_coef=0.03)
    # This is a CPU BF16 component gate, with separate predeclared tolerances;
    # it does not qualify CUDA or the MCore mixed-precision optimizer wrapper.
    low = dtype == torch.bfloat16
    loss_atol, grad_atol, grad_rtol, relative_limit = (3e-3, 3e-3, 0.04, 0.025) if low else (2e-6, 1e-5, 8e-5, 2e-6)
    measurements = []
    for update in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad()
        before = [[p.detach().clone() for p in model.parameters()] for model in models]
        iterator, step = plan(data, groups, wave, reduction, generation=10 + update)
        scores, entropy = dense_scores(oracle, items)
        credit = torch.arange(1, depth + 1).float().pow(1.5)
        credit /= credit.sum()
        ratio = ref - scores[:, -1]
        weights = torch.cat([step.weights[i] for i in range(5)])
        expected = (
            (
                -(scores * credit).sum(-1) * torch.cat(data["advantages"])
                + 0.1 * (ratio.expm1() - ratio)
                - 0.03 * entropy
            )
            * weights
        ).sum()
        (expected * scale).backward()
        legacy.config.grad_scale_func = lambda loss: loss * scale
        legacy.config.deallocate_pipeline_outputs = False
        loss_sum = expected.new_zeros(())
        log_sum = expected.new_zeros(())
        for group in groups:
            output = legacy(**packed([items[i] for i in group], all_loops=True))
            batch = {key: [data[key][i] for i in group] for key in ("advantages", "ref_log_probs")}
            logs = []
            loss, _ = forward_step_calc_loss(
                legacy,
                output,
                partial(megatron_loss, args, batch, [step.weights[i] for i in group], len(groups), 5),
                legacy.config,
                None,
                False,
                len(groups),
                logs,
                cp_group_size=1,
                is_last_stage=True,
            )
            loss_sum += loss.detach()
            log_sum += logs[0]["values"][1] / 5
            backward_step(None, loss, None, None, legacy.config)
        shared_sum = expected.new_zeros(())
        for group in step.groups:
            prompt = data["tokens"][group.indices[0]][: len(group.identity.prompt_tokens)]
            state = PrefixReplay(
                shared.prefix_program(prompt),
                group.identity,
                shared,
                shared.lm_head.weight,
                vocab_tile=5,
                temperature=0.7,
                all_loops=True,
                entropy=True,
            )
            responses = tuple(data["tokens"][i][len(prompt) :] for i in group.indices)
            local = {index: position for position, index in enumerate(group.indices)}
            waves = tuple(tuple(local[i] for i in indices) for indices in group.waves)

            def objective(output, indices, group=group, step=step):
                actual = [group.indices[i] for i in indices]
                batch = {key: [data[key][i] for i in actual] for key in ("advantages", "ref_log_probs")}
                loss, metrics = logical_rltt_loss(args, batch, [step.weights[i] for i in actual], output)
                torch.testing.assert_close(loss.detach(), metrics["loss"], atol=0, rtol=0)
                return loss

            shared_sum += state.backward_suffixes(
                responses,
                waves,
                objective,
                identity=group.identity,
                batch_suffixes=True,
                grad_scale_func=lambda loss: loss * scale,
            )
        loss_pass = [
            bool(torch.isclose(value, expected.detach(), atol=loss_atol, rtol=loss_atol))
            for value in (loss_sum, log_sum, shared_sum)
        ]
        gradients = [
            torch.cat([p.grad.float().flatten() for p in model.parameters() if p.requires_grad]) / scale
            for model in models
        ]
        assert all(bool(torch.isfinite(gradient).all()) and bool(gradient.norm() > 0) for gradient in gradients)
        relative, gradient_pass = [], []
        for actual in gradients[1:]:
            error = float((actual - gradients[0]).double().norm() / gradients[0].double().norm())
            gradient_pass.append(
                bool(torch.isclose(actual, gradients[0], atol=grad_atol, rtol=grad_rtol).all())
                and error < relative_limit
            )
            relative.append(error)
        for model, optimizer in zip(models, optimizers, strict=True):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad.div_(scale)
            optimizer.step()
        deltas = [
            torch.cat(
                [
                    (p.detach().float() - old.float()).flatten()
                    for p, old in zip(model.parameters(), previous, strict=True)
                ]
            )
            for model, previous in zip(models, before, strict=True)
        ]
        assert all(bool(delta.norm() > 0) for delta in deltas)
        delta_pass = [
            bool(torch.isclose(actual, deltas[0], atol=2e-4 if low else 2e-6, rtol=0.08 if low else 8e-5).all())
            for actual in deltas[1:]
        ]
        assert any(not torch.equal(p, q) for p, q in zip(shared.parameters(), reference.parameters(), strict=True))
        assert all(p.grad is None for p in reference.parameters())
        measurements.append(
            {
                "update": update,
                "loss": float(expected.detach()),
                "loss_pass": loss_pass,
                "gradient_relative_l2": relative,
                "gradient_pass": gradient_pass,
                "delta_norms": [float(d.norm()) for d in deltas],
                "delta_max_errors": [float((d - deltas[0]).abs().max()) for d in deltas[1:]],
                "delta_pass": delta_pass,
            }
        )
    record_property("two_update_oracle", json.dumps(measurements))
    if not all(all(row[key]) for row in measurements for key in ("loss_pass", "gradient_pass", "delta_pass")):
        raise PrecisionMismatch("Trajectory exceeds the original precision tolerance; see two_update_oracle")
