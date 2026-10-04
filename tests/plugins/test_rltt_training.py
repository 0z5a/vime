"""Packed replay, frozen reference and MCore-scaled RLTT gradient oracles."""

import copy
from argparse import Namespace
from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from test_looped_response_readout import make_actor

from vime.backends.megatron_utils.data import DataIterator
from vime.utils.types import RecurrentTrace
from vime_plugins.looped.response import ResponseReadout
from vime_plugins.looped.training import (
    collect_log_probs,
    masked_token_weights,
    megatron_loss,
    weights_for_step,
    validate_rltt_args,
)


def trace(family, depth, seed, length):
    return RecurrentTrace(
        schema_version=1,
        model_family="huginn_raven" if family == "huginn" else family,
        model_revision="tiny-cpu",
        engine_revision="test",
        runtime_epoch="test",
        policy_version=3,
        publication_digest="test",
        request_id=str(seed),
        seed=seed,
        prefill_depth=depth,
        decode_depths=[depth] * length,
        temperature=0.7,
        finish_reason="length",
        latent_seed=seed if family == "huginn" else None,
        latent_profile="like-init-cpu-f32-v1" if family == "huginn" else None,
    )


def packed(items, *, all_loops, entropy=True, temperature=0.7):
    tokens = [item[0] for item in items]
    lengths = [item[1] for item in items]
    boundaries = torch.tensor([0, *torch.tensor([len(t) for t in tokens]).cumsum(0).tolist()], dtype=torch.int32)
    boundaries = torch.cat([boundaries, boundaries[-1:] + 2])
    values = torch.cat([*tokens, torch.zeros(2, dtype=torch.long)])[None]
    mask = torch.zeros_like(values)
    return dict(
        input_ids=values,
        packed_seq_params=PackedSeqParams(cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries, qkv_format="thd"),
        recurrent_inputs=[item[2] for item in items],
        loss_mask=mask,
        readout=ResponseReadout(tuple(lengths), 5, temperature, all_loops, entropy),
    )


def dense_replays(actor, family, items):
    depth = actor.readout_depth
    scores, entropies = [], []
    for tokens, length, identity in items:
        begin = len(tokens) - length - 1
        loops = []
        for loop in range(1, depth + 1):
            if family == "huginn":
                actor.huginn_config = replace(actor.huginn_config, mean_recurrence=loop)
                logits = actor._sequence(tokens[:-1], identity.latent_seed)
            else:
                actor.loop_budget = loop
                logits = actor._sequence(tokens[:-1])
            log_p = F.log_softmax(logits[begin : begin + length].float() / 0.7, -1)
            loops.append(log_p.gather(1, tokens[begin + 1 :, None])[:, 0])
        scores.append(torch.stack(loops, -1))
        entropies.append(-(log_p.exp() * log_p).sum(-1))
    if family == "huginn":
        actor.huginn_config = replace(actor.huginn_config, mean_recurrence=depth)
    else:
        actor.loop_budget = depth
    return torch.cat(scores), torch.cat(entropies)


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("recompute", [False, True])
def test_packed_reference_and_logical_microbatch_gradient(family, reduction, recompute):
    source, depth = make_actor(family, recompute)
    items = [
        (torch.tensor(tokens), length, trace(family, depth, 19 + i, length))
        for i, (tokens, length) in enumerate([([1, 3, 5, 7, 9], 3), ([2, 4], 0), ([8, 2, 3, 5], 1)])
    ]
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.empty(0), torch.ones(1)]
    advantages = [torch.tensor([0.7, -0.2, 0.4]), torch.empty(0), torch.tensor([-0.3])]
    reference = copy.deepcopy(source).requires_grad_(False)
    with torch.no_grad():
        # A distinct frozen reference catches accidental current/old-policy substitution.
        reference.lm_head.weight.mul_(0.9)
        reference_output = reference(**packed(items, all_loops=False))
        _, ref = collect_log_probs(reference_output, response_lengths=[3, 0, 1], with_entropy=True)
        ref_dense, ref_entropy = dense_replays(reference, family, items)
        torch.testing.assert_close(torch.cat(ref["log_probs"]), ref_dense[:, -1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(torch.cat(ref["entropy"]), ref_entropy, atol=2e-6, rtol=2e-6)

    args = Namespace(rltt_progressive_alpha=1.5, kl_loss_coef=0.1, entropy_coef=0.03)
    oracle = copy.deepcopy(source)
    scores, entropy = dense_replays(oracle, family, items)
    credit = torch.arange(1, depth + 1).float().pow(1.5)
    credit /= credit.sum()
    ratio = torch.cat(ref["log_probs"]) - scores[:, -1]
    per_token = -(scores * credit).sum(-1) * torch.cat(advantages) + 0.1 * (ratio.exp() - ratio - 1) - 0.03 * entropy
    chunks = per_token.split([3, 0, 1])
    if reduction == "token_mean":
        expected = sum((x * mask).sum() for x, mask in zip(chunks, masks, strict=True)) / 3
    else:
        expected = sum((x * mask).sum() / mask.sum().clamp_min(1) for x, mask in zip(chunks, masks, strict=True)) / 2
    expected.backward()

    for groups in ([[0, 1, 2]], [[0], [1], [2]], [[2], [0, 1]]):
        actor = copy.deepcopy(source)
        iterator = DataIterator({"loss_masks": masks}, [[2], *groups])
        iterator.offset = 1  # Ignore another logical step in the same rollout iterator.
        weights = weights_for_step(iterator, len(groups), reduction)
        total = 0.0
        logged = 0.0
        for group in groups:
            output = actor(**packed([items[i] for i in group], all_loops=True))
            loss_fn = partial(
                megatron_loss,
                args,
                {"advantages": [advantages[i] for i in group], "ref_log_probs": [ref["log_probs"][i] for i in group]},
                [weights[i] for i in group],
                len(groups),
                3,
            )
            logs = []
            loss, count = forward_step_calc_loss(
                actor,
                output,
                loss_fn,
                actor.config,
                None,
                False,
                len(groups),
                logs,
                cp_group_size=1,
                is_last_stage=True,
            )
            log = logs[0]
            assert count.item() == 1
            loss.backward()
            total += loss.item()
            logged += log["values"][1].item() / 3
        assert total == pytest.approx(expected.item(), abs=2e-6)
        assert logged == pytest.approx(expected.item(), abs=2e-6)
        for actual, wanted in zip(actor.parameters(), oracle.parameters(), strict=True):
            if actual.requires_grad:
                assert actual.grad is not None and wanted.grad is not None
                torch.testing.assert_close(actual.grad, wanted.grad, atol=5e-6, rtol=1e-4)
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_empty_logical_batch_is_not_an_optimizer_update():
    with pytest.raises(ValueError, match="loss-contributing"):
        masked_token_weights([torch.empty(0), torch.zeros(2)], "response_mean")


def test_reference_is_mandatory_at_zero_kl():
    args = Namespace(rltt_progressive_alpha=0, kl_loss_coef=0, entropy_coef=0)
    with pytest.raises(ValueError, match="frozen initial reference"):
        megatron_loss(
            args, {"advantages": [torch.ones(1)], "ref_log_probs": None}, [torch.ones(1)], 1, 1, torch.zeros(1, 3)
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout_backend": "vllm"}, "native recurrent"),
        ({"actor_num_gpus_per_node": 2}, "one learner"),
        ({"context_parallel_size": 2}, "TP=PP=CP"),
        ({"advantage_estimator": "ppo"}, "grouped GRPO"),
        ({"ref_load": None}, "initial reference"),
        ({"compute_advantages_and_returns": False}, "initial reference"),
        ({"ref_update_interval": 1}, "frozen initial reference"),
        ({"kl_coef": 0.1}, "unclipped PG"),
        ({"calculate_per_token_loss": True}, "logical denominator"),
    ],
)
def test_reject_unqualified_runtime_contracts(overrides, match):
    args = Namespace(
        rollout_backend="vllm-rlt",
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        advantage_estimator="grpo",
        n_samples_per_prompt=4,
        kl_coef=0,
        kl_loss_coef=0,
        flow_dppo_divergence_budget=None,
        use_tis=False,
        use_opsm=False,
        calculate_per_token_loss=False,
        enable_mtp_training=False,
        dspark_enabled=False,
        ref_load="initial-checkpoint",
        ref_update_interval=None,
        compute_advantages_and_returns=True,
        rltt_vocab_tile=4,
        rltt_progressive_alpha=0,
    )
    validate_rltt_args(args)
    vars(args).update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_rltt_args(args)
