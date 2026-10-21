"""All-loop readout matches independent depth replays and every trainable gradient."""

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime_plugins.huginn.model import HuginnMegatronModel
from vime_plugins.looped.response import response_log_probs
from vime_plugins.nanbeige.model import NanbeigeMegatronModel
from vime_plugins.ouro.model import OuroMegatronModel


def make_actor(family, recompute):
    torch.manual_seed(81)
    config = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2, kv_channels=4, ffn_hidden_size=16)
    if family == "ouro":
        native = OuroForCausalLM(
            OuroConfig(
                vocab_size=13,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=4,
                total_ut_steps=4,
                max_position_embeddings=32,
            )
        )
        return OuroMegatronModel(config, native, recompute=recompute), 4
    if family == "nanbeige":
        native = NanbeigeForCausalLM(
            NanbeigeConfig(
                vocab_size=13,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
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
        return NanbeigeMegatronModel(config, native, recompute=recompute), 2
    native = HuginnForCausalLM(
        HuginnConfig(
            n_embd=8,
            n_heads=2,
            n_layers=3,
            n_layers_in_prelude=1,
            n_layers_in_recurrent_block=1,
            n_layers_in_coda=1,
            intermediate_size=16,
            mean_recurrence=3,
            block_size=32,
            vocab_size=13,
            padded_vocab_size=13,
        )
    )
    return HuginnMegatronModel(config, native, recompute=recompute), 3


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("response_length", [0, 1, 4])
@pytest.mark.parametrize("readout", ["streamed", "dense"])
def test_response_boundary_loops_and_shared_parameter_gradients(
    family, recompute, response_length, readout, record_property
):
    actor, depth = make_actor(family, recompute)
    oracle = copy.deepcopy(actor)
    tokens = torch.tensor([1, 3, 5, 7, 9, 11, 4])
    begin = len(tokens) - response_length - 1
    reference = []
    for loop in range(1, depth + 1):
        if family == "huginn":
            oracle.huginn_config = replace(oracle.huginn_config, mean_recurrence=loop)
            logits = oracle._sequence(tokens[:-1], 19)
        else:
            oracle.loop_budget = loop
            logits = oracle._sequence(tokens[:-1])
        reference.append(F.log_softmax(logits[begin:].float() / 0.7, -1).gather(1, tokens[begin + 1 :, None])[:, 0])
    reference = torch.stack(reference, -1)
    if readout == "streamed":
        actual = response_log_probs(
            actor, tokens, response_length, vocab_tile=5, temperature=0.7, all_loops=True, latent_seed=19
        )
    else:
        actual = torch.stack(
            [
                F.log_softmax(actor.lm_head(hidden[begin:]).float() / 0.7, -1).gather(1, tokens[begin + 1 :, None])[
                    :, 0
                ]
                for hidden in actor.iter_readout_states(tokens[:-1], all_loops=True, latent_seed=19)
            ],
            -1,
        )
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)
    factor = torch.arange(1, depth + 1).float()[None] * torch.linspace(-0.5, 1.0, response_length)[:, None]
    (actual * factor).sum().backward()
    (reference * factor).sum().backward()
    max_abs, error_squared, norm_squared = 0.0, 0.0, 0.0
    for (name, parameter), (other_name, other) in zip(
        actor.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and other.grad is not None, name
            error = parameter.grad - other.grad
            max_abs = max(max_abs, error.abs().max().item())
            error_squared += error.double().square().sum().item()
            norm_squared += other.grad.double().square().sum().item()
            # Shared recurrence sums contributions in a different FP32 order
            # from independent depth replays, including with a dense readout.
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-5, rtol=8e-5)
    relative_l2 = (error_squared / max(norm_squared, 1e-30)) ** 0.5
    assert relative_l2 < 2e-6
    record_property("gradient_max_abs", max_abs)
    record_property("gradient_relative_l2", relative_l2)


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
def test_terminal_readout_keeps_only_final_loop(family):
    actor, _ = make_actor(family, False)
    tokens = torch.tensor([1, 2, 3, 4])
    all_scores = response_log_probs(actor, tokens, 2, vocab_tile=4, temperature=1, all_loops=True, latent_seed=31)
    final = response_log_probs(actor, tokens, 2, vocab_tile=4, temperature=1, all_loops=False, latent_seed=31)
    torch.testing.assert_close(final, all_scores[:, -1:])


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
def test_one_token_prompt_empty_response(family):
    actor, depth = make_actor(family, True)
    scores = response_log_probs(actor, torch.tensor([1]), 0, vocab_tile=4, temperature=1, all_loops=True)
    assert scores.shape == (0, depth)
    scores.sum().backward()
    for parameter in actor.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.equal(parameter.grad, torch.zeros_like(parameter))
