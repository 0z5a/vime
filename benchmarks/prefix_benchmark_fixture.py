"""Identical tiny providers with an explicit positional capacity for each workload."""

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime_plugins.huginn.model import HuginnMegatronModel
from vime_plugins.nanbeige.model import NanbeigeMegatronModel
from vime_plugins.ouro.model import OuroMegatronModel


def make_actor(family, recompute, context_length=32):
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
                max_position_embeddings=context_length,
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
                max_position_embeddings=context_length,
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
            block_size=context_length,
            vocab_size=13,
            padded_vocab_size=13,
        )
    )
    return HuginnMegatronModel(config, native, recompute=recompute), 3
