"""Differentiable replay of the native full-prefill and mixed-depth KV policy."""

import copy
import unittest

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt import LLM, CacheConfig, ExitConfig, SamplingParams
from vllm_rlt.models import OuroConfig, OuroForCausalLM

from vime_plugins.ouro.model import OuroMegatronModel


class OuroExecutionReplayTest(unittest.TestCase):
    def test_mixed_depth_probabilities_and_recompute_gradients(self):
        torch.manual_seed(23)
        native = OuroForCausalLM(
            OuroConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                total_ut_steps=4,
                eos_token_id=0,
            )
        )
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=True,
        )
        actor = OuroMegatronModel(config, copy.deepcopy(native))
        prompt = [3, 4, 7]
        for depths in ((4, 2, 2, 2, 2), (4, 2, 4, 3, 2), (4, 4, 4, 4, 4)):
            engine = LLM(
                native,
                cache_config=CacheConfig(num_blocks=128),
                exit_config=ExitConfig("trace", depths_by_request={"0": depths}),
            )
            output = engine.generate(
                [prompt],
                SamplingParams(
                    max_tokens=5,
                    temperature=1,
                    logprobs=0,
                    ignore_eos=True,
                ),
            )[0]
            tokens = torch.tensor(prompt + output.token_ids)[None]
            inputs = torch.tensor([4] * len(prompt) + output.exit_depths[1:] + [1])
            actor.recompute = False
            actor.zero_grad(set_to_none=True)
            logits = actor(tokens, execution_depths=inputs)[0, len(prompt) - 1 : -1]
            selected = logits.float().log_softmax(-1).gather(1, torch.tensor(output.token_ids)[:, None])[:, 0]
            torch.testing.assert_close(selected, torch.tensor(output.log_probs), atol=2e-6, rtol=2e-6)
            (-selected.mean()).backward()
            gradients = {name: p.grad.clone() for name, p in actor.named_parameters() if p.requires_grad}
            actor.zero_grad(set_to_none=True)
            actor.recompute = True
            logits = actor(tokens, execution_depths=inputs)[0, len(prompt) - 1 : -1]
            selected = logits.log_softmax(-1).gather(1, torch.tensor(output.token_ids)[:, None])[:, 0]
            (-selected.mean()).backward()
            for name, parameter in actor.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-7, rtol=2e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
