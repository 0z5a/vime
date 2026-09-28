import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt.models.config import OuroConfig
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.models.reference import dense_reference

from vime_plugins.ouro.model import OuroMegatronModel


def test_packed_sequences_match_independent_dense_ouro_and_recompute_gradients():
    torch.manual_seed(9)
    native = OuroForCausalLM(OuroConfig.tiny())
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4,
                               gradient_accumulation_fusion=False, use_cpu_initialization=True)
    model = OuroMegatronModel(config, native)
    sequences = [torch.tensor([1, 4, 7]), torch.tensor([8, 3, 2, 9])]
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries,
                             max_seqlen_q=4, max_seqlen_kv=4)
    tokens = torch.cat(sequences)[None]
    physical_count = sum(parameter.numel() for parameter in model.parameters())
    for k in (2, 3, 4):
        model.set_loop_budget(k)
        model.recompute = False
        model.zero_grad(set_to_none=True)
        actual = model(tokens, packed_seq_params=packed)
        expected = torch.cat([dense_reference(native, seq, k)[-1][2] for seq in sequences])[None]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        actual.square().mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        model.zero_grad(set_to_none=True)
        model.recompute = True
        model(tokens, packed_seq_params=packed).square().mean().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name])
        assert sum(parameter.numel() for parameter in model.parameters()) == physical_count
        assert model.model.early_exit_gate.weight.grad is None
