import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm_rlt.models import OuroConfig
from vllm_rlt.models.ouro import OuroForCausalLM

from vime.backends.megatron_utils.hf_to_megatron import _LOADERS
from vime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from vime.backends.megatron_utils.megatron_to_hf import convert_to_hf


class TestOuroConversion(unittest.TestCase):
    def test_physical_parameters_round_trip_with_ddp_names(self):
        torch.manual_seed(9)
        model = OuroForCausalLM(
            OuroConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            )
        )
        args = Namespace(vocab_size=model.config.vocab_size)
        with tempfile.TemporaryDirectory() as folder:
            save_file(model.state_dict(), str(Path(folder) / "model.safetensors"))
            reader = SafetensorReader(folder)
            for prefix in ("", "module.", "module.module.language_model."):
                exported = {}
                for name, parameter in model.named_parameters():
                    wrapped = prefix + name
                    loaded = _LOADERS["ouro"](wrapped, reader, model.config)
                    torch.testing.assert_close(loaded, parameter, rtol=0, atol=0)
                    exported.update(convert_to_hf(args, "OuroConfig", wrapped, loaded))
                self.assertEqual(exported.keys(), model.state_dict().keys())
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(exported[name], parameter, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
