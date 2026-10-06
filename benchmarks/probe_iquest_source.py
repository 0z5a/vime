"""Execute pinned, unmodified IQuest code with a tiny CPU configuration."""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("cached", "uncached", "checkpointed", "decode"),
        required=True,
    )
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--explicit-mask", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "benchmarks/results/iquest-contract/reference"
    package = types.ModuleType("pinned_iquest")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    config_module = importlib.import_module("pinned_iquest.configuration_iquestloopcoder")
    model_module = importlib.import_module("pinned_iquest.modeling_iquestloopcoder")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = config_module.IQuestLoopCoderConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=131072,
        loop_num=2,
        loop_window_size=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = model_module.IQuestLoopCoderForCausalLM(config).float()
    with torch.no_grad():
        for gate in model.model.gate_projections:
            gate.weight.normal_(std=0.1)
            gate.bias.normal_(std=0.1)
    tokens = torch.randint(3, config.vocab_size, (1, args.length))
    receipt = {
        "case": args.case,
        "length": args.length,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "revision": json.loads((source / "source_manifest.json").read_text())["revision"],
        "device": "cpu",
        "official_weights": False,
        "config": config.to_dict(),
        "unique_parameter_count": sum(p.numel() for p in model.parameters()),
        "explicit_causal_mask_input": args.explicit_mask,
    }
    args.output.with_suffix(".input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"state": "TINY_INPUT_READY", "case": args.case, "length": args.length}),
        flush=True,
    )

    def forward(input_ids: torch.Tensor, cache=None):
        kwargs = {}
        if args.explicit_mask:
            seen = 0 if cache is None else cache.get_seq_length()
            positions = torch.arange(seen, seen + input_ids.shape[1])
            allowed = torch.arange(seen + input_ids.shape[1])[None, :] <= positions[:, None]
            mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]
            kwargs = {
                "attention_mask": {"full_attention": mask},
                "cache_position": positions,
                "position_ids": positions[None],
            }
        return model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=args.case != "uncached",
            **kwargs,
        )

    if args.case == "checkpointed":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.case == "decode":
        model.eval()
        with torch.no_grad():
            full = forward(tokens).logits
            cache = None
            chunks = []
            for position in range(args.length):
                result = forward(tokens[:, position : position + 1], cache)
                cache = result.past_key_values
                chunks.append(result.logits)
            decoded = torch.cat(chunks, dim=1)
        receipt["prefill_decode_max_abs"] = (full - decoded).abs().max().item()
        receipt["prefill_decode_allclose_1e-5"] = torch.allclose(full, decoded, atol=1e-5, rtol=1e-5)
        receipt["prefill_finite"] = bool(torch.isfinite(full).all())
        receipt["decode_finite"] = bool(torch.isfinite(decoded).all())
    else:
        result = forward(tokens)
        loss = result.logits.square().mean()
        loss.backward()
        receipt["loss"] = loss.item()
        receipt["all_parameters_have_finite_gradients"] = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        receipt["gradient_norm"] = sum(p.grad.square().sum().item() for p in model.parameters()) ** 0.5
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "config"}))


if __name__ == "__main__":
    main()
