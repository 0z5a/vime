"""Measure full-prefill/decode trace replay against uniform-depth recomputation."""

import argparse
import json
from pathlib import Path

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import AutoTokenizer
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.models import OuroForCausalLM

from vime_plugins.ouro.model import OuroMegatronModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    cfg = native.config
    actor = OuroMegatronModel(
        TransformerConfig(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=True,
        ),
        native,
    )
    prompt = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False).encode("Compute 17 + 25.\n")
    report = {}
    for name, loops, threshold, asynchronous in (
        ("decode-2", 2, 1, False),
        ("decode-3", 3, 1, False),
        ("decode-4", 4, 1, False),
        ("early-exit", 4, 0, False),
        ("delayed-exit", 4, 0, True),
    ):
        engine = LLM(
            native,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=128),
            scheduler_config=SchedulerConfig(max_num_seqs=1),
            exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
            execution_config=ExecutionConfig(cuda_graphs=True, async_scheduling=asynchronous),
        )
        output = engine.generate(
            [prompt],
            SamplingParams(
                max_tokens=8,
                max_loops=loops,
                temperature=0.9,
                seed=None,
                logprobs=0,
                logprobs_mode="processed",
                exit_threshold=threshold,
                ignore_eos=True,
            ),
        )[0]
        assert cfg.total_ut_steps == 4 and output.exit_depths[0] == 4
        replay = engine.generate([prompt], output.sampling_params)[0]
        assert output.token_ids == replay.token_ids and output.log_probs == replay.log_probs
        tokens = torch.tensor(prompt + output.token_ids, device="cuda")[None]
        depths = torch.tensor([4] * len(prompt) + output.exit_depths[1:] + [1], device="cuda")
        selected = torch.tensor(output.token_ids, device="cuda")[:, None]
        expected = torch.tensor(output.log_probs, device="cuda")
        errors = {}
        for mode, budget, trace in (
            ("actual_trace", loops, depths),
            ("uniform_decode_budget", loops, None),
            ("uniform_full_depth", 4, None),
        ):
            actor.set_loop_budget(budget)
            logits = actor(tokens, execution_depths=trace)[0, len(prompt) - 1 : -1]
            scores = (logits.float() / 0.9).log_softmax(-1).gather(1, selected)[:, 0]
            difference = (scores - expected).abs()
            errors[mode] = {"mean_absolute": difference.mean().item(), "maximum_absolute": difference.max().item()}
        report[name] = {
            "prompt_tokens": len(prompt),
            "exit_depths": output.exit_depths,
            "effective_seed": output.sampling_params.seed,
            "seed_replay_exact": True,
            "score_errors": errors,
        }
        engine.close()
        print(name, report[name], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
