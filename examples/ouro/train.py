"""Synchronous Ouro recipe using VIME's existing Megatron train/loss/checkpoint path.

Launch with torchrun; every DP rank runs the same global K and local full groups.
The optional vllm-rlt RL engine is a companion dependency, not upstream vLLM.
"""

import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model
from transformers import AutoTokenizer
from vllm_rlt.engine.rl_engine import RLEngine
from vllm_rlt.models.ouro import OuroForCausalLM

from vime.backends.megatron_utils.checkpoint import load_checkpoint
from vime.backends.megatron_utils.data import get_data_iterator
from vime.backends.megatron_utils.initialize import init
from vime.backends.megatron_utils.loss import compute_advantages_and_returns
from vime.backends.megatron_utils.model import save, setup_model_and_optimizer, train
from vime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from vime.utils.arguments import parse_args
from vime.utils.distributed_utils import init_gloo_group
from vime.utils.reward_normalization import normalize_rewards
from vime_plugins.ouro.budget import BudgetSchedule, ExecutionPlan, synchronize_plan


def add_arguments(parser):
    parser.add_argument("--ouro-depths", type=int, nargs="+", default=[4])
    parser.add_argument("--ouro-run-dir", type=Path, required=True)
    parser.add_argument("--ouro-resume", action="store_true")
    parser.add_argument("--ouro-eval-data", type=Path)
    parser.add_argument("--ouro-eval-prompts", type=int, default=4)
    parser.add_argument("--ouro-eval-interval", type=int, default=3)
    parser.add_argument("--ouro-kv-blocks", type=int, default=4096)
    return parser


def encode_prompt(tokenizer, question) -> list[int]:
    if isinstance(question, list):
        question = "\n".join(message["content"] for message in question)
    return tokenizer.encode(question + "\nGive your final answer in \\boxed{}.\n###Response\n")


def evaluate(args, engine: RLEngine, tokenizer, update: int, started: float) -> None:
    records = [json.loads(line) for line in args.ouro_eval_data.read_text().splitlines()]
    records = records[: args.ouro_eval_prompts][args.rank :: args.world_size]
    if not records:
        raise ValueError("Evaluation needs at least one prompt per DP rank")
    for loops in (2, 3, 4):
        results = engine.generate_batch(
            [encode_prompt(tokenizer, row["prompt"]) for row in records],
            loops=loops,
            max_tokens=args.rollout_max_response_len,
            seed=args.seed + 100000 + args.rank * 100,
            temperature=args.rollout_temperature,
        )
        with (args.ouro_run_dir / f"evaluations-rank{args.rank}.jsonl").open("a") as output:
            for row, result in zip(records, results, strict=True):
                text = tokenizer.decode(result.token_ids)
                output.write(
                    json.dumps(
                        {
                            "update": update,
                            "K": loops,
                            "policy_version": result.policy_version,
                            "problem_id": row["metadata"]["problem_id"],
                            "reward": get_deepscaler_rule_based_reward("###Response\n" + text, row["label"]),
                            "response_tokens": len(result.token_ids),
                            "finish_reason": result.finish_reason,
                            "elapsed_seconds": time.monotonic() - started,
                            "response": text,
                        }
                    )
                    + "\n"
                )
    dist.barrier()


def main():
    started = time.monotonic()
    args = parse_args(add_arguments)
    if args.kl_coef or args.use_kl_loss or args.use_critic or args.rollout_top_p != 1:
        raise ValueError("First Ouro recipe supports GRPO without reference/critic and full-vocab sampling")
    if args.advantage_estimator != "grpo" or args.rollout_batch_size % args.world_size:
        raise ValueError("Use GRPO and a prompt count divisible by DP size")
    schedule = BudgetSchedule(tuple(args.ouro_depths))
    args.ouro_run_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    init_gloo_group()
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    init(args)
    chunks, optimizer, scheduler = setup_model_and_optimizer(args)
    chunks[0].role = "actor"
    actor = unwrap_model(chunks)[0]
    contract = {
        "depths": list(schedule.depths),
        "seed": args.seed,
        "temperature": args.rollout_temperature,
        "prompts_per_update": args.rollout_batch_size,
        "group_size": args.n_samples_per_prompt,
        "max_response_tokens": args.rollout_max_response_len,
        "data_sha256": hashlib.sha256(Path(args.prompt_data).read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((Path(args.hf_checkpoint) / "config.json").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((Path(args.hf_checkpoint) / "tokenizer.json").read_bytes()).hexdigest(),
    }
    start = 0
    if args.ouro_resume:
        manifest = json.loads((Path(args.load) / "ouro-plan.json").read_text())
        if manifest["contract"] != contract:
            raise ValueError("Checkpoint budget schedule differs from requested schedule")
        iteration, _ = load_checkpoint(chunks, optimizer, scheduler, {}, False)
        start = iteration + 1
        if manifest["next_update"] != start:
            raise ValueError("Checkpoint and budget/data cursor disagree")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
    rollout = OuroForCausalLM.from_pretrained(
        args.hf_checkpoint, device=torch.cuda.current_device(), dtype=torch.bfloat16
    )
    engine = RLEngine(rollout, num_blocks=args.ouro_kv_blocks)
    engine.publish(dict(actor.named_parameters()), version=start + 1)
    (args.ouro_run_dir / f"contract-rank{args.rank}.json").write_text(json.dumps(contract, indent=2))
    records = [json.loads(line) for line in Path(args.prompt_data).read_text().splitlines()]
    local_prompts = args.rollout_batch_size // args.world_size
    normalization_args = copy.copy(args)
    normalization_args.rollout_batch_size = local_prompts
    log = args.ouro_run_dir / f"metrics-rank{args.rank}.jsonl"
    if args.ouro_eval_data is not None:
        evaluate(args, engine, tokenizer, start, started)
    for update in range(start, args.num_rollout):
        step_started = time.monotonic()
        plan = ExecutionPlan(schedule.at(update), engine.policy_version, args.rollout_temperature)
        synchronize_plan(plan)
        actor.set_loop_budget(plan.loop_budget)
        first = update * args.rollout_batch_size + args.rank * local_prompts
        batch = [records[index % len(records)] for index in range(first, first + local_prompts)]
        prompts, labels, metadata = [], [], []
        for item in batch:
            prompt = encode_prompt(tokenizer, item["prompt"])
            for _ in range(args.n_samples_per_prompt):
                prompts.append(prompt)
                labels.append(item["label"])
                metadata.append(plan.metadata(item["metadata"]["problem_id"]))
        plan.validate(metadata, args.n_samples_per_prompt)
        generated = engine.generate_batch(
            prompts,
            loops=plan.loop_budget,
            max_tokens=args.rollout_max_response_len,
            seed=args.seed + update * args.global_batch_size + args.rank * len(prompts),
            temperature=args.rollout_temperature,
        )
        rollout_seconds = time.monotonic() - step_started
        for result in generated:
            if result.policy_version != plan.policy_version or result.exit_depths != [plan.loop_budget] * len(
                result.token_ids
            ):
                raise ValueError("Rollout did not execute the requested policy")
        rewards = [
            get_deepscaler_rule_based_reward("###Response\n" + tokenizer.decode(result.token_ids), label)
            for result, label in zip(generated, labels, strict=True)
        ]
        sequences = [prompt + result.token_ids for prompt, result in zip(prompts, generated, strict=True)]
        responses = [len(result.token_ids) for result in generated]
        data = {
            "tokens": [torch.tensor(sequence, device="cuda") for sequence in sequences],
            "total_lengths": [len(sequence) for sequence in sequences],
            "response_lengths": responses,
            "loss_masks": [torch.ones(length, device="cuda") for length in responses],
            "rollout_log_probs": [torch.tensor(result.log_probs, device="cuda") for result in generated],
            "rewards": normalize_rewards(normalization_args, rewards),
            "rollout_mask_sums": [torch.tensor(n, device="cuda") for n in responses],
            "micro_batch_indices": [[index] for index in range(len(sequences))],
        }
        compute_advantages_and_returns(args, data)
        actor.block_tokens = 0
        train_start = time.monotonic()
        train(
            update, chunks, optimizer, scheduler, get_data_iterator(data), [len(sequences)], [args.global_batch_size]
        )
        train_seconds = time.monotonic() - train_start
        publish_start = time.monotonic()
        engine.publish(dict(actor.named_parameters()), version=plan.policy_version + 1)
        stats = {
            "update": update,
            "K": plan.loop_budget,
            "policy_version": plan.policy_version,
            "execution_config_hash": plan.config_hash,
            "rewards": rewards,
            "response_lengths": responses,
            "truncated": [result.finish_reason == "length" for result in generated],
            "rollout_seconds": rollout_seconds,
            "train_seconds": train_seconds,
            "publish_seconds": time.monotonic() - publish_start,
            "step_seconds": time.monotonic() - step_started,
            "job_elapsed_seconds": time.monotonic() - started,
            "allocated_gpu_hours": args.world_size * (time.monotonic() - started) / 3600,
            "rollout_prefill_block_tokens": engine.block_tokens["prefill"],
            "rollout_decode_block_tokens": engine.block_tokens["decode"],
            "trainer_layer_invocation_tokens_including_recompute": actor.block_tokens,
            "trainer_backward_profiled_cost": None,
        }
        with log.open("a") as stream:
            stream.write(json.dumps(stats) + "\n")
        if args.ouro_eval_data is not None and (update + 1) % args.ouro_eval_interval == 0:
            evaluate(args, engine, tokenizer, update + 1, started)
    # Retain the newest checkpoint only after its distributed save completes.
    save(args.num_rollout - 1, chunks, optimizer, scheduler)
    dist.barrier()
    if args.rank == 0:
        (Path(args.save) / "ouro-plan.json").write_text(
            json.dumps({"contract": contract, "next_update": args.num_rollout})
        )
        checkpoint_root = Path(args.save)
        newest = checkpoint_root / f"iter_{args.num_rollout - 1:07d}"
        if not (newest / ".metadata").is_file():
            raise RuntimeError("Latest checkpoint is incomplete; keeping previous checkpoints")
        for older in checkpoint_root.glob("iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]"):
            if older != newest and older.is_dir() and not older.is_symlink():
                shutil.rmtree(older)
    dist.barrier()
    dist.destroy_process_group()
    (args.ouro_run_dir / f"complete-rank{args.rank}.json").write_text(
        json.dumps(
            {"completed": True, "next_update": args.num_rollout, "job_elapsed_seconds": time.monotonic() - started}
        )
    )


if __name__ == "__main__":
    main()
