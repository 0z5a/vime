"""Complete held-out datasets through the committed native recurrent policy."""

import copy
import hashlib
import json
import time
from argparse import Namespace
from typing import TYPE_CHECKING

import ray

from vime.rollout.base_types import RolloutFnEvalOutput
from vime.rollout.vllm_rlt_rollout import _tokenizer, decode_and_reward
from vime.utils.data import Dataset
from vime.utils.eval_config import EvalDatasetConfig
from vime.utils.types import Sample

if TYPE_CHECKING:
    from vllm_rlt import SamplingParams


def evaluation_seed(seed: int, dataset: str, prompt: int, completion: int) -> int:
    identity = json.dumps(("native-heldout-v1", seed, dataset, prompt, completion), separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**63)


def evaluation_sampling(config: EvalDatasetConfig) -> "SamplingParams":
    from vllm_rlt import SamplingParams

    if config.stop or config.min_new_tokens not in (None, 0) or config.repetition_penalty not in (None, 1):
        raise ValueError("Native evaluation supports token stops, without minimum length or repetition penalties")
    if config.no_stop_trim is False:
        raise ValueError("Native evaluation retains generated stop tokens")
    if config.max_response_len is None or config.temperature is None or config.top_p is None or config.top_k is None:
        raise ValueError("Native evaluation requires resolved length, temperature, top-p and top-k settings")
    return SamplingParams(
        max_tokens=config.max_response_len,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        stop_token_ids=tuple(config.stop_token_ids or ()),
    )


def validate_evaluation(args: Namespace) -> None:
    if not args.eval_datasets or args.group_rm:
        raise ValueError("Native evaluation requires held-out datasets and per-sample rewards")
    if len({config.name for config in args.eval_datasets}) != len(args.eval_datasets):
        raise ValueError("Native evaluation dataset names must be unique")
    for config in args.eval_datasets:
        if config.n_samples_per_eval_prompt is None or config.n_samples_per_eval_prompt < 1:
            raise ValueError("Native evaluation requires a positive completion count")
        if (
            config.multimodal_keys
            or args.multimodal_keys
            or config.tool_key
            or config.custom_generate_function_path
            or config.app_service
            or config.message_processor
            or config.reward_model
            or config.remote_environment
        ):
            raise ValueError("Native evaluation uses text-only native generation and per-sample reward hooks")
        if (
            config.eval_task_timeout is not None
            or config.eval_early_stop_remaining is not None
            or config.eval_early_stop_idle_timeout is not None
        ):
            raise ValueError("Native evaluation completes the entire held-out dataset without early stopping")
        evaluation_sampling(config)


def _reward(sample: Sample, key: str | None) -> float:
    value = sample.reward
    if key:
        if not isinstance(value, dict):
            raise ValueError("Evaluation reward key requires a reward mapping")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise ValueError("Native evaluation requires a scalar reward or an explicit evaluation reward key")
    return float(value)


def evaluate(args: Namespace) -> RolloutFnEvalOutput:
    """Do not read, shuffle or advance the training data source or its RNG."""
    validate_evaluation(args)
    started = time.perf_counter()
    tokenizer = _tokenizer(args.hf_checkpoint)
    results = {}
    cohort: tuple[str, int, str] | None = None
    total_samples = 0
    for config in args.eval_datasets:
        chat = args.apply_chat_template if config.apply_chat_template is None else config.apply_chat_template
        chat_kwargs = (
            args.apply_chat_template_kwargs
            if config.apply_chat_template_kwargs is None
            else config.apply_chat_template_kwargs
        )
        dataset = Dataset(
            config.path,
            tokenizer,
            None,
            None,
            prompt_key=config.input_key,
            label_key=config.label_key,
            metadata_key=config.metadata_key,
            apply_chat_template=chat,
            apply_chat_template_kwargs=chat_kwargs,
        )
        if not dataset.samples:
            raise ValueError(f"Native evaluation dataset {config.name!r} is empty")
        sampling = evaluation_sampling(config)
        samples, seeds = [], []
        for prompt_index, original in enumerate(dataset.samples):
            tokens = tokenizer.encode(original.prompt, add_special_tokens=False)
            if not tokens or (args.eval_max_prompt_len is not None and len(tokens) > args.eval_max_prompt_len):
                raise ValueError("Held-out prompts must be nonempty and fit the declared limit; none are dropped")
            if args.eval_max_context_len is not None and len(tokens) + sampling.max_tokens > args.eval_max_context_len:
                raise ValueError("Held-out prompt plus response budget exceeds the evaluation context limit")
            for completion in range(config.n_samples_per_eval_prompt):
                sample = copy.deepcopy(original)
                sample.index, sample.group_index = len(samples), prompt_index
                sample.tokens = list(tokens)
                sample.metadata = config.inject_metadata(sample.metadata)
                sample.metadata["native_eval"] = {
                    "dataset": config.name,
                    "prompt_index": prompt_index,
                    "completion": completion,
                    "seed_profile": "native-heldout-v1",
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                    "max_tokens": sampling.max_tokens,
                    "stop_token_ids": list(sampling.stop_token_ids),
                }
                sample.custom_rm_path = config.custom_rm_path
                samples.append(sample)
                seeds.append(evaluation_seed(args.seed, config.name, prompt_index, completion))
        if config.min_eval_samples is not None and len(samples) < config.min_eval_samples:
            raise ValueError("Held-out dataset has fewer completions than min_eval_samples")
        generated = []
        for begin in range(0, len(samples), args.rlt_max_num_seqs):
            batch = samples[begin : begin + args.rlt_max_num_seqs]
            identities = [(sample.group_index, sample.index) for sample in batch]
            outputs = ray.get(
                args.rlt_engine.evaluate.remote(batch, sampling, tuple(seeds[begin : begin + len(batch)]))
            )
            if [(sample.group_index, sample.index) for sample in outputs] != identities:
                raise ValueError("Native evaluation must preserve every held-out sample in order")
            for sample in outputs:
                trace = sample.recurrent_trace
                if trace is None:
                    raise RuntimeError("Native evaluation requires a committed policy trace")
                identity = (trace.runtime_epoch, trace.policy_version, trace.publication_digest)
                if cohort is not None and identity != cohort:
                    raise RuntimeError("Held-out evaluation crossed committed policy cohorts")
                cohort = identity
            decode_and_reward(
                args,
                outputs,
                tokenizer,
                skip_special_tokens=True if config.skip_special_tokens is None else config.skip_special_tokens,
            )
            generated.extend(outputs)
        results[config.name] = {
            "rewards": [_reward(sample, args.eval_reward_key or args.reward_key) for sample in generated],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in generated],
            "samples": generated,
            "n_samples_per_prompt": config.n_samples_per_eval_prompt,
        }
        total_samples += len(generated)
    assert cohort is not None
    return RolloutFnEvalOutput(
        data=results,
        metrics={
            "eval/native_seconds": time.perf_counter() - started,
            "eval/native_policy_version": cohort[1],
            "eval/native_samples": total_samples,
        },
    )
