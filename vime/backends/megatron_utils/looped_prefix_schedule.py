"""Opt-in single-learner looped prefix forward/backward through MCore DDP."""

from argparse import Namespace
from collections.abc import Callable
import json
from pathlib import Path
from typing import cast

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from vime_plugins.looped.logical_step import PrefixGroup, plan_step
from vime.utils.types import RecurrentTrace
from vime_plugins.huginn.model import HuginnMegatronModel
from vime_plugins.looped.response import ResponseReadout
from vime_plugins.looped.prefix import PrefixReplay
from vime_plugins.looped.training import logical_rltt_loss
from vime_plugins.ouro.model import OuroMegatronModel

from .data import DataIterator


def forward_backward_prefix(
    args: Namespace,
    iterator: DataIterator,
    model: DDP,
    num_microbatches: int,
    step_global_batch_size: int,
    actor_generation: int,
) -> list[dict[str, list[str] | torch.Tensor]]:
    """Finalize once; leave optimizer, scheduler and cursor commit to the caller."""
    if not isinstance(model, DDP) or not isinstance(model.module, (OuroMegatronModel, HuginnMegatronModel)):
        raise ValueError("The prefix schedule requires the actual FP32 MCore DDP looped wrapper")
    actor, config = model.module, model.config
    if model.dp_cp_group.size() != 1:
        raise ValueError("The prefix schedule requires one actual learner rank")
    if any(parameter.dtype != torch.float32 for parameter in actor.parameters()):
        raise ValueError("The prefix learner schedule requires FP32 until mixed-precision qualification")
    if model.ddp_config.overlap_grad_reduce or model.ddp_config.use_distributed_optimizer:
        raise ValueError("The first prefix schedule requires non-overlapped, unsharded gradient buffers")
    if config.cuda_graph_scope or config.calculate_per_token_loss or config.finalize_model_grads_func is None:
        raise ValueError("The prefix schedule requires eager logical reduction and MCore gradient finalization")
    if (
        actor.recompute
        or args.rltt_layer_checkpoint
        or args.rltt_token_chunk
        or args.rltt_loop_checkpoint not in (0, actor.readout_depth)
    ):
        raise ValueError("Prefix rematerialization requires one whole-depth segment without layer/query subdivision")
    if isinstance(actor, HuginnMegatronModel) and args.rltt_loop_checkpoint:
        raise ValueError("Huginn joint prefix rematerialization is not qualified")
    step = plan_step(
        iterator,
        num_microbatches,
        actor_generation=actor_generation,
        model_revision=args.rlt_model_revision,
        loop_depth=actor.readout_depth,
        model_family=actor.prefix_model_family,
        suffix_wave_size=args.rltt_prefix_wave_size,
        reduction=args.rltt_reduction,
    )
    tokens = cast(list[torch.Tensor], iterator.rollout_data["tokens"])
    advantages = cast(list[torch.Tensor], iterator.rollout_data["advantages"])
    references = cast(list[torch.Tensor] | None, iterator.rollout_data["ref_log_probs"])
    if references is None:
        raise ValueError("Prefix RLTT requires the frozen reference forward")
    traces = cast(list[RecurrentTrace], iterator.rollout_data["recurrent_inputs"])
    logs = []
    for group in step.groups:
        step.check_current(actor_generation, iterator.offset)

        def objective(output, positions, group=group):
            indices = [group.indices[position] for position in positions]
            loss, metrics = logical_rltt_loss(
                args,
                {"advantages": [advantages[i] for i in indices], "ref_log_probs": [references[i] for i in indices]},
                [step.weights[i] for i in indices],
                output,
            )
            logs.append(
                {
                    "keys": list(metrics),
                    "values": torch.stack(
                        [loss.new_zeros(()), *[value * step_global_batch_size for value in metrics.values()]]
                    ),
                }
            )
            return loss

        backward_group(model, actor, group, tokens, traces, args, objective)
    config.finalize_model_grads_func([model], None)
    if args.rlt_runtime_report_dir is not None:
        shared = sum(len(group.indices) for group in step.groups if len(group.indices) > 1)
        fallback = sum(
            len(group.indices)
            for group in step.groups
            if group.identity.latent_seed is not None and len(group.indices) == 1
        )
        folder = Path(args.rlt_runtime_report_dir) / "prefix-plans"
        folder.mkdir(parents=True, exist_ok=True)
        report = {
            "phase": "backward_completed",
            "actor_generation": actor_generation,
            "iterator_offset": iterator.offset,
            "model_family": actor.prefix_model_family,
            "samples": len(step.indices),
            "shared_samples": shared,
            "shared_fraction": shared / len(step.indices),
            "full_replay_samples": fallback,
            "sample_ids": step.sample_ids,
            "group_ids": step.group_ids,
        }
        (folder / f"generation{actor_generation}-offset{iterator.offset}.json").write_text(json.dumps(report) + "\n")
    return logs


def backward_group(
    model: torch.nn.Module,
    actor: OuroMegatronModel | HuginnMegatronModel,
    group: PrefixGroup,
    tokens: list[torch.Tensor],
    traces: list[RecurrentTrace],
    args: Namespace,
    objective: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor],
) -> None:
    """Shared identity reuses its graph; an independent Huginn latent replays in full."""
    length = len(group.identity.prompt_tokens)
    scale = actor.config.grad_scale_func
    if group.identity.latent_seed is not None and len(group.indices) == 1:
        index = group.indices[0]
        output = model(
            input_ids=tokens[index][None],
            recurrent_inputs=[traces[index]],
            readout=ResponseReadout(
                (len(tokens[index]) - length,),
                args.rltt_vocab_tile,
                args.rollout_temperature,
                True,
                args.entropy_coef != 0,
            ),
        )
        loss = objective(output, (0,))
        (scale(loss) if scale is not None else loss).backward()
        return
    prompt = tokens[group.indices[0]][:length]
    # Keep DDP.forward and its real AccumulateGrad hooks on both paths.
    program = (
        model(input_ids=prompt[None], prefix_only=True, prefix_latent_seed=group.identity.latent_seed)
        if isinstance(actor, HuginnMegatronModel)
        else model(input_ids=prompt[None], prefix_only=True, prefix_rematerialize=args.rltt_loop_checkpoint != 0)
    )
    replay = PrefixReplay(
        program,
        group.identity,
        actor,
        actor.lm_head.weight,
        vocab_tile=args.rltt_vocab_tile,
        temperature=args.rollout_temperature,
        all_loops=True,
        entropy=args.entropy_coef != 0,
    )
    local = {index: position for position, index in enumerate(group.indices)}
    waves = tuple(tuple(local[index] for index in wave) for wave in group.waves)
    replay.backward_suffixes(
        tuple(tokens[i][length:] for i in group.indices),
        waves,
        objective,
        identity=group.identity,
        batch_suffixes=True,
        grad_scale_func=scale,
    )
