"""Opt-in single-learner Ouro prefix forward/backward through MCore DDP."""

from argparse import Namespace
from typing import cast

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from vime_plugins.looped.logical_step import plan_step
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
    if not isinstance(model, DDP) or not isinstance(model.module, OuroMegatronModel):
        raise ValueError("The prefix schedule requires the actual FP32 MCore DDP Ouro wrapper")
    actor, config = model.module, model.config
    if model.dp_cp_group.size() != 1:
        raise ValueError("The prefix schedule requires one actual learner rank")
    if any(parameter.dtype != torch.float32 for parameter in actor.parameters()):
        raise ValueError("The prefix learner schedule requires FP32 until mixed-precision qualification")
    if model.ddp_config.overlap_grad_reduce or model.ddp_config.use_distributed_optimizer:
        raise ValueError("The first prefix schedule requires non-overlapped, unsharded gradient buffers")
    if config.cuda_graph_scope or config.calculate_per_token_loss or config.finalize_model_grads_func is None:
        raise ValueError("The prefix schedule requires eager logical reduction and MCore gradient finalization")
    if actor.recompute or any((args.rltt_loop_checkpoint, args.rltt_layer_checkpoint, args.rltt_token_chunk)):
        raise ValueError("Joint prefix rematerialization is a separate schedule, not yet enabled")
    step = plan_step(
        iterator,
        num_microbatches,
        actor_generation=actor_generation,
        model_revision=args.rlt_model_revision,
        loop_depth=actor.readout_depth,
        suffix_wave_size=args.rltt_prefix_wave_size,
        reduction=args.rltt_reduction,
    )
    tokens = cast(list[torch.Tensor], iterator.rollout_data["tokens"])
    advantages = cast(list[torch.Tensor], iterator.rollout_data["advantages"])
    references = cast(list[torch.Tensor] | None, iterator.rollout_data["ref_log_probs"])
    if references is None:
        raise ValueError("Prefix RLTT requires the frozen reference forward")
    logs = []
    for group in step.groups:
        step.check_current(actor_generation, iterator.offset)
        length = len(group.identity.prompt_tokens)
        prompt = tokens[group.indices[0]][:length]
        # Enter through DDP.forward; its real AccumulateGrad hooks continue to
        # move each suffix/prefix contribution into the MCore main_grad buffers.
        program = model(input_ids=prompt.unsqueeze(0), prefix_only=True)
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

        replay.backward_suffixes(
            tuple(tokens[i][length:] for i in group.indices),
            waves,
            objective,
            identity=group.identity,
            batch_suffixes=True,
            grad_scale_func=config.grad_scale_func,
        )
        del replay, program
    config.finalize_model_grads_func([model], None)
    return logs
