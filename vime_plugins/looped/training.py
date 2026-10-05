"""MCore bridge for fixed-depth RLTT with a logical-batch denominator."""

from argparse import Namespace
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import torch

from .response import ResponseReadout
from .execution import RematPlan
from .rltt import loop_weights, rltt_loss

if TYPE_CHECKING:
    from vime.backends.megatron_utils.data import DataIterator


class RLTTBatch(TypedDict):
    advantages: list[torch.Tensor]
    ref_log_probs: list[torch.Tensor] | None


def validate_rltt_args(args: Namespace) -> None:
    RematPlan(args.rltt_loop_checkpoint, args.rltt_layer_checkpoint, args.rltt_token_chunk)
    if args.rollout_backend != "vllm-rlt":
        raise ValueError("RLTT requires the native recurrent rollout/provider contract")
    if args.actor_num_nodes * args.actor_num_gpus_per_node != 1:
        raise ValueError("RLTT's logical denominator is currently qualified for one learner rank")
    if (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.context_parallel_size) != (1, 1, 1):
        raise ValueError("RLTT response readout currently requires TP=PP=CP=1")
    if args.advantage_estimator != "grpo" or args.n_samples_per_prompt < 2:
        raise ValueError("RLTT requires grouped GRPO advantages with at least two completions")
    if args.kl_coef != 0 or args.flow_dppo_divergence_budget is not None or args.use_tis or args.use_opsm:
        raise ValueError("RLTT uses its explicit unclipped PG and terminal sampled-k3 loss")
    if args.calculate_per_token_loss or args.enable_mtp_training or args.dspark_enabled:
        raise ValueError("RLTT uses its own logical denominator and recurrent output contract")
    if args.ref_load is None or not args.compute_advantages_and_returns:
        raise ValueError("RLTT requires an initial reference checkpoint and its fresh forward pass")
    if args.ref_update_interval is not None:
        raise ValueError("RLTT requires a frozen initial reference throughout training")
    if args.rltt_vocab_tile < 1 or not 0 <= args.rltt_progressive_alpha < float("inf"):
        raise ValueError("RLTT tile must be positive and progressive alpha finite/nonnegative")
    if not 0 <= args.kl_loss_coef < float("inf"):
        raise ValueError("RLTT KL coefficient must be finite/nonnegative")


def masked_token_weights(
    masks: list[torch.Tensor], reduction: Literal["token_mean", "response_mean"]
) -> list[torch.Tensor]:
    totals = torch.stack([mask.float().sum() for mask in masks])
    denominator = totals.sum() if reduction == "token_mean" else (totals > 0).sum()
    if not bool(denominator > 0):
        raise ValueError("RLTT logical batch must contain a loss-contributing response")
    if reduction == "token_mean":
        return [mask.float() / denominator for mask in masks]
    if reduction != "response_mean":
        raise ValueError("unknown RLTT logical reduction")
    return [mask.float() / (denominator * total.clamp_min(1)) for mask, total in zip(masks, totals, strict=True)]


def weights_for_step(
    iterator: "DataIterator", num_microbatches: int, reduction: Literal["token_mean", "response_mean"]
) -> dict[int, torch.Tensor]:
    indices = [
        index
        for group in iterator.micro_batch_indices[iterator.offset : iterator.offset + num_microbatches]
        for index in group
    ]
    masks = cast(list[torch.Tensor], iterator.rollout_data["loss_masks"])
    weights = masked_token_weights([masks[index] for index in indices], reduction)
    return dict(zip(indices, weights, strict=True))


def readout_request(
    args: Namespace, lengths: list[int], total_lengths: list[int], *, all_loops: bool, entropy: bool
) -> ResponseReadout:
    intervals = (args.rltt_loop_checkpoint, args.rltt_layer_checkpoint, args.rltt_token_chunk)
    return ResponseReadout(
        tuple(lengths),
        args.rltt_vocab_tile,
        args.rollout_temperature,
        all_loops,
        entropy,
        tuple(total_lengths),
        args.rltt_attention_backend,
        RematPlan(*intervals) if any(intervals) else None,
    )


def collect_log_probs(
    output: torch.Tensor, *, response_lengths: list[int], with_entropy: bool, non_loss_data: bool = True
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    assert non_loss_data
    result = {"log_probs": list(output[:, -2].split(response_lengths))}
    if with_entropy:
        result["entropy"] = list(output[:, -1].split(response_lengths))
    return output.new_empty(0), result


def logical_rltt_loss(
    args: Namespace,
    batch: RLTTBatch,
    weights: list[torch.Tensor],
    output: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use step-global token weights; apply no schedule, optimizer or log scale."""
    if batch["ref_log_probs"] is None:
        raise ValueError("RLTT requires scores from the frozen initial reference")
    scores, entropy = output[:, :-1], output[:, -1]
    token_weight = torch.cat(weights)
    loss = rltt_loss(
        scores,
        torch.cat(batch["advantages"]),
        loop_weights(scores.shape[1], alpha=args.rltt_progressive_alpha, like=scores),
        token_weight,
        reference_log_probs=torch.cat(batch["ref_log_probs"]),
        kl_coefficient=args.kl_loss_coef,
        credit_stopgrad=True,
    )
    log = {"loss": loss.detach(), "rltt_terminal_logprob": (scores[:, -1].detach() * token_weight).sum()}
    if args.entropy_coef != 0:
        entropy_mean = (entropy * token_weight).sum()
        loss = loss - args.entropy_coef * entropy_mean
        log.update(loss=loss.detach(), entropy_loss=entropy_mean.detach())
    return loss, log


def megatron_loss(
    args: Namespace,
    batch: RLTTBatch,
    weights: list[torch.Tensor],
    num_microbatches: int,
    step_global_batch_size: int,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    loss, log = logical_rltt_loss(args, batch, weights, output)
    # The single-rank MCore schedule divides each loss by num_microbatches.
    # Logging is subsequently divided by VIME's step_global_batch_size.
    return (
        loss * num_microbatches,
        output.new_ones((), dtype=torch.int64),
        {
            "keys": list(log),
            "values": torch.stack([output.new_zeros(()), *[value * step_global_batch_size for value in log.values()]]),
        },
    )
