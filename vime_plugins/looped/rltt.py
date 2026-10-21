"""Explicit RLTT objective reimplementation; not a reproduction of the original stack."""

from typing import Literal

import torch


def loop_weights(depth: int, *, alpha: float, like: torch.Tensor) -> torch.Tensor:
    """alpha=0 gives uniform credit; alpha>0 gives normalized progressive credit."""
    if depth < 1 or not 0 <= alpha < float("inf"):
        raise ValueError("depth must be positive and alpha finite/nonnegative")
    logits = torch.arange(1, depth + 1, device=like.device, dtype=like.dtype).log() * alpha
    return logits.softmax(0)


def token_weights(
    response_lengths: list[int], *, reduction: Literal["token_mean", "response_mean"], like: torch.Tensor
) -> torch.Tensor:
    """Build once for the logical batch, then slice without renormalizing microbatches."""
    if not response_lengths or min(response_lengths) < 0 or sum(response_lengths) == 0:
        raise ValueError("logical batch requires at least one response token")
    if reduction == "token_mean":
        return like.new_full((sum(response_lengths),), 1 / sum(response_lengths))
    if reduction == "response_mean":
        nonempty = sum(length > 0 for length in response_lengths)
        return torch.cat([like.new_full((length,), 1 / (nonempty * length)) for length in response_lengths if length])
    raise ValueError("unknown logical reduction")


def rltt_loss(
    per_loop_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    credit: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    reference_log_probs: torch.Tensor,
    kl_coefficient: float,
    credit_stopgrad: bool,
    terminal_full_kl: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unclipped weighted-logprob PG plus terminal sampled-k3 or supplied full KL.

    Inputs are response-only [tokens, loops], with global logical token weights.
    Empty local partitions contribute a differentiable zero. A frozen reference
    is mandatory even when the KL coefficient is zero; old policy is not a fallback.
    Sampled k3 follows the visible actor (not the generic aggregated KL helper).
    """
    scores = per_loop_log_probs
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("per-loop scores must have shape [response tokens, loops]")
    if any(value.shape != scores.shape[:1] for value in (advantages, token_weight, reference_log_probs)):
        raise ValueError("advantages, weights and reference must be response-aligned")
    if credit.shape not in (scores.shape[1:], scores.shape):
        raise ValueError("credit must be [loops] or [response tokens, loops]")
    if reference_log_probs.requires_grad:
        raise ValueError("reference policy must be frozen")
    if not 0 <= kl_coefficient < float("inf"):
        raise ValueError("KL coefficient must be finite and nonnegative")
    weights = credit.detach() if credit_stopgrad else credit
    policy_gradient = -advantages.detach() * (scores * weights).sum(-1)
    if kl_coefficient == 0:
        return (policy_gradient * token_weight).sum()
    if terminal_full_kl is None:
        log_ratio = reference_log_probs - scores[:, -1]
        penalty = log_ratio.expm1() - log_ratio
    else:
        if terminal_full_kl.shape != scores.shape[:1]:
            raise ValueError("full-vocabulary KL must be response-aligned")
        penalty = terminal_full_kl
    return ((policy_gradient + kl_coefficient * penalty) * token_weight).sum()
