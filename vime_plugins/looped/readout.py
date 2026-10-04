"""Vocabulary-tiled first-order readout without saved token-by-vocabulary logits."""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class Readout:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    full_kl: torch.Tensor


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, reference_hidden, reference_weight, tile, temperature, entropy):
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        h = hidden.to(dtype)
        log_z = torch.full((len(h),), -torch.inf, dtype=dtype, device=h.device)
        ref_z = torch.zeros_like(log_z)
        has_reference = reference_hidden is not None
        rh = reference_hidden.to(dtype) if has_reference else None
        if has_reference:
            ref_z.fill_(-torch.inf)
        for start in range(0, len(weight), tile):
            logits = h @ weight[start : start + tile].to(dtype).T / temperature
            log_z = torch.logaddexp(log_z, logits.logsumexp(-1))
            if has_reference:
                ref_logits = rh @ reference_weight[start : start + tile].to(dtype).T / temperature
                ref_z = torch.logaddexp(ref_z, ref_logits.logsumexp(-1))
        selected = (h * weight[targets].to(dtype)).sum(-1) / temperature - log_z
        ent, kl = torch.zeros_like(log_z), torch.zeros_like(log_z)
        if entropy or has_reference:
            for start in range(0, len(weight), tile):
                log_p = h @ weight[start : start + tile].to(dtype).T / temperature - log_z[:, None]
                prob = log_p.exp()
                if entropy:
                    ent -= (prob * log_p).sum(-1)
                if has_reference:
                    log_q = rh @ reference_weight[start : start + tile].to(dtype).T / temperature - ref_z[:, None]
                    kl += (prob * (log_p - log_q)).sum(-1)
        ctx.save_for_backward(hidden, weight, targets, log_z, ent, kl, reference_hidden, reference_weight, ref_z)
        ctx.tile, ctx.temperature = tile, temperature
        ctx.with_entropy = entropy
        ctx.set_materialize_grads(False)
        return selected, ent, kl

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_selected, grad_entropy, grad_kl):
        hidden, weight, targets, log_z, entropy, kl, reference_hidden, reference_weight, ref_z = ctx.saved_tensors
        dtype = log_z.dtype
        h = hidden.to(dtype)
        grad_hidden, grad_weight = torch.zeros_like(h), torch.empty_like(weight)
        has_reference = reference_hidden is not None
        rh = reference_hidden.to(dtype) if has_reference else None
        for start in range(0, len(weight), ctx.tile):
            w = weight[start : start + ctx.tile].to(dtype)
            log_p = h @ w.T / ctx.temperature - log_z[:, None]
            prob = log_p.exp()
            grad_logits = torch.zeros_like(prob)
            if grad_selected is not None:
                grad_logits -= grad_selected[:, None] * prob
                offset = targets - start
                included = (offset >= 0) & (offset < len(w))
                grad_logits.scatter_add_(1, offset.clamp(0, len(w) - 1)[:, None], (grad_selected * included)[:, None])
            if grad_entropy is not None and ctx.with_entropy:
                grad_logits -= grad_entropy[:, None] * prob * (log_p + entropy[:, None])
            if grad_kl is not None and has_reference:
                log_q = rh @ reference_weight[start : start + ctx.tile].to(dtype).T / ctx.temperature - ref_z[:, None]
                grad_logits += grad_kl[:, None] * prob * (log_p - log_q - kl[:, None])
            grad_logits /= ctx.temperature
            grad_hidden += grad_logits @ w
            grad_weight[start : start + ctx.tile] = grad_logits.T @ h
        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, None, None, None, None, None


def streamed_readout(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_tile: int,
    temperature: float = 1.0,
    entropy: bool = False,
    reference_hidden: torch.Tensor | None = None,
    reference_weight: torch.Tensor | None = None,
) -> Readout:
    """Return raw temperature-scaled scores and optional entropy/full KL.

    Linear products accumulate in FP32 (FP64 for FP64 inputs); there is no
    top-k/p truncation, gradient filtering, or differentiation of the reference.
    Memory is bounded by response rows times vocab_tile, plus parameter gradients.
    Only first-order derivatives are supported.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("readout expects [response, hidden] and [vocabulary, hidden]")
    if targets.shape != hidden.shape[:1] or targets.dtype != torch.long:
        raise ValueError("one int64 target is required per response position")
    if vocab_tile < 1 or not 0 < temperature < float("inf") or len(weight) == 0:
        raise ValueError("readout tile, temperature and vocabulary must be positive")
    if bool(((targets < 0) | (targets >= len(weight))).any()):
        raise ValueError("target outside vocabulary")
    if (reference_hidden is None) != (reference_weight is None):
        raise ValueError("full KL requires both frozen reference hidden states and weights")
    if reference_hidden is not None:
        if reference_hidden.shape != hidden.shape or reference_weight.shape != weight.shape:
            raise ValueError("reference shape must match actor")
        if reference_hidden.requires_grad or reference_weight.requires_grad:
            raise ValueError("reference policy must be frozen")
    return Readout(
        *_Readout.apply(hidden, weight, targets, reference_hidden, reference_weight, vocab_tile, temperature, entropy)
    )
