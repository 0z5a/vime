"""Differentiable IQuest replay with explicit first-loop KV dependencies."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Protocol

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Attention(Protocol):
    head_dim: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear


class Layer(Protocol):
    self_attn: Attention
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    mlp: nn.Module


class Config(Protocol):
    loop_num: int
    loop_window_size: int


class Core(Protocol):
    embed_tokens: nn.Embedding
    layers: Sequence[Layer]
    gate_projections: Sequence[Callable[[torch.Tensor], torch.Tensor]]
    norm: nn.Module
    rotary_emb: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class PhysicalModel(Protocol):
    config: Config
    model: Core
    lm_head: nn.Linear


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1] if enable_gqa else 1
    if groups > 1:
        batch, heads, length, width = key.shape
        key = key[:, :, None].expand(batch, heads, groups, length, width).reshape(batch, heads * groups, length, width)
        value = value[:, :, None].expand(batch, heads, groups, length, width).reshape(batch, heads * groups, length, width)
    scores = torch.matmul(query, key.transpose(2, 3)) * query.shape[-1] ** -0.5
    if is_causal:
        attn_mask = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device).tril()
    if attn_mask is not None:
        bias = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
        scores = scores + bias
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def layer_forward(
    layer: Layer,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    global_key: torch.Tensor | None = None,
    global_value: torch.Tensor | None = None,
    *,
    gate: Callable[[torch.Tensor], torch.Tensor] | None = None,
    window: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = layer.self_attn
    value = layer.input_layernorm(hidden)
    shape = (*hidden.shape[:2], -1, attention.head_dim)
    query, key, local_value = (projection(value).view(shape).transpose(1, 2) for projection in (attention.q_proj, attention.k_proj, attention.v_proj))

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos[:, None] + torch.cat((-second, first), dim=-1) * sin[:, None]

    query, key = rotate(query), rotate(key)
    if global_key is None:
        attended = dense_attention(query, key, local_value, is_causal=True, enable_gqa=True)
    else:
        assert global_value is not None and gate is not None
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        distance = positions[:, None] - positions[None, :]
        local_mask = (distance >= 0) & (distance < window)
        local = dense_attention(query, key, local_value, attn_mask=local_mask, enable_gqa=True)
        global_output = dense_attention(query, global_key, global_value, is_causal=True, enable_gqa=True)
        mixing = gate(query)
        attended = local * (1 - mixing) + global_output * mixing
    projected = attention.o_proj(attended.transpose(1, 2).reshape(*hidden.shape[:2], -1))
    hidden = hidden + projected
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), key, local_value


def replay(model: PhysicalModel, tokens: torch.Tensor, *, recompute: bool = False) -> torch.Tensor:
    """Return complete logits; no serving cache mutation or detached boundary KV."""
    if model.config.loop_num != 2:
        raise ValueError("IQuest replay requires the checkpoint's two-loop contract")
    core = model.model
    hidden = core.embed_tokens(tokens)
    positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
    cos, sin = core.rotary_emb(hidden, positions)
    shared = []
    for layer in core.layers:
        forward = partial(layer_forward, layer)
        hidden, key, value = checkpoint(forward, hidden, cos, sin, use_reentrant=False) if recompute else forward(hidden, cos, sin)
        shared.append((key, value))
    for layer, gate, (key, value) in zip(core.layers, core.gate_projections, shared, strict=True):
        forward = partial(layer_forward, layer, gate=gate, window=model.config.loop_window_size)
        hidden, _, _ = checkpoint(forward, hidden, cos, sin, key, value, use_reentrant=False) if recompute else forward(hidden, cos, sin, key, value)
    return model.lm_head(core.norm(hidden)).float()
