"""
Modified MIT License 

Software Copyright© 2025 IQuest Research

Our only modification is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services, you shall
prominently display "IQuest Coder" on the user interface of such product or
service.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
from typing import Any, Callable, Optional, Union, Tuple, List

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from .configuration_iquestloopcoder import IQuestLoopCoderConfig


logger = logging.getLogger(__name__)


def needs_iquestloopcoder_cache(
    cache: Optional[Cache]
) -> bool:
    # need to test more conditions
    if cache is None:
        return True
    if isinstance(cache, IQuestLoopCoderCache):
        return False
    return True

class IQuestLoopCoderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class IQuestLoopCoderCache(Cache):
    """Cache implementation for IQuestLoopCoder that manages shared and local KV caches.
    
    - shared_key_cache/shared_value_cache: Stores KV from Loop 1 (global context)
    - local_key_cache/local_value_cache: Stores KV from Loop 2+ (local window, only window_size tokens)
    """
    
    def __init__(self, window_size: int, num_layers: int, loop_num: int=2):
        # We intentionally don't call super().__init__ because the parent assumes static cache sizes.
        self.window_size = window_size
        self.num_layers = num_layers
        self.loop_num = loop_num
        
        # Shared cache: stores Loop 1 KV (global context)   
        self.shared_key_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.shared_value_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers
        
        # Local cache: stores Loop 2+ KV (sliding window, only window_size tokens)
        self.local_key_cache: List[Optional[torch.Tensor]] = [None] * (self.loop_num-1) * self.num_layers
        self.local_value_cache: List[Optional[torch.Tensor]] = [None] * (self.loop_num-1) * self.num_layers
        
        self.layers: List[Any] = []  # attribute expected by HF Cache utilities
        self._seen_tokens = 0
    
    def update_shared(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update shared cache (Loop 1 KV)."""
        # only store the first loop's kv cache
        loop_idx = cache_kwargs.get("loop_idx", 0)
        assert loop_idx == 0
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        
        cached_key = self.shared_key_cache[layer_idx]
        cached_value = self.shared_value_cache[layer_idx]
        
        if cached_key is None:
            self.shared_key_cache[layer_idx] = key_states
            self.shared_value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert key_states.shape[2] == 1
            assert value_states.shape[2] == 1
            self.shared_key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.shared_value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)
        
        result_key = self.shared_key_cache[layer_idx]
        result_value = self.shared_value_cache[layer_idx]
        assert result_key is not None and result_value is not None
        
        # Track sequence length
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value
    
    def update_local(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update local cache (Loop 2+ KV) with sliding window management.
        
        Ensures the local cache always contains at most window_size tokens.
        Local cache only stores loop_idx > 0 (i.e., loop_idx = 1, 2, ...).
        For loop_idx = 1, cache_idx = layer_idx + 0 * num_layers = layer_idx (0 to num_layers-1)
        For loop_idx = 2, cache_idx = layer_idx + 1 * num_layers (num_layers to 2*num_layers-1)
        """
        # only store the local kv cache for loop_idx > 0
        loop_idx = cache_kwargs.get("loop_idx", 0)
        assert loop_idx > 0, f"update_local should only be called for loop_idx > 0, got {loop_idx}"
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        
        # Local cache size is (loop_num-1) * num_layers
        # loop_idx = 1 maps to indices 0 to num_layers-1
        # loop_idx = 2 maps to indices num_layers to 2*num_layers-1
        # So offset = (loop_idx - 1) * num_layers
        cache_idx = layer_idx + (loop_idx - 1) * self.num_layers
        
        # Validate cache_idx is within bounds
        max_cache_idx = (self.loop_num - 1) * self.num_layers
        if cache_idx >= max_cache_idx:
            raise IndexError(
                f"cache_idx {cache_idx} out of range. "
                f"loop_idx={loop_idx}, layer_idx={layer_idx}, "
                f"max_cache_idx={max_cache_idx - 1}"
            )
        cached_key = self.local_key_cache[cache_idx]
        cached_value = self.local_value_cache[cache_idx]
        
        if cached_key is None:
            # First token in local cache, for prefill
            # If prefill sequence is longer than window_size, only keep the last window_size tokens
            seq_len = key_states.shape[2]
            if seq_len > self.window_size:
                # Keep only the last window_size tokens
                start_idx = seq_len - self.window_size
                self.local_key_cache[cache_idx] = key_states[:, :, start_idx:, :]
                self.local_value_cache[cache_idx] = value_states[:, :, start_idx:, :]
            else:
                self.local_key_cache[cache_idx] = key_states
                self.local_value_cache[cache_idx] = value_states
        else:
            # store the local kv cache for decode
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            assert key_states.shape[2] == 1
            assert value_states.shape[2] == 1
            # Concatenate new tokens
            new_key = torch.cat([cached_key, key_states], dim=2)
            new_value = torch.cat([cached_value, value_states], dim=2)
            
            # Ensure the total length doesn't exceed window_size
            total_len = new_key.shape[2]
            if total_len > self.window_size:
                # Keep only the last window_size tokens
                self.local_key_cache[cache_idx] = new_key[:, :, -self.window_size:, :]
                self.local_value_cache[cache_idx] = new_value[:, :, -self.window_size:, :]
            else:
                self.local_key_cache[cache_idx] = new_key
                self.local_value_cache[cache_idx] = new_value
        
        result_key = self.local_key_cache[cache_idx]
        result_value = self.local_value_cache[cache_idx]
        assert result_key is not None and result_value is not None
        # Ensure the result is at most window_size (can be less during prefill when sequence is shorter)
        assert result_key.shape[2] <= self.window_size, f"Local cache size {result_key.shape[2]} exceeds window_size {self.window_size}"
        
        return result_key, result_value
    
    def get_shared(self, layer_idx: int|List[int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get shared cache for some layer."""
        if isinstance(layer_idx, list):
            return [self.get_shared(layer_idx) for layer_idx in layer_idx]
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        return self.shared_key_cache[layer_idx], self.shared_value_cache[layer_idx]
    
    def get_local(self, layer_idx: int|List[int], loop_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get local cache for a layer."""
        assert loop_idx > 0, f"get_local should only be called for loop_idx > 0, got {loop_idx}"
        if isinstance(layer_idx, list):
            return [self.get_local(layer_idx, loop_idx) for layer_idx in layer_idx]
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        
        # Local cache size is (loop_num-1) * num_layers
        # loop_idx = 1 maps to indices 0 to num_layers-1
        # loop_idx = 2 maps to indices num_layers to 2*num_layers-1
        # So offset = (loop_idx - 1) * num_layers
        cache_idx = layer_idx + (loop_idx - 1) * self.num_layers
        
        # Validate cache_idx is within bounds
        max_cache_idx = (self.loop_num - 1) * self.num_layers
        if cache_idx >= max_cache_idx:
            raise IndexError(
                f"cache_idx {cache_idx} out of range. "
                f"loop_idx={loop_idx}, layer_idx={layer_idx}, "
                f"max_cache_idx={max_cache_idx - 1}"
            )
        
        return self.local_key_cache[cache_idx], self.local_value_cache[cache_idx]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default update method (for compatibility, updates shared cache)."""
        loop_idx = cache_kwargs.get("loop_idx", 0)
        assert loop_idx < self.loop_num
        if loop_idx == 0:
            return self.update_shared(key_states, value_states, layer_idx, cache_kwargs)
        else:
            return self.update_local(key_states, value_states, layer_idx, cache_kwargs)
        
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length from shared cache."""
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or layer_idx >= self.loop_num * self.num_layers:
            return 0
        cached_key = self.shared_key_cache[layer_idx]
        if cached_key is None:
            return 0
        return cached_key.shape[2]
    
    def get_max_length(self) -> Optional[int]:
        return None
    
    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)
    
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        # pass
        raise NotImplementedError("Reorder cache for beam search is not implemented")
        """Reorder cache for beam search.
        
        Reorders both shared cache (Loop 1) and local cache (Loop 2+) according to beam_idx.
        """
        # Reorder shared cache (Loop 1, loop_idx=0)
        for layer_idx in range(self.num_layers):
            if self.shared_key_cache[layer_idx] is not None:
                device = self.shared_key_cache[layer_idx].device
                self.shared_key_cache[layer_idx] = self.shared_key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.shared_value_cache[layer_idx] = self.shared_value_cache[layer_idx].index_select(0, beam_idx.to(device))
        
        # Reorder local cache (Loop 2+, loop_idx > 0)
        # Local cache size is (loop_num-1) * num_layers
        for cache_idx in range(len(self.local_key_cache)):
            if self.local_key_cache[cache_idx] is not None:
                device = self.local_key_cache[cache_idx].device
                self.local_key_cache[cache_idx] = self.local_key_cache[cache_idx].index_select(0, beam_idx.to(device))
                self.local_value_cache[cache_idx] = self.local_value_cache[cache_idx].index_select(0, beam_idx.to(device))
    
    @property
    def is_compileable(self) -> bool:
        return False
    
    def clear(self) -> None:
        """Clear all caches."""
        logger.debug("Clearing IQuestLoopCoderCache")
        self.shared_key_cache = [None] * self.num_layers
        self.shared_value_cache = [None] * self.num_layers
        self.local_key_cache = [None] * self.num_layers * (self.loop_num-1)
        self.local_value_cache = [None] * self.num_layers * (self.loop_num-1)
        self._seen_tokens = 0


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LoopGateProjection(nn.Module):
    """Gate projection for mixed attention in Loop 2+.
    
    Computes: g = sigmoid(linear(Q)) for each head independently.
    This gate determines how much to use Loop1's KV (global) vs current loop's KV (local).
    """
    
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Each head has its own gate: Linear(head_dim -> 1) per head
        # Implemented as [num_heads, head_dim] weight + [num_heads] bias
        self.weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads))
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Compute gate values from query tensor.
        
        Args:
            query: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            gate: [batch, num_heads, seq_len, 1]
        """
        # query: [batch, num_heads, seq_len, head_dim]
        # weight: [num_heads, head_dim]
        # For each head h: gate_h = query[:, h, :, :] @ weight[h, :].T + bias[h]
        # Using einsum: gate = einsum('bhsd,hd->bhs', query, weight) + bias
        gate_logits = torch.einsum('bhsd,hd->bhs', query, self.weight)  # [batch, num_heads, seq_len]
        gate_logits = gate_logits + self.bias[None, :, None]  # broadcast bias
        gate = torch.sigmoid(gate_logits)
        return gate.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]

class IQuestLoopCoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: IQuestLoopCoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        assert layer_idx >= 0 and layer_idx < config.num_hidden_layers
        self.layer_idx = layer_idx

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loop_idx: int = 0,
        gate_proj: Optional[LoopGateProjection] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if loop_idx == 0:
            return self.forward_loop1(hidden_states, loop_idx, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        else:
            return self.forward_loop2(hidden_states, loop_idx, position_embeddings, attention_mask, past_key_value, cache_position, gate_proj, **kwargs)
        
    def forward_loop1(
        self, 
        hidden_states: torch.Tensor, 
        loop_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor], 
        attention_mask: Optional[torch.Tensor], 
        past_key_value: Optional[IQuestLoopCoderCache] = None, 
        cache_position: Optional[torch.LongTensor] = None, 
        **kwargs: Unpack[FlashAttentionKwargs]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "loop_idx": loop_idx}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, (attn_weights)


    def forward_loop2(
        self, 
        hidden_states: torch.Tensor, 
        loop_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor], 
        attention_mask: Optional[torch.Tensor], 
        past_key_value: Optional[IQuestLoopCoderCache] = None, 
        cache_position: Optional[torch.LongTensor] = None, 
        gate_proj: Optional[LoopGateProjection] = None,
        **kwargs: Unpack[FlashAttentionKwargs]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states_local = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states_local = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states_local = apply_rotary_pos_emb(
            query_states, key_states_local, cos, sin
        )

        key_states_share, value_states_share = None, None
        if past_key_value is not None:
            # get key_share, value_share from past_key_value
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "loop_idx": loop_idx}
            key_states_share, value_states_share = past_key_value.get_shared(self.layer_idx)
            key_states_local, value_states_local = past_key_value.update(
                key_states_local,
                value_states_local,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        
        # Create masks for global and local attention
        # Global attention: full causal mask (can see all tokens in shared cache)
        # Local attention: causal mask for local window (can only see window_size tokens in local cache)
        attention_mask_global = attention_mask  # Use full causal mask for global attention
        
        # For local attention, create a mask that matches the local cache size
        # The local cache already contains only the last window_size tokens,
        # so we need a causal mask that allows attention within this window
        attention_mask_local = None
        if key_states_local is not None and value_states_local is not None:
            # Local cache has shape [batch, num_heads, local_seq_len, head_dim]
            # where local_seq_len <= window_size
            local_seq_len = key_states_local.shape[2]
            bsz = query_states.shape[0]
            q_len = query_states.shape[2]
            
            # Create a causal mask for local attention
            # This allows each query position to attend to all positions up to and including itself
            # within the local window (which is already the last window_size tokens)
            device = query_states.device
            dtype = query_states.dtype
            
            if attention_mask is not None:
                # If we have a global mask, we need to adapt it for local attention
                # The global mask shape is [batch, 1, q_len, global_kv_len]
                # For local attention, we only need the last local_seq_len positions
                global_kv_len = attention_mask.shape[-1]
                
                if global_kv_len >= local_seq_len:
                    # Extract the last local_seq_len columns from the global mask
                    # This represents attention to the last window_size tokens
                    attention_mask_local = attention_mask[..., -local_seq_len:]
                else:
                    # If global mask is shorter than local_seq_len, create a simple causal mask
                    # This can happen during prefill when local cache is being built
                    attention_mask_local = torch.triu(
                        torch.ones((q_len, local_seq_len), device=device, dtype=dtype) * float("-inf"),
                        diagonal=1
                    ).unsqueeze(0).expand(bsz, -1, -1, -1)  # [batch, 1, q_len, local_seq_len]
            else:
                # No global mask provided, create a simple causal mask for local attention
                # This allows full attention within the local window (causal)
                attention_mask_local = torch.triu(
                    torch.ones((q_len, local_seq_len), device=device, dtype=dtype) * float("-inf"),
                    diagonal=1
                ).unsqueeze(0).expand(bsz, -1, -1, -1)  # [batch, 1, q_len, local_seq_len]
        
        # global attn: attend to all tokens in shared cache
        attn_output_global, attn_weights_global = attention_interface(
            self,
            query_states,
            key_states_share,
            value_states_share,
            attention_mask_global,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # local attn: attend only to tokens in local cache (window_size)
        attn_output_local, attn_weights_local = attention_interface(
            self,
            query_states,
            key_states_local,
            value_states_local,
            attention_mask_local,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # attention_interface returns [batch, seq_len, num_heads, head_dim] for eager_attention_forward
        # but Flash Attention might return [batch, num_heads, seq_len, head_dim]
        # We need [batch, num_heads, seq_len, head_dim] to match gate shape
        q_len = query_states.shape[2]  # Query sequence length
        num_heads = query_states.shape[1]
        
        # Normalize attn_output_global to [batch, num_heads, q_len, head_dim]
        if attn_output_global.dim() == 4:
            # Check if shape is [batch, seq_len, num_heads, head_dim] (eager) or [batch, num_heads, seq_len, head_dim] (flash)
            if attn_output_global.shape[1] == q_len:
                # Shape is [batch, seq_len, num_heads, head_dim], transpose to [batch, num_heads, seq_len, head_dim]
                attn_output_global = attn_output_global.transpose(1, 2)
            # Ensure sequence length matches query length (take first q_len tokens)
            if attn_output_global.shape[2] > q_len:
                attn_output_global = attn_output_global[:, :, :q_len, :]
            elif attn_output_global.shape[2] < q_len:
                # This shouldn't happen, but handle it gracefully
                raise ValueError(f"attn_output_global seq_len {attn_output_global.shape[2]} < q_len {q_len}")
        
        # Normalize attn_output_local to [batch, num_heads, q_len, head_dim]
        if attn_output_local.dim() == 4:
            # Check if shape is [batch, seq_len, num_heads, head_dim] (eager) or [batch, num_heads, seq_len, head_dim] (flash)
            if attn_output_local.shape[1] == q_len:
                # Shape is [batch, seq_len, num_heads, head_dim], transpose to [batch, num_heads, seq_len, head_dim]
                attn_output_local = attn_output_local.transpose(1, 2)
            # Ensure sequence length matches query length (take first q_len tokens)
            if attn_output_local.shape[2] > q_len:
                attn_output_local = attn_output_local[:, :, :q_len, :]
            elif attn_output_local.shape[2] < q_len:
                # This shouldn't happen, but handle it gracefully
                raise ValueError(f"attn_output_local seq_len {attn_output_local.shape[2]} < q_len {q_len}")

        assert gate_proj is not None
        gate = gate_proj(query_states)  # [batch, num_heads, seq_len, 1]
        mixed_attn_output = attn_output_local * (1 - gate) + attn_output_global * gate

        mixed_attn_output = mixed_attn_output.reshape(*input_shape, -1).contiguous()
        mixed_attn_output = self.o_proj(mixed_attn_output)
        return mixed_attn_output, (attn_weights_global, attn_weights_local, attn_output_global, attn_output_local, gate)


@use_kernel_forward_from_hub("RMSNorm")
class IQuestLoopCoderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        IQuestLoopCoderRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IQuestLoopCoderDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: IQuestLoopCoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = IQuestLoopCoderAttention(config=config, layer_idx=layer_idx)

        self.mlp = IQuestLoopCoderMLP(config)
        self.input_layernorm = IQuestLoopCoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = IQuestLoopCoderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int = 0,
        gate_proj: Optional[LoopGateProjection] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            loop_idx=loop_idx,
            position_embeddings=position_embeddings,
            gate_proj=gate_proj if loop_idx > 0 else None,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class IQuestLoopCoderPreTrainedModel(PreTrainedModel):
    config: IQuestLoopCoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IQuestLoopCoderDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": IQuestLoopCoderDecoderLayer,
        "attentions": IQuestLoopCoderAttention,
    }

    # Important for inference with `device_map` / low_cpu_mem_usage:
    # Avoid initializing parameters that are not present in the checkpoint.
    # Those should keep their constructor-time initialization (e.g. zeros for LoopGateProjection),
    # instead of being materialized from meta/empty tensors which can contain NaNs.
    def _init_weights(self, module: nn.Module) -> None:
        return


class IQuestLoopCoderRotaryEmbedding(nn.Module):
    def __init__(self, config: IQuestLoopCoderConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class IQuestLoopCoderModel(IQuestLoopCoderPreTrainedModel):
    def __init__(self, config: IQuestLoopCoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                IQuestLoopCoderDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = IQuestLoopCoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = IQuestLoopCoderRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.loop_num = getattr(self.config, "loop_num", 2)
        self.loop_window_size = getattr(self.config, "loop_window_size", 64)

        # Gate projections for Loop 2+ (one per layer)
        self.gate_projections = nn.ModuleList([
            LoopGateProjection(config.num_attention_heads, config.head_dim)
            for _ in range(config.num_hidden_layers)
        ])

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache

        if use_cache:
            if needs_iquestloopcoder_cache(past_key_values):
                past_key_values = IQuestLoopCoderCache(self.loop_window_size, self.config.num_hidden_layers, self.loop_num)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the full causal mask for all layers
            # All layers use full_attention (no sliding window layers)
            full_attention_mask = create_causal_mask(**mask_kwargs)
            causal_mask_mapping = {
                "full_attention": full_attention_mask,
            }

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []

        for loop_idx in range(self.loop_num):
            # For each loop, use the full_attention mask
            # Loop 1: uses full_attention mask directly
            # Loop 2+: forward_loop2 will create local mask internally, but uses full_attention mask for global attention
            loop_attention_mask = causal_mask_mapping["full_attention"]
            
            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                hidden_states = decoder_layer(
                    hidden_states,
                    loop_idx,
                    gate_proj=self.gate_projections[layer_idx] if loop_idx > 0 else None,
                    attention_mask=loop_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            if loop_idx < self.loop_num - 1:
                hidden_states_list.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states_list.append(hidden_states)
        
        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            ),
            hidden_states_list,
        )


@auto_docstring
class IQuestLoopCoderForCausalLM(IQuestLoopCoderPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = IQuestLoopCoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 分块大小配置
        self.chunk_size = getattr(config, "chunk_size", 2)  # 默认分块大小为2

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs, hidden_states_list = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        def _select_token_positions(tensor: torch.Tensor) -> torch.Tensor:
            if isinstance(slice_indices, slice):
                return tensor[:, slice_indices, ...]
            if isinstance(slice_indices, torch.Tensor):
                return tensor.index_select(1, slice_indices.to(tensor.device))
            raise TypeError(
                f"Unsupported index type for logits_to_keep: {type(slice_indices)}"
            )

        stacked_exit_pdf = None

        expected_logits_cache: Optional[torch.Tensor] = None

        def compute_expected_logits() -> Optional[torch.Tensor]:
            nonlocal expected_logits_cache
            if expected_logits_cache is not None:
                return expected_logits_cache
            if stacked_exit_pdf is None or not hidden_states_list:
                return None
            token_exit_pdf = _select_token_positions(stacked_exit_pdf)
            expected_logits = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_hidden = _select_token_positions(hidden)
                step_logits = self.lm_head(step_hidden)
                weight = (
                    token_exit_pdf[..., step_idx].unsqueeze(-1).to(step_logits.dtype)
                )
                expected_logits = (
                    step_logits * weight
                    if expected_logits is None
                    else expected_logits + step_logits * weight
                )
            expected_logits_cache = expected_logits
            return expected_logits_cache

        logits: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return result
