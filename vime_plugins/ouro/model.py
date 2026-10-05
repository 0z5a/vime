"""Shared-parameter Ouro for Megatron's optimizer, DDP and checkpoint contracts.

The optional vllm-rlt dependency supplies checkpoint-compatible physical modules.
Training uses differentiable SDPA, not the serving KV writes or a HF model wrapper.
"""

from collections.abc import Iterator
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.utils.checkpoint import checkpoint
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroDecoderLayer, OuroForCausalLM

from vime.utils.types import RecurrentTrace
from vime_plugins.looped.packing import (
    ReplayLayout,
    SuffixLayout,
    causal_attention,
    prefix_attention,
    prefix_readout,
    readout_boundaries,
)
from vime_plugins.looped.response import ResponseReadout, packed_response_readout
from vime_plugins.looped.execution import RecurrentProgram, RematPlan, run_layers
from vime_plugins.looped.prefix import PrefixProgram


def decoder_forward(
    layer: OuroDecoderLayer,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    previous_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    layout: ReplayLayout | None = None,
    prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    suffix_layout: SuffixLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = layer.self_attn
    value = layer.input_layernorm(hidden)
    shape = (hidden.shape[0], -1, attention.config.head_dim)
    q, k, v = (projection(value).view(shape) for projection in (attention.q_proj, attention.k_proj, attention.v_proj))

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos + torch.cat((-second, first), dim=-1) * sin

    q, k = rotate(q), rotate(k)
    if previous_kv is not None:
        k = previous_kv[0].index_copy(0, positions, k)
        v = previous_kv[1].index_copy(0, positions, v)
    mask = None if positions is None else torch.arange(k.shape[0], device=k.device)[None, :] <= positions[:, None]
    output = (
        prefix_attention(q, k, v, prefix_kv, suffix_layout).reshape(hidden.shape[0], -1)
        if prefix_kv is not None
        else causal_attention(q, k, v, layout).reshape(hidden.shape[0], -1)
        if layout is not None
        else (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                is_causal=positions is None,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(hidden.shape[0], -1)
        )
    )
    hidden = hidden + layer.input_layernorm_2(attention.o_proj(output))
    return hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden))), k, v


class OuroMegatronModel(MegatronModule):
    prefix_model_family: Literal["ouro", "nanbeige"] = "ouro"

    def __init__(
        self,
        config: TransformerConfig,
        native: OuroForCausalLM | NanbeigeForCausalLM,
        *,
        recompute: bool = False,
        role: Literal["actor", "critic"] = "actor",
        freeze_exit_gate: bool = True,
    ):
        super().__init__(config)
        if (config.tensor_model_parallel_size, config.pipeline_model_parallel_size, config.context_parallel_size) != (
            1,
            1,
            1,
        ):
            raise ValueError("Ouro currently requires TP=PP=CP=1")
        if config.sequence_parallel or config.gradient_accumulation_fusion:
            raise ValueError("Ouro SDPA provider requires sequence parallel and grad fusion disabled")
        self.ouro_config = native.config
        self.model = native.model
        self.role = role
        if role == "critic":
            from vime.backends.megatron_utils.model_provider import LinearForLastLayer

            self.output_layer = LinearForLastLayer(config.hidden_size, 1, config=config, bias=False)
        else:
            self.lm_head = native.lm_head
        self.requires_grad_(True)
        if freeze_exit_gate:
            self.model.early_exit_gate.requires_grad_(False)
        self.pre_process = self.post_process = True
        self.share_embeddings_and_output_weights = False
        self.vp_stage = None
        self.loop_budget = self.ouro_config.total_ut_steps
        self.recompute = recompute
        self.block_tokens = 0
        self.recompute_block_tokens = 0

    def set_input_tensor(self, input_tensor: list[torch.Tensor | None]) -> None:
        if any(value is not None for value in input_tensor):
            raise ValueError("Ouro does not accept pipeline input")

    def set_loop_budget(self, loops: int) -> None:
        if type(loops) is not int or not 1 <= loops <= self.ouro_config.total_ut_steps:
            raise ValueError("Loop budget exceeds the checkpoint's supported depth")
        self.loop_budget = loops

    def _layer(
        self,
        layer: OuroDecoderLayer,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        layout: ReplayLayout | None = None,
    ) -> torch.Tensor:
        self.block_tokens += hidden.shape[0]
        return decoder_forward(layer, hidden, cos, sin, layout=layout)[0]

    def _traced_layer(
        self,
        layer: OuroDecoderLayer,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        positions: torch.Tensor,
        previous_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.block_tokens += hidden.shape[0]
        return decoder_forward(layer, hidden, cos, sin, positions=positions, previous_kv=previous_kv)

    def _prefix_layer(
        self,
        layer: OuroDecoderLayer,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        suffix_layout: SuffixLayout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.block_tokens += hidden.shape[0]
        return decoder_forward(layer, hidden, cos, sin, prefix_kv=prefix_kv, suffix_layout=suffix_layout)

    def prefix_program(self, tokens: torch.Tensor, *, rematerialize: bool = False) -> PrefixProgram:
        """Capture every physical-layer/loop KV and first-response hidden boundary."""
        if self.role != "actor":
            raise ValueError("shared response prefix requires an actor")
        layers, depth, length = len(self.model.layers), self.loop_budget, len(tokens)

        def prefix(tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
            hidden = self.model.embed_tokens(tokens)
            cos, sin = self.model.rotary_emb(hidden, torch.arange(len(tokens), device=tokens.device))
            boundaries = []
            for _ in range(depth):
                for layer in self.model.layers:
                    hidden, k, v = self._prefix_layer(layer, hidden, cos, sin)
                    boundaries.extend((k, v))
                hidden = self._loop_output(hidden)
                boundaries.append(hidden[-1:])
            return tuple(boundaries)

        boundaries = (
            checkpoint(prefix, tokens, use_reentrant=False)
            if rematerialize and torch.is_grad_enabled()
            else prefix(tokens)
        )

        def suffix(
            tokens: torch.Tensor, boundary: tuple[torch.Tensor, ...], layout: SuffixLayout | None
        ) -> Iterator[torch.Tensor]:
            hidden = self.model.embed_tokens(tokens)
            positions = torch.arange(len(tokens), device=tokens.device) if layout is None else layout.positions
            cos, sin = self.model.rotary_emb(hidden, length + positions)
            for loop in range(depth):
                base = loop * (2 * layers + 1)
                if len(tokens):
                    for index, layer in enumerate(self.model.layers):
                        hidden, _, _ = self._prefix_layer(
                            layer,
                            hidden,
                            cos,
                            sin,
                            boundary[base + 2 * index : base + 2 * index + 2],
                            suffix_layout=layout,
                        )
                    hidden = self._loop_output(hidden)
                yield prefix_readout(boundary[base + 2 * layers], hidden, layout)

        return PrefixProgram(
            boundaries, depth, suffix, tuple(tokens.tolist()), self, self.lm_head.weight, rematerialize=rematerialize
        )

    def _traced_sequence(self, tokens: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """Replay LAST_EXITED: skipped deeper KV planes reuse the last computed KV."""
        hidden = self.model.embed_tokens(tokens)
        cos, sin = self.model.rotary_emb(hidden, torch.arange(tokens.numel(), device=tokens.device))
        previous = [None] * len(self.model.layers)
        for depth in range(int(depths.max())):
            positions = (depths > depth).nonzero().flatten().to(tokens.device)
            active = hidden[positions]
            for index, layer in enumerate(self.model.layers):
                forward = partial(self._traced_layer, layer, positions=positions, previous_kv=previous[index])
                inputs = (active, cos[positions], sin[positions])
                active, k, v = (
                    checkpoint(forward, *inputs, use_reentrant=False)
                    if self.recompute and torch.is_grad_enabled()
                    else forward(*inputs)
                )
                previous[index] = (k, v)
            hidden = hidden.index_copy(0, positions, self.model.norm(active))
        return self._readout(hidden)

    def _readout(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_layer(hidden)[0] if self.role == "critic" else self.lm_head(hidden)

    @property
    def readout_depth(self) -> int:
        return self.loop_budget

    def _loop_output(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.norm(hidden)

    def recurrent_program(
        self, tokens: torch.Tensor, *, latent_seed: int, layout: ReplayLayout, plan: RematPlan
    ) -> RecurrentProgram:
        hidden = self.model.embed_tokens(tokens)
        context = self.model.rotary_emb(hidden, layout.positions)

        def layer_forward(layer, state: torch.Tensor, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return self._layer(layer, state, *context, layout=layout)

        functions = tuple(partial(layer_forward, layer) for layer in self.model.layers)

        def step(state: torch.Tensor, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return self._loop_output(run_layers(functions, state, context, plan.layer_interval))

        return RecurrentProgram(hidden, context, self.loop_budget, step, lambda state, context: state)

    def iter_readout_states(
        self,
        tokens: torch.Tensor,
        *,
        all_loops: bool = False,
        latent_seed: int = 0,
        layout: ReplayLayout | None = None,
    ) -> Iterator[torch.Tensor]:
        hidden = self.model.embed_tokens(tokens)
        positions = torch.arange(tokens.numel(), device=tokens.device) if layout is None else layout.positions
        cos, sin = self.model.rotary_emb(hidden, positions)
        # Capture K in the forward, so later schedule changes cannot alter recomputation.
        for depth in range(self.loop_budget):
            for layer in self.model.layers:
                forward = partial(self._layer, layer, layout=layout)
                if self.recompute and torch.is_grad_enabled():
                    hidden = checkpoint(forward, hidden, cos, sin, use_reentrant=False)
                else:
                    hidden = forward(hidden, cos, sin)
            hidden = self.model.norm(hidden)
            if all_loops or depth + 1 == self.loop_budget:
                yield hidden

    def _sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._readout(next(self.iter_readout_states(tokens)))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params: PackedSeqParams | None = None,
        loss_mask=None,
        execution_depths: torch.Tensor | None = None,
        recurrent_inputs: list[RecurrentTrace] | None = None,
        readout: ResponseReadout | None = None,
        prefix_only: bool = False,
        prefix_rematerialize: bool = False,
    ) -> torch.Tensor | PrefixProgram:
        if labels is not None or attention_mask is not None or position_ids is not None:
            raise ValueError("Use VIME's masked RL loss and unmodified per-sequence positions")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Ouro expects one packed token stream")
        if prefix_only:
            if input_ids.shape[1] == 0 or any(
                value is not None
                for value in (packed_seq_params, loss_mask, execution_depths, recurrent_inputs, readout)
            ):
                raise ValueError("A prefix forward requires only one nonempty unmodified prompt")
            return self.prefix_program(input_ids[0], rematerialize=prefix_rematerialize)
        if prefix_rematerialize:
            raise ValueError("Joint rematerialization requires a prefix forward")
        if recurrent_inputs is not None and any(
            trace.prefill_depth != self.loop_budget or any(depth != self.loop_budget for depth in trace.decode_depths)
            for trace in recurrent_inputs
        ):
            raise ValueError("Native training requires the rollout's fixed full depth")
        if execution_depths is not None:
            if (
                execution_depths.ndim != 1
                or execution_depths.numel() > input_ids.numel()
                or execution_depths.dtype not in (torch.int32, torch.int64)
                or not bool(((execution_depths >= 1) & (execution_depths <= self.ouro_config.total_ut_steps)).all())
            ):
                raise ValueError("Execution depths must specify a supported depth per input token")
            # VIME adds a masked padding sequence after the real packed samples.
            execution_depths = F.pad(execution_depths, (0, input_ids.numel() - execution_depths.numel()), value=1)
        if readout is not None:
            if self.role != "actor" or execution_depths is not None or recurrent_inputs is None:
                raise ValueError("response readout requires an actor and fixed-depth recurrent traces")
            if packed_seq_params is not None and packed_seq_params.qkv_format != "thd":
                raise ValueError("response readout expects thd packed sequences")
            boundaries = (
                readout_boundaries(input_ids.numel(), readout.sequence_lengths)
                if readout.attention_backend != "serial"
                else [0, input_ids.numel()]
                if packed_seq_params is None
                else packed_seq_params.cu_seqlens_q.tolist()
            )
            if len(recurrent_inputs) != len(readout.response_lengths):
                raise ValueError("every response requires a recurrent trace")
            return packed_response_readout(
                self, input_ids, boundaries, [0] * len(recurrent_inputs), readout, loss_mask
            )
        if packed_seq_params is None:
            return (
                self._sequence(input_ids[0])
                if execution_depths is None
                else self._traced_sequence(input_ids[0], execution_depths)
            ).unsqueeze(0)
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Ouro expects thd packed sequences")
        boundaries = packed_seq_params.cu_seqlens_q.tolist()
        outputs = [
            (
                self._sequence(input_ids[0, start:end])
                if execution_depths is None
                else self._traced_sequence(input_ids[0, start:end], execution_depths[start:end])
            )
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
        return torch.cat(outputs).unsqueeze(0)


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    role: Literal["actor", "critic"] = "actor",
) -> OuroMegatronModel:
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Ouro supports one non-pipelined model chunk")
    args = get_args()
    config = core_transformer_config_from_args(args)
    native = OuroForCausalLM.from_pretrained(args.hf_checkpoint, dtype=config.params_dtype)
    return OuroMegatronModel(config, native, recompute=args.recompute_granularity is not None, role=role)
