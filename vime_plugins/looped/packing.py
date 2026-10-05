"""Packed fixed-depth replay metadata and explicit attention backends."""

from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

AttentionBackend = Literal["sdpa-reference", "varlen"]


@dataclass(frozen=True)
class SuffixLayout:
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]
    positions: torch.Tensor

    @classmethod
    def create(cls, lengths: tuple[int, ...], device: torch.device) -> "SuffixLayout":
        if not lengths or min(lengths) < 0:
            raise ValueError("suffix inputs require nonnegative lengths")
        return cls(
            lengths,
            (0, *accumulate(lengths)),
            torch.tensor([p for length in lengths for p in range(length)], dtype=torch.long, device=device),
        )


def prefix_readout(first: torch.Tensor, hidden: torch.Tensor, layout: SuffixLayout | None) -> torch.Tensor:
    if layout is None:
        return torch.cat((first, hidden))
    return torch.cat([torch.cat((first, part)) for part in hidden.split(layout.lengths)])


def prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix: tuple[torch.Tensor, torch.Tensor],
    layout: SuffixLayout | None = None,
) -> torch.Tensor:
    """Suffix queries see the complete shared prefix and their causal suffix."""
    if layout is not None:
        return torch.cat(
            [
                prefix_attention(q[begin:end], k[begin:end], v[begin:end], prefix)
                for begin, end in zip(layout.boundaries[:-1], layout.boundaries[1:], strict=True)
                if begin < end
            ]
        )
    length = prefix[0].shape[0]
    k, v = torch.cat((prefix[0], k)), torch.cat((prefix[1], v))
    mask = torch.arange(k.shape[0], device=q.device)[None, :] <= (
        length + torch.arange(q.shape[0], device=q.device)[:, None]
    )
    return (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            enable_gqa=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )


@dataclass(frozen=True)
class QueryChunk:
    prefix: int
    begin: int
    end: int
    cu_query: torch.Tensor
    cu_key: torch.Tensor


@dataclass(frozen=True)
class ReplayLayout:
    lengths: tuple[int, ...]
    seeds: tuple[int, ...]
    backend: AttentionBackend
    boundaries: tuple[int, ...]
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    chunks: tuple[QueryChunk, ...] = ()

    @classmethod
    def create(
        cls,
        lengths: tuple[int, ...],
        seeds: tuple[int, ...],
        backend: AttentionBackend,
        device: torch.device,
        *,
        token_chunk: int = 0,
    ) -> "ReplayLayout":
        if not lengths or min(lengths) < 1 or len(seeds) != len(lengths):
            raise ValueError("packed replay requires positive lengths and one seed per sequence")
        boundaries = (0, *accumulate(lengths))
        positions = torch.tensor([p for length in lengths for p in range(length)], device=device)
        chunks = (
            tuple(
                QueryChunk(
                    prefix,
                    begin,
                    min(begin + token_chunk, end),
                    torch.tensor([0, min(token_chunk, end - begin)], dtype=torch.int32, device=device),
                    torch.tensor([0, min(begin + token_chunk, end) - prefix], dtype=torch.int32, device=device),
                )
                for prefix, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                for begin in range(prefix, end, token_chunk)
            )
            if token_chunk
            else ()
        )
        return cls(
            lengths,
            seeds,
            backend,
            boundaries,
            positions,
            torch.tensor(boundaries, dtype=torch.int32, device=device),
            chunks,
        )


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: ReplayLayout | None = None
) -> torch.Tensor:
    """Flat THD tensors; the reference batches projections but not attention."""
    if layout is not None and layout.chunks:
        return _chunked_attention(q, k, v, layout)
    if layout is not None and layout.backend == "varlen":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("varlen replay requires CUDA FP16/BF16; select sdpa-reference for the CPU oracle")
        from torch.nn.attention.varlen import varlen_attn

        maximum = max(layout.lengths)
        return cast(
            torch.Tensor,
            varlen_attn(
                q,
                k,
                v,
                layout.cu_seqlens,
                layout.cu_seqlens,
                maximum,
                maximum,
                window_size=(-1, 0),
                enable_gqa=True,
            ),
        )
    boundaries = (0, q.shape[0]) if layout is None else layout.boundaries
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                q[begin:end].transpose(0, 1).unsqueeze(0),
                k[begin:end].transpose(0, 1).unsqueeze(0),
                v[begin:end].transpose(0, 1).unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )


def _chunked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: ReplayLayout) -> torch.Tensor:
    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk: QueryChunk) -> torch.Tensor:
        if layout.backend == "varlen":
            if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("varlen replay requires CUDA FP16/BF16")
            from torch.nn.attention.varlen import varlen_attn

            # The pinned Torch varlen implementation uses FlashAttention's
            # bottom-right causal mask. The CUDA oracle covers unequal Q/K lengths.
            return cast(
                torch.Tensor,
                varlen_attn(
                    q,
                    k,
                    v,
                    chunk.cu_query,
                    chunk.cu_key,
                    q.shape[0],
                    k.shape[0],
                    window_size=(-1, 0),
                    enable_gqa=True,
                ),
            )
        mask = (
            torch.arange(k.shape[0], device=q.device)[None, :]
            <= torch.arange(chunk.begin - chunk.prefix, chunk.end - chunk.prefix, device=q.device)[:, None]
        )
        return (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    outputs = []
    for chunk in layout.chunks:
        inputs = (q[chunk.begin : chunk.end], k[chunk.prefix : chunk.end], v[chunk.prefix : chunk.end])
        # Binding each chunk avoids replaying the last chunk's mask in backward.
        forward = partial(attend, chunk=chunk)
        outputs.append(
            checkpoint(forward, *inputs, use_reentrant=False) if torch.is_grad_enabled() else forward(*inputs)
        )
    return torch.cat(outputs)


def readout_boundaries(total_tokens: int, lengths: tuple[int, ...]) -> list[int]:
    if not lengths or min(lengths) < 1 or sum(lengths) > total_tokens:
        raise ValueError("CPU sequence lengths must cover the real packed token spans")
    boundaries = [0, *accumulate(lengths)]
    if boundaries[-1] < total_tokens:
        boundaries.append(total_tokens)
    return boundaries
