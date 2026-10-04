"""Packed fixed-depth replay metadata and explicit attention backends."""

from dataclasses import dataclass
from itertools import accumulate
from typing import Literal, cast

import torch
import torch.nn.functional as F

AttentionBackend = Literal["sdpa-reference", "varlen"]


@dataclass(frozen=True)
class ReplayLayout:
    lengths: tuple[int, ...]
    seeds: tuple[int, ...]
    backend: AttentionBackend
    boundaries: tuple[int, ...]
    positions: torch.Tensor
    cu_seqlens: torch.Tensor

    @classmethod
    def create(
        cls, lengths: tuple[int, ...], seeds: tuple[int, ...], backend: AttentionBackend, device: torch.device
    ) -> "ReplayLayout":
        if not lengths or min(lengths) < 1 or len(seeds) != len(lengths):
            raise ValueError("packed replay requires positive lengths and one seed per sequence")
        boundaries = (0, *accumulate(lengths))
        positions = torch.tensor([p for length in lengths for p in range(length)], device=device)
        return cls(
            lengths, seeds, backend, boundaries, positions, torch.tensor(boundaries, dtype=torch.int32, device=device)
        )


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: ReplayLayout | None = None
) -> torch.Tensor:
    """Flat THD tensors; the reference batches projections but not attention."""
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


def readout_boundaries(total_tokens: int, lengths: tuple[int, ...]) -> list[int]:
    if not lengths or min(lengths) < 1 or sum(lengths) > total_tokens:
        raise ValueError("CPU sequence lengths must cover the real packed token spans")
    boundaries = [0, *accumulate(lengths)]
    if boundaries[-1] < total_tokens:
        boundaries.append(total_tokens)
    return boundaries
