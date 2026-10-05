"""Plan one fixed-depth actor update without changing its samples or loss denominator."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from vime.utils.types import RecurrentTrace

from .prefix import PrefixIdentity
from .training import masked_token_weights

if TYPE_CHECKING:
    from vime.backends.megatron_utils.data import DataIterator


@dataclass(frozen=True)
class PrefixGroup:
    identity: PrefixIdentity
    indices: tuple[int, ...]
    waves: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LogicalStep:
    actor_generation: int
    iterator_offset: int
    microbatches: tuple[tuple[int, ...], ...]
    indices: tuple[int, ...]
    sample_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    weights: dict[int, torch.Tensor]
    groups: tuple[PrefixGroup, ...]

    def check_current(self, actor_generation: int, iterator_offset: int) -> None:
        if (actor_generation, iterator_offset) != (self.actor_generation, self.iterator_offset):
            raise ValueError("Logical prefix plan belongs to another actor update or iterator step")


def plan_step(
    iterator: "DataIterator",
    num_microbatches: int,
    *,
    actor_generation: int,
    model_revision: str,
    loop_depth: int,
    model_family: Literal["ouro", "nanbeige"] = "ouro",
    suffix_wave_size: int,
    reduction: Literal["token_mean", "response_mean"],
) -> LogicalStep:
    """Keep original identities; group only canonical causal, fixed-depth Ouro/Nanbeige.

    Prefix positions start at zero and use the provider's unmodified causal
    mask. Response loss masks remain per sample and do not change prefix keys.
    The caller supplies the current actor generation, not a behavior version.
    """
    if min(num_microbatches, suffix_wave_size, loop_depth) < 1 or iterator.offset < 0:
        raise ValueError("A logical step requires positive microbatch, wave and depth counts")
    if model_family not in ("ouro", "nanbeige"):
        raise ValueError("Prefix planning supports Ouro and Nanbeige; Huginn requires latent-aware replay")
    data = iterator.rollout_data
    if data.get("position_ids") is not None or data.get("attention_mask") is not None:
        raise ValueError("Prefix sharing requires canonical positions and causal attention")
    microbatches = tuple(
        tuple(group) for group in iterator.micro_batch_indices[iterator.offset : iterator.offset + num_microbatches]
    )
    indices = tuple(index for group in microbatches for index in group)
    tokens = cast(list[torch.Tensor], data["tokens"])
    if (
        len(microbatches) != num_microbatches
        or not all(microbatches)
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= len(tokens) for index in indices)
    ):
        raise ValueError("Microbatches must contain the complete step's valid indices exactly once")
    samples = cast(list[int], data["sample_indices"])
    group_ids = cast(list[int], data["group_indices"])
    sample_ids = tuple(samples[index] for index in indices)
    original_groups = tuple(group_ids[index] for index in indices)
    if len(set(sample_ids)) != len(indices) or any(value is None for value in (*sample_ids, *original_groups)):
        raise ValueError("Logical samples require unique sample IDs and original GRPO group IDs")
    masks = cast(list[torch.Tensor], data["loss_masks"])
    lengths = cast(list[int], data["response_lengths"])
    traces = cast(list[RecurrentTrace], data["recurrent_inputs"])
    grouped: dict[PrefixIdentity, list[int]] = {}
    for index in indices:
        sequence, length, trace, mask = tokens[index], lengths[index], traces[index], masks[index]
        if sequence.ndim != 1 or not 0 <= length < len(sequence) or mask.shape != (length,):
            raise ValueError("Each response and loss mask must follow a nonempty token-vector prompt")
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
            raise ValueError("Logical loss masks must be finite and nonnegative")
        if (
            trace.model_family != model_family
            or trace.model_revision != model_revision
            or trace.prefill_depth != loop_depth
            or trace.decode_depths != [loop_depth] * length
            or trace.latent_seed is not None
            or trace.latent_profile is not None
        ):
            raise ValueError("Logical prefix plan requires the recorded fixed-depth model family and revision")
        prompt = tuple(sequence[: len(sequence) - length].tolist())
        identity = PrefixIdentity(model_revision, actor_generation, prompt, loop_depth)
        grouped.setdefault(identity, []).append(index)
    weights = masked_token_weights([masks[index] for index in indices], reduction)
    groups = tuple(
        PrefixGroup(
            identity,
            tuple(group),
            tuple(tuple(group[i : i + suffix_wave_size]) for i in range(0, len(group), suffix_wave_size)),
        )
        for identity, group in grouped.items()
    )
    return LogicalStep(
        actor_generation,
        iterator.offset,
        microbatches,
        indices,
        sample_ids,
        original_groups,
        dict(zip(indices, weights, strict=True)),
        groups,
    )
