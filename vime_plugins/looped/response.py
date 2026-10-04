"""Response alignment shared by recurrent providers and RL objectives."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import torch

from .readout import streamed_readout


class RecurrentProvider(Protocol):
    lm_head: torch.nn.Linear

    @property
    def readout_depth(self) -> int: ...

    def iter_readout_states(
        self, tokens: torch.Tensor, *, all_loops: bool, latent_seed: int
    ) -> Iterator[torch.Tensor]: ...


@dataclass(frozen=True)
class ResponseReadout:
    response_lengths: tuple[int, ...]
    vocab_tile: int
    temperature: float
    all_loops: bool
    entropy: bool = False


def response_log_probs(
    model: RecurrentProvider,
    tokens: torch.Tensor,
    response_length: int,
    *,
    vocab_tile: int,
    temperature: float,
    all_loops: bool,
    latent_seed: int = 0,
) -> torch.Tensor:
    """Return [response tokens, supervised loops], including the prompt boundary.

    This explicit fixed-depth replay entry is independent of Megatron's default
    dense-logit loss bridge. The caller supplies the recorded latent identity.
    """
    request = ResponseReadout((response_length,), vocab_tile, temperature, all_loops)
    return _response_readout(model, tokens, response_length, request, latent_seed)[:, :-1]


def _response_readout(
    model: RecurrentProvider, tokens: torch.Tensor, response_length: int, request: ResponseReadout, seed: int
) -> torch.Tensor:
    if tokens.ndim != 1 or not 0 <= response_length < len(tokens):
        raise ValueError("response must follow a nonempty prompt")
    begin = len(tokens) - response_length - 1
    labels = tokens[begin + 1 :]
    # A one-token prompt with no response still anchors a differentiable zero.
    replay_tokens = tokens[: max(1, len(tokens) - 1)]
    scores = []
    depth = model.readout_depth if request.all_loops else 1
    for index, hidden in enumerate(
        model.iter_readout_states(replay_tokens, all_loops=request.all_loops, latent_seed=seed), 1
    ):
        result = streamed_readout(
            hidden[begin : begin + response_length],
            model.lm_head.weight,
            labels,
            vocab_tile=request.vocab_tile,
            temperature=request.temperature,
            entropy=request.entropy and index == depth,
        )
        scores.append(result.log_probs)
    return torch.stack([*scores, result.entropy], dim=-1)


def packed_response_readout(
    model: RecurrentProvider,
    tokens: torch.Tensor,
    boundaries: list[int],
    seeds: list[int],
    request: ResponseReadout,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return [response tokens, loop scores + terminal entropy] for MCore.

    The caller validates the model's fixed-depth/latent trace contract. Only a
    fully masked trailing padding sequence may be omitted from the request.
    """
    count = len(request.response_lengths)
    if count not in (len(boundaries) - 1, len(boundaries) - 2) or not count:
        raise ValueError("one response length is required per real packed sequence")
    if count == len(boundaries) - 2 and (loss_mask is None or bool(loss_mask[0, boundaries[-2] :].any())):
        raise ValueError("only fully masked padding may omit response readout")
    return torch.cat(
        [
            _response_readout(model, tokens[0, boundaries[i] : boundaries[i + 1]], length, request, seeds[i])
            for i, length in enumerate(request.response_lengths)
        ]
    )
