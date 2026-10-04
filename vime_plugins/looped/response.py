"""Response alignment shared by recurrent providers and RL objectives."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from .readout import streamed_readout
from .packing import ReplayLayout
from .execution import RecurrentProgram, RematPlan, recurrent_scores


class RecurrentProvider(Protocol):
    lm_head: torch.nn.Linear

    @property
    def readout_depth(self) -> int: ...

    def iter_readout_states(
        self, tokens: torch.Tensor, *, all_loops: bool, latent_seed: int, layout: ReplayLayout | None = None
    ) -> Iterator[torch.Tensor]: ...

    def recurrent_program(
        self, tokens: torch.Tensor, *, latent_seed: int, layout: ReplayLayout, plan: RematPlan
    ) -> RecurrentProgram: ...


@dataclass(frozen=True)
class ResponseReadout:
    response_lengths: tuple[int, ...]
    vocab_tile: int
    temperature: float
    all_loops: bool
    entropy: bool = False
    sequence_lengths: tuple[int, ...] = ()
    attention_backend: Literal["serial", "sdpa-reference", "varlen"] = "serial"
    rematerialization: RematPlan | None = None


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
    if request.rematerialization is not None:
        layout = ReplayLayout.create(
            (len(replay_tokens),),
            (seed,),
            "sdpa-reference",
            tokens.device,
            token_chunk=request.rematerialization.token_chunk,
        )
        rows = torch.arange(begin, begin + response_length, device=tokens.device)
        return _rematerialized_readout(model, replay_tokens, rows, labels, request, layout)
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
    if request.attention_backend != "serial":
        return _batched_response_readout(model, tokens, boundaries, seeds[:count], request)
    return torch.cat(
        [
            _response_readout(model, tokens[0, boundaries[i] : boundaries[i + 1]], length, request, seeds[i])
            for i, length in enumerate(request.response_lengths)
        ]
    )


def _batched_response_readout(
    model: RecurrentProvider, tokens: torch.Tensor, boundaries: list[int], seeds: list[int], request: ResponseReadout
) -> torch.Tensor:
    lengths = tuple(boundaries[i + 1] - boundaries[i] for i in range(len(request.response_lengths)))
    if any(not 0 <= response < length for response, length in zip(request.response_lengths, lengths, strict=True)):
        raise ValueError("response must follow a nonempty prompt")
    replay_lengths = tuple(max(1, length - 1) for length in lengths)
    assert request.attention_backend != "serial"
    layout = ReplayLayout.create(
        replay_lengths,
        tuple(seeds),
        request.attention_backend,
        tokens.device,
        token_chunk=0 if request.rematerialization is None else request.rematerialization.token_chunk,
    )
    replay = torch.cat([tokens[0, boundaries[i] : boundaries[i] + length] for i, length in enumerate(replay_lengths)])
    rows, labels = [], []
    for i, (length, response) in enumerate(zip(lengths, request.response_lengths, strict=True)):
        begin = length - response - 1
        rows.extend(range(layout.boundaries[i] + begin, layout.boundaries[i] + begin + response))
        labels.append(tokens[0, boundaries[i] + begin + 1 : boundaries[i + 1]])
    rows = torch.tensor(rows, device=tokens.device, dtype=torch.long)
    labels = torch.cat(labels)
    if request.rematerialization is not None:
        return _rematerialized_readout(model, replay, rows, labels, request, layout)
    depth = model.readout_depth if request.all_loops else 1
    scores = []
    for index, hidden in enumerate(
        model.iter_readout_states(replay, all_loops=request.all_loops, latent_seed=0, layout=layout), 1
    ):
        result = streamed_readout(
            hidden.index_select(0, rows),
            model.lm_head.weight,
            labels,
            vocab_tile=request.vocab_tile,
            temperature=request.temperature,
            entropy=request.entropy and index == depth,
        )
        scores.append(result.log_probs)
    return torch.stack([*scores, result.entropy], dim=-1)


def _rematerialized_readout(
    model: RecurrentProvider,
    tokens: torch.Tensor,
    rows: torch.Tensor,
    labels: torch.Tensor,
    request: ResponseReadout,
    layout: ReplayLayout,
) -> torch.Tensor:
    assert request.rematerialization is not None
    program = model.recurrent_program(tokens, latent_seed=0, layout=layout, plan=request.rematerialization)

    def score(hidden: torch.Tensor, terminal: bool) -> torch.Tensor:
        result = streamed_readout(
            hidden.index_select(0, rows),
            model.lm_head.weight,
            labels,
            vocab_tile=request.vocab_tile,
            temperature=request.temperature,
            entropy=terminal and request.entropy,
        )
        return torch.stack((result.log_probs, result.entropy), dim=-1)

    return recurrent_scores(
        program, score, all_loops=request.all_loops, response_tokens=len(labels), plan=request.rematerialization
    )
