"""Response alignment shared by recurrent providers and RL objectives."""

from collections.abc import Iterator
from typing import Protocol

import torch

from .readout import streamed_readout


class RecurrentProvider(Protocol):
    lm_head: torch.nn.Linear

    def iter_readout_states(
        self, tokens: torch.Tensor, *, all_loops: bool, latent_seed: int
    ) -> Iterator[torch.Tensor]: ...


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
    if tokens.ndim != 1 or not 0 <= response_length < len(tokens):
        raise ValueError("response must follow a nonempty prompt")
    begin = len(tokens) - response_length - 1
    labels = tokens[begin + 1 :]
    # A one-token prompt with no response still anchors a differentiable zero.
    replay_tokens = tokens[: max(1, len(tokens) - 1)]
    scores = [
        streamed_readout(
            hidden[begin : begin + response_length],
            model.lm_head.weight,
            labels,
            vocab_tile=vocab_tile,
            temperature=temperature,
        ).log_probs
        for hidden in model.iter_readout_states(replay_tokens, all_loops=all_loops, latent_seed=latent_seed)
    ]
    return torch.stack(scores, dim=-1)
