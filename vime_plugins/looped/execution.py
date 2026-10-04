"""Exact loop/layer rematerialization with response reductions inside segments."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

StateFunction = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]


@dataclass(frozen=True)
class RematPlan:
    loop_interval: int = 0
    layer_interval: int = 0
    token_chunk: int = 0

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.loop_interval, self.layer_interval, self.token_chunk)
        ):
            raise ValueError("rematerialization intervals must be nonnegative integers")


@dataclass(frozen=True)
class RecurrentProgram:
    state: torch.Tensor
    context: tuple[torch.Tensor, ...]
    depth: int
    step: StateFunction
    readout: StateFunction


def run_layers(
    functions: Sequence[StateFunction], state: torch.Tensor, context: tuple[torch.Tensor, ...], interval: int
) -> torch.Tensor:
    def group(state: torch.Tensor, *context: torch.Tensor, functions: tuple[StateFunction, ...]) -> torch.Tensor:
        for function in functions:
            state = function(state, context)
        return state

    width = interval or max(1, len(functions))
    for begin in range(0, len(functions), width):
        forward = partial(group, functions=tuple(functions[begin : begin + width]))
        state = (
            checkpoint(forward, state, *context, use_reentrant=False)
            if interval and torch.is_grad_enabled()
            else forward(state, *context)
        )
    return state


def recurrent_scores(
    program: RecurrentProgram,
    score: Callable[[torch.Tensor, bool], torch.Tensor],
    *,
    all_loops: bool,
    response_tokens: int,
    plan: RematPlan,
) -> torch.Tensor:
    """Retain segment boundaries and small scores, not every loop's hidden state.

    Readout stays inside checkpointed segments. In particular, Huginn's coda
    contributes gradients without becoming the state passed to the next loop.
    All differentiable repeated inputs (including Huginn injection) are explicit.
    """

    def segment(
        state: torch.Tensor, *context: torch.Tensor, begin: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        columns = []
        for depth in range(begin, end):
            state = program.step(state, context)
            terminal = depth + 1 == program.depth
            if all_loops or terminal:
                output = score(program.readout(state, context), terminal)
                columns.append(output[:, 0])
                if terminal:
                    columns.append(output[:, 1])
        return state, torch.stack(columns, dim=-1) if columns else state.new_empty((response_tokens, 0))

    state, outputs = program.state, []
    width = plan.loop_interval or program.depth
    for begin in range(0, program.depth, width):
        forward = partial(segment, begin=begin, end=min(begin + width, program.depth))
        state, scores = (
            checkpoint(forward, state, *program.context, use_reentrant=False)
            if plan.loop_interval and torch.is_grad_enabled()
            else forward(state, *program.context)
        )
        outputs.append(scores)
    return torch.cat(outputs, dim=-1)
