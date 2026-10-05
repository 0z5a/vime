"""Three-phase differentiable prefix reuse with every recurrent readout boundary.

This is a serial SDPA reference for the schedule of arXiv:2606.01143v3,
extended to fixed recurrent calls and per-loop response supervision.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from .readout import streamed_readout

SuffixReplay = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], Iterator[torch.Tensor]]


@dataclass(frozen=True)
class PrefixProgram:
    boundaries: tuple[torch.Tensor, ...]
    depth: int
    suffix: SuffixReplay
    prompt_tokens: tuple[int, ...]
    latent_seed: int | None = None
    latent_profile: str | None = None


@dataclass(frozen=True)
class PrefixIdentity:
    model_revision: str
    policy_version: int
    prompt_tokens: tuple[int, ...]
    loop_depth: int
    latent_seed: int | None = None
    latent_profile: str | None = None


class PrefixReplay:
    """One current-actor logical group; no persistent rollout-cache reuse.

    Backward each suffix microbatch with the logical batch's weights, then call
    backward_prefix once before any optimizer step. Detached boundary leaves
    accumulate all KV and direct-readout adjoints without retaining suffix graphs.
    """

    def __init__(
        self,
        program: PrefixProgram,
        identity: PrefixIdentity,
        actor: torch.nn.Module,
        weight: torch.Tensor,
        *,
        vocab_tile: int,
        temperature: float,
        all_loops: bool,
        entropy: bool,
    ):
        if (
            program.depth != identity.loop_depth
            or program.prompt_tokens != identity.prompt_tokens
            or not identity.prompt_tokens
            or program.latent_seed != identity.latent_seed
            or program.latent_profile != identity.latent_profile
        ):
            raise ValueError("prefix identity requires the current nonempty prompt and fixed depth")
        self.program, self.identity, self.actor = program, identity, actor
        self.versions = self._parameter_versions()
        self.leaves = tuple(value.detach().requires_grad_(value.requires_grad) for value in program.boundaries)
        self.weight, self.vocab_tile, self.temperature = weight, vocab_tile, temperature
        self.all_loops, self.entropy, self.closed = all_loops, entropy, False

    def _check(self, identity: PrefixIdentity) -> None:
        if self.closed or identity != self.identity:
            raise ValueError("prefix replay is closed or belongs to a different policy/prompt/depth")
        if self.versions != self._parameter_versions():
            raise RuntimeError("actor parameters changed before logical prefix backward")

    def _parameter_versions(self) -> tuple[tuple[int, int, torch.dtype, torch.device], ...]:
        return tuple((id(p), p._version, p.dtype, p.device) for p in self.actor.parameters())

    def scores(self, response: torch.Tensor, *, identity: PrefixIdentity) -> torch.Tensor:
        self._check(identity)
        if response.ndim != 1:
            raise ValueError("shared-prefix response must be a token vector")
        if not len(response):
            zero = sum(leaf.reshape(-1)[:1].sum() * 0 for leaf in self.leaves) + self.weight[:1, :1].sum() * 0
            return zero.expand(0, (self.program.depth if self.all_loops else 1) + 1)
        columns = []
        for loop, hidden in enumerate(self.program.suffix(response[:-1], self.leaves), 1):
            terminal = loop == self.program.depth
            if self.all_loops or terminal:
                result = streamed_readout(
                    hidden,
                    self.weight,
                    response,
                    vocab_tile=self.vocab_tile,
                    temperature=self.temperature,
                    entropy=terminal and self.entropy,
                )
                columns.append(result.log_probs)
                if terminal:
                    columns.append(result.entropy)
        return torch.stack(columns, dim=-1)

    def backward_prefix(self, *, identity: PrefixIdentity) -> None:
        self._check(identity)
        outputs, gradients = [], []
        for original, leaf in zip(self.program.boundaries, self.leaves, strict=True):
            if original.requires_grad:
                outputs.append(original)
                gradients.append(torch.zeros_like(original) if leaf.grad is None else leaf.grad)
        if outputs:
            torch.autograd.backward(outputs, gradients)
        self.closed = True

    def backward_suffixes(
        self,
        responses: tuple[torch.Tensor, ...],
        microbatches: tuple[tuple[int, ...], ...],
        objective: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor],
        *,
        identity: PrefixIdentity,
    ) -> torch.Tensor:
        """Run B/C with caller-supplied logical loss weights; optimizer follows C."""
        indices = [index for group in microbatches for index in group]
        if sorted(indices) != list(range(len(responses))) or not all(microbatches):
            raise ValueError("suffix microbatches must partition the logical group exactly once")
        total = self.weight.new_zeros(())
        for group in microbatches:
            scores = torch.cat([self.scores(responses[index], identity=identity) for index in group])
            loss = objective(scores, group)
            loss.backward()
            total = total + loss.detach()
        self.backward_prefix(identity=identity)
        return total
