"""Three-phase differentiable prefix reuse with every recurrent readout boundary.

The schedule of arXiv:2606.01143v3 extended to fixed recurrent calls and
per-loop response supervision, with optional packed suffix projections.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch

from .readout import streamed_readout
from .packing import SuffixLayout

SuffixReplay = Callable[[torch.Tensor, tuple[torch.Tensor, ...], SuffixLayout | None], Iterator[torch.Tensor]]
ParameterVersions = tuple[tuple[int, int, torch.dtype, torch.device, bool], ...]


def parameter_versions(actor: torch.nn.Module) -> ParameterVersions:
    return tuple((id(p), p._version, p.dtype, p.device, p.requires_grad) for p in actor.parameters())


@dataclass(frozen=True)
class PrefixProgram:
    boundaries: tuple[torch.Tensor, ...]
    depth: int
    suffix: SuffixReplay
    prompt_tokens: tuple[int, ...]
    actor: torch.nn.Module = field(repr=False)
    readout_weight: torch.Tensor = field(repr=False)
    latent_seed: int | None = None
    latent_profile: str | None = None
    versions: ParameterVersions = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", parameter_versions(self.actor))


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
        if (
            program.actor is not actor
            or program.readout_weight is not weight
            or program.versions != parameter_versions(actor)
        ):
            raise RuntimeError("prefix actor binding changed before replay creation")
        self.program, self.identity, self.actor = program, identity, actor
        self.leaves = tuple(value.detach().requires_grad_(value.requires_grad) for value in program.boundaries)
        self.weight, self.vocab_tile, self.temperature = weight, vocab_tile, temperature
        self.all_loops, self.entropy, self.closed = all_loops, entropy, False

    def _check(self, identity: PrefixIdentity) -> None:
        if self.closed or identity != self.identity:
            raise ValueError("prefix replay is closed or belongs to a different policy/prompt/depth")
        if self.program.versions != parameter_versions(self.actor):
            raise RuntimeError("actor parameters changed before logical prefix backward")

    def scores(self, response: torch.Tensor, *, identity: PrefixIdentity) -> torch.Tensor:
        self._check(identity)
        if response.ndim != 1:
            raise ValueError("shared-prefix response must be a token vector")
        if not len(response):
            return self._empty_scores()
        return self._readout(self.program.suffix(response[:-1], self.leaves, None), response)

    def scores_batch(self, responses: tuple[torch.Tensor, ...], *, identity: PrefixIdentity) -> torch.Tensor:
        """Pack projections/readout; each suffix attends only to its own history."""
        self._check(identity)
        if not responses or any(response.ndim != 1 for response in responses):
            raise ValueError("shared-prefix batch requires response token vectors")
        nonempty = tuple(response for response in responses if len(response))
        if not nonempty:
            return self._empty_scores()
        labels = torch.cat(nonempty)
        layout = SuffixLayout.create(tuple(len(response) - 1 for response in nonempty), labels.device)
        states = self.program.suffix(torch.cat([response[:-1] for response in nonempty]), self.leaves, layout)
        return self._readout(states, labels)

    def _empty_scores(self) -> torch.Tensor:
        zero = sum(leaf.reshape(-1)[:1].sum() * 0 for leaf in self.leaves) + self.weight[:1, :1].sum() * 0
        return zero.expand(0, (self.program.depth if self.all_loops else 1) + 1)

    def _readout(self, states: Iterator[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        columns = []
        for loop, hidden in enumerate(states, 1):
            terminal = loop == self.program.depth
            if self.all_loops or terminal:
                result = streamed_readout(
                    hidden,
                    self.weight,
                    labels,
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
        batch_suffixes: bool = False,
        grad_scale_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run B/C with caller-supplied logical loss weights; optimizer follows C."""
        indices = [index for group in microbatches for index in group]
        if sorted(indices) != list(range(len(responses))) or not all(microbatches):
            raise ValueError("suffix microbatches must partition the logical group exactly once")
        total = self.weight.new_zeros(())
        for group in microbatches:
            scores = (
                self.scores_batch(tuple(responses[index] for index in group), identity=identity)
                if batch_suffixes
                else torch.cat([self.scores(responses[index], identity=identity) for index in group])
            )
            loss = objective(scores, group)
            backward_loss = loss if grad_scale_func is None else grad_scale_func(loss)
            backward_loss.backward()
            total = total + loss.detach()
        # Boundary adjoints already carry the optimizer scale from each suffix.
        self.backward_prefix(identity=identity)
        return total
