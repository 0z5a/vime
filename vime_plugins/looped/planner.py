"""Choose among measured schedules for one immutable workload contract."""

from dataclasses import dataclass
from itertools import product
from typing import Literal

from .execution import RematPlan

MemoryMetric = Literal["cuda_peak_allocated", "forward_observed_saved_storage"]
TimingScope = Literal["actor_update", "full_rl"]


@dataclass(frozen=True)
class ScheduleProfile:
    plan: RematPlan
    contract: str
    scope: TimingScope
    memory_metric: MemoryMetric
    seconds: float
    memory_bytes: int
    gradient_verified: bool


def candidate_plans(loops: tuple[int, ...], layers: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[RematPlan, ...]:
    return tuple(dict.fromkeys(RematPlan(*values) for values in product(loops, layers, chunks)))


def select_plan(
    profiles: tuple[ScheduleProfile, ...],
    *,
    contract: str,
    scope: TimingScope,
    memory_metric: MemoryMetric,
    budget_bytes: int,
) -> RematPlan:
    """No extrapolation or mixing micro/whole-step, CPU/GPU or model identities."""
    eligible = [
        profile
        for profile in profiles
        if profile.contract == contract
        and profile.scope == scope
        and profile.memory_metric == memory_metric
        and profile.gradient_verified
        and 0 < profile.seconds < float("inf")
        and 0 <= profile.memory_bytes <= budget_bytes
    ]
    if not eligible:
        raise ValueError("no gradient-verified measured schedule fits this contract and memory budget")
    return min(eligible, key=lambda profile: (profile.seconds, profile.memory_bytes)).plan
