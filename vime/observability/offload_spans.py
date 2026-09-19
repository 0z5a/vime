"""Opt-in wall-clock spans for the colocate offload/onload dance.

In colocate mode every step hands the GPUs back and forth between Megatron and
vLLM, and `Actor.sleep()` / `Actor.wake_up()` sit on the critical path doing it.
The actor's own `@timer` reports the two totals, which on a small-model
short-response workload is the largest single block in the step -- but not where
inside them the time goes. `sleep()` alone is `clear_memory(clear_host_memory=True)`
+ `destroy_process_groups()` + `torch_memory_saver.pause()` + two `print_memory()`
calls, and those have very different fixes.

This module measures the breakdown. It is disabled unless
``VIME_OFFLOAD_SPANS=1``, so the default path pays nothing but one dict lookup per
span, and it only accumulates ``perf_counter()`` deltas: no tensor work, no extra
Ray calls, no change to any value that is produced or returned.

Spans nest, so a parent's ``total_s`` includes its children. The spans recorded
in `Actor.sleep()` / `Actor.wake_up()` are disjoint segments of the same call,
which is what makes their sum comparable to the call's own total.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

_ENABLED = os.environ.get("VIME_OFFLOAD_SPANS") == "1"


class Spans:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextlib.contextmanager
    def span(self, name: str):
        if not _ENABLED:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1


_current = Spans()


def span(name: str):
    return _current.span(name)


def reset() -> None:
    _current.totals.clear()
    _current.counts.clear()


def snapshot() -> tuple[dict[str, float], dict[str, int]]:
    """Seconds per span name and call counts recorded since the last reset."""
    return dict(_current.totals), dict(_current.counts)


def emit(label: str, total_s: float, logger) -> None:
    if not _ENABLED:
        return
    totals, counts = snapshot()
    payload = {
        name: {"total_s": round(value, 6), "calls": counts.get(name, 0)}
        for name, value in sorted(totals.items(), key=lambda kv: -kv[1])
    }
    # The recorded spans are disjoint segments of one call, so their sum is what
    # the breakdown actually accounts for; the remainder is call overhead. Summing
    # the rounded entries keeps `accounted_s` equal to the table next to it.
    accounted = round(sum(entry["total_s"] for entry in payload.values()), 6)
    logger.info(
        "OFFLOAD_SPANS %s total_s=%.6f accounted_s=%.6f unaccounted_s=%.6f %s",
        label,
        total_s,
        accounted,
        round(total_s - accounted, 6),
        json.dumps(payload, separators=(",", ":")),
    )
