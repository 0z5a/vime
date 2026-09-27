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

Spans nest, so a parent's ``total_s`` includes its children. ``accounted_s``
only includes root spans, avoiding double counting nested measurements.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

_ENABLED = os.environ.get("VIME_OFFLOAD_SPANS") == "1"
_clock = time.perf_counter


def enabled() -> bool:
    return _ENABLED


class Spans:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.root_total_s = 0.0
        self.depth = 0

    @contextlib.contextmanager
    def span(self, name: str):
        if not _ENABLED:
            yield
            return
        start = _clock()
        self.depth += 1
        try:
            yield
        finally:
            elapsed = _clock() - start
            self.depth -= 1
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            if self.depth == 0:
                self.root_total_s += elapsed


_current = Spans()


def span(name: str):
    return _current.span(name)


def reset() -> None:
    _current.totals.clear()
    _current.counts.clear()
    _current.root_total_s = 0.0


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
    accounted = _current.root_total_s
    logger.info(
        "OFFLOAD_SPANS %s total_s=%.6f accounted_s=%.6f unaccounted_s=%.6f %s",
        label,
        total_s,
        accounted,
        total_s - accounted,
        json.dumps(payload, separators=(",", ":")),
    )
