"""The offload spans are opt-in and must stay free when they are off."""

from __future__ import annotations

import importlib
import json
import time

import pytest

from vime.observability import offload_spans


class _Recorder:
    """Stand-in for a logger: `emit` only needs the %-style ``info`` call."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, fmt: str, *args) -> None:
        self.messages.append(fmt % args)


@pytest.fixture(autouse=True)
def _restore_default_module():
    """Every case reloads the module; leave it as the rest of the suite found it."""
    yield
    importlib.reload(offload_spans)


def _module(monkeypatch, *, enabled: bool):
    if enabled:
        monkeypatch.setenv("VIME_OFFLOAD_SPANS", "1")
    else:
        monkeypatch.delenv("VIME_OFFLOAD_SPANS", raising=False)
    return importlib.reload(offload_spans)


def test_disabled_by_default_records_nothing(monkeypatch):
    module = _module(monkeypatch, enabled=False)
    module.reset()
    with module.span("sleep.memory_saver_pause"):
        time.sleep(0.001)
    assert module.snapshot() == ({}, {})


def test_disabled_emit_is_silent(monkeypatch):
    module = _module(monkeypatch, enabled=False)
    recorder = _Recorder()
    module.emit("sleep", 1.0, recorder)
    assert recorder.messages == []


def test_enabled_accumulates_per_name(monkeypatch):
    module = _module(monkeypatch, enabled=True)
    module.reset()
    for _ in range(3):
        with module.span("sleep.memory_saver_pause"):
            time.sleep(0.001)
    totals, counts = module.snapshot()
    assert counts == {"sleep.memory_saver_pause": 3}
    assert totals["sleep.memory_saver_pause"] >= 0.003


def test_parent_includes_children(monkeypatch):
    module = _module(monkeypatch, enabled=True)
    module.reset()
    with module.span("sleep.clear_memory"):
        with module.span("sleep.clear_memory.empty_cache"):
            time.sleep(0.001)
    totals, counts = module.snapshot()
    assert counts == {"sleep.clear_memory": 1, "sleep.clear_memory.empty_cache": 1}
    assert totals["sleep.clear_memory"] >= totals["sleep.clear_memory.empty_cache"]


def test_reset_clears_the_accumulator(monkeypatch):
    module = _module(monkeypatch, enabled=True)
    with module.span("wake.memory_saver_resume"):
        pass
    assert module.snapshot() != ({}, {})
    module.reset()
    assert module.snapshot() == ({}, {})


def test_emit_reports_payload_and_accounted_share(monkeypatch):
    module = _module(monkeypatch, enabled=True)
    module.reset()
    with module.span("sleep.clear_memory"):
        time.sleep(0.002)
    with module.span("sleep.memory_saver_pause"):
        pass

    recorder = _Recorder()
    module.emit("sleep", 1.5, recorder)
    prefix, payload = recorder.messages[-1].rsplit(" ", 1)

    assert prefix.startswith("OFFLOAD_SPANS sleep total_s=1.500000 accounted_s=")
    parsed = json.loads(payload)
    assert set(parsed) == {"sleep.clear_memory", "sleep.memory_saver_pause"}
    accounted = float(prefix.split("accounted_s=")[1].split(" ")[0])
    unaccounted = float(prefix.split("unaccounted_s=")[1])
    assert accounted == pytest.approx(sum(entry["total_s"] for entry in parsed.values()), abs=1e-6)
    assert unaccounted == pytest.approx(1.5 - accounted, abs=1e-6)


def test_emit_payload_is_sorted_by_time(monkeypatch):
    module = _module(monkeypatch, enabled=True)
    module.reset()
    with module.span("sleep.slow"):
        time.sleep(0.002)
    with module.span("sleep.fast"):
        pass

    recorder = _Recorder()
    module.emit("sleep", 0.0, recorder)
    parsed = json.loads(recorder.messages[-1].rsplit(" ", 1)[1])
    assert list(parsed) == ["sleep.slow", "sleep.fast"]
