"""The offload spans are opt-in and account for nested calls once."""

from __future__ import annotations

import json

import pytest

from vime.observability import offload_spans


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, fmt: str, *args) -> None:
        self.messages.append(fmt % args)


@pytest.fixture(autouse=True)
def _reset_spans():
    offload_spans.reset()
    yield
    offload_spans.reset()


def _module(monkeypatch, *, enabled: bool, ticks: list[float]):
    monkeypatch.setattr(offload_spans, "_ENABLED", enabled)
    monkeypatch.setattr(offload_spans, "_clock", iter(ticks).__next__)
    return offload_spans


def _emitted(module, total_s: float) -> tuple[str, dict]:
    recorder = _Recorder()
    module.emit("sleep", total_s, recorder)
    prefix, payload = recorder.messages[-1].rsplit(" ", 1)
    return prefix, json.loads(payload)


def test_disabled_does_not_read_clock_or_emit(monkeypatch):
    module = _module(monkeypatch, enabled=False, ticks=[])
    with module.span("sleep.pause"):
        pass
    recorder = _Recorder()
    module.emit("sleep", 1.0, recorder)
    assert module.snapshot() == ({}, {})
    assert recorder.messages == []


def test_enabled_accumulates_repeated_spans(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.125, 0.2, 0.45])
    with module.span("sleep.pause"):
        pass
    with module.span("sleep.pause"):
        pass
    assert module.snapshot() == ({"sleep.pause": pytest.approx(0.375)}, {"sleep.pause": 2})
    prefix, _ = _emitted(module, 0.5)
    assert "accounted_s=0.375000" in prefix


def test_nested_child_is_not_double_counted(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.1, 0.3, 0.5])
    with module.span("sleep.clear_memory"):
        with module.span("sleep.clear_memory.empty_cache"):
            pass
    prefix, payload = _emitted(module, 0.6)
    assert payload["sleep.clear_memory"] == {"total_s": 0.5, "calls": 1}
    assert payload["sleep.clear_memory.empty_cache"] == {"total_s": 0.2, "calls": 1}
    assert "accounted_s=0.500000 unaccounted_s=0.100000" in prefix


def test_dotted_siblings_both_count_as_roots(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.2, 0.3, 0.6])
    with module.span("sleep.copy"):
        pass
    with module.span("sleep.copy.h2d"):
        pass
    prefix, _ = _emitted(module, 0.7)
    assert "accounted_s=0.500000" in prefix


def test_exception_unwinds_nesting_before_next_span(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(RuntimeError, match="boom"):
        with module.span("sleep.parent"):
            with module.span("sleep.child"):
                raise RuntimeError("boom")
    with module.span("sleep.next"):
        pass
    prefix, _ = _emitted(module, 0.5)
    assert "accounted_s=0.400000" in prefix
    assert module._current.depth == 0


def test_reset_starts_a_fresh_call(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.1, 0.2, 0.25])
    with module.span("sleep.first"):
        pass
    module.reset()
    with module.span("sleep.second"):
        pass
    prefix, payload = _emitted(module, 0.1)
    assert set(payload) == {"sleep.second"}
    assert "accounted_s=0.050000" in prefix


def test_accounting_uses_raw_durations_before_rounding(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.0000006, 0.0000007, 0.0000013])
    with module.span("sleep.a"):
        pass
    with module.span("sleep.b"):
        pass
    prefix, payload = _emitted(module, 0.0000023)
    assert sum(entry["total_s"] for entry in payload.values()) == pytest.approx(0.000002)
    assert "accounted_s=0.000001 unaccounted_s=0.000001" in prefix


def test_emit_payload_is_sorted_by_duration(monkeypatch):
    module = _module(monkeypatch, enabled=True, ticks=[0.0, 0.2, 0.3, 0.4])
    with module.span("sleep.slow"):
        pass
    with module.span("sleep.fast"):
        pass
    _, payload = _emitted(module, 0.5)
    assert list(payload) == ["sleep.slow", "sleep.fast"]
