"""Pre-dispatch environment gating for passes beyond the main loop.

The main executor pass drives ``EnvironmentGuard.tick()`` before every
dispatch; the passes that run outside it reach the same machinery
through ``environment.make_dispatch_gate`` (and the
``run_executor_sync`` adapter ``make_executor_on_tick``). These tests
cover the helper contract and the shared re-review driver
(``_collect_reviews_until_budget``): a concluding guard stops new
dispatches with in-flight work harvested, a pause defers dispatch and
resumes, and a guard-less run keeps its pre-existing paths.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from core.audit.environment import (
    make_dispatch_gate,
    make_executor_on_tick,
)
from core.audit.orchestrator import _collect_reviews_until_budget


class _StubGuard:
    """Duck-typed guard: countable ticks, settable conclusion, and an
    optional pause (tick blocks until released) — the three behaviours
    the gate consumes."""

    def __init__(self) -> None:
        self.ticks = 0
        self.concluded = False
        self._lock = threading.Lock()
        self._release = threading.Event()
        self._release.set()

    def pause(self) -> None:
        self._release.clear()

    def resume(self) -> None:
        self._release.set()

    def tick(self) -> None:
        with self._lock:
            self.ticks += 1
        self._release.wait(timeout=10.0)


def _config(guard: Any) -> SimpleNamespace:
    return SimpleNamespace(environment_guard_state=guard)


# ── make_dispatch_gate ───────────────────────────────────────────────


class TestMakeDispatchGate:
    def test_no_guard_returns_none(self):
        assert make_dispatch_gate(SimpleNamespace()) is None
        assert make_dispatch_gate(_config(None)) is None

    def test_ticks_then_rechecks_stop_rails(self):
        guard = _StubGuard()
        order: list[str] = []

        def stop_check() -> bool:
            order.append(f"rails-after-{guard.ticks}-ticks")
            return False

        gate = make_dispatch_gate(_config(guard), stop_check=stop_check)
        assert gate is not None
        assert gate() is False
        # The stop-rail re-check ran AFTER the tick (main-pass order).
        assert order == ["rails-after-1-ticks"]

    def test_stop_check_verdict_is_returned(self):
        guard = _StubGuard()
        gate = make_dispatch_gate(_config(guard), stop_check=lambda: True)
        assert gate is not None
        assert gate() is True

    def test_without_stop_check_reports_conclusion(self):
        guard = _StubGuard()
        gate = make_dispatch_gate(_config(guard))
        assert gate is not None
        assert gate() is False
        guard.concluded = True
        assert gate() is True
        assert guard.ticks == 2


class TestMakeExecutorOnTick:
    def test_no_guard_returns_none(self):
        assert make_executor_on_tick(SimpleNamespace()) is None

    def test_tick_drives_the_guard(self):
        guard = _StubGuard()
        on_tick = make_executor_on_tick(_config(guard))
        assert on_tick is not None
        assert on_tick({"file": "a.c", "name": "f"}) is None
        assert guard.ticks == 1


# ── _collect_reviews_until_budget with a dispatch gate ───────────────


def _items(n: int) -> list[int]:
    return list(range(n))


class TestCollectReviewsDispatchGate:
    def test_serial_concluding_gate_stops_before_dispatch(self):
        guard = _StubGuard()
        calls: list[int] = []

        def do_review(item: int) -> int:
            calls.append(item)
            if item == 1:
                guard.concluded = True
            return item

        gate = make_dispatch_gate(
            _config(guard), stop_check=lambda: guard.concluded,
        )
        collected = _collect_reviews_until_budget(
            _items(5), do_review, lambda: guard.concluded, 1,
            phase_label="test", dispatch_gate=gate,
        )
        # Items 0 and 1 dispatched; the gate concluded before item 2.
        assert calls == [0, 1]
        assert collected == [0, 1]
        # One tick per attempted dispatch (2 dispatched + 1 refused).
        assert guard.ticks == 3

    def test_serial_no_gate_is_the_pre_existing_path(self):
        calls: list[int] = []
        collected = _collect_reviews_until_budget(
            _items(3), lambda i: calls.append(i) or i, lambda: False, 1,
            phase_label="test",
        )
        assert calls == [0, 1, 2]
        assert collected == [0, 1, 2]

    def test_parallel_gate_refuses_at_worker_entry(self):
        """A gate that trips after two dispatches stops every later
        item at worker entry — futures were all submitted up front, so
        without the worker-entry gate they would all still run."""
        calls: list[int] = []
        calls_lock = threading.Lock()
        allowed = [2]

        def gate() -> bool:
            with calls_lock:
                if allowed[0] > 0:
                    allowed[0] -= 1
                    return False
                return True

        def do_review(item: int) -> int:
            with calls_lock:
                calls.append(item)
            return item

        collected = _collect_reviews_until_budget(
            _items(6), do_review, lambda: False, 2,
            phase_label="test", dispatch_gate=gate,
        )
        assert len(calls) == 2
        assert sorted(collected) == sorted(calls)

    def test_parallel_already_concluded_dispatches_nothing(self):
        guard = _StubGuard()
        guard.concluded = True
        calls: list[int] = []
        gate = make_dispatch_gate(
            _config(guard), stop_check=lambda: guard.concluded,
        )
        collected = _collect_reviews_until_budget(
            _items(4), lambda i: calls.append(i) or i,
            lambda: guard.concluded, 2,
            phase_label="test", dispatch_gate=gate,
        )
        assert calls == []
        assert collected == []

    def test_parallel_pause_defers_dispatch_then_resumes(self):
        """A paused guard blocks every worker at its entry gate — no
        new dispatch starts — and releasing the pause lets the pass
        complete normally."""
        guard = _StubGuard()
        guard.pause()
        calls: list[int] = []
        calls_lock = threading.Lock()

        def do_review(item: int) -> int:
            with calls_lock:
                calls.append(item)
            return item

        gate = make_dispatch_gate(_config(guard), stop_check=lambda: False)
        out: list[list[int]] = []

        def run() -> None:
            out.append(_collect_reviews_until_budget(
                _items(4), do_review, lambda: False, 2,
                phase_label="test", dispatch_gate=gate,
            ))

        t = threading.Thread(target=run)
        t.start()
        # Every worker blocks inside the paused tick before its first
        # dispatch, so no review can have run while the pause holds.
        t.join(timeout=0.3)
        assert t.is_alive()
        with calls_lock:
            assert calls == []
        guard.resume()
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert sorted(out[0]) == [0, 1, 2, 3]

    def test_parallel_no_gate_is_the_pre_existing_path(self):
        calls: list[int] = []
        calls_lock = threading.Lock()

        def do_review(item: int) -> int:
            with calls_lock:
                calls.append(item)
            return item

        collected = _collect_reviews_until_budget(
            _items(4), do_review, lambda: False, 2,
            phase_label="test",
        )
        assert sorted(calls) == [0, 1, 2, 3]
        assert sorted(collected) == [0, 1, 2, 3]
