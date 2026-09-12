"""Tests for core.audit.environment: preflight floors and the
resource watchdog (pause/resume hysteresis, bounded-wait conclude,
run-deadline clamp)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.audit.environment import (
    PREFLIGHT_REFUSE_FREE_BYTES,
    PREFLIGHT_REFUSE_FREE_INODES,
    PREFLIGHT_WARN_FREE_BYTES,
    WATCHDOG_CHECK_INTERVAL_S,
    WATCHDOG_MAX_PAUSE_S,
    WATCHDOG_PAUSE_FREE_BYTES,
    WATCHDOG_RESUME_FREE_BYTES,
    EnvironmentGuard,
    EnvironmentPreflightError,
    preflight_environment,
)

GIB = 1024 * 1024 * 1024


def _stat(free_bytes: int, free_inodes: int = 1_000_000,
          total_inodes: int = 2_000_000,
          total_bytes: int = 64 * 1024 * 1024 * 1024) -> SimpleNamespace:
    return SimpleNamespace(
        f_bavail=free_bytes, f_frsize=1, f_blocks=total_bytes,
        f_favail=free_inodes, f_files=total_inodes,
    )


class _FakeFs:
    """statvfs stub returning a mutable per-call state."""

    def __init__(self, stat: SimpleNamespace) -> None:
        self.stat = stat
        self.calls = 0

    def __call__(self, path: str) -> SimpleNamespace:
        self.calls += 1
        return self.stat


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


# ── Preflight ─────────────────────────────────────────────────────────


class TestPreflight:
    def test_healthy_passes_silently(self, caplog):
        fs = _FakeFs(_stat(4 * GIB))
        with caplog.at_level("WARNING"):
            preflight_environment([Path("/x")], statvfs_fn=fs)
        assert not caplog.records

    def test_refuses_below_byte_floor(self):
        fs = _FakeFs(_stat(PREFLIGHT_REFUSE_FREE_BYTES - 1))
        with pytest.raises(EnvironmentPreflightError) as ei:
            preflight_environment([Path("/x")], statvfs_fn=fs)
        # Loud: the measured numbers and the path are in the message.
        assert "/x" in str(ei.value)
        assert "MiB free" in str(ei.value)

    def test_refuses_below_inode_floor(self):
        fs = _FakeFs(_stat(4 * GIB, free_inodes=PREFLIGHT_REFUSE_FREE_INODES - 1))
        with pytest.raises(EnvironmentPreflightError) as ei:
            preflight_environment([Path("/x")], statvfs_fn=fs)
        assert "free inodes" in str(ei.value)

    def test_warns_between_floors_and_continues(self, caplog):
        fs = _FakeFs(_stat(PREFLIGHT_WARN_FREE_BYTES - 1))
        with caplog.at_level("WARNING"):
            preflight_environment([Path("/x")], statvfs_fn=fs)
        assert any("low" in r.message for r in caplog.records)

    def test_inode_unlimited_fs_never_trips_inode_floor(self):
        # btrfs-style: f_files == 0 means "no inode accounting", not
        # "zero inodes left".
        fs = _FakeFs(_stat(4 * GIB, free_inodes=0, total_inodes=0))
        preflight_environment([Path("/x")], statvfs_fn=fs)

    def test_unmeasurable_path_is_skipped(self):
        def _raise(path: str) -> SimpleNamespace:
            raise OSError("no such filesystem")

        preflight_environment([Path("/x")], statvfs_fn=_raise)

    def test_no_statvfs_degrades_to_noop(self, monkeypatch):
        import core.audit.environment as env

        monkeypatch.setattr(env, "_default_statvfs", lambda: None)
        preflight_environment([Path("/x")])


# ── Watchdog ──────────────────────────────────────────────────────────


def _guard(fs: Any, clock: _Clock, **kw: Any) -> EnvironmentGuard:
    return EnvironmentGuard(
        tmp_dir=Path("/x"),
        clock=clock,
        sleep_fn=clock.sleep,
        statvfs_fn=fs,
        **kw,
    )


class TestWatchdog:
    def test_healthy_tick_is_cheap_and_rate_limited(self):
        clock = _Clock()
        fs = _FakeFs(_stat(4 * GIB))
        g = _guard(fs, clock)
        g.tick()
        first = fs.calls
        assert first >= 1
        g.tick()  # within the interval: no re-measure
        assert fs.calls == first
        clock.t += WATCHDOG_CHECK_INTERVAL_S + 0.1
        g.tick()
        assert fs.calls > first
        assert not g.concluded

    def test_pause_then_recovery_above_resume_floor(self):
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))

        real_sleep = clock.sleep

        def recovering_sleep(s: float) -> None:
            real_sleep(s)
            if clock.t >= 20.0:
                fs.stat = _stat(WATCHDOG_RESUME_FREE_BYTES + 1)

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=recovering_sleep, statvfs_fn=fs,
        )
        g.tick()
        assert not g.concluded
        assert clock.t < WATCHDOG_MAX_PAUSE_S  # resumed, not timed out

    def test_hysteresis_between_pause_and_resume_floor_stays_paused(self):
        # Recovery to just above the PAUSE floor is not enough — the
        # resume floor is 2x (hysteresis), so the guard keeps waiting
        # and eventually concludes.
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))

        real_sleep = clock.sleep

        def partial_recovery_sleep(s: float) -> None:
            real_sleep(s)
            fs.stat = _stat(WATCHDOG_PAUSE_FREE_BYTES + 1)

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=partial_recovery_sleep, statvfs_fn=fs,
        )
        g.tick()
        assert g.concluded
        assert "resource pressure" in g.conclude_reason

    def test_bounded_pause_concludes_with_reason(self):
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        g = _guard(fs, clock)
        g.tick()
        assert g.concluded
        assert "/x" in g.conclude_reason
        assert clock.t >= WATCHDOG_MAX_PAUSE_S
        # A concluded guard never re-enters the pause loop.
        t = clock.t
        g.tick()
        assert clock.t == t

    def test_inode_pressure_pauses_too(self):
        clock = _Clock()
        fs = _FakeFs(_stat(4 * GIB, free_inodes=1))
        g = _guard(fs, clock)
        g.tick()
        assert g.concluded
        assert "inodes" in g.conclude_reason

    def test_abort_check_exits_pause_without_conclude(self):
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        aborted = {"v": False}
        g = _guard(fs, clock, abort_check=lambda: aborted["v"])

        real_sleep = clock.sleep

        def aborting_sleep(s: float) -> None:
            real_sleep(s)
            aborted["v"] = True

        g._sleep = aborting_sleep
        g.tick()
        # The shutdown rails own the conclusion; the guard stands down.
        assert not g.concluded

    def test_no_statvfs_degrades_to_noop(self):
        clock = _Clock()
        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=clock.sleep, statvfs_fn=None,
        )
        # statvfs_fn=None falls back to the platform default; force
        # the degraded state directly.
        g._statvfs = None
        g.tick()
        assert not g.concluded


# ── Run-deadline clamp ───────────────────────────────────────────────


class TestDeadlineClamp:
    """Pause/probe waits are clamped to the run deadline minus the
    drain margin — a pause entered near the wall cap concludes with
    room to drain and report instead of blocking into the
    supervisor's kill, and cumulative pauses can never overshoot
    ``--max-time``."""

    _MARGIN = 300.0

    def _deadline_guard(self, fs: Any, clock: _Clock,
                        deadline: float) -> EnvironmentGuard:
        return EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock, sleep_fn=clock.sleep,
            statvfs_fn=fs, deadline_monotonic=deadline,
            drain_margin_s=self._MARGIN,
        )

    def test_pause_near_deadline_concludes_before_it(self):
        clock = _Clock()
        deadline = 100.0 + self._MARGIN  # 100s of room, << 600s bound
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        g = self._deadline_guard(fs, clock, deadline)
        g.tick()
        assert g.concluded
        assert clock.t < deadline  # concluded BEFORE the deadline
        assert clock.t < WATCHDOG_MAX_PAUSE_S  # never took the full bound
        assert "deadline" in g.conclude_reason

    def test_no_room_left_concludes_without_waiting(self):
        clock = _Clock()
        clock.t = 50.0
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        g = self._deadline_guard(fs, clock, 50.0 + self._MARGIN - 1.0)
        g.tick()
        assert g.concluded
        assert clock.t == 50.0  # no sleep at all

    def test_cumulative_pauses_cannot_exceed_deadline(self):
        # Oscillating filesystem: each pause recovers, pressure comes
        # back, a fresh pause begins. Without the deadline clamp every
        # cycle re-bought the full 600s; with it the clock (which runs
        # THROUGH pauses) caps the total at the deadline.
        clock = _Clock()
        deadline = 900.0 + self._MARGIN
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))

        real_sleep = clock.sleep
        state = {"recover_at": 20.0}

        def oscillating_sleep(s: float) -> None:
            real_sleep(s)
            if clock.t >= state["recover_at"]:
                fs.stat = _stat(WATCHDOG_RESUME_FREE_BYTES + 1)

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock, sleep_fn=oscillating_sleep,
            statvfs_fn=fs, deadline_monotonic=deadline,
            drain_margin_s=self._MARGIN,
        )
        cycles = 0
        while not g.concluded and cycles < 500:
            fs.stat = _stat(WATCHDOG_PAUSE_FREE_BYTES - 1)  # pressure back
            state["recover_at"] = clock.t + 20.0
            clock.t += WATCHDOG_CHECK_INTERVAL_S + 0.1
            g.tick()
            cycles += 1
        assert g.concluded
        assert clock.t <= deadline
        assert cycles > 1  # genuinely oscillated before concluding

    def test_behavior_unchanged_without_deadline(self):
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        g = _guard(fs, clock)  # no deadline
        g.tick()
        assert g.concluded
        assert clock.t >= WATCHDOG_MAX_PAUSE_S  # full bound, as before


# ── Operator floor override ──────────────────────────────────────────


class TestFloorOverride:
    """RAPTOR_ENV_FLOOR_MIB is the sanctioned way to run on a
    deliberately tight filesystem (container tmpfs)."""

    def test_preflight_refuses_then_override_admits(self, monkeypatch):
        fs = _FakeFs(_stat(16 * 1024 * 1024))  # 16 MiB free
        with pytest.raises(EnvironmentPreflightError):
            preflight_environment([Path("/x")], statvfs_fn=fs)
        monkeypatch.setenv("RAPTOR_ENV_FLOOR_MIB", "8")
        preflight_environment([Path("/x")], statvfs_fn=fs)

    def test_override_still_refuses_below_it(self, monkeypatch):
        monkeypatch.setenv("RAPTOR_ENV_FLOOR_MIB", "8")
        fs = _FakeFs(_stat(4 * 1024 * 1024))  # 4 MiB < 8 MiB floor
        with pytest.raises(EnvironmentPreflightError) as ei:
            preflight_environment([Path("/x")], statvfs_fn=fs)
        assert "RAPTOR_ENV_FLOOR_MIB" in str(ei.value)

    def test_zero_disables_byte_floors(self, monkeypatch):
        monkeypatch.setenv("RAPTOR_ENV_FLOOR_MIB", "0")
        fs = _FakeFs(_stat(0))
        preflight_environment([Path("/x")], statvfs_fn=fs)
        clock = _Clock()
        g = _guard(fs, clock)
        g.tick()
        assert not g.concluded  # watchdog byte floor disabled too

    def test_watchdog_pause_floor_overridden(self, monkeypatch):
        monkeypatch.setenv("RAPTOR_ENV_FLOOR_MIB", "8")
        clock = _Clock()
        # 16 MiB free: below the default 64 MiB pause floor, above the
        # 8 MiB override — no pause.
        fs = _FakeFs(_stat(16 * 1024 * 1024))
        g = _guard(fs, clock)
        g.tick()
        assert not g.concluded
        assert clock.t == 0.0  # never paused

    def test_invalid_override_keeps_defaults(self, monkeypatch):
        monkeypatch.setenv("RAPTOR_ENV_FLOOR_MIB", "lots")
        fs = _FakeFs(_stat(PREFLIGHT_REFUSE_FREE_BYTES - 1))
        with pytest.raises(EnvironmentPreflightError):
            preflight_environment([Path("/x")], statvfs_fn=fs)


# ── Resume-floor reachability clamp ──────────────────────────────────


class TestResumeFloorClamp:
    _MIB = 1024 * 1024

    def test_small_fs_recovery_is_reachable_and_stays_resumed(self):
        # Filesystem total (100 MiB) below the configured resume floor
        # (128 MiB): the clamps (resume = half of total = 50 MiB,
        # pause = quarter = 25 MiB) make genuine recovery detectable
        # instead of guaranteeing a full pause + conclude — and the
        # recovered state must SURVIVE the next measuring tick.
        total = 100 * self._MIB
        assert total < WATCHDOG_RESUME_FREE_BYTES
        clock = _Clock()
        # 10 MiB free: below the clamped 25 MiB pause floor.
        fs = _FakeFs(_stat(10 * self._MIB, total_bytes=total))

        real_sleep = clock.sleep
        sleeps = {"n": 0}

        def recovering_sleep(s: float) -> None:
            sleeps["n"] += 1
            real_sleep(s)
            if clock.t >= 20.0:
                # 60 MiB free: above half-of-total (50 MiB), below the
                # unclamped 128 MiB floor.
                fs.stat = _stat(60 * self._MIB, total_bytes=total)

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=recovering_sleep, statvfs_fn=fs,
        )
        g.tick()
        assert not g.concluded  # resumed
        assert clock.t < WATCHDOG_MAX_PAUSE_S
        # The NEXT measuring tick must not re-pause: 60 MiB free is
        # above the clamped 25 MiB pause floor. With only the resume
        # floor clamped, the unclamped 64 MiB pause floor sat ABOVE
        # the 50 MiB resume floor — inverted hysteresis, an immediate
        # re-pause.
        paused_sleeps = sleeps["n"]
        clock.t += WATCHDOG_CHECK_INTERVAL_S + 0.1
        g.tick()
        assert sleeps["n"] == paused_sleeps  # no new pause
        assert not g.concluded

    def test_small_fs_free_between_clamped_floors_never_flaps(self):
        # The inverted-hysteresis shape: 55 MiB free on a 100 MiB
        # filesystem. Unclamped pause floor (64 MiB) > clamped resume
        # floor (50 MiB) inverted the hysteresis: every measuring
        # tick paused (55 < 64) and recovered on its first poll
        # (55 >= 50). With the pause floor clamped in step
        # (total // 4 = 25 MiB), 55 MiB free is simply healthy.
        total = 100 * self._MIB
        clock = _Clock()
        fs = _FakeFs(_stat(55 * self._MIB, total_bytes=total))
        sleeps = {"n": 0}

        def counting_sleep(s: float) -> None:
            sleeps["n"] += 1
            clock.t += s

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=counting_sleep, statvfs_fn=fs,
        )
        for _ in range(5):
            g.tick()
            clock.t += WATCHDOG_CHECK_INTERVAL_S + 0.1
        assert sleeps["n"] == 0  # never paused, not even once
        assert not g.concluded

    def test_small_fs_inode_floors_clamped_in_step(self):
        # Same inversion on the inode pair: 400 total inodes clamp the
        # resume floor to 200 (< the 256 pause floor). The pause floor
        # clamps to 100 in step, so 150 free inodes is healthy.
        clock = _Clock()
        fs = _FakeFs(_stat(
            4 * GIB, free_inodes=150, total_inodes=400,
        ))
        sleeps = {"n": 0}

        def counting_sleep(s: float) -> None:
            sleeps["n"] += 1
            clock.t += s

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=counting_sleep, statvfs_fn=fs,
        )
        for _ in range(5):
            g.tick()
            clock.t += WATCHDOG_CHECK_INTERVAL_S + 0.1
        assert sleeps["n"] == 0
        assert not g.concluded

    def test_large_fs_hysteresis_preserved(self):
        # Other direction: on a filesystem with ample total capacity
        # the clamp is inert — recovery to just above the PAUSE floor
        # still does not resume (the 2x hysteresis holds).
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))

        real_sleep = clock.sleep

        def partial_recovery_sleep(s: float) -> None:
            real_sleep(s)
            fs.stat = _stat(WATCHDOG_PAUSE_FREE_BYTES + 1)

        g = EnvironmentGuard(
            tmp_dir=Path("/x"), clock=clock,
            sleep_fn=partial_recovery_sleep, statvfs_fn=fs,
        )
        g.tick()
        assert g.concluded


# ── Stop-rail integration ────────────────────────────────────────────


class TestStopRails:
    def _config_and_result(self, tmp_path: Path):
        from core.audit.orchestrator import (
            OrchestratorConfig,
            OrchestratorResult,
        )

        config = OrchestratorConfig(
            target_path=tmp_path, out_dir=tmp_path / "out",
        )
        (tmp_path / "out").mkdir()
        return config, OrchestratorResult()

    def test_check_budget_reports_concluded_guard(self, tmp_path):
        import time as _time

        from core.audit.orchestrator import _check_budget

        config, result = self._config_and_result(tmp_path)
        clock = _Clock()
        fs = _FakeFs(_stat(WATCHDOG_PAUSE_FREE_BYTES - 1))
        g = _guard(fs, clock)
        g.tick()
        assert g.concluded
        config.environment_guard_state = g
        assert _check_budget(config, _time.monotonic(), result) is True
        assert result.terminated_by == "environment"
        assert result.environment_fault == g.conclude_reason

    def test_check_budget_ignores_healthy_guard(self, tmp_path):
        import time as _time

        from core.audit.orchestrator import _check_budget

        config, result = self._config_and_result(tmp_path)
        clock = _Clock()
        g = _guard(_FakeFs(_stat(4 * GIB)), clock)
        g.tick()
        config.environment_guard_state = g
        assert _check_budget(config, _time.monotonic(), result) is False
        assert result.terminated_by == "complete"

    def test_executor_stops_after_tick_concludes(self, tmp_path):
        """A tick that concludes the run stops the serial executor
        BEFORE the popped task's review dispatches."""
        from unittest.mock import MagicMock

        from core.audit.executor import ExecutorConfig, run_executor_sync
        from core.audit.task_graph import TaskGraph

        wq = [
            {"file": "a.py", "name": f"f{i}",
             "priority_score": 0.5, "line_start": 1}
            for i in range(4)
        ]
        graph = TaskGraph.from_workqueue(wq, [])
        reviewed: list[str] = []
        concluded = {"v": False}

        def tick(gap: dict) -> None:
            if len(reviewed) >= 2:
                concluded["v"] = True

        def review(gap, shared, config, review_fn, result_obj, **kw):
            reviewed.append(gap["name"])
            outcome = MagicMock()
            outcome.status = "clean"
            outcome.cost_usd = 0.0
            return outcome

        result = MagicMock()
        stats = run_executor_sync(
            graph, MagicMock(), MagicMock(), MagicMock(), result,
            ExecutorConfig(max_workers=1),
            review_one_fn=review,
            on_tick=tick,
            budget_check=lambda: concluded["v"],
        )
        assert stats.budget_stopped
        assert len(reviewed) == 2  # the third dispatch never happened

    def _glance_setup(self, monkeypatch, n: int = 3):
        """A serial executor whose every task is a GLANCE task, with
        the batch machinery stubbed so no LLM plumbing is needed."""
        import core.audit.executor as executor_mod
        from core.audit.task_graph import TaskGraph

        wq = [
            {"file": "a.py", "name": f"g{i}",
             "priority_score": 0.5, "line_start": 1}
            for i in range(n)
        ]
        graph = TaskGraph.from_workqueue(wq, [])
        batches: list[list] = []

        monkeypatch.setattr(
            executor_mod, "_get_batch_review_fn",
            lambda shared, config: (lambda *a, **kw: None),
        )
        monkeypatch.setattr(
            executor_mod, "_is_glance", lambda task, shared: True,
        )

        def fake_process(batch, *args: object, **kwargs: object) -> None:
            batches.append(list(batch))

        monkeypatch.setattr(
            executor_mod, "_process_glance_batch", fake_process,
        )
        return graph, batches

    def test_serial_glance_flush_ticks_before_dispatch(self, monkeypatch):
        """The serial glance-batch flush is a dispatch site: it must
        drive the guard's tick (once per batch) before the LLM batch
        call — a glance-heavy serial run otherwise never pauses."""
        from unittest.mock import MagicMock

        from core.audit.executor import ExecutorConfig, run_executor_sync

        graph, batches = self._glance_setup(monkeypatch)
        ticks: list[dict] = []
        stats = run_executor_sync(
            graph, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            ExecutorConfig(max_workers=1),
            review_one_fn=MagicMock(),
            on_tick=ticks.append,
            budget_check=lambda: False,
        )
        assert batches  # the batch dispatched
        assert ticks  # ...and the tick ran for it
        assert stats.completed == 3

    def test_serial_glance_flush_honors_concluding_tick(self, monkeypatch):
        """A tick that concludes the run stops the serial executor
        BEFORE the glance batch dispatches; the queued tasks stay
        unreviewed gaps."""
        from unittest.mock import MagicMock

        from core.audit.executor import ExecutorConfig, run_executor_sync

        graph, batches = self._glance_setup(monkeypatch)
        concluded = {"v": False}

        def tick(gap: dict) -> None:
            concluded["v"] = True  # the guard concluded during the tick

        stats = run_executor_sync(
            graph, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            ExecutorConfig(max_workers=1),
            review_one_fn=MagicMock(),
            on_tick=tick,
            budget_check=lambda: concluded["v"],
        )
        assert not batches  # the batch never dispatched
        assert stats.budget_stopped
        assert stats.completed == 0  # every glance task stays a gap
