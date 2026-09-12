"""Environmental fault tolerance for audit runs.

Two mechanisms share this module:

* **Preflight** (:func:`preflight_environment`) — statvfs on the
  effective TMPDIR and the run output dir at run start. Below the
  refuse floors the run fails loudly before any LLM spend; below the
  warn floors it banners and continues.
* **Resource watchdog** (:class:`EnvironmentGuard`) — an executor-tick
  hook that re-measures the same filesystems during the run
  (rate-limited), pauses dispatch with hysteresis while they are
  under pressure, and — when the pressure does not clear within a
  bounded wait — concludes the run gracefully through the same stop
  rails the wall/cost budgets use (``_check_budget`` consults
  :attr:`EnvironmentGuard.concluded`), so in-flight reviews are
  harvested, the report is written, and the run stays resumable.
Scope: the guard's pause machinery is driven ONLY by the main
executor pass's pre-dispatch tick (both executor paths, including the
serial glance-batch flush). The study consumer, deepen, error-retry,
trivial-batch, and edge passes do not tick — they observe the guard
solely through ``_check_budget``, so they STOP dispatching once a
conclusion is reached but are never paused mid-pass; in particular
the study consumer keeps dispatching during a main-pass pause,
bounded by that pause's own conclusion. Wiring ticks into those
passes is deliberately out of scope here.

Every bounded pause is additionally clamped to the run
deadline (when one is set) minus a drain margin, so a pause entered
near the wall cap concludes with enough room to harvest, report, and
transition the lifecycle instead of blocking into the supervisor's
kill.

Platforms without ``os.statvfs`` degrade silently to no-ops.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ── Preflight floors ──────────────────────────────────────────────────
# Trade-off, both directions: floors set too high refuse or spam
# warnings on small-but-healthy scratch filesystems (containers with
# deliberately tight tmpfs), too low and the run starts into a
# filesystem that exhausts mid-run before the watchdog can react —
# every dispatch then fails identically until something correlates
# the failures. The refuse floors are sized to what a single review
# dispatch plus its run artifacts can plausibly write; the warn
# floors give the operator a run-start signal well before the
# watchdog's pause floor is in sight.
PREFLIGHT_REFUSE_FREE_BYTES = 64 * 1024 * 1024
PREFLIGHT_WARN_FREE_BYTES = 512 * 1024 * 1024
# Inode floors are far below any healthy filesystem: a run creates
# tens of files, not thousands, so only near-exhaustion should trip.
PREFLIGHT_REFUSE_FREE_INODES = 256
PREFLIGHT_WARN_FREE_INODES = 4096

# Operator override for the BYTE floors (preflight refuse/warn,
# watchdog pause/resume): a deliberately tight container tmpfs is a
# legitimate environment, and the fixed floors above would refuse it
# at start (or pause it permanently). The value is the refuse/pause
# floor in MiB; the warn floor scales at 8x and the resume floor at
# 2x, preserving the default ratios. ``0`` disables the byte floors
# entirely. Inode floors are unaffected. Documented in
# docs/environment.md.
_FLOOR_OVERRIDE_ENV = "RAPTOR_ENV_FLOOR_MIB"


def _floor_override_bytes() -> int | None:
    """The byte-floor override in bytes, or ``None`` when unset or
    unparseable (invalid values log once per call and keep the
    defaults — a typo must never silently disable the guard)."""
    raw = os.environ.get(_FLOOR_OVERRIDE_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        mib = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number — keeping the default floors",
            _FLOOR_OVERRIDE_ENV, raw,
        )
        return None
    if mib < 0:
        logger.warning(
            "%s=%r is negative — keeping the default floors",
            _FLOOR_OVERRIDE_ENV, raw,
        )
        return None
    return int(mib * 1024 * 1024)

# ── Watchdog floors ───────────────────────────────────────────────────
# Trade-off, both directions: a pause floor set too high pauses
# spuriously on busy hosts where co-tenant processes legitimately
# work near the floor; too low and the run dies mid-write (journal
# append, sysprompt staging, report) before the rate-limited watchdog
# ever observes the pressure. The resume floor is 2x the pause floor
# (hysteresis): resuming at the pause floor itself would flap —
# dispatch a review, drop below, pause again — on any filesystem
# hovering at the boundary.
WATCHDOG_PAUSE_FREE_BYTES = 64 * 1024 * 1024
WATCHDOG_RESUME_FREE_BYTES = 2 * WATCHDOG_PAUSE_FREE_BYTES
WATCHDOG_PAUSE_FREE_INODES = 256
WATCHDOG_RESUME_FREE_INODES = 2 * WATCHDOG_PAUSE_FREE_INODES
# One statvfs per interval, not per dispatch: statvfs is cheap but
# the tick runs before EVERY dispatch — unbounded polling adds noise
# on network filesystems; too long an interval and a fast-filling
# disk outruns the watchdog.
WATCHDOG_CHECK_INTERVAL_S = 5.0
# Poll cadence while paused. Shorter reacts faster to a cleared
# disk; longer wastes less time re-measuring a filesystem that
# typically needs operator action to clear.
WATCHDOG_POLL_S = 5.0
# Bounded pause. Longer rides out slow external cleanup (log
# rotation, another run finishing) without losing the run; shorter
# returns control to the operator sooner when nothing is going to
# clear the pressure without intervention. After this the run
# concludes gracefully with a resume hint.
WATCHDOG_MAX_PAUSE_S = 600.0


class EnvironmentPreflightError(RuntimeError):
    """Run start refused: a required filesystem is below the refuse floor."""


def _default_statvfs() -> Callable[[str], Any] | None:
    return getattr(os, "statvfs", None)


def _free_space(
    statvfs_fn: Callable[[str], Any], path: Path,
) -> tuple[int, int | None, int, int | None] | None:
    """(free bytes, free inodes, total bytes, total inodes) for
    *path*; the inode figures are ``None`` when the filesystem does
    not account inodes (``f_files == 0``, e.g. btrfs) — a zero there
    means "unlimited", never "exhausted". ``None`` overall when the
    path cannot be measured (missing, EACCES): the caller degrades to
    no-op rather than guessing."""
    try:
        st = statvfs_fn(str(path))
    except OSError:
        return None
    free_bytes = int(st.f_bavail) * int(st.f_frsize)
    total_bytes = int(st.f_blocks) * int(st.f_frsize)
    if int(st.f_files) > 0:
        free_inodes: int | None = int(st.f_favail)
        total_inodes: int | None = int(st.f_files)
    else:
        free_inodes = None
        total_inodes = None
    return free_bytes, free_inodes, total_bytes, total_inodes


def effective_tmp_dir() -> Path:
    """The tempdir every dispatch stages files in (TMPDIR-resolved)."""
    return Path(tempfile.gettempdir())


def preflight_environment(
    paths: list[Path],
    *,
    statvfs_fn: Callable[[str], Any] | None = None,
) -> None:
    """Refuse (raise) or warn when any of *paths* is under-resourced.

    Raises :class:`EnvironmentPreflightError` naming the path and the
    measured free bytes/inodes when a refuse floor is breached; logs
    one warning banner per path between the warn and refuse floors.
    Silently no-ops where ``statvfs`` is unavailable (non-POSIX) or a
    path cannot be measured.
    """
    fn = statvfs_fn if statvfs_fn is not None else _default_statvfs()
    if fn is None:
        return
    override = _floor_override_bytes()
    refuse_bytes = (
        override if override is not None else PREFLIGHT_REFUSE_FREE_BYTES
    )
    # 8x preserves the default warn/refuse ratio under an override.
    warn_bytes = (
        override * 8 if override is not None else PREFLIGHT_WARN_FREE_BYTES
    )
    for path in paths:
        measured = _free_space(fn, path)
        if measured is None:
            continue
        free_bytes, free_inodes, _total_bytes, _total_inodes = measured
        if free_bytes < refuse_bytes or (
            free_inodes is not None
            and free_inodes < PREFLIGHT_REFUSE_FREE_INODES
        ):
            msg = (
                f"environment preflight: {path} has "
                f"{free_bytes / (1024 * 1024):.0f} MiB free"
                + (
                    f" and {free_inodes} free inodes"
                    if free_inodes is not None else ""
                )
                + f" — below the refuse floor "
                f"({refuse_bytes // (1024 * 1024)} MiB / "
                f"{PREFLIGHT_REFUSE_FREE_INODES} inodes). Free space "
                f"there before starting the run, or override the floor "
                f"with {_FLOOR_OVERRIDE_ENV} for a deliberately tight "
                f"environment."
            )
            raise EnvironmentPreflightError(msg)
        if free_bytes < warn_bytes or (
            free_inodes is not None
            and free_inodes < PREFLIGHT_WARN_FREE_INODES
        ):
            logger.warning(
                "environment preflight: %s has %.0f MiB free%s — above "
                "the refuse floor but low; a long run may pause or "
                "conclude early if it fills",
                path,
                free_bytes / (1024 * 1024),
                (
                    f" and {free_inodes} free inodes"
                    if free_inodes is not None else ""
                ),
            )


class EnvironmentGuard:
    """Run-scoped environment sentinel driven from the executor tick.

    ``tick()`` is called from the MAIN executor pass before every
    dispatch (both executor paths, including the serial glance-batch
    flush) on the dispatch loop's own thread, so blocking inside it
    IS the pause mechanism: no new main-pass task is dispatched while
    a tick is waiting, and in-flight reviews keep running on their
    worker threads and are harvested when the tick returns. The other
    dispatching passes (study consumer, deepen, error retry, trivial
    batches, edges) do not tick — they stop only once the guard has
    CONCLUDED, via ``_check_budget`` (see the module docstring).

    When a pause exceeds its bounded wait — or would run past the run
    deadline minus the drain margin — the guard *concludes*: it never
    kills anything itself — ``_check_budget`` reports the run as
    stopped (``terminated_by="environment"``), which drains the
    executor, writes the report, and marks the lifecycle interrupted
    with a resume hint, exactly like the wall-budget stop rails.
    """

    def __init__(
        self,
        *,
        tmp_dir: Path | None = None,
        out_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        statvfs_fn: Callable[[str], Any] | None = None,
        abort_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
        drain_margin_s: float | None = None,
    ) -> None:
        self._dirs: list[Path] = []
        for d in (tmp_dir if tmp_dir is not None else effective_tmp_dir(),
                  out_dir):
            if d is not None and d not in self._dirs:
                self._dirs.append(Path(d))
        self._clock = clock
        self._sleep = sleep_fn
        self._statvfs = (
            statvfs_fn if statvfs_fn is not None else _default_statvfs()
        )
        # Aborting the wait on operator shutdown: the SIGTERM/stop
        # rails own that conclusion — the guard must not out-wait them.
        self._abort_check = abort_check or (lambda: False)
        # Run deadline on the SAME monotonic clock as ``clock``: every
        # pause/probe wait is clamped to (deadline - now - drain
        # margin) so a pause entered near the wall cap concludes with
        # room to drain, report, and transition the lifecycle instead
        # of blocking through the supervisor's kill — and because the
        # clock keeps running while paused, cumulative pauses can
        # never push the run past its --max-time either.
        self._deadline = deadline_monotonic
        if drain_margin_s is None:
            from core.run.supervisor import DRAIN_MARGIN_S
            drain_margin_s = DRAIN_MARGIN_S
        self._drain_margin_s = drain_margin_s
        # Operator byte-floor override (RAPTOR_ENV_FLOOR_MIB):
        # pause = the override, resume = 2x (hysteresis preserved).
        override = _floor_override_bytes()
        self._pause_free_bytes = (
            override if override is not None else WATCHDOG_PAUSE_FREE_BYTES
        )
        self._resume_free_bytes = (
            override * 2 if override is not None
            else WATCHDOG_RESUME_FREE_BYTES
        )
        self._lock = threading.Lock()
        self._concluded_reason: str | None = None
        self._last_watchdog_check = float("-inf")
    # ── Conclusion state ─────────────────────────────────────────────

    @property
    def concluded(self) -> bool:
        with self._lock:
            return self._concluded_reason is not None

    @property
    def conclude_reason(self) -> str:
        with self._lock:
            return self._concluded_reason or ""

    def _conclude(self, reason: str) -> None:
        with self._lock:
            if self._concluded_reason is not None:
                return
            self._concluded_reason = reason
        logger.error(
            "environment guard: %s — concluding the run gracefully "
            "(in-flight reviews are harvested, unreviewed functions "
            "stay gaps; resume once the environment is fixed)",
            reason,
        )

    # ── Executor tick ────────────────────────────────────────────────

    def tick(self) -> None:
        """Pre-dispatch gate: pause on resource pressure, conclude
        when the pause exceeds its bounded wait. Cheap when healthy —
        at most one statvfs sweep per ``WATCHDOG_CHECK_INTERVAL_S``."""
        if self.concluded:
            return
        self._watchdog_gate()

    def _wait_budget(self, bounded: float) -> float | None:
        """*bounded* clamped to the room left before the run deadline
        (minus the drain margin), or ``None`` when no room remains —
        the caller concludes immediately instead of waiting into the
        supervisor's kill."""
        if self._deadline is None:
            return bounded
        room = self._deadline - self._clock() - self._drain_margin_s
        if room <= 0:
            return None
        return min(bounded, room)

    def _resume_floors(self, d: Path) -> tuple[int, int]:
        """(byte, inode) resume floors effective for *d*'s filesystem.

        The configured resume floor (2x pause, hysteresis) is clamped
        to half the filesystem's TOTAL capacity: on a filesystem
        smaller than the floor, recovery would otherwise be
        structurally unreachable — a guaranteed full pause + conclude
        even after genuine cleanup. The pause floor is clamped in
        step (``_pause_floors``, a quarter of total) so the 2x
        hysteresis gap survives the clamp: clamping only the resume
        floor would drop it BELOW the unclamped pause floor on small
        filesystems, inverting the hysteresis into a pause/recover
        flap on every measuring tick. Trade-off, both directions: a
        larger fraction re-creates the unreachable-floor failure on
        small filesystems; a smaller one resumes with so little
        headroom that the very next dispatch drops below the pause
        floor again (flap the hysteresis exists to prevent).
        """
        byte_floor = self._resume_free_bytes
        inode_floor = WATCHDOG_RESUME_FREE_INODES
        if self._statvfs is not None:
            measured = _free_space(self._statvfs, d)
            if measured is not None:
                _fb, _fi, total_bytes, total_inodes = measured
                byte_floor = min(byte_floor, total_bytes // 2)
                if total_inodes is not None:
                    inode_floor = min(inode_floor, total_inodes // 2)
        return byte_floor, inode_floor

    # ── Resource watchdog ────────────────────────────────────────────

    def _pause_floors(self, d: Path) -> tuple[int, int]:
        """(byte, inode) pause floors effective for *d*'s filesystem,
        clamped to a quarter of TOTAL capacity — half the resume
        floor's clamp, preserving the 2x hysteresis ratio on
        filesystems small enough for the clamps to bind (see
        ``_resume_floors`` for the inversion this prevents and the
        fraction trade-off)."""
        byte_floor = self._pause_free_bytes
        inode_floor = WATCHDOG_PAUSE_FREE_INODES
        if self._statvfs is not None:
            measured = _free_space(self._statvfs, d)
            if measured is not None:
                _fb, _fi, total_bytes, total_inodes = measured
                byte_floor = min(byte_floor, total_bytes // 4)
                if total_inodes is not None:
                    inode_floor = min(inode_floor, total_inodes // 4)
        return byte_floor, inode_floor

    def _pressure(self, *, resume: bool) -> str | None:
        """Description of the worst floor breach, or ``None`` when all
        measured dirs are at/above the (pause or resume) floors."""
        if self._statvfs is None:
            return None
        for d in self._dirs:
            measured = _free_space(self._statvfs, d)
            if measured is None:
                continue
            free_bytes, free_inodes, _tb, _ti = measured
            if resume:
                byte_floor, inode_floor = self._resume_floors(d)
            else:
                byte_floor, inode_floor = self._pause_floors(d)
            if free_bytes < byte_floor:
                return (
                    f"{d}: {free_bytes / (1024 * 1024):.0f} MiB free "
                    f"(< {byte_floor // (1024 * 1024)} MiB)"
                )
            if free_inodes is not None and free_inodes < inode_floor:
                return f"{d}: {free_inodes} free inodes (< {inode_floor})"
        return None

    def _watchdog_gate(self) -> None:
        if self._statvfs is None:
            return
        now = self._clock()
        if now - self._last_watchdog_check < WATCHDOG_CHECK_INTERVAL_S:
            return
        self._last_watchdog_check = now
        pressure = self._pressure(resume=False)
        if pressure is None:
            return
        max_pause = self._wait_budget(WATCHDOG_MAX_PAUSE_S)
        if max_pause is None:
            self._conclude(
                f"resource pressure with no wall-budget room left to "
                f"pause (run deadline reached) ({pressure})",
            )
            return
        logger.warning(
            "resource watchdog: %s — pausing dispatch (in-flight "
            "reviews finish; resumes at 2x the pause floor, concludes "
            "after %.0fs)",
            pressure, max_pause,
        )
        start = self._clock()
        while True:
            if self._abort_check():
                return
            if self._clock() - start >= max_pause:
                self._conclude(
                    f"resource pressure did not clear within "
                    f"{self._clock() - start:.0f}s of paused dispatch"
                    + ("" if max_pause >= WATCHDOG_MAX_PAUSE_S
                       else " (clamped to the run deadline)")
                    + f" ({pressure})",
                )
                return
            self._sleep(WATCHDOG_POLL_S)
            still = self._pressure(resume=True)
            if still is None:
                logger.warning(
                    "resource watchdog: resources recovered — "
                    "resuming dispatch",
                )
                self._last_watchdog_check = self._clock()
                return
            pressure = still
