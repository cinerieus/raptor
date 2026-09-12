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
* **Systemic-fault circuit breaker** (same guard) — correlates
  terminal dispatch failures across DISTINCT functions by systemic
  class (disk/fd/memory errnos anywhere in the exception chain,
  auth-layer refusal, connection/DNS on the ``__cause__`` chain).
  When enough of the recent failures share one class the breaker
  trips: dispatch stops, a direct probe for that class runs with
  bounded backoff, and the run either resumes (probe recovered —
  window reset) or concludes through the same stop rails as above.

Scope: the guard's pause/probe machinery is driven ONLY by the main
executor pass's pre-dispatch tick (both executor paths, including the
serial glance-batch flush). The study consumer, deepen, error-retry,
trivial-batch, and edge passes do not tick — they observe the guard
solely through ``_check_budget``, so they STOP dispatching once a
conclusion is reached but are never paused mid-pass; in particular
the study consumer keeps dispatching during a main-pass pause,
bounded by that pause's own conclusion. Wiring ticks into those
passes is deliberately out of scope here.

Every bounded pause/probe wait is additionally clamped to the run
deadline (when one is set) minus a drain margin, so a pause entered
near the wall cap concludes with enough room to harvest, report, and
transition the lifecycle instead of blocking into the supervisor's
kill.

Platforms without ``os.statvfs`` degrade silently to no-ops.
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.audit.orchestrator import OrchestratorConfig

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


# ── Circuit-breaker window ────────────────────────────────────────────
# Trade-off, both directions: a looser correlation (smaller trip count
# or larger window) trips on coincidental clusters — a burst of
# unrelated per-function errors would pause a healthy run; a stricter
# one keeps paying for doomed dispatches while a genuinely broken
# environment fails function after function. 5-of-6 across DISTINCT
# functions is conservative: one function retried to death can never
# trip it, and two interleaved unrelated failures keep it open.
BREAKER_WINDOW = 6
BREAKER_TRIP_COUNT = 5
# Probe backoff. Starting smaller re-probes a transient fault (a
# co-tenant's spike freed the disk) quickly; capping the growth keeps
# a slow-clearing fault from being probed so rarely the bounded wait
# expires between probes. The overall bound mirrors the watchdog's:
# past it, nothing is going to clear without operator action, so the
# run concludes resumably instead of holding the pause forever.
BREAKER_PROBE_BACKOFF_INITIAL_S = 5.0
BREAKER_PROBE_BACKOFF_MAX_S = 60.0
BREAKER_MAX_PROBE_WAIT_S = 600.0
# Memory-probe allocation size. Larger proves more headroom before
# resuming (a resumed run immediately builds multi-MiB prompts);
# smaller keeps the probe itself from destabilising a barely-
# recovered host.
_MEMORY_PROBE_BYTES = 16 * 1024 * 1024

# Systemic failure classes.
CLASS_DISK = "disk"
CLASS_FDS = "fds"
CLASS_MEMORY = "memory"
CLASS_AUTH = "auth"
CLASS_NETWORK = "network"

# Classes with no cheap in-process recovery probe: proving an auth
# credential or a provider endpoint healthy again requires a real
# client call (spends budget, embeds provider specifics), so a trip
# on these is treated as non-recoverable — the run concludes
# resumably and the operator fixes the credential/network first.
_NON_RECOVERABLE_CLASSES = frozenset({CLASS_AUTH, CLASS_NETWORK})

_DISK_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS})
_FD_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE})
_MEMORY_ERRNOS = frozenset({errno.ENOMEM})
_NETWORK_ERRNOS = frozenset({
    errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ENETUNREACH,
})


def _cause_only_chain(exc: BaseException):
    """Yield *exc* and its explicit ``__cause__`` chain only (bounded,
    cycle-safe). Unlike ``client._exception_chain`` this never follows
    implicit ``__context__``: a network errno that merely happened to
    be in flight while another failure was being handled is ambient,
    not causal."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < 8:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__


def classify_systemic(exc: BaseException) -> str | None:
    """The systemic failure class of *exc*, or ``None`` for ordinary
    per-function failures.

    Reuses the LLM client's error taxonomy rather than growing a
    parallel one: the causal-chain walk is ``client._exception_chain``
    (the same walk its content-filter/auth classifiers use) and the
    credential class is ``client.is_auth_refusal`` (structural
    signals first; response-shape failures can never impersonate
    auth). Budget exhaustion, model refusals and schema-validation
    failures classify as ``None`` — they have their own handling and
    must never trip the breaker.

    Disk/fd/memory errnos count anywhere in the chain (an error
    raised while HANDLING an ENOSPC is still an ENOSPC environment).
    The network class counts only on the explicit ``__cause__`` chain:
    a ``TimeoutError`` raised while an ECONNREFUSED sat in implicit
    ``__context__`` is a timeout, not a network fault — classifying it
    network would strip the timeout's own recovery handling.
    """
    from core.llm.client import (
        _exception_chain,
        is_auth_refusal,
        is_budget_exceeded_error,
    )

    if is_budget_exceeded_error(exc):
        return None
    for e in _exception_chain(exc):
        if isinstance(e, OSError):
            eno = e.errno
            if eno in _DISK_ERRNOS:
                return CLASS_DISK
            if eno in _FD_ERRNOS:
                return CLASS_FDS
            if eno in _MEMORY_ERRNOS:
                return CLASS_MEMORY
    for e in _cause_only_chain(exc):
        if isinstance(e, socket.gaierror):
            # DNS resolution failures carry EAI_* codes, not errnos.
            return CLASS_NETWORK
        if isinstance(e, OSError) and e.errno in _NETWORK_ERRNOS:
            return CLASS_NETWORK
    if isinstance(exc, Exception) and is_auth_refusal(exc):
        return CLASS_AUTH
    return None


def _fault_dir_hint(
    exc: BaseException, allowed_dirs: list[Path],
) -> Path | None:
    """The directory implicated by a disk-class failure, or ``None``.

    The filename is taken ONLY from the disk-errno OSError itself —
    never from arbitrary chain members, where an unrelated
    target-tree path (e.g. a FileNotFoundError joining via implicit
    ``__context__`` during ENOSPC handling) would redirect the probe.
    The hint is realpath-resolved and accepted only when it lies
    under one of *allowed_dirs* (the guard's own directories): the
    filename can be attacker-influenced (a path inside the scanned
    target, a symlink planted there, a relative name resolving into
    the harness CWD), and the probe WRITES a canary in the hinted
    directory — a probe write must never land outside guard-owned
    space. A rejected hint on the same filesystem as a guard dir
    loses nothing (the guard-dir probe covers that filesystem); a
    rejected hint on a foreign filesystem is exactly the untrusted
    case.
    """
    from core.llm.client import _exception_chain

    for e in _exception_chain(exc):
        if not isinstance(e, OSError) or e.errno not in _DISK_ERRNOS:
            continue
        filename = getattr(e, "filename", None)
        if not filename:
            return None
        try:
            parent = Path(os.path.realpath(str(filename))).parent
        except (OSError, ValueError):
            return None
        if not parent.is_dir():
            return None
        for d in allowed_dirs:
            try:
                base = Path(os.path.realpath(str(d)))
            except (OSError, ValueError):
                continue
            if parent == base or base in parent.parents:
                return parent
        return None
    return None


def note_dispatch_failure(
    config: OrchestratorConfig, key: str, exc: BaseException,
) -> None:
    """Feed a terminal dispatch failure to the run's guard (no-op when
    the run carries none). The single hook every error-outcome writer
    calls so the breaker sees the same failures the journal does."""
    guard = getattr(config, "environment_guard_state", None)
    if guard is not None:
        guard.note_dispatch_failure(key, exc)


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
        breaker_enabled: bool = True,
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
        # Circuit-breaker state. The window holds (function key,
        # systemic class-or-None) for the last BREAKER_WINDOW terminal
        # dispatch failures; heterogeneous entries dilute it by
        # construction, so mixed failure causes never trip.
        self._breaker_enabled = breaker_enabled
        self._window: deque[tuple[str, str | None]] = deque(
            maxlen=BREAKER_WINDOW,
        )
        self._tripped_class: str | None = None
        self._trip_hint_dir: Path | None = None

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
        """Pre-dispatch gate: probe a tripped breaker, pause on
        resource pressure, conclude when either exceeds its bounded
        wait. Cheap when healthy — no probe, at most one statvfs
        sweep per ``WATCHDOG_CHECK_INTERVAL_S``."""
        if self.concluded:
            return
        self._breaker_gate()
        if self.concluded:
            return
        self._watchdog_gate()

    # ── Systemic-fault circuit breaker ───────────────────────────────

    def note_dispatch_failure(self, key: str, exc: BaseException) -> None:
        """Record a terminal dispatch failure for *key* (a
        ``file:function`` identity). Called from review worker
        threads; trips the breaker when ``BREAKER_TRIP_COUNT`` of the
        last ``BREAKER_WINDOW`` failures across DISTINCT functions
        share one systemic class. The trip itself only sets state —
        the pause/probe runs on the next dispatch tick."""
        if not self._breaker_enabled:
            return
        cls = classify_systemic(exc)
        with self._lock:
            if self._concluded_reason is not None:
                return
            self._window.append((key, cls))
            if cls is None or self._tripped_class is not None:
                return
            matching_keys = {
                k for k, c in self._window if c == cls
            }
            if len(matching_keys) < BREAKER_TRIP_COUNT:
                return
            self._tripped_class = cls
            self._trip_hint_dir = (
                _fault_dir_hint(exc, self._dirs)
                if cls == CLASS_DISK else None
            )
        logger.warning(
            "circuit breaker: %d of the last %d dispatch failures "
            "across distinct functions share systemic class %r — "
            "pausing dispatch to probe the environment",
            len(matching_keys), BREAKER_WINDOW, cls,
        )

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

    def _breaker_gate(self) -> None:
        with self._lock:
            cls = self._tripped_class
            hint = self._trip_hint_dir
        if cls is None:
            return
        if cls in _NON_RECOVERABLE_CLASSES:
            self._conclude(
                f"systemic {cls} failures on the dispatch path (no "
                f"cheap recovery probe exists for this class)",
            )
            return
        max_wait = self._wait_budget(BREAKER_MAX_PROBE_WAIT_S)
        if max_wait is None:
            self._conclude(
                f"systemic {cls} failures with no wall-budget room "
                f"left to probe (run deadline reached)",
            )
            return
        start = self._clock()
        delay = BREAKER_PROBE_BACKOFF_INITIAL_S
        while True:
            if self._probe(cls, hint):
                with self._lock:
                    self._tripped_class = None
                    self._trip_hint_dir = None
                    self._window.clear()
                logger.warning(
                    "circuit breaker: %s probe succeeded — resuming "
                    "dispatch (failure window reset)", cls,
                )
                return
            if self._abort_check():
                return
            if self._clock() - start >= max_wait:
                self._conclude(
                    f"systemic {cls} failures persisted through "
                    f"{self._clock() - start:.0f}s of probing"
                    + ("" if max_wait >= BREAKER_MAX_PROBE_WAIT_S
                       else " (clamped to the run deadline)"),
                )
                return
            self._sleep(delay)
            delay = min(delay * 2, BREAKER_PROBE_BACKOFF_MAX_S)

    def _probe(self, cls: str, hint: Path | None) -> bool:
        """Direct, cheap health probe for a recoverable class. Never
        an LLM call; never raises."""
        try:
            if cls == CLASS_DISK:
                return self._disk_probe(hint)
            if cls == CLASS_FDS:
                fd = os.open(os.devnull, os.O_RDONLY)
                os.close(fd)
                return True
            if cls == CLASS_MEMORY:
                buf = bytearray(_MEMORY_PROBE_BYTES)
                del buf
                return True
        except (OSError, MemoryError):
            return False
        return False

    def _disk_probe(self, hint: Path | None) -> bool:
        """Disk recovery = the guard's OWN dirs (tmp, run out) are
        healthy. The (containment-checked) hint dir is probed too but
        is advisory only: a hint that stays unwritable while tmp and
        out are healthy is a read-only-by-design location (an
        RO-mounted checkout hitting EROFS, a swept-away staging
        subdir) — gating recovery on it would hold the pause to its
        full bound and conclude a run whose environment is fine."""
        for d in self._dirs:
            try:
                if not self._disk_ok(d):
                    return False
            except OSError:
                return False
        if hint is not None and hint not in self._dirs:
            try:
                hint_ok = self._disk_ok(hint)
            except OSError:
                hint_ok = False
            if not hint_ok:
                logger.warning(
                    "circuit breaker: implicated dir %s is still "
                    "unwritable but %s are healthy — treating the "
                    "environment as recovered (read-only-by-design "
                    "location)", hint,
                    ", ".join(str(d) for d in self._dirs),
                )
        return True

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

    def _disk_ok(self, d: Path) -> bool:
        """Resume-floor headroom AND a canary write in *d*: statvfs
        alone misses quota exhaustion (EDQUOT) and a filesystem
        remounted read-only (EROFS) — only an actual write proves the
        failing operation works again. Raises OSError when the canary
        cannot be written (caller decides whether that dir gates
        recovery)."""
        if self._statvfs is not None:
            measured = _free_space(self._statvfs, d)
            if measured is not None:
                free_bytes, free_inodes, _tb, _ti = measured
                byte_floor, inode_floor = self._resume_floors(d)
                if free_bytes < byte_floor:
                    return False
                if free_inodes is not None and free_inodes < inode_floor:
                    return False
        fd, path = tempfile.mkstemp(prefix=".env-probe-", dir=str(d))
        try:
            os.write(fd, b"\0" * 4096)
        finally:
            os.close(fd)
            os.unlink(path)
        return True

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
