"""Pool teardown must actually kill process-pool workers.

The trap pinned here: ``ProcessPoolExecutor.shutdown()`` clears its
``_processes`` map before returning, so a teardown that snapshots the
map AFTER shutdown terminates nothing — the wedged worker leaks,
holding the process's inherited stdout/stderr and blocking the
executor manager thread's join at interpreter exit (the process then
never exits and its output pipe never closes). The snapshot must be
taken first, and SIGTERM must escalate to SIGKILL: fork-context
workers inherit the parent's SIGTERM handler, and a handler that
blocks makes the worker survive ``terminate()`` indefinitely.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import FrameType

import pytest

from core.inventory.builder import (
    _init_inventory_worker,
    _shutdown_pool_nowait,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="fork start method required (deterministic handler "
           "inheritance)",
)


def _wedge() -> None:
    time.sleep(600)


def _blocking_handler(signum: int, frame: FrameType | None) -> None:
    # Deterministic stand-in for the fork-frozen-lock class: an
    # inherited handler that never returns, so SIGTERM never kills.
    time.sleep(600)


def _install_blocking_sigterm() -> None:
    signal.signal(signal.SIGTERM, _blocking_handler)


def _wait_dead(procs: list, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(p.is_alive() for p in procs):
            return True
        time.sleep(0.05)
    return False


def test_teardown_kills_sigterm_immune_worker_via_sigkill() -> None:
    """A worker whose inherited SIGTERM handler blocks must still die:
    the teardown escalates to SIGKILL after its grace window."""
    ctx = multiprocessing.get_context("fork")
    pool = ProcessPoolExecutor(
        max_workers=1, mp_context=ctx,
        initializer=_install_blocking_sigterm,
    )
    procs: list = []
    try:
        pool.submit(_wedge)
        time.sleep(0.5)  # let the worker start the task
        procs = list(pool._processes.values())
        assert procs, "worker never spawned"
        t0 = time.monotonic()
        _shutdown_pool_nowait(pool, kill_grace_s=1.0)
        assert time.monotonic() - t0 < 15.0
        assert _wait_dead(procs, 10.0), (
            "SIGTERM-immune worker survived teardown"
        )
        assert procs[0].exitcode == -signal.SIGKILL
    finally:
        for p in procs:  # leak-guard for assertion failures above
            if p.is_alive():
                p.kill()


def test_teardown_lets_responsive_worker_die_on_sigterm() -> None:
    """A worker with the default disposition dies to terminate() —
    no gratuitous SIGKILL inside the grace window."""
    ctx = multiprocessing.get_context("fork")
    pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    procs: list = []
    try:
        pool.submit(_wedge)
        time.sleep(0.5)
        procs = list(pool._processes.values())
        assert procs, "worker never spawned"
        _shutdown_pool_nowait(pool, kill_grace_s=10.0)
        assert _wait_dead(procs, 10.0)
        assert procs[0].exitcode == -signal.SIGTERM
    finally:
        for p in procs:  # leak-guard for assertion failures above
            if p.is_alive():
                p.kill()


def test_teardown_snapshot_precedes_shutdown() -> None:
    """The regression itself: shutdown() nulls ``_processes``, so a
    post-shutdown snapshot sees nothing to kill and a plainly wedged
    worker leaks. The helper must reap it regardless."""
    ctx = multiprocessing.get_context("fork")
    pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    procs: list = []
    try:
        pool.submit(_wedge)
        time.sleep(0.5)
        procs = list(pool._processes.values())
        assert procs
        _shutdown_pool_nowait(pool, kill_grace_s=5.0)
        # shutdown() has dropped the executor's own reference…
        assert getattr(pool, "_processes", None) in (None, {})
        # …but the worker still died, because the snapshot came first.
        assert _wait_dead(procs, 10.0), "wedged worker leaked"
    finally:
        for p in procs:  # leak-guard for assertion failures above
            if p.is_alive():
                p.kill()


def test_teardown_no_ops_on_thread_pool() -> None:
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(lambda: None).result(timeout=30)
    _shutdown_pool_nowait(pool)  # must not raise


def _report_sigterm_is_default() -> bool:
    return signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


_INIT_ARGS = (Path("."), [], True, {}, False, None, None, None)


def test_init_inventory_worker_resets_sigterm() -> None:
    """The worker initializer must restore the default SIGTERM
    disposition — inherited parent handlers must never run inside an
    extractor worker."""
    prior = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, _blocking_handler)
        _init_inventory_worker(*_INIT_ARGS)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, prior)


def test_fork_worker_sheds_inherited_sigterm_handler() -> None:
    """End to end through a real fork pool: a parent-installed
    SIGTERM handler must not survive into the worker."""
    prior = signal.getsignal(signal.SIGTERM)
    pool = None
    try:
        signal.signal(signal.SIGTERM, _blocking_handler)
        ctx = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=1, mp_context=ctx,
            initializer=_init_inventory_worker, initargs=_INIT_ARGS,
        )
        assert pool.submit(_report_sigterm_is_default).result(
            timeout=60,
        ) is True
    finally:
        signal.signal(signal.SIGTERM, prior)
        if pool is not None:
            # The bounded teardown, not shutdown(wait=True): a fork
            # worker can wedge on a fork-frozen lock even AFTER its
            # task completed (pytest runs threads too), and an
            # unbounded join then hangs the whole test session.
            _shutdown_pool_nowait(pool)
