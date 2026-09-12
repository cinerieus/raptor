"""RLIMIT_NPROC on the macOS seatbelt lane: relative ceiling, not an
absolute cap.

macOS counts RLIMIT_NPROC against the user's TOTAL simultaneous
processes, so an absolute cap equal to the configured budget sits
BELOW current usage on any busy host and makes every fork inside the
sandbox fail EAGAIN (and admits budget-minus-count forks on a quiet
one). The spawn layer now mirrors the Linux no-namespace lane:
ceiling = current same-UID process count + budget, clamped to the
hard limit; when ps cannot produce a count the cap is skipped with a
warning (Linux /proc-unreadable parity).

One more darwin quirk, confirmed on a real macOS runner: setrlimit
silently stores at most kern.maxprocperuid for RLIMIT_NPROC whatever
value is requested (even under an infinite hard limit), so the spawn
layer requests min(count + budget, kernel clamp) and these tests
compute their expectations with the same clamp — via the real sysctl
helper on macOS, and via a monkeypatched clamp for the emulated
darwin-shape cases on Linux (where the helper is platform-gated to
None).

Cross-platform: SANDBOX_EXEC is swapped for a pass-through script so
the REAL shim + trampoline apply the preexec on Linux too; the count
is monkeypatched to a deterministic value well above any plausible
live usage so lowering the soft limit never breaks the test's own
process tree.
"""

from __future__ import annotations

import logging
import resource
import sys

import pytest

from core.sandbox import _macos_spawn

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only",
)

_SOFT_DUMP = [sys.executable, "-c",
              "import resource; "
              "print(resource.getrlimit(resource.RLIMIT_NPROC)[0])"]


def _fake_sandbox_exec(tmp_path):
    fake = tmp_path / "fake-sandbox-exec"
    fake.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')  # drop -p <profile> --
    fake.chmod(0o755)
    return str(fake)


def test_same_uid_process_count_works_here():
    """ps -xo pid= -U <uid> is portable across the BSD and procps ps
    families — the helper must produce a plausible count on this
    host (at minimum, this very process)."""
    count = _macos_spawn._same_uid_process_count()
    assert isinstance(count, int) and count >= 1


def _expected_soft(requested: int, kernel_clamp: "int | None") -> int:
    """The soft limit the child must observe: count+budget, bounded by
    the inherited hard limit and — on darwin — the kernel's silent
    kern.maxprocperuid clamp (which applies whether or not we ask)."""
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    expected = (requested if hard == resource.RLIM_INFINITY
                else min(requested, hard))
    if kernel_clamp is not None:
        expected = min(expected, kernel_clamp)
    return expected


def test_ceiling_is_count_plus_budget_clamped(tmp_path, monkeypatch):
    fake_count = 100000  # far above live usage: the ceiling can only RAISE
    budget = 64
    monkeypatch.setattr(_macos_spawn, "_same_uid_process_count",
                        lambda: fake_count)
    monkeypatch.setattr(_macos_spawn, "SANDBOX_EXEC",
                        _fake_sandbox_exec(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    r = _macos_spawn.run_sandboxed(
        list(_SOFT_DUMP), output=str(out),
        nproc_limit=budget,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0, r.stderr
    soft = int(r.stdout.strip())
    # The REAL kernel-clamp helper: None on Linux (platform-gated),
    # the live kern.maxprocperuid on a macOS runner — the same value
    # the production path folds into its request.
    expected = _expected_soft(fake_count + budget,
                              _macos_spawn._darwin_nproc_kernel_clamp())
    assert soft == expected, (
        f"child soft NPROC {soft} != count+budget ceiling {expected} "
        f"(absolute-cap regression?)")


def test_kernel_clamp_bounds_the_request(tmp_path, monkeypatch):
    """Darwin shape emulated on any host: a kernel clamp BELOW
    count+budget wins — the spawn layer must request the honoured
    value, so the child reads exactly the clamp."""
    fake_count = 100000
    budget = 64
    fake_clamp = 1333
    monkeypatch.setattr(_macos_spawn, "_same_uid_process_count",
                        lambda: fake_count)
    monkeypatch.setattr(_macos_spawn, "_darwin_nproc_kernel_clamp",
                        lambda: fake_clamp)
    monkeypatch.setattr(_macos_spawn, "SANDBOX_EXEC",
                        _fake_sandbox_exec(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    r = _macos_spawn.run_sandboxed(
        list(_SOFT_DUMP), output=str(out),
        nproc_limit=budget,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == _expected_soft(fake_count + budget,
                                                   fake_clamp)


def test_clamp_above_ceiling_keeps_relative_arithmetic(tmp_path,
                                                       monkeypatch):
    """Anti-degeneration guard: with the kernel clamp ABOVE
    count+budget, the child must still observe the RELATIVE ceiling —
    the clamp must never swallow the count+budget assertion (an
    absolute-cap regression stays red under a sub-clamp count)."""
    fake_count = 100000
    budget = 64
    fake_clamp = 200000  # above the ceiling: must not matter
    monkeypatch.setattr(_macos_spawn, "_same_uid_process_count",
                        lambda: fake_count)
    monkeypatch.setattr(_macos_spawn, "_darwin_nproc_kernel_clamp",
                        lambda: fake_clamp)
    monkeypatch.setattr(_macos_spawn, "SANDBOX_EXEC",
                        _fake_sandbox_exec(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    r = _macos_spawn.run_sandboxed(
        list(_SOFT_DUMP), output=str(out),
        nproc_limit=budget,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == _expected_soft(fake_count + budget,
                                                   fake_clamp)


def test_kernel_clamp_helper_platform_gated():
    """On non-darwin hosts the helper must return None without
    touching sysctl (Linux sysctl exists with different semantics)."""
    if sys.platform == "darwin":
        clamp = _macos_spawn._darwin_nproc_kernel_clamp()
        assert clamp is None or clamp > 0
    else:
        assert _macos_spawn._darwin_nproc_kernel_clamp() is None


def test_count_unavailable_skips_cap_with_warning(tmp_path, monkeypatch,
                                                  caplog):
    """No count → no cap (never a blind absolute one), and the skip is
    named to the operator."""
    monkeypatch.setattr(_macos_spawn, "_same_uid_process_count",
                        lambda: None)
    monkeypatch.setattr(_macos_spawn, "SANDBOX_EXEC",
                        _fake_sandbox_exec(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    inherited_soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    with caplog.at_level(logging.WARNING, logger="core.sandbox._macos_spawn"):
        r = _macos_spawn.run_sandboxed(
            list(_SOFT_DUMP), output=str(out),
            nproc_limit=64,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=20,
        )
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == inherited_soft
    assert any("skipping the RLIMIT_NPROC" in rec.message
               for rec in caplog.records)


def test_no_budget_means_no_count_and_no_cap(tmp_path, monkeypatch):
    """nproc_limit unset/zero: the counting helper must not even run
    (no ps cost on the default path) and the limit stays inherited."""
    def boom():
        raise AssertionError("counted with no nproc budget")

    monkeypatch.setattr(_macos_spawn, "_same_uid_process_count", boom)
    monkeypatch.setattr(_macos_spawn, "SANDBOX_EXEC",
                        _fake_sandbox_exec(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    inherited_soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    r = _macos_spawn.run_sandboxed(
        list(_SOFT_DUMP), output=str(out),
        nproc_limit=0,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == inherited_soft
