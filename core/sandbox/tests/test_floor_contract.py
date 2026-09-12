"""The containment-floor contract: lattice, entry check, dispatch
assertions.

Four load-bearing properties, each pinned here:

1. The tier lattice is a total order per platform, and floors resolve
   per caller class exactly as the per-lane gates they subsumed
   demanded (consent-chain matrix).
2. Every dispatch site carries a declared tier and a hard pre-exec
   floor check (source tripwire), so a FUTURE lane wired into the
   demotion ladder without a thought for the contract fails closed
   (the fake-lane simulation) — under the old per-lane-gate
   architecture it silently executed.
3. The assertion is a runtime raise, not a debug artifact: python -O
   cannot strip it, no flag skips it, and the error type inherits the
   BaseException fail-loud semantics.
4. The mount and mountless spawn lanes stamp the same posture surface,
   differing exactly as declared (parity contract).
"""

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from core.sandbox import tiers as _tiers
from core.sandbox.errors import SandboxFloorError, SandboxSetupError
from core.sandbox.tiers import ContainmentTier

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------- unit tier

def test_lattice_is_total_and_strictly_ordered_per_platform():
    linux = [ContainmentTier.BARE, ContainmentTier.LANDLOCK_ONLY,
             ContainmentTier.NS_NOMOUNT, ContainmentTier.MOUNTLESS_NS,
             ContainmentTier.MOUNT_NS]
    assert linux == sorted(linux)
    assert len({int(t) for t in linux}) == len(linux)
    # macOS values live far above the Linux band so an accidental
    # cross-platform compare is loudly wrong rather than subtly wrong.
    assert ContainmentTier.SEATBELT > ContainmentTier.MOUNT_NS


def test_tier_labels_round_trip():
    for tier in ContainmentTier:
        assert _tiers.label_tier(_tiers.tier_label(tier)) is tier
    with pytest.raises(KeyError):
        _tiers.label_tier("no-such-tier")


def test_floor_class_defaults_per_platform(monkeypatch):
    assert _tiers.untrusted_default_floor() in (
        ContainmentTier.MOUNT_NS, ContainmentTier.SEATBELT)
    monkeypatch.setattr(_tiers, "sys",
                        types.SimpleNamespace(platform="darwin"))
    assert _tiers.untrusted_default_floor() is ContainmentTier.SEATBELT
    assert _tiers.waived_untrusted_floor() is ContainmentTier.BARE
    monkeypatch.setattr(_tiers, "sys",
                        types.SimpleNamespace(platform="linux"))
    assert _tiers.untrusted_default_floor() is ContainmentTier.MOUNT_NS
    assert _tiers.waived_untrusted_floor() is (
        ContainmentTier.LANDLOCK_ONLY)


def test_consent_chain_matrix(monkeypatch):
    """Every (disable × contract-flag × workload) cell resolves to
    exactly the floor the subsumed gates demanded: operator disable is
    globally authoritative (BARE), the resolved contract gets the
    platform's untrusted floor, waived untrusted work gets the frozen
    env-var mapping (LANDLOCK_ONLY on Linux, BARE on macOS — the
    documented arm-3 semantics), and plain trusted calls get BARE with
    the default source. Nothing consents untrusted work to BARE on
    Linux."""
    monkeypatch.setattr(_tiers, "sys",
                        types.SimpleNamespace(platform="linux"))
    cases = {
        # (disabled, require_fresh_procfs, untrusted): (floor, source)
        (True, True, True): (ContainmentTier.BARE, "operator-disable"),
        (True, None, False): (ContainmentTier.BARE, "operator-disable"),
        (False, True, True): (ContainmentTier.MOUNT_NS, "default"),
        (False, True, False): (ContainmentTier.MOUNT_NS, "default"),
        (False, False, True): (ContainmentTier.LANDLOCK_ONLY, "env"),
        (False, False, False): (ContainmentTier.LANDLOCK_ONLY, "env"),
        (False, None, True): (ContainmentTier.LANDLOCK_ONLY, "env"),
        (False, None, False): (ContainmentTier.BARE, "default"),
    }
    for (disabled, rfp, untrusted), expected in cases.items():
        got = _tiers.resolve_call_floor(
            operator_disabled=disabled, require_fresh_procfs=rfp,
            untrusted_workload=untrusted)
        assert got == expected, (disabled, rfp, untrusted, got)
    monkeypatch.setattr(_tiers, "sys",
                        types.SimpleNamespace(platform="darwin"))
    assert _tiers.resolve_call_floor(
        operator_disabled=False, require_fresh_procfs=True,
        untrusted_workload=True) == (ContainmentTier.SEATBELT, "default")
    assert _tiers.resolve_call_floor(
        operator_disabled=False, require_fresh_procfs=False,
        untrusted_workload=True) == (ContainmentTier.BARE, "env")


def test_assert_floor_semantics():
    """For every (delivered, floor) pair exactly one of {returns,
    SandboxFloorError} by ``delivered >= floor`` — never a run below
    floor, never a refusal at/above it."""
    linux = [ContainmentTier.BARE, ContainmentTier.LANDLOCK_ONLY,
             ContainmentTier.NS_NOMOUNT, ContainmentTier.MOUNTLESS_NS,
             ContainmentTier.MOUNT_NS]
    for delivered in linux:
        for floor in linux:
            if delivered >= floor:
                _tiers.assert_floor(delivered, floor, lane="t")
            else:
                with pytest.raises(SandboxFloorError) as excinfo:
                    _tiers.assert_floor(delivered, floor, lane="t")
                assert excinfo.value.achievable is delivered
                assert excinfo.value.floor is floor


def test_assert_floor_chains_cause_and_lifts_category():
    cause = SandboxSetupError("backend died", setup_category="U")
    with pytest.raises(SandboxFloorError) as excinfo:
        _tiers.assert_floor(
            ContainmentTier.LANDLOCK_ONLY, ContainmentTier.MOUNT_NS,
            lane="fallback", cause=cause, detail="why-text",
            remedy="set THE_OVERRIDE")
    e = excinfo.value
    assert e.__cause__ is cause
    assert e.setup_category == "U"     # lifted from the cause
    assert "why-text" in str(e)
    assert "set THE_OVERRIDE" in str(e)
    assert isinstance(e, SandboxSetupError)   # subtype, existing catches
    assert isinstance(e, BaseException) and not isinstance(e, Exception)


def test_assert_floor_survives_python_O():
    """The floor assertion is a plain runtime raise — optimized
    bytecode (-O, which strips ``assert`` statements and
    ``__debug__`` blocks) cannot remove it."""
    code = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, os.environ["RAPTOR_DIR"])
        from core.sandbox.tiers import ContainmentTier, assert_floor
        from core.sandbox.errors import SandboxFloorError
        try:
            assert_floor(ContainmentTier.LANDLOCK_ONLY,
                         ContainmentTier.MOUNT_NS, lane="probe")
        except SandboxFloorError:
            print("floor-held-under-O")
    """)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "RAPTOR_DIR": str(_REPO_ROOT),
    }
    r = subprocess.run([sys.executable, "-O", "-c", code], env=env,
                       capture_output=True, text=True, timeout=60,
                       check=False)
    assert r.returncode == 0, r.stderr
    assert "floor-held-under-O" in r.stdout
