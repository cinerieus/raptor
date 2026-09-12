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
        # Untrusted-marked call that never derived the contract kwarg:
        # fail CLOSED at the class default — the waived floor (and its
        # "env" attribution) belongs only to callers that carried the
        # env-var-honouring derivation through; the resolver must not
        # attribute consent nobody verified.
        (False, None, True): (ContainmentTier.MOUNT_NS, "default"),
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


def test_rfp_kwarg_falsy_literals_normalise_to_false(
        tmp_path, monkeypatch):
    """A literal falsy require_fresh_procfs (0, '') must resolve like
    the derived False (waived untrusted class), not like an absent
    kwarg — the tri-state boundary normalises before resolution."""
    import subprocess as _subprocess

    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx

    def ok_spawn(cmd, **kwargs):
        return _subprocess.CompletedProcess(cmd, returncode=0,
                                            stdout="", stderr="")

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", ok_spawn)
    try:
        r = _ctx.run(["true"], target=str(tmp_path),
                     output=str(tmp_path), timeout=60,
                     require_fresh_procfs=0)
    except BaseException as e:  # noqa: BLE001 — host capability gate
        pytest.skip(f"mount-ns lane unavailable: {e}")
    assert r.sandbox_info["floor_source"] == "env"
    assert r.sandbox_info["containment_floor"] == "landlock"


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


def test_no_debug_gating_in_contract_source():
    """Source pin: neither the lattice nor the dispatch checks hide
    behind ``__debug__`` or an ``assert`` statement, so no bytecode
    optimisation level and no flag can disable the boundary."""
    tiers_src = (_REPO_ROOT / "core/sandbox/tiers.py").read_text(
        encoding="utf-8")
    assert "__debug__" not in tiers_src
    assert "\nassert " not in tiers_src.replace("    assert_floor", "")
    ctx_src = (_REPO_ROOT / "core/sandbox/context.py").read_text(
        encoding="utf-8")
    check_def = ctx_src[ctx_src.index("def _dispatch_floor_check"):]
    check_def = check_def[:check_def.index("\n        # NOTE")
                          if "\n        # NOTE" in check_def[:4000]
                          else 4000]
    assert "__debug__" not in check_def
    assert "_tiers.assert_floor" in check_def


def test_every_dispatch_site_carries_a_floor_check():
    """Dispatch-site completeness tripwire: every executor hand-off in
    context.py is lexically preceded by a ``_dispatch_floor_check(``
    within its dispatch block. Crude but effective (same spirit as the
    pid1-shim tuple-sync tripwire): a new executor added without a
    declared tier fails this test before it can ship an ungated lane.
    """
    src = (_REPO_ROOT / "core/sandbox/context.py").read_text(
        encoding="utf-8")
    anchors = [
        "_macos_mod.run_sandboxed(",
        "result = _run_spawn_backend(",
        "_la.run_landlock_audit(",
        "result = subprocess.run(",
        "result = _run_teardown_first_timeout(",
    ]
    check_positions = [
        i for i in range(len(src))
        if src.startswith("_dispatch_floor_check(", i)
        and not src.startswith("def _dispatch_floor_check(", i - 4)
    ]
    assert check_positions, "no dispatch floor checks found"
    window = 7000
    for anchor in anchors:
        start = 0
        found_any = False
        while True:
            pos = src.find(anchor, start)
            if pos == -1:
                break
            found_any = True
            assert any(pos - window < c < pos for c in check_positions), (
                f"executor {anchor!r} at offset {pos} has no "
                f"_dispatch_floor_check within its dispatch block — "
                f"a lane is shipping without a declared tier")
            start = pos + 1
        assert found_any, f"executor anchor {anchor!r} vanished — " \
                          f"update the tripwire with the new spelling"


def test_lane_registry_covers_every_lane_and_matches_the_lattice():
    from core.sandbox import context as _ctx
    expected = {
        "seatbelt spawn": ContainmentTier.SEATBELT,
        "mount-ns spawn": ContainmentTier.MOUNT_NS,
        "mountless namespace backend": ContainmentTier.MOUNTLESS_NS,
        "unshare-CLI subprocess": ContainmentTier.NS_NOMOUNT,
        "Landlock-only subprocess": ContainmentTier.LANDLOCK_ONLY,
    }
    assert _ctx._LANE_TIERS == expected


# --------------------------------------------------- integration tier

def _untrusted_preflight(ctx_mod, spawn_mod, monkeypatch, tmp_path):
    """Prove run_untrusted() reaches the spawn dispatch on this host
    with a stubbed-successful backend; skip (host environment) if not."""

    def ok_spawn(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0,
                                           stdout="", stderr="")

    monkeypatch.setattr(spawn_mod, "run_sandboxed", ok_spawn)
    try:
        r = ctx_mod.run_untrusted(["true"], target=str(tmp_path),
                                  output=str(tmp_path), timeout=60)
    except BaseException as e:  # noqa: BLE001 — includes SandboxSetupError
        pytest.skip(f"untrusted lane unavailable on this host: {e}")
    if r.returncode != 0:
        pytest.skip("untrusted lane pre-flight did not run cleanly")


@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_future_below_floor_lane_is_caught_by_the_dispatch_assert(
        tmp_path, monkeypatch):
    """THE contract test: someone adds (or reroutes to) a demotion lane
    below the floor and forgets every gate. Simulated by injecting the
    environmental spawn-exception shape and re-declaring the fallback
    lane's tier as LANDLOCK_ONLY in the lane registry — the dispatch
    site itself carries the declaration and the assertion, so the stub
    lane cannot execute the sentinel: SandboxFloorError with the
    structured fields, the injected error chained, exactly one spawn
    attempt, sentinel absent. Under the per-lane-gate architecture this
    scenario silently executed."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", raising=False)
    _untrusted_preflight(_ctx, _spawn_mod, monkeypatch, tmp_path)

    injected = OSError("forced spawn setup failure")
    attempts: list[int] = []
    sentinel = tmp_path / "future-lane-ran.marker"

    def raising_spawn(cmd, **kwargs):
        attempts.append(1)
        raise injected

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", raising_spawn)
    # The "future lane": the ladder's fallback dispatch now claims to
    # deliver only LANDLOCK_ONLY (e.g. a rewritten fallback that lost
    # its namespaces). The floor must catch it all the same.
    monkeypatch.setitem(_ctx._LANE_TIERS, "unshare-CLI subprocess",
                        ContainmentTier.LANDLOCK_ONLY)
    with pytest.raises(SandboxFloorError) as excinfo:
        _ctx.run_untrusted(["touch", str(sentinel)],
                           target=str(tmp_path), output=str(tmp_path),
                           timeout=60)
    e = excinfo.value
    assert e.floor is ContainmentTier.MOUNT_NS
    assert e.achievable is ContainmentTier.LANDLOCK_ONLY
    assert e.__cause__ is injected
    assert "RAPTOR_ALLOW_DEGRADED_UNTRUSTED" in str(e)
    assert len(attempts) == 1, "expected exactly one spawn attempt"
    assert not sentinel.exists(), (
        "the below-floor stub lane executed the target")


@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_mount_and_mountless_lanes_stamp_the_same_posture_surface(
        tmp_path, monkeypatch):
    """Parity contract: the mount and mountless spawn lanes produce the
    same sandbox_info posture keys with the same truth values,
    differing exactly as declared — containment_tier names the lane's
    tier, the mountless lane additionally stamps its backend, and
    mount_ns_active flips."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    calls: list[dict] = []
    fail_first = [False]

    def fake_spawn(cmd, **kwargs):
        calls.append(kwargs)
        cp = subprocess.CompletedProcess(cmd, returncode=0,
                                         stdout="", stderr="")
        cp._setup_status = (
            ("M", "forced mount-ns failure")
            if fail_first[0] and len(calls) == 1 else None)
        return cp

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", fake_spawn)
    try:
        mount_r = _ctx.run(["true"], target=str(tmp_path),
                           output=str(tmp_path), timeout=60)
    except BaseException as e:  # noqa: BLE001 — host capability gate
        pytest.skip(f"mount-ns lane unavailable: {e}")
    if not mount_r.sandbox_info.get("mount_ns_active"):
        pytest.skip("mount-ns lane not taken on this host")

    calls.clear()
    fail_first[0] = True
    mountless_r = _ctx.run(["true"], target=str(tmp_path),
                           output=str(tmp_path), timeout=60)

    mi, li = mount_r.sandbox_info, mountless_r.sandbox_info
    assert mi["containment_tier"] == "mount-ns"
    assert li["containment_tier"] == "mountless-ns"
    assert mi["containment_floor"] == li["containment_floor"] == "none"
    assert mi["floor_source"] == li["floor_source"] == "default"
    assert mi["mount_ns_active"] is True
    assert li["mount_ns_active"] is False
    assert "backend" not in mi
    assert li["backend"] == "landlock-pidns"
    # Same posture surface otherwise: every floor-contract key present
    # in one is present in the other with the same shape.
    for key in ("containment_tier", "containment_floor", "floor_source",
                "mount_ns_active", "restrict_reads"):
        assert key in mi and key in li, key


@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_delivered_tier_stamps_follow_the_lane(tmp_path, monkeypatch):
    """containment_tier / containment_floor / floor_source stamp on
    every result: the trusted default floor is BARE ('none'), the
    unwaived untrusted contract records the mount-tier floor, and the
    operator disable records its own source."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx

    def ok_spawn(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0,
                                           stdout="", stderr="")

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", ok_spawn)
    monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", raising=False)
    try:
        r = _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(tmp_path), timeout=60)
    except BaseException as e:  # noqa: BLE001 — host capability gate
        pytest.skip(f"untrusted lane unavailable: {e}")
    assert r.sandbox_info["containment_tier"] == "mount-ns"
    assert r.sandbox_info["containment_floor"] == "mount-ns"
    assert r.sandbox_info["floor_source"] == "default"

    disabled = _ctx.run(["true"], disabled=True, timeout=60)
    assert disabled.sandbox_info["containment_tier"] == "none"
    assert disabled.sandbox_info["containment_floor"] == "none"
    assert disabled.sandbox_info["floor_source"] == "operator-disable"


@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_skip_pid_ns_caps_spawn_lane_tier_and_warning(
        tmp_path, monkeypatch, caplog):
    """skip_pid_ns keeps the HOST procfs on both spawn lanes (the
    fresh-proc remount rides the pid-ns grandchild fork), so the
    declared tier, the posture stamp, AND the consented-degrade
    warning must all treat such a run as ns-only — a mountless-ns
    label would overstate delivery and silence the per-call warning
    on a genuinely host-procfs-visible waived run."""
    import logging as _logging
    import subprocess as _subprocess

    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx

    def ok_spawn(cmd, **kwargs):
        return _subprocess.CompletedProcess(cmd, returncode=0,
                                            stdout="", stderr="")

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", ok_spawn)
    try:
        trusted = _ctx.run(["true"], target=str(tmp_path),
                           output=str(tmp_path), timeout=60,
                           skip_mount_ns=True, skip_pid_ns=True)
    except BaseException as e:  # noqa: BLE001 — host capability gate
        pytest.skip(f"spawn lane unavailable: {e}")
    assert trusted.sandbox_info["containment_tier"] == "ns-only"

    # Waived untrusted-class shape on the same lane: the capped
    # delivered tier sits at/below ns-only, so the per-call HOST
    # process table warning must fire.
    with caplog.at_level(_logging.WARNING, logger="core.sandbox.context"):
        waived = _ctx.run(["true"], target=str(tmp_path),
                          output=str(tmp_path), timeout=60,
                          skip_mount_ns=True, skip_pid_ns=True,
                          require_fresh_procfs=False)
    assert waived.sandbox_info["containment_tier"] == "ns-only"
    assert waived.sandbox_info["floor_source"] == "env"
    assert any("HOST process table" in rec.getMessage()
               for rec in caplog.records), caplog.text


@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_run_posture_record_carries_weakest_tier(tmp_path, monkeypatch):
    """record_run_posture merges containment tier/floor weakest-wins,
    like the existing posture booleans."""
    from core.sandbox import summary as _summary
    _summary.record_run_posture(
        tmp_path, mount_ns_active=True, restrict_reads=True,
        containment_tier="mount-ns", containment_floor="mount-ns")
    _summary.record_run_posture(
        tmp_path, mount_ns_active=False, restrict_reads=True,
        containment_tier="landlock", containment_floor="landlock")
    _summary.record_run_posture(
        tmp_path, mount_ns_active=True, restrict_reads=True,
        containment_tier="mountless-ns", containment_floor="mount-ns")
    posture = _summary.get_run_posture(tmp_path)
    assert posture is not None
    assert posture["containment_tier"] == "landlock"
    assert posture["containment_floor"] == "landlock"


@pytest.mark.skipif(sys.platform != "linux", reason="linux probe seam")
def test_floor_lowered_banner_fires_once_per_process(
        tmp_path, monkeypatch, caplog):
    """The consent banner names the lowered floor and its source once
    per process when the env waiver is in force for untrusted-class
    work."""
    import logging as _logging
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    from core.sandbox import state

    def ok_spawn(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0,
                                           stdout="", stderr="")

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", ok_spawn)
    monkeypatch.setenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", "1")
    state.reset_warn_once("_floor_lowered_banner_warned")
    with caplog.at_level(_logging.WARNING, logger="core.sandbox.context"):
        try:
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(tmp_path), timeout=60)
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(tmp_path), timeout=60)
        except BaseException as e:  # noqa: BLE001 — host capability gate
            pytest.skip(f"untrusted lane unavailable: {e}")
    banners = [rec for rec in caplog.records
               if "containment floor lowered" in rec.getMessage()]
    assert len(banners) == 1, caplog.text
    assert "RAPTOR_ALLOW_DEGRADED_UNTRUSTED" in banners[0].getMessage()


@pytest.mark.skipif(sys.platform != "linux", reason="linux netns lanes")
def test_inherit_netns_drop_is_stamped_warned_and_floor_gated(
        tmp_path, monkeypatch, caplog):
    """inherit_netns=True keeps the caller's netns, dropping the
    requested network block from every Linux lane. That drop is now
    explicit: stamped per run (netns_inherited), warned once per
    process, and REFUSED for the untrusted contract (whose network
    block cannot be inherited away) — pre-fix a 'network-blocked'
    trusted run silently kept host interfaces and the host TCP table
    with nothing in sandbox_info."""
    import logging as _logging
    import subprocess as _subprocess

    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    from core.sandbox import state

    def ok_spawn(cmd, **kwargs):
        return _subprocess.CompletedProcess(cmd, returncode=0,
                                            stdout="", stderr="")

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", ok_spawn)
    monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", raising=False)
    state.reset_warn_once("_inherit_netns_block_warned")
    with caplog.at_level(_logging.WARNING, logger="core.sandbox.context"):
        try:
            r = _ctx.run(["true"], block_network=True,
                         inherit_netns=True, target=str(tmp_path),
                         output=str(tmp_path), timeout=60)
        except BaseException as e:  # noqa: BLE001 — host capability gate
            pytest.skip(f"spawn lane unavailable: {e}")
    assert r.sandbox_info["netns_inherited"] is True
    assert any("inherit_netns" in rec.getMessage()
               for rec in caplog.records), caplog.text

    # A run without the network block inherits nothing away — no stamp.
    plain = _ctx.run(["true"], block_network=False,
                     inherit_netns=True, target=str(tmp_path),
                     output=str(tmp_path), timeout=60)
    assert "netns_inherited" not in plain.sandbox_info

    # The untrusted contract refuses the drop outright (run_untrusted*
    # already reject the kwarg at their allowlist; this pins the
    # direct-caller path).
    with pytest.raises(SandboxFloorError) as excinfo:
        _ctx.run(["true"], block_network=True, inherit_netns=True,
                 target=str(tmp_path), output=str(tmp_path),
                 timeout=60, require_fresh_procfs=True)
    assert "inherited away" in str(excinfo.value)
