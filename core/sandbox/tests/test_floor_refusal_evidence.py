"""Run-level containment-floor refusal evidence.

A refused run must be distinguishable from a run that recorded
nothing, even after the fact: the refusal chokepoint in context.run()
records a ``floor_refusal`` record into the run-dir evidence stream,
and ``sandbox-summary.json`` carries the unverifiable-environment
count line whose shape the verification seams' per-finding mapping
expects — same status string, same line wording, MAC-bound
only-when-set so pre-existing tokens keep verifying.
"""

from __future__ import annotations

import sys

import pytest

from core.sandbox.errors import SandboxFloorError

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux namespace lanes")


def _simulate_ns_capable(monkeypatch):
    """Pin the namespace-capability probes True (spawn stubbed /
    refused before any spawn), so the refusal shape under test fires
    identically on degraded feature-matrix lanes."""
    from core.sandbox import context as _ctx
    from core.sandbox import probes as _probes_mod
    monkeypatch.setattr(_ctx, "check_net_available", lambda: True)
    monkeypatch.setattr(_probes_mod, "check_unshare_engages",
                        lambda flags: (True, ""))


def test_refused_run_leaves_count_in_summary(tmp_path, monkeypatch):
    """A containment-floor refusal is recorded into the run-dir
    evidence stream at the refusal chokepoint, and summary generation
    surfaces the unverifiable-environment count line."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    from core.sandbox import summary as _summary
    monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", raising=False)
    _simulate_ns_capable(monkeypatch)
    monkeypatch.setattr(_ctx, "check_mount_available", lambda: False)
    monkeypatch.setattr(_ctx, "check_landlock_available", lambda: True)
    monkeypatch.setattr(_ctx, "_get_landlock_abi", lambda: 4)
    monkeypatch.setattr(_spawn_mod, "mount_ns_available", lambda: True)
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(SandboxFloorError):
        _ctx.run_untrusted(["true"], target=str(tmp_path),
                           output=str(out), timeout=60)
    result = _summary.summarize_and_write(out)
    assert result is not None
    assert result["total_floor_refusals"] >= 1
    line = result["floor_refusal_line"]
    assert "execution(s) refused" in line
    assert "containment floor" in line
    assert "mount-ns required" in line
    rec = result["floor_refusals"][0]
    assert rec["status"] == "unverifiable_environment"
    assert rec["floor"] == "mount-ns"
    # No denial ever happened — the refusal alone must produce the
    # summary, and it must not masquerade as a denial.
    assert result["total_denials"] == 0
    # And the file landed on disk for post-hoc readers.
    assert (out / _summary.SUMMARY_FILE).exists()


def test_refusal_status_matches_witness_seam_constant():
    """The sandbox-side literal and the witness seam's canonical
    status must never drift (the dependency arrow forbids an
    import)."""
    from core.sandbox import summary as _summary
    from core.witness.sandbox_outcome import UNVERIFIABLE_ENVIRONMENT
    assert (_summary.UNVERIFIABLE_ENVIRONMENT_STATUS
            == UNVERIFIABLE_ENVIRONMENT)


def test_refusal_line_wording_matches_witness_renderer():
    """The run-level line and the per-finding renderer share one
    wording so operator output cannot drift."""
    from core.sandbox import summary as _summary
    from core.witness.sandbox_outcome import refusal_summary_line
    detail = {"status": "unverifiable_environment", "floor": "mount-ns",
              "achievable": "landlock", "remedies": "fix the host"}
    rendered = refusal_summary_line(detail, count=2)
    record = {"floor": "mount-ns", "achievable": "landlock",
              "remedies": "fix the host"}
    # Rebuild the summary line exactly as summarize_and_write does.
    line = (
        f"2 execution(s) refused: environment cannot meet the "
        f"containment floor ({record['floor']} required, "
        f"{record['achievable']} achievable) — remedies: "
        f"{record['remedies']}"
    )
    assert line == rendered
    _summary  # namespace parity

class TestPreRunArmsRecord:
    """The floor-class refusals that fire BEFORE run() — the untrusted
    entry gate's userns / libseccomp / seatbelt arms and strict's
    construction gate — leave the same ``floor_refusal`` evidence as
    the in-run chokepoint: a refused run must be distinguishable from
    a run that recorded nothing, whichever gate refused it."""

    @pytest.fixture(autouse=True)
    def _unwaived(self, monkeypatch):
        monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED",
                           raising=False)

    def _summary_of(self, out):
        from core.sandbox import summary as _summary
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] >= 1
        return result["floor_refusals"][0]

    def test_userns_arm_records(self, tmp_path, monkeypatch):
        from core.sandbox import context as _ctx
        from core.sandbox import seccomp as _seccomp_mod
        if not _seccomp_mod.check_seccomp_available():
            pytest.skip("libseccomp required to isolate the userns arm")
        monkeypatch.setattr(_ctx, "check_net_available", lambda: False)
        monkeypatch.setattr(_ctx, "check_landlock_available",
                            lambda: True)
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxFloorError) as excinfo:
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(out), timeout=60)
        assert "user namespaces" in str(excinfo.value)
        rec = self._summary_of(out)
        assert rec["status"] == "unverifiable_environment"
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == "landlock"

    def test_seccomp_arm_records(self, tmp_path, monkeypatch):
        from core.sandbox import context as _ctx
        monkeypatch.setattr(_ctx._seccomp, "check_seccomp_available",
                            lambda: False)
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxFloorError) as excinfo:
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(out), timeout=60)
        assert "libseccomp" in str(excinfo.value)
        rec = self._summary_of(out)
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == "none"

    def test_seccomp_pin_arm_records(self, tmp_path, monkeypatch):
        """The explicit-pin refusal (no tier waives seccomp absence)
        records the PINNED floor."""
        from core.sandbox import context as _ctx
        from core.sandbox import state
        monkeypatch.setattr(_ctx._seccomp, "check_seccomp_available",
                            lambda: False)
        monkeypatch.setenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", "1")
        state._cli_sandbox_floor = "landlock"
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxFloorError) as excinfo:
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               output=str(out), timeout=60)
        assert "does not override" in str(excinfo.value)
        rec = self._summary_of(out)
        assert rec["floor"] == "landlock"
        assert rec["achievable"] == "none"

    def test_darwin_seatbelt_arm_records(self, tmp_path):
        from unittest import mock

        from core.sandbox import context as _ctx
        out = tmp_path / "run"
        out.mkdir()
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(_ctx, "check_seatbelt_available",
                                  return_value=False), \
                pytest.raises(SandboxFloorError) as excinfo:
            _ctx._require_userns_or_optin("run_untrusted",
                                          record_dir=str(out))
        assert "seatbelt" in str(excinfo.value)
        assert excinfo.value.floor.name == "SEATBELT"
        rec = self._summary_of(out)
        assert rec["floor"] == "seatbelt"
        assert rec["achievable"] == "none"

    def test_strict_gate_records_then_raises_existing_type(
            self, tmp_path, monkeypatch):
        """strict's construction abort is floor-class in substance
        and now records — but the exception type stays a plain
        SandboxSetupError (record-then-raise): the gate aggregates
        mixed tier + seccomp-axis requirements that one typed
        floor/achievable pair would overclaim."""
        from core.sandbox import context as _ctx
        from core.sandbox import seccomp as _seccomp_mod
        from core.sandbox.errors import SandboxSetupError
        if not _seccomp_mod.check_seccomp_available():
            pytest.skip("libseccomp required for the simulated host")
        monkeypatch.setattr(_ctx, "check_net_available", lambda: False)
        monkeypatch.setattr(_ctx, "check_mount_available",
                            lambda: False)
        monkeypatch.setattr(_ctx, "check_landlock_available",
                            lambda: True)
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError) as excinfo:
            with _ctx.sandbox(profile="strict", target=str(tmp_path),
                              output=str(out)):
                pass  # pragma: no cover — construction must refuse
        assert type(excinfo.value) is SandboxSetupError
        assert "strict" in str(excinfo.value)
        rec = self._summary_of(out)
        assert rec["status"] == "unverifiable_environment"
        # target/output present: strict demands the mount tier; the
        # policy layers still engage on this simulated host.
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == "landlock"

    def test_target_only_refusal_attributes_to_active_run_dir(
            self, tmp_path, monkeypatch):
        """A call with no audit/output dir of its own (target-only
        run_untrusted) no longer skips the record silently: the
        refusal attributes to the process's active run dir."""
        from core.sandbox import context as _ctx
        from core.sandbox import seccomp as _seccomp_mod
        from core.sandbox import summary as _summary
        if not _seccomp_mod.check_seccomp_available():
            pytest.skip("libseccomp required to isolate the userns arm")
        monkeypatch.setattr(_ctx, "check_net_available", lambda: False)
        monkeypatch.setattr(_ctx, "check_landlock_available",
                            lambda: True)
        run_dir = tmp_path / "active-run"
        run_dir.mkdir()
        _summary.set_active_run_dir(run_dir)
        try:
            with pytest.raises(SandboxFloorError):
                _ctx.run_untrusted(["true"], target=str(tmp_path),
                                   timeout=60)
            rec = self._summary_of(run_dir)
            assert rec["floor"] == "mount-ns"
        finally:
            _summary.set_active_run_dir(None)

    def test_no_dir_anywhere_still_raises_loud(self, tmp_path,
                                               monkeypatch):
        """With no call dir AND no active run dir there is nowhere
        durable to record — the typed raise itself is the signal
        (documented limitation); the refusal must not be masked by
        the recording attempt."""
        from core.sandbox import context as _ctx
        from core.sandbox import seccomp as _seccomp_mod
        from core.sandbox import summary as _summary
        if not _seccomp_mod.check_seccomp_available():
            pytest.skip("libseccomp required to isolate the userns arm")
        monkeypatch.setattr(_ctx, "check_net_available", lambda: False)
        _summary.set_active_run_dir(None)
        with pytest.raises(SandboxFloorError):
            _ctx.run_untrusted(["true"], target=str(tmp_path),
                               timeout=60)


class TestDeliveryTimeContractFailuresRecord:
    """The delivery-time contract failures that are floor-class in
    substance — the fresh-procfs 'F' status byte and the seatbelt 'E'
    readiness byte on a floored call — record the same floor_refusal
    evidence as the entry-time refusals. Their exception TYPE stays a
    plain SandboxSetupError (child-side engagement truth, kept by
    design); only the evidence gap closes."""

    @staticmethod
    def _status_stub(status):
        import subprocess as _sp

        def fake(cmd, **kwargs):
            cp = _sp.CompletedProcess(cmd, returncode=1,
                                      stdout="", stderr="")
            cp._setup_status = status
            return cp
        return fake

    def _linux_spawn_route(self, monkeypatch):
        from core.sandbox import _spawn as _spawn_mod
        from core.sandbox import context as _ctx
        _simulate_ns_capable(monkeypatch)
        monkeypatch.setattr(_ctx, "check_mount_available", lambda: True)
        monkeypatch.setattr(_ctx, "check_landlock_available",
                            lambda: True)
        monkeypatch.setattr(_spawn_mod, "mount_ns_available",
                            lambda: True)
        monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED",
                           raising=False)
        return _spawn_mod, _ctx

    def test_f_status_on_contract_call_records(self, tmp_path,
                                               monkeypatch):
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub(("F", "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60,
                     require_fresh_procfs=(
                         _ctx.untrusted_fresh_procfs_required()))
        assert excinfo.value.setup_category == "F"
        assert not isinstance(excinfo.value, SandboxFloorError)
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] == 1
        rec = result["floor_refusals"][0]
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == "landlock"

    def test_f_status_on_trusted_call_stays_unrecorded(
            self, tmp_path, monkeypatch):
        """Floor BARE: the F arm keeps its plain fail-loud raise with
        no floor_refusal record — no floor was demanded."""
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub(("F", "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError):
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60)
        result = _summary.summarize_and_write(out)
        assert result is None or result.get(
            "total_floor_refusals", 0) == 0

    @pytest.mark.parametrize("byte,ach", [
        ("S", "none"),       # filter is in every tier's contract
        ("L", "ns-only"),    # namespace tier's own promise holds
    ])
    def test_ls_status_on_contract_call_records(self, tmp_path,
                                                monkeypatch, byte,
                                                ach):
        """A terminal layer-apply failure in the spawn child
        ('L'/'S') on a floored call is floor-class in substance and
        records, with achievable following the failed layer
        (understating)."""
        from core.sandbox import seccomp as _seccomp_mod
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        if not _seccomp_mod.check_seccomp_available():
            pytest.skip("libseccomp required for the 'L' label truth")
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub((byte, "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60,
                     require_fresh_procfs=(
                         _ctx.untrusted_fresh_procfs_required()))
        assert excinfo.value.setup_category == byte
        assert not isinstance(excinfo.value, SandboxFloorError)
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] == 1
        rec = result["floor_refusals"][0]
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == ach

    def test_ls_status_on_trusted_call_stays_unrecorded(
            self, tmp_path, monkeypatch):
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub(("L", "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError):
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60)
        result = _summary.summarize_and_write(out)
        assert result is None or result.get(
            "total_floor_refusals", 0) == 0

    def test_u_status_on_contract_call_refuses_at_fallback_and_records(
            self, tmp_path, monkeypatch):
        """The status-byte 'U' shape rides the demotion ladder (like
        its exception-shape twin): a floored call is refused at the
        fallback dispatch — SandboxFloorError with the child's 'U'
        diagnostic chained — and records exactly ONCE. Pre-fix the
        status shape fell through the ladder's catch with the SETUP
        CHILD's CompletedProcess returned as the target's result (no
        raise, no record, fabricated rc)."""
        from core.sandbox import summary as _summary
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub(("U", "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxFloorError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60,
                     require_fresh_procfs=(
                         _ctx.untrusted_fresh_procfs_required()))
        # The environment diagnosis stays reachable: the original 'U'
        # failure is the refusal's cause and its category is carried.
        assert excinfo.value.setup_category == "U"
        _chain = excinfo.value.__cause__
        assert _chain is not None and "unshare" in str(_chain)
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] == 1
        assert result["floor_refusals"][0]["floor"] == "mount-ns"

    def test_u_status_on_trusted_call_reruns_on_the_fallback_lane(
            self, tmp_path, monkeypatch):
        """Trusted parity for the same fix: the ladder-demoted call
        actually RE-RUNS on the fallback lane and returns the real
        command's result — never the dead setup child's."""
        from core.sandbox import summary as _summary
        _spawn_mod, _ctx = self._linux_spawn_route(monkeypatch)
        monkeypatch.setattr(
            _spawn_mod, "run_sandboxed",
            self._status_stub(("U", "forced by test")))
        out = tmp_path / "run"
        out.mkdir()
        marker = out / "ran-for-real"
        try:
            r = _ctx.run(["touch", str(marker)], target=str(tmp_path),
                         output=str(out), timeout=60)
        except BaseException as e:  # noqa: BLE001 — host capability gate
            pytest.skip(f"fallback lane unavailable: {e}")
        assert r.returncode == 0
        assert marker.exists(), (
            "fallback returned a result without re-running the command")
        assert getattr(r, "_setup_status", None) is None
        result = _summary.summarize_and_write(out)
        assert result is None or result.get(
            "total_floor_refusals", 0) == 0

    def test_engage_probe_refusal_on_contract_call_records(
            self, tmp_path, monkeypatch):
        """The pre-spawn definitive kernel refusal
        (check_unshare_engages is False) on a floored call records
        with the userns-arm mapping."""
        from core.sandbox import _spawn as _spawn_mod
        from core.sandbox import context as _ctx
        from core.sandbox import probes as _probes_mod
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED",
                           raising=False)
        # Entry-time views say the mount tier is intendable — the
        # narrow probes pass; only the FULL flag-set probe refuses
        # (the rootless-podman / nested-userns shape).
        monkeypatch.setattr(_ctx, "check_net_available", lambda: True)
        monkeypatch.setattr(_ctx, "check_mount_available",
                            lambda: True)
        monkeypatch.setattr(_spawn_mod, "mount_ns_available",
                            lambda: True)
        monkeypatch.setattr(_ctx, "check_landlock_available",
                            lambda: True)
        monkeypatch.setattr(_probes_mod, "check_unshare_engages",
                            lambda flags: (False, "denied by test"))
        out = tmp_path / "run"
        out.mkdir()
        with pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path), output=str(out),
                     timeout=60,
                     require_fresh_procfs=(
                         _ctx.untrusted_fresh_procfs_required()))
        assert "namespace setup failed" in str(excinfo.value)
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] >= 1
        rec = result["floor_refusals"][0]
        assert rec["floor"] == "mount-ns"
        assert rec["achievable"] == "landlock"

    def test_seatbelt_e_status_on_contract_call_records(
            self, tmp_path, monkeypatch):
        from unittest import mock

        from core.sandbox import _macos_spawn as macos_mod
        from core.sandbox import context as _ctx
        from core.sandbox import summary as _summary
        from core.sandbox.errors import SandboxSetupError
        monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED",
                           raising=False)
        out = tmp_path / "run"
        out.mkdir()
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(_ctx, "check_seatbelt_available",
                                  return_value=True), \
                mock.patch.object(_ctx, "check_mount_available",
                                  return_value=False), \
                mock.patch.object(_ctx, "check_net_available",
                                  return_value=False), \
                mock.patch.object(
                    macos_mod, "run_sandboxed",
                    self._status_stub(("E", "no readiness byte"))), \
                pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["/usr/bin/true"], target=str(tmp_path),
                     output=str(out), timeout=30,
                     require_fresh_procfs=True)
        assert excinfo.value.setup_category == "E"
        assert not isinstance(excinfo.value, SandboxFloorError)
        result = _summary.summarize_and_write(out)
        assert result is not None
        assert result["total_floor_refusals"] == 1
        rec = result["floor_refusals"][0]
        assert rec["floor"] == "seatbelt"
        assert rec["achievable"] == "none"


def test_refusal_survives_whole_file_deletion(tmp_path, monkeypatch):
    """A target that DELETES the evidence JSONL outright (not just
    edits or truncates it) must not erase the refusal: the parent-
    memory record re-asserts through the same summary assembly, MAC
    included."""
    from core.sandbox import summary as _summary
    out = tmp_path / "run"
    out.mkdir()
    _summary.record_floor_refusal(
        out, floor="mount-ns", achievable="landlock",
        reason="probe refusal", remedies="fix the host")
    ev = out / ".audit" / _summary.DENIALS_FILE
    assert ev.exists()
    ev.unlink()
    result = _summary.summarize_and_write(out)
    assert result is not None
    assert result["total_floor_refusals"] == 1
    assert result["floor_refusals"][0]["status"] == (
        "unverifiable_environment")
    assert (out / _summary.SUMMARY_FILE).exists()


def test_pre_run_arm_refusal_survives_whole_file_deletion(
        tmp_path, monkeypatch):
    """Same deletion-survival guarantee for a record produced by a
    PRE-run() arm (the untrusted entry gate) rather than a direct
    record_floor_refusal call: parent memory re-asserts it."""
    from core.sandbox import context as _ctx
    from core.sandbox import seccomp as _seccomp_mod
    from core.sandbox import summary as _summary
    if not _seccomp_mod.check_seccomp_available():
        pytest.skip("libseccomp required to isolate the userns arm")
    monkeypatch.delenv("RAPTOR_ALLOW_DEGRADED_UNTRUSTED", raising=False)
    monkeypatch.setattr(_ctx, "check_net_available", lambda: False)
    monkeypatch.setattr(_ctx, "check_landlock_available", lambda: True)
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(SandboxFloorError):
        _ctx.run_untrusted(["true"], target=str(tmp_path),
                           output=str(out), timeout=60)
    ev = out / ".audit" / _summary.DENIALS_FILE
    assert ev.exists()
    ev.unlink()
    result = _summary.summarize_and_write(out)
    assert result is not None
    assert result["total_floor_refusals"] >= 1
    assert result["floor_refusals"][0]["floor"] == "mount-ns"
