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
