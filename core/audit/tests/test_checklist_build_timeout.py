"""A timed-out checklist build is reported, not a traceback.

The raptor-build-checklist children in `gaps` and `run` set
timeout=300 but nothing caught subprocess.TimeoutExpired: `gaps` died
with a raw traceback, and `run` — where the build fires after
lifecycle start — left the run wedged in status=running with no fail
transition, so the advertised resume refused the directory.
"""

from __future__ import annotations

import importlib.util
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "libexec" / "raptor-audit"


def _load_cli():
    loader = SourceFileLoader("raptor_audit_cli_cktimeout", str(_SCRIPT))
    spec = importlib.util.spec_from_loader(
        "raptor_audit_cli_cktimeout", loader,
    )
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_gaps_checklist_build_timeout_reported(
        tmp_path, monkeypatch, capsys):
    mod = _load_cli()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.py").write_text("def f():\n    return 1\n")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Only out/target are read before the timeout path returns.
    rc = mod.cmd_gaps(SimpleNamespace(out=str(out_dir), target=str(target)))
    captured = capsys.readouterr()
    assert rc == 1
    assert "timed out" in captured.err


def test_run_checklist_build_timeout_fails_lifecycle(
        tmp_path, monkeypatch, capsys):
    mod = _load_cli()
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.py").write_text("def f():\n    return 1\n")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        cmd_s = [str(c) for c in cmd]
        calls.append(cmd_s)
        joined = " ".join(cmd_s)
        if "raptor-run-lifecycle" in joined:
            stdout = f"OUTPUT_DIR={out_dir}\n" if "start" in cmd_s else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if "raptor-build-checklist" in joined:
            raise subprocess.TimeoutExpired(cmd, 300)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = mod.cmd_run(SimpleNamespace(
        target=str(target), out=str(out_dir),
    ))
    captured = capsys.readouterr()
    assert rc == 1
    assert "timed out" in captured.err
    # The run must transition to failed, not stay wedged in running.
    fail_calls = [c for c in calls
                  if "raptor-run-lifecycle" in " ".join(c) and "fail" in c]
    assert fail_calls, f"no lifecycle fail transition in {calls}"
