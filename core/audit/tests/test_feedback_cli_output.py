"""`feedback` reports journal corrections, not annotation writes.

Post-migration the importer appends review-journal correction
entries and only READS annotations (human-note veto); the CLI output
still claimed "annotation(s) updated" — machinery must not claim to
write /annotate content.
"""

from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "libexec" / "raptor-audit"


def _load_cli():
    loader = SourceFileLoader("raptor_audit_cli_fbout", str(_SCRIPT))
    spec = importlib.util.spec_from_loader("raptor_audit_cli_fbout", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_feedback_output_names_journal_not_annotations(tmp_path, capsys):
    mod = _load_cli()
    report = tmp_path / "findings.json"
    report.write_text(json.dumps({"findings": []}))
    ann_dir = tmp_path / "annotations"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    rc = mod.cmd_feedback(SimpleNamespace(
        validation_report=str(report),
        annotations_dir=str(ann_dir),
        audit_out=str(out_dir),
    ))
    captured = capsys.readouterr()
    assert rc == 0
    assert "journal correction(s)" in captured.out
    assert "annotation(s) updated" not in captured.out
