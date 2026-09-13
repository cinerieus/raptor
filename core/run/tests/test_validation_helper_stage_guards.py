"""Defensive-input tests for libexec/raptor-validation-helper stage preps.

findings.json and the stage files are LLM-authored with advisory-only
schema validation, so the stage preps must tolerate missing containers,
id-less findings, string 'line' fields, empty-but-valid working docs,
and numeric-suffix id collisions without raw tracebacks or duplicate
ids. Colocated with the run-lifecycle CLI tests.
"""

import importlib.util
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from core.json import load_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_helper():
    os.environ.setdefault("_RAPTOR_TRUSTED", "1")
    script = str(REPO_ROOT / "libexec" / "raptor-validation-helper")
    loader = SourceFileLoader("raptor_validation_helper", script)
    spec = importlib.util.spec_from_loader("raptor_validation_helper", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _write(workdir: Path, name: str, data) -> None:
    (workdir / name).write_text(json.dumps(data))


def _finding(fid: str, **overrides) -> dict:
    base = {
        "id": fid,
        "file": "a.c",
        "function": "add",
        "line": 3,
        "vuln_type": "buffer_overflow",
        "status": "not_disproven",
        "stage_a_summary": {"confidence": "medium",
                            "status": "not_disproven"},
        "origin": "claude_native",
    }
    base.update(overrides)
    return base


class TestValidateFileEmptyContainer:

    def test_empty_list_is_not_reported_missing(self, tmp_path, capsys):
        # This helper deliberately writes hypotheses.json as [] —
        # an existing empty-valid file must not print 'not found'.
        mod = _load_helper()
        _write(tmp_path, "hypotheses.json", [])
        valid, errors = mod._validate_file(
            str(tmp_path), "hypotheses.json", lambda data: (True, []))
        assert (valid, errors) == (True, [])
        assert "not found" not in capsys.readouterr().err

    def test_absent_file_still_reports_missing(self, tmp_path, capsys):
        mod = _load_helper()
        valid, errors = mod._validate_file(
            str(tmp_path), "hypotheses.json", lambda data: (True, []))
        assert (valid, errors) == (False, ["missing"])
        assert "not found" in capsys.readouterr().err


class TestStageMergeIdlessFinding:

    def test_merge_skips_idless_finding_and_applies_updates(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "B",
            "findings": [
                {"file": "b.c", "line": 1},          # no id
                _finding("FIND-1"),
            ],
        })
        _write(tmp_path, "stage-b.json", {
            "stage": "B",
            "updates": {"FIND-1": {"note": "updated"}},
        })
        assert mod._apply_stage_file(str(tmp_path), "B") is True
        merged = load_json(tmp_path / "findings.json")
        by_id = {f.get("id"): f for f in merged["findings"]}
        assert by_id["FIND-1"]["note"] == "updated"
        # The id-less finding survives the merge untouched.
        assert any("id" not in f for f in merged["findings"])


class TestMissingFindingsGuards:

    @pytest.mark.parametrize("stage", ["B", "C", "E", "F"])
    def test_missing_findings_exits_cleanly(self, tmp_path, stage, capsys):
        mod = _load_helper()
        prep = getattr(mod, f"prepare_{stage}")
        with pytest.raises(SystemExit) as exc:
            prep(str(tmp_path))
        assert exc.value.code == 1
        assert "findings.json not found" in capsys.readouterr().err

    def test_prepare_c_missing_checklist_exits_cleanly(self, tmp_path,
                                                       capsys):
        mod = _load_helper()
        _write(tmp_path, "findings.json",
               {"stage": "B", "findings": [_finding("FIND-1")]})
        with pytest.raises(SystemExit) as exc:
            mod.prepare_C(str(tmp_path))
        assert exc.value.code == 1
        assert "checklist.json not found" in capsys.readouterr().err


class TestStringLineFields:

    def test_prepare_b_tolerates_string_line(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "A",
            "findings": [
                _finding("FIND-1", line="42", function="f1"),
                _finding("FIND-2", line=40, function="f2"),
            ],
        })
        mod.prepare_B(str(tmp_path))  # str-int proximity must not raise
        data = load_json(tmp_path / "findings.json")
        assert len(data["findings"]) == 2

    def test_prepare_b_still_flags_int_line_duplicates(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "A",
            "findings": [
                _finding("FIND-1", line=40, function="f1"),
                _finding("FIND-2", line=42, function="f2"),
            ],
        })
        mod.prepare_B(str(tmp_path))
        data = load_json(tmp_path / "findings.json")
        by_id = {f["id"]: f for f in data["findings"]}
        assert by_id["FIND-2"].get("dedup_flag") == "potential_dup:FIND-1"

    def test_prepare_c_inventory_check_tolerates_string_line(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "checklist.json",
               {"files": [{"path": "a.c", "lines": 10}]})
        _write(tmp_path, "findings.json", {
            "stage": "B",
            "findings": [
                _finding("FIND-1", line="42",
                         stage_b_summary={"hypothesis_id": "H1",
                                          "hypothesis_status": "open",
                                          "proximity": 5}),
                _finding("FIND-2", line=42,
                         stage_b_summary={"hypothesis_id": "H2",
                                          "hypothesis_status": "open",
                                          "proximity": 5}),
            ],
        })
        mod.prepare_C(str(tmp_path))  # str > int comparison must not raise
        data = load_json(tmp_path / "findings.json")
        by_id = {f["id"]: f for f in data["findings"]}
        # Non-int line can't be range-checked — not failed for that.
        assert by_id["FIND-1"]["checklist_verified"] is True
        # Int line beyond file length still fails the check.
        assert by_id["FIND-2"]["checklist_verified"] is False


class TestFastPathIdCollisions:

    @staticmethod
    def _poc_finding(fid: str) -> dict:
        return _finding(
            fid,
            status="poc_success",
            stage_a_summary={"confidence": "high", "status": "poc_success"},
            poc={"result": "crash observed", "description": "run poc"},
        )

    def test_shared_numeric_suffix_gets_distinct_ids(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "A",
            "findings": [self._poc_finding("FIND-3"),
                         self._poc_finding("SARIF-3")],
        })
        mod.prepare_B(str(tmp_path))

        hypotheses = load_json(tmp_path / "hypotheses.json")
        attack_paths = load_json(tmp_path / "attack-paths.json")
        hyp_ids = [h["id"] for h in hypotheses]
        path_ids = [p["id"] for p in attack_paths]
        assert len(hyp_ids) == len(set(hyp_ids)) == 2
        assert len(path_ids) == len(set(path_ids)) == 2

        data = load_json(tmp_path / "findings.json")
        summaries = {f["id"]: f["stage_b_summary"] for f in data["findings"]}
        assert (summaries["FIND-3"]["hypothesis_id"]
                != summaries["SARIF-3"]["hypothesis_id"])
        # Each summary points at the hypothesis owned by its finding.
        owner = {h["id"]: h["finding"] for h in hypotheses}
        for fid, summary in summaries.items():
            assert owner[summary["hypothesis_id"]] == fid

    def test_fast_path_is_idempotent_across_reruns(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "A",
            "findings": [self._poc_finding("FIND-3"),
                         self._poc_finding("SARIF-3")],
        })
        mod.prepare_B(str(tmp_path))
        first = load_json(tmp_path / "hypotheses.json")
        mod.prepare_B(str(tmp_path))
        second = load_json(tmp_path / "hypotheses.json")
        assert [h["id"] for h in second] == [h["id"] for h in first]
        assert len(load_json(tmp_path / "attack-paths.json")) == 2


class TestIdlessFindingDiagnostics:
    """The id-less finding is exactly the one that takes the WARN
    branch — a KeyError there wedges the pipeline AFTER the stage
    file was consumed and deleted, and every re-run crashes the same
    way from the persisted findings.json."""

    def test_prepare_b_warn_path_tolerates_missing_id(self, tmp_path,
                                                      capsys):
        mod = _load_helper()
        f = _finding("FIND-1")
        del f["id"]
        del f["stage_a_summary"]
        _write(tmp_path, "findings.json", {"stage": "A", "findings": [f]})
        mod.prepare_B(str(tmp_path))  # must not KeyError
        err = capsys.readouterr().err
        assert "missing stage_a_summary" in err

    def test_prepare_c_warn_and_c0_paths_tolerate_missing_id(
            self, tmp_path, capsys):
        mod = _load_helper()
        _write(tmp_path, "checklist.json",
               {"files": [{"path": "a.c", "lines": 10}]})
        f = _finding("FIND-1", file="not-in-inventory.c")
        del f["id"]
        _write(tmp_path, "findings.json", {"stage": "B", "findings": [f]})
        mod.prepare_C(str(tmp_path))  # must not KeyError
        err = capsys.readouterr().err
        assert "missing stage_b_summary" in err
        assert "C0 FAIL" in err

    def test_prepare_d_warn_path_tolerates_missing_id(self, tmp_path,
                                                      capsys):
        mod = _load_helper()
        f = _finding("FIND-1")
        del f["id"]
        _write(tmp_path, "findings.json", {"stage": "C", "findings": [f]})
        mod.prepare_D(str(tmp_path))  # must not KeyError
        err = capsys.readouterr().err
        assert "missing stage_c_summary" in err


class TestNonDictSummaries:
    """poc / stage summaries may be bare strings — findings.json is
    LLM-authored and schema validation is advisory; the same file
    already guards this shape in _finding_poc_payload."""

    def test_prepare_b_tolerates_string_poc_and_summary(self, tmp_path):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "A",
            "findings": [_finding(
                "FIND-1", status="poc_success",
                poc="segfault at 0x41414141",
                stage_a_summary="high",
            )],
        })
        mod.prepare_B(str(tmp_path))  # must not AttributeError
        data = load_json(tmp_path / "findings.json")
        assert len(data["findings"]) == 1

    def test_prepare_d_tolerates_string_stage_c_summary(self, tmp_path,
                                                        capsys):
        mod = _load_helper()
        _write(tmp_path, "findings.json", {
            "stage": "C",
            "findings": [_finding("FIND-1", stage_c_summary="passed")],
        })
        mod.prepare_D(str(tmp_path))  # must not AttributeError
        err = capsys.readouterr().err
        assert "must be a dict" in err


class TestBinaryDiscoveryFailuresAreLoud:
    """A find timeout/failure must not masquerade as "no binary
    found" attrition — the old contextlib.suppress(Exception) wrapped
    the whole discovery body and the operator never learned discovery
    itself failed."""

    def test_timeout_warns_and_returns_empty(self, tmp_path, capsys,
                                             monkeypatch):
        import subprocess
        mod = _load_helper()

        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="find", timeout=30)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = mod._discover_executables(str(tmp_path))
        assert result == {}
        err = capsys.readouterr().err
        assert "binary discovery timed out" in err

    def test_oserror_warns_and_returns_empty(self, tmp_path, capsys,
                                             monkeypatch):
        import subprocess
        mod = _load_helper()

        def fake_run(*a, **kw):
            raise OSError("find: command not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = mod._discover_executables(str(tmp_path))
        assert result == {}
        err = capsys.readouterr().err
        assert "binary discovery failed" in err
        assert "OSError" in err
