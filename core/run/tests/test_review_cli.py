"""Tests for libexec/raptor-review argument dispatch.

Colocated with the run-lifecycle CLI tests: raptor-review is the
operator CLI navigating run/project output, and its dispatch bugs are
invisible to CI without a direct test.
"""

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_review_module():
    cli_path = str(REPO_ROOT / "libexec" / "raptor-review")
    loader = SourceFileLoader("raptor_review_cli", cli_path)
    spec = importlib.util.spec_from_loader("raptor_review_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class TestDefaultSubcommandSniffer:
    """The sniffer must not mistake a flag VALUE for the subcommand."""

    def test_flag_value_before_subcommand_is_skipped(self):
        # Regression: `--project /x stats` picked `/x` as the first
        # positional and prepended "show", misparsing `stats`.
        mod = _load_review_module()
        assert mod._first_positional(["--project", "/x", "stats"]) == "stats"

    def test_inline_flag_value_consumes_nothing(self):
        mod = _load_review_module()
        assert mod._first_positional(["--project=/x", "stats"]) == "stats"

    def test_bare_positional_wins(self):
        mod = _load_review_module()
        assert mod._first_positional(["src/a.c", "fn"]) == "src/a.c"

    def test_boolean_flag_does_not_consume(self):
        mod = _load_review_module()
        assert mod._first_positional(["--raw", "findings"]) == "findings"

    def test_no_positional(self):
        mod = _load_review_module()
        assert mod._first_positional(["--project", "/x"]) is None


class TestNoteEditWriteBase:
    """note/edit writes must land in the SAME annotations dir the
    read side resolves (run's project via --out / --project), never
    in the ambient active project — the write-path equivalent of the
    read side's cross-project-bleed fix."""

    @staticmethod
    def _run_dir(tmp_path):
        import json
        project = tmp_path / "proj"
        run = project / "run_001"
        run.mkdir(parents=True)
        (run / ".raptor-run.json").write_text(
            json.dumps({"command": "audit"}), encoding="utf-8")
        return run

    def _capture_delegate(self, mod, monkeypatch):
        import subprocess
        calls = []

        def fake_call(cmd, env=None):
            calls.append(cmd)
            return 0

        monkeypatch.setattr(subprocess, "call", fake_call)
        return calls

    def test_note_derives_base_from_out(self, tmp_path, monkeypatch):
        import argparse

        import pytest
        mod = _load_review_module()
        calls = self._capture_delegate(mod, monkeypatch)
        run = self._run_dir(tmp_path)
        args = argparse.Namespace(
            file="src/a.c", function="f", body="note", status=None,
            base=None, out=str(run), project=None,
        )
        with pytest.raises(SystemExit):
            mod.cmd_note(args)
        cmd = calls[0]
        assert "--base" in cmd
        base = cmd[cmd.index("--base") + 1]
        assert base == str(run.parent / "annotations")

    def test_edit_derives_base_from_out(self, tmp_path, monkeypatch):
        import argparse

        import pytest
        mod = _load_review_module()
        calls = self._capture_delegate(mod, monkeypatch)
        run = self._run_dir(tmp_path)
        args = argparse.Namespace(
            file="src/a.c", function="f",
            base=None, out=str(run), project=None,
        )
        with pytest.raises(SystemExit):
            mod.cmd_edit(args)
        cmd = calls[0]
        assert "--base" in cmd
        assert cmd[cmd.index("--base") + 1] == str(
            run.parent / "annotations")

    def test_explicit_base_wins(self, tmp_path, monkeypatch):
        import argparse

        import pytest
        mod = _load_review_module()
        calls = self._capture_delegate(mod, monkeypatch)
        run = self._run_dir(tmp_path)
        args = argparse.Namespace(
            file="src/a.c", function="f", body=None, status=None,
            base="/explicit/base", out=str(run), project=None,
        )
        with pytest.raises(SystemExit):
            mod.cmd_note(args)
        cmd = calls[0]
        assert cmd[cmd.index("--base") + 1] == "/explicit/base"


class TestProjectFlagBeforeSubcommand:
    """`--project X <subcmd>` must not silently drop the value.

    ``common`` parents both the main parser and every subparser; the
    subparser used to re-apply its None default over the value the
    main parser had already stored (the hard-error for a bad
    --project only fired for flag-after-subcommand order).
    """

    def test_bad_project_before_subcommand_hard_errors(
            self, monkeypatch, capsys):
        mod = _load_review_module()
        monkeypatch.setattr(
            "sys.argv",
            ["raptor-review", "--project", "/nonexistent-project-xyz",
             "stats"],
        )
        try:
            mod.main()
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError(
                "bad --project before the subcommand was dropped "
                "instead of hard-erroring"
            )
        err = capsys.readouterr().err
        assert "neither a project directory nor a registered" in err

    def test_bad_project_after_subcommand_still_hard_errors(
            self, monkeypatch, capsys):
        mod = _load_review_module()
        monkeypatch.setattr(
            "sys.argv",
            ["raptor-review", "stats", "--project",
             "/nonexistent-project-xyz"],
        )
        try:
            mod.main()
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("bad --project accepted")
