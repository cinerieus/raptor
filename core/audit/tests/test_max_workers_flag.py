"""--max-workers CLI surface: parse-time validation and the plumbing
into OrchestratorConfig.max_workers (which _resolve_max_workers
honours over the transport-derived default)."""

from __future__ import annotations

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


def _load_cli():
    cli_path = str(
        Path(__file__).resolve().parents[3] / "libexec" / "raptor-audit",
    )
    loader = SourceFileLoader("raptor_audit_maxworkers_test", cli_path)
    spec = importlib.util.spec_from_loader(
        "raptor_audit_maxworkers_test", loader,
    )
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _parser(mod) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=mod._max_workers_arg,
                        default=0)
    return parser


class TestParse:
    def test_valid_value_parses(self):
        mod = _load_cli()
        args = _parser(mod).parse_args(["--max-workers", "8"])
        assert args.max_workers == 8

    def test_above_cap_clamped(self):
        from core.llm.concurrency import MAX_WORKERS_CAP
        mod = _load_cli()
        args = _parser(mod).parse_args(["--max-workers", "64"])
        assert args.max_workers == MAX_WORKERS_CAP

    def test_invalid_values_rejected_at_parse(self):
        mod = _load_cli()
        parser = _parser(mod)
        for bad in ("0", "-3", "abc", "2.5"):
            with pytest.raises(SystemExit):
                parser.parse_args(["--max-workers", bad])

    def test_default_is_transport_derived_auto(self):
        mod = _load_cli()
        assert _parser(mod).parse_args([]).max_workers == 0


class TestReachesOrchestratorConfig:
    def test_value_lands_on_config_and_wins_resolution(self, tmp_path):
        from core.audit.orchestrator import _resolve_max_workers
        from core.audit.pipeline import (
            AuditPipelineOpts,
            ReviewMode,
            _build_orchestrator_config,
        )
        opts = AuditPipelineOpts(
            target_path=tmp_path, out_dir=tmp_path, max_workers=7,
        )
        config = _build_orchestrator_config(
            opts, client=None, models=["some-model"],
            mode=ReviewMode.ENSEMBLE,
        )
        assert config.max_workers == 7
        assert _resolve_max_workers(config) == 7

    def test_zero_defers_to_derived_default(self, tmp_path):
        from core.audit.pipeline import (
            AuditPipelineOpts,
            ReviewMode,
            _build_orchestrator_config,
        )
        opts = AuditPipelineOpts(target_path=tmp_path, out_dir=tmp_path)
        config = _build_orchestrator_config(
            opts, client=None, models=["some-model"],
            mode=ReviewMode.ENSEMBLE,
        )
        assert config.max_workers == 0
