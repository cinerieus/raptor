"""SARIF read-cap parity across the core/dataflow consumers.

The budget lives in ``core.sarif.parser.SARIF_MAX_BYTES``; the
corpus/bridge/pipeline loaders and ``finding_diff`` alias it. Before
the aliasing, ``finding_diff`` carried a hand-copied value that had
drifted to 128 MiB while its docstring claimed to match the parser's
policy — these tests keep the aliases from re-drifting and pin the
cap's behavior in both directions.
"""

from __future__ import annotations

import json

import pytest

from core.dataflow import (
    barrier_synth,
    cvefix_bridge,
    cvefix_pipeline,
    finding_diff,
    owasp_corpus_generator,
)
from core.sarif.parser import SARIF_MAX_BYTES


def test_every_dataflow_alias_matches_the_parser_policy():
    for mod in (
        barrier_synth, cvefix_bridge, cvefix_pipeline,
        owasp_corpus_generator,
    ):
        assert mod._MAX_SARIF_BYTES == SARIF_MAX_BYTES, mod.__name__


_EMPTY_SARIF = json.dumps({"version": "2.1.0", "runs": []})


def test_diff_sarif_files_refuses_over_cap(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.sarif"
    augmented = tmp_path / "augmented.sarif"
    baseline.write_text(_EMPTY_SARIF)
    augmented.write_text(_EMPTY_SARIF)
    monkeypatch.setattr(
        finding_diff, "SARIF_MAX_BYTES", len(_EMPTY_SARIF) - 1,
    )
    with pytest.raises(RuntimeError, match="exceeds .*-byte cap"):
        finding_diff.diff_sarif_files(baseline, augmented)


def test_diff_sarif_files_accepts_under_cap(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.sarif"
    augmented = tmp_path / "augmented.sarif"
    baseline.write_text(_EMPTY_SARIF)
    augmented.write_text(_EMPTY_SARIF)
    monkeypatch.setattr(
        finding_diff, "SARIF_MAX_BYTES", len(_EMPTY_SARIF),
    )
    diff = finding_diff.diff_sarif_files(baseline, augmented)
    assert diff.baseline_count == 0
    assert diff.augmented_count == 0
