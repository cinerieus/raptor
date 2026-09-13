"""Round-trip tests for the SAGE evidence-row grammar.

The writer (``core.sage.hooks.store_study_concepts`` via
``core.concepts.model.sage_evidence_row``) and the parsers
(``core.concepts.study._SAGE_EVIDENCE_RE`` / ``_EVIDENCE_HASH_RE`` and
the staleness verifier) share one grammar; a drift silently disables
the cross-run study skip (parse fails closed to the seed path). These
tests pin the round trip on both the regex level and the
hash-verification level.
"""

from __future__ import annotations

from pathlib import Path

from core.concepts.model import Evidence, sage_evidence_row
from core.concepts.study import (
    _EVIDENCE_HASH_RE,
    _SAGE_EVIDENCE_RE,
    _extract_evidence_hashes,
    _verify_evidence_hashes,
)


def test_row_with_line_and_hash_round_trips():
    ev = Evidence(type="code_path", file="src/mm.c", line=42,
                  observation="alloc without size check", hash="ab12cd34ef56")
    row = sage_evidence_row(ev)
    m = _SAGE_EVIDENCE_RE.match(row.strip())
    assert m is not None
    assert m.group(1) == "code_path"
    assert m.group(2).strip() == "src/mm.c"
    assert m.group(3) == "42"
    assert m.group(4) == "ab12cd34ef56"
    assert m.group(5).strip() == "alloc without size check"


def test_row_without_hash_round_trips():
    ev = Evidence(type="api_pattern", file="src/api.c", line=7,
                  observation="callback registered")
    row = sage_evidence_row(ev)
    m = _SAGE_EVIDENCE_RE.match(row.strip())
    assert m is not None
    assert m.group(2).strip() == "src/api.c"
    assert m.group(3) == "7"
    assert m.group(4) is None
    assert m.group(5).strip() == "callback registered"


def test_row_without_line_round_trips():
    ev = Evidence(type="doc", file="README.md",
                  observation="documented ownership transfer")
    row = sage_evidence_row(ev)
    m = _SAGE_EVIDENCE_RE.match(row.strip())
    assert m is not None
    assert m.group(2).strip() == "README.md"
    assert m.group(3) is None
    assert m.group(5).strip() == "documented ownership transfer"


def test_hash_extraction_sees_written_hashes():
    ev = Evidence(type="code_path", file="a.c", line=1,
                  observation="x", hash="deadbeef1234")
    content = "Concept [c.x] in scope: d\n" + sage_evidence_row(ev)
    assert _extract_evidence_hashes(content) == {"deadbeef1234"}


def test_written_row_verifies_and_detects_drift(tmp_path: Path):
    # Writer -> staleness-verifier round trip: a row rendered by the
    # shared helper over a real source span verifies fresh, and stops
    # verifying when the span changes.
    from core.staleness import hash_spans

    src = tmp_path / "mm.c"
    src.write_text("int a;\nchar *p = malloc(n);\nint b;\n")
    h = hash_spans(src, [(2, 2)])[0]
    assert h
    ev = Evidence(type="code_path", file="mm.c", line=2,
                  observation="unchecked alloc", hash=h)
    content = "Concept [c.mm] in scope: d\n" + sage_evidence_row(ev)
    assert _verify_evidence_hashes(content, tmp_path) is True
    src.write_text("int a;\nchar *p = calloc(1, n);\nint b;\n")
    assert _verify_evidence_hashes(content, tmp_path) is False


def test_hash_grammar_matches_stamped_hash_alphabet(tmp_path: Path):
    # hash_spans emits lowercase-hex prefixes; the [h=...] grammar
    # ([a-f0-9]+) must accept every hash the stamper produces.
    from core.staleness import hash_spans

    src = tmp_path / "x.c"
    src.write_text("line one\n")
    h = hash_spans(src, [(1, 1)])[0]
    assert h
    assert _EVIDENCE_HASH_RE.fullmatch(f"[h={h}]")
