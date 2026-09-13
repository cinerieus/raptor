"""Tests for core.audit.proto_length_checker — hermetic, synthetic C.

The suppression direction is the dangerous one: only a comparison
that caps the length ABOVE may make an unbounded protocol length read
as bounded.
"""

from __future__ import annotations

from core.audit.proto_length_checker import (
    _var_has_upper_bound,
    check_proto_length,
)

# Classic recv → ntohs → malloc(len) → copy shape, with a slot for a
# guard between extraction and allocation.
_TEMPLATE = (
    "void handle(int fd, char *pkt) {\n"
    "    unsigned short len;\n"
    "    len = ntohs(*(unsigned short *)pkt);\n"
    "    %s\n"
    "    char *buf = malloc(len);\n"
    "    memcpy(buf, pkt + 2, len);\n"
    "}\n"
)


class TestUpperBoundDirection:
    def test_unbounded_length_detected(self):
        findings = check_proto_length("handle", _TEMPLATE % "")
        assert len(findings) == 1

    def test_copy_loop_bound_on_i_does_not_suppress(self):
        # `while (i < len)` bounds i, not len — symmetric acceptance
        # read it as an upper bound ON len and hid the bug behind its
        # own copy loop.
        guarded = _TEMPLATE % "int i = 0;\n    while (i < len) { i++; }"
        findings = check_proto_length("handle", guarded)
        assert len(findings) == 1

    def test_real_upper_bound_suppresses(self):
        guarded = _TEMPLATE % "if (len < 512) {"
        assert check_proto_length("handle", guarded) == []

    def test_reversed_operand_upper_bound_suppresses(self):
        guarded = _TEMPLATE % "if (512 >= len) {"
        assert check_proto_length("handle", guarded) == []

    def test_bail_check_suppresses(self):
        guarded = _TEMPLATE % "if (len > 512) { return; }"
        assert check_proto_length("handle", guarded) == []

    def test_lower_bound_does_not_suppress(self):
        guarded = _TEMPLATE % "if (len > 2) {"
        findings = check_proto_length("handle", guarded)
        assert len(findings) == 1

    def test_var_has_upper_bound_direction(self):
        src = "while (i < len) { }"
        assert _var_has_upper_bound(src, "len", len(src)) is None
        assert _var_has_upper_bound(src, "i", len(src)) == "len"
        src2 = "if (len <= 64) { }"
        assert _var_has_upper_bound(src2, "len", len(src2)) == "64"


# Cross-function (xref) scoping: the xref blob concatenates several
# independent functions, each introduced by the producer's
# "// --- caller|callee: name ---" chunk marker.
_PRIMARY_DISPATCHES = (
    "void handle(int fd, char *pkt) {\n"
    "    unsigned short len;\n"
    "    len = ntohs(*(unsigned short *)pkt);\n"
    "    dispatch(pkt, len);\n"
    "}\n"
)

_XREF_UNRELATED_GUARD = (
    "\n// --- caller: unrelated_reader ---\n"
    "void unrelated_reader(char *q) {\n"
    "    int len2 = 0;\n"
    "    if (len < 512) { len2 = 1; }\n"
    "}\n"
)

_XREF_SINK = (
    "\n// --- callee: dispatch ---\n"
    "void dispatch(char *pkt, unsigned short len) {\n"
    "    char *buf = malloc(len);\n"
    "    memcpy(buf, pkt + 2, len);\n"
    "}\n"
)

_XREF_SINK_GUARDED = (
    "\n// --- callee: dispatch ---\n"
    "void dispatch(char *pkt, unsigned short len) {\n"
    "    if (len < 512) {\n"
    "        char *buf = malloc(len);\n"
    "        memcpy(buf, pkt + 2, len);\n"
    "    }\n"
    "}\n"
)

_XREF_TWO_SINKS = (
    "\n// --- callee: dispatch ---\n"
    "void dispatch(char *pkt, unsigned short len) {\n"
    "    char *a = malloc(len);\n"
    "    memcpy(a, pkt + 2, len);\n"
    "    char *b = malloc(len);\n"
    "    memcpy(b, pkt + 4, len);\n"
    "}\n"
)


class TestXrefSegmentScoping:
    def test_xref_sink_found(self):
        findings = check_proto_length(
            "handle", _PRIMARY_DISPATCHES, xref_source=_XREF_SINK,
        )
        assert len(findings) == 1
        assert "[cross-function]" in findings[0].evidence

    def test_unrelated_xref_guard_does_not_suppress(self):
        # A same-named bound in a DIFFERENT xref function must not
        # read as a guard on this chain (suppression is the dangerous
        # direction).
        findings = check_proto_length(
            "handle", _PRIMARY_DISPATCHES,
            xref_source=_XREF_UNRELATED_GUARD + _XREF_SINK,
        )
        assert len(findings) == 1

    def test_same_segment_guard_still_suppresses(self):
        findings = check_proto_length(
            "handle", _PRIMARY_DISPATCHES, xref_source=_XREF_SINK_GUARDED,
        )
        assert findings == []

    def test_distinct_xref_findings_not_collapsed(self):
        # Two independent alloc+copy chains on the same length var
        # are two findings, not one row that hides the other.
        findings = check_proto_length(
            "handle", _PRIMARY_DISPATCHES, xref_source=_XREF_TWO_SINKS,
        )
        assert len(findings) == 2
