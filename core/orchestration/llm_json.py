"""Defensive accessors for LLM-authored JSON artifacts.

context-map.json, attack-paths.json, and their siblings are written by
an LLM and routinely arrive shape-drifted: a section rendered as a list
of STRINGS instead of objects, a scalar where a list belongs, ``null``
under a present key. ``or []`` fallbacks only catch the falsy cases —
a truthy non-list still hits ``for x in 42`` and raises, and a string
element crashes the first ``x.get(...)``.

understand_bridge hardened its own iterations with a private
``_list_at``; the sibling enrichers (mechanical sink discovery, taint
pairing, the audit bridge) consumed the same artifacts with bare
``.get()`` iteration, so one malformed entry silently killed a whole
enrichment stage (the libexec shims wrap stages in ``except
Exception``). Use these helpers at every iteration over LLM-authored
sections so the drift class stays fixed in one place.
"""

from __future__ import annotations

from typing import Any


def list_at(d: Any, key: str) -> list[Any]:
    """Return ``d[key]`` if it is a list, else an empty list."""
    if not isinstance(d, dict):
        return []
    v = d.get(key)
    return v if isinstance(v, list) else []


def dict_rows(d: Any, key: str) -> list[dict[str, Any]]:
    """The dict elements of ``d[key]`` — non-dict rows are skipped."""
    return [row for row in list_at(d, key) if isinstance(row, dict)]
