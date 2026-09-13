"""Shared building blocks for manifest parsers.

Every parser used to hand-roll the same purl assembly; this module is
the single implementation. Per-ecosystem semantics stay at the call
site as explicit arguments (purl type, pre-canonicalised name,
namespace segment) — the shared helper only owns the assembly rule,
so an ecosystem can never be silently unified into another's
behavior.
"""

from __future__ import annotations


def build_purl(
    purl_type: str,
    name: str,
    version: str | None,
    *,
    namespace: str | None = None,
) -> str:
    """Assemble ``pkg:<type>/[<namespace>/]<name>[@<version>]``.

    ``name`` is spliced verbatim — per-ecosystem canonicalisation
    (PEP 503 for PyPI, scoped ``@`` preserved for npm) is the
    CALLER's job, exactly as it was when each parser owned its own
    copy. ``namespace`` is the group segment for two-part
    coordinates (Maven ``group/artifact``). A falsy ``version``
    (None or empty) yields a version-less purl.
    """
    base = (
        f"pkg:{purl_type}/{namespace}/{name}"
        if namespace is not None
        else f"pkg:{purl_type}/{name}"
    )
    if version:
        return f"{base}@{version}"
    return base


__all__ = ["build_purl"]
