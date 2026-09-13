"""Shared building blocks for manifest parsers.

Every parser used to hand-roll the same purl assembly; this module is
the single implementation. Per-ecosystem semantics stay at the call
site as explicit arguments (purl type, pre-canonicalised name,
namespace segment) — the shared helper only owns the assembly rule,
so an ecosystem can never be silently unified into another's
behavior.
"""

from __future__ import annotations

from ..models import Confidence, PinStyle


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


def lockfile_confidence(
    pin_style: PinStyle,
    version: str | None,
    *,
    git_reason: str,
    path_reason: str,
    unversioned_reason: str,
    resolved_reason: str,
) -> Confidence:
    """The lockfile-family pin-confidence ladder: GIT source ->
    medium, PATH source -> medium, entry without version -> low,
    resolved entry -> high.

    The ladder is shared; the reason strings are the caller's
    per-ecosystem wording (a lockfile row without a version is
    anomalous, hence low — contrast the manifest ladder where
    unpinned is normal and only demotes to medium).
    """
    if pin_style is PinStyle.GIT:
        return Confidence("medium", reason=git_reason)
    if pin_style is PinStyle.PATH:
        return Confidence("medium", reason=path_reason)
    if version is None:
        return Confidence("low", reason=unversioned_reason)
    return Confidence("high", reason=resolved_reason)


def manifest_confidence(
    pin_style: PinStyle,
    version: str | None,
    *,
    unrecognised_reason: str,
    git_path_reason: str,
    unpinned_reason: str,
    pinned_reason: str,
) -> Confidence:
    """The manifest-family pin-confidence ladder: unrecognised spec
    -> low, git/path source -> medium, unpinned / wildcard -> medium,
    structured spec -> high.

    Reason strings are the caller's per-ecosystem wording.
    """
    if pin_style is PinStyle.UNKNOWN:
        return Confidence("low", reason=unrecognised_reason)
    if pin_style in (PinStyle.GIT, PinStyle.PATH):
        return Confidence("medium", reason=git_path_reason)
    if version is None:
        return Confidence("medium", reason=unpinned_reason)
    return Confidence("high", reason=pinned_reason)


__all__ = [
    "build_purl",
    "lockfile_confidence",
    "manifest_confidence",
]
