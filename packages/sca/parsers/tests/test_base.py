"""Tests for ``packages.sca.parsers._base`` — shared parser helpers."""

from __future__ import annotations

from packages.sca.parsers._base import build_purl


def test_build_purl_basic() -> None:
    assert build_purl("cargo", "serde", "1.0.0") == "pkg:cargo/serde@1.0.0"


def test_build_purl_versionless_on_none_and_empty() -> None:
    assert build_purl("npm", "lodash", None) == "pkg:npm/lodash"
    assert build_purl("npm", "lodash", "") == "pkg:npm/lodash"


def test_build_purl_namespace() -> None:
    assert (
        build_purl("maven", "spring-core", "5.0", namespace="org.springframework")
        == "pkg:maven/org.springframework/spring-core@5.0"
    )


def test_build_purl_empty_namespace_is_preserved() -> None:
    """An empty (but present) group keeps the double slash the
    per-parser copies emitted — byte-compatibility for Maven/Gradle
    rows with a blank group."""
    assert build_purl("maven", "x", "1", namespace="") == "pkg:maven//x@1"


def test_build_purl_npm_scope_verbatim() -> None:
    assert build_purl("npm", "@scope/pkg", "2.0") == "pkg:npm/@scope/pkg@2.0"


def test_lockfile_ladder_shape() -> None:
    from packages.sca.models import PinStyle
    from packages.sca.parsers._base import lockfile_confidence

    kw = {
        "git_reason": "g", "path_reason": "p",
        "unversioned_reason": "u", "resolved_reason": "r",
    }
    c = lockfile_confidence(PinStyle.GIT, "1.0", **kw)
    assert (c.level, c.reason) == ("medium", "g")
    c = lockfile_confidence(PinStyle.PATH, "1.0", **kw)
    assert (c.level, c.reason) == ("medium", "p")
    # A lockfile row without a version is anomalous -> low.
    c = lockfile_confidence(PinStyle.EXACT, None, **kw)
    assert (c.level, c.reason) == ("low", "u")
    c = lockfile_confidence(PinStyle.EXACT, "1.0", **kw)
    assert (c.level, c.reason) == ("high", "r")


def test_manifest_ladder_shape() -> None:
    from packages.sca.models import PinStyle
    from packages.sca.parsers._base import manifest_confidence

    kw = {
        "unrecognised_reason": "u", "git_path_reason": "g",
        "unpinned_reason": "w", "pinned_reason": "s",
    }
    c = manifest_confidence(PinStyle.UNKNOWN, None, **kw)
    assert (c.level, c.reason) == ("low", "u")
    c = manifest_confidence(PinStyle.GIT, "ref", **kw)
    assert (c.level, c.reason) == ("medium", "g")
    c = manifest_confidence(PinStyle.PATH, None, **kw)
    assert (c.level, c.reason) == ("medium", "g")
    # An unpinned manifest entry is normal -> medium, not low.
    c = manifest_confidence(PinStyle.CARET, None, **kw)
    assert (c.level, c.reason) == ("medium", "w")
    c = manifest_confidence(PinStyle.EXACT, "1.0", **kw)
    assert (c.level, c.reason) == ("high", "s")
