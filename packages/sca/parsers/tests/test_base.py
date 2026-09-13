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
