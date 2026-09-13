"""The ``.validated`` skip-cache of ``libexec/raptor-validate-schema``
must be keyed by schema type, not just filename + content hash.

A file validated as one type must not short-circuit a later validation
request under a different type — that would claim a pass for a check
that never ran.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "libexec" / "raptor-validate-schema"


@pytest.fixture(scope="module")
def validator():
    prior = os.environ.get("_RAPTOR_TRUSTED")
    os.environ["_RAPTOR_TRUSTED"] = "1"
    try:
        loader = importlib.machinery.SourceFileLoader(
            "raptor_validate_schema", str(SCRIPT),
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        yield mod
    finally:
        if prior is None:
            os.environ.pop("_RAPTOR_TRUSTED", None)
        else:
            os.environ["_RAPTOR_TRUSTED"] = prior


class TestTypeAwareCache:
    def test_same_type_hits_cache(self, validator, tmp_path: Path):
        p = tmp_path / "foo.json"
        p.write_text("{}", encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p), "findings",
        )
        assert validator.is_already_validated(p, "findings")

    def test_different_type_misses_cache(self, validator, tmp_path: Path):
        p = tmp_path / "foo.json"
        p.write_text("{}", encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p), "findings",
        )
        assert not validator.is_already_validated(p, "attack-tree")

    def test_content_change_misses_cache(self, validator, tmp_path: Path):
        p = tmp_path / "foo.json"
        p.write_text("{}", encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p), "findings",
        )
        p.write_text('{"changed": true}', encoding="utf-8")
        assert not validator.is_already_validated(p, "findings")

    def test_legacy_typeless_record_not_claimed_for_type(
        self, validator, tmp_path: Path,
    ):
        # A pre-existing type-less record must trigger one
        # re-validation under the typed lookup rather than being
        # claimed for an arbitrary type.
        p = tmp_path / "foo.json"
        p.write_text("{}", encoding="utf-8")
        validator._write_validated(tmp_path, p.name, validator._hash_file(p))
        assert validator.is_already_validated(p)
        assert not validator.is_already_validated(p, "findings")

    def test_typed_records_roundtrip_through_index(
        self, validator, tmp_path: Path,
    ):
        # Two files with different types coexist in one .validated.
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("{}", encoding="utf-8")
        b.write_text("[]", encoding="utf-8")
        validator._write_validated(
            tmp_path, a.name, validator._hash_file(a), "findings",
        )
        validator._write_validated(
            tmp_path, b.name, validator._hash_file(b), "attack-paths",
        )
        assert validator.is_already_validated(a, "findings")
        assert validator.is_already_validated(b, "attack-paths")
        assert not validator.is_already_validated(a, "attack-paths")


class TestHelperReaderRoundTrip:
    """The validation helper's ``_is_validated`` reads the SAME
    ``.validated`` index this writer produces — the two hand-rolled
    parsers drifted once (writer gained the ``type:hash`` form, the
    reader kept comparing the whole value against a bare hash, so the
    documented skip never fired and every stage prep re-validated
    every doc). This pins the cross-tool round trip."""

    @staticmethod
    @pytest.fixture(scope="class")
    def helper():
        prior = os.environ.get("_RAPTOR_TRUSTED")
        os.environ["_RAPTOR_TRUSTED"] = "1"
        try:
            script = REPO_ROOT / "libexec" / "raptor-validation-helper"
            loader = importlib.machinery.SourceFileLoader(
                "raptor_validation_helper_rt", str(script),
            )
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            yield mod
        finally:
            if prior is None:
                os.environ.pop("_RAPTOR_TRUSTED", None)
            else:
                os.environ["_RAPTOR_TRUSTED"] = prior

    def test_typed_record_recognised_by_helper(
        self, validator, helper, tmp_path: Path,
    ):
        p = tmp_path / "findings.json"
        p.write_text('{"findings": []}', encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p), "findings",
        )
        assert helper._is_validated(str(tmp_path), p.name)

    def test_legacy_bare_hash_recognised_by_helper(
        self, validator, helper, tmp_path: Path,
    ):
        p = tmp_path / "findings.json"
        p.write_text('{"findings": []}', encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p),
        )
        assert helper._is_validated(str(tmp_path), p.name)

    def test_stale_hash_not_recognised(
        self, validator, helper, tmp_path: Path,
    ):
        p = tmp_path / "findings.json"
        p.write_text('{"findings": []}', encoding="utf-8")
        validator._write_validated(
            tmp_path, p.name, validator._hash_file(p), "findings",
        )
        p.write_text('{"findings": [1]}', encoding="utf-8")
        assert not helper._is_validated(str(tmp_path), p.name)
