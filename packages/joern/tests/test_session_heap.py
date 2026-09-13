"""joern_session must pass the operator's heap tunable to the CPG
build — the joern-parse JVM on a large target is exactly where
joern_heap_mb matters, not just the query server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import packages.joern as joern_pkg
from packages.joern.tunables import JoernTunables


def _run_session(tmp_path: Path, tunables: JoernTunables, use_cache: bool):
    captured: dict = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        cpg = MagicMock()
        cpg.exists.return_value = False
        return cpg

    server = MagicMock()
    with patch.object(joern_pkg, "is_available", return_value=True), \
            patch.object(joern_pkg.JoernServer, "from_tunables",
                         return_value=server), \
            patch.object(joern_pkg, "build_cpg_cached",
                         side_effect=fake_build), \
            patch.object(joern_pkg, "build_cpg", side_effect=fake_build):
        with joern_pkg.joern_session(
            tmp_path,
            cache_dir=(tmp_path if use_cache else None),
            tunables=tunables,
            register_reach_audit=False,
        ) as srv:
            assert srv is server
    return captured


def test_cached_build_receives_heap_mb(tmp_path):
    captured = _run_session(
        tmp_path, JoernTunables(heap_mb=8192), use_cache=True)
    assert captured.get("heap_mb") == 8192


def test_uncached_build_receives_heap_mb(tmp_path):
    captured = _run_session(
        tmp_path, JoernTunables(heap_mb=4096), use_cache=False)
    assert captured.get("heap_mb") == 4096
