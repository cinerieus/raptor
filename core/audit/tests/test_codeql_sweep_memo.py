"""Tests for the whole-database CodeQL result memo in run_codeql_sweep.

The memo collapses N per-hypothesis ``codeql database analyze``
invocations over the same (database, query) pair into one: the parsed
whole-DB result set is cached, and later hypotheses filter it for
their own file/line window without re-running the CLI. Correctness
hinges on the key embedding content stamps for both the database and
the query, so these tests drive the stamp behaviour as hard as the
hit path.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.audit.sweep as sweep_mod
from core.audit.run_memo import BoundedMemo
from core.audit.sweep import run_codeql_sweep


@pytest.fixture(autouse=True)
def _fresh_memo():
    sweep_mod._reset_codeql_memo()
    yield
    sweep_mod._reset_codeql_memo()


def _sarif_result(uri: str, line: int) -> dict:
    return {
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": uri},
                "region": {"startLine": line},
            },
        }],
    }


def _counting_analyze(results: list[dict], calls: list[dict]):
    """Stand-in for codeql_augmented_run.analyze recording each call."""

    def fake(db_path, queries, output_path, *, extension_pack=None,
             codeql_bin="codeql", timeout_seconds=0, runner=None,
             extra_args=()):
        calls.append({"db": db_path, "queries": tuple(queries)})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"runs": [{"results": results}]}), encoding="utf-8",
        )
        return SimpleNamespace(
            sarif_path=output_path,
            queries=tuple(queries),
            extension_pack=extension_pack,
            elapsed_seconds=0.0,
        )

    return fake


def _make_db(tmp_path: Path, manifest: str = "sourceLocationPrefix: /src\n") -> Path:
    db = tmp_path / "codeql-db"
    db.mkdir(exist_ok=True)
    (db / "codeql-database.yml").write_text(manifest, encoding="utf-8")
    return db


def _make_query(tmp_path: Path, text: str = "select 1", name: str = "q.ql") -> Path:
    query = tmp_path / name
    query.write_text(text, encoding="utf-8")
    return query


def _sweep(tmp_path: Path, query: Path, db: Path, *, file_path: str = "a.c",
           line_start: int = 10, line_end: int = 20):
    return run_codeql_sweep(
        target_path=tmp_path,
        file_path=file_path,
        function_name="foo",
        query_path=str(query),
        database_path=str(db),
        line_start=line_start,
        line_end=line_end,
    )


class TestMemoHit:
    def test_second_hypothesis_skips_analyze(self, tmp_path: Path, monkeypatch):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12), _sarif_result("src/a.c", 99)],
            calls,
        ))

        first = _sweep(tmp_path, query, db, line_start=10, line_end=20)
        second = _sweep(tmp_path, query, db, line_start=90, line_end=110)

        assert len(calls) == 1
        assert first.outcome == "confirmed"
        assert second.outcome == "confirmed"
        assert first.matches != second.matches  # per-window filtering intact

    def test_hit_serves_refutation_from_cached_results(
        self, tmp_path: Path, monkeypatch,
    ):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], calls,
        ))

        assert _sweep(tmp_path, query, db).outcome == "confirmed"
        miss = _sweep(tmp_path, query, db, line_start=500, line_end=600)
        assert miss.outcome == "refuted"
        assert len(calls) == 1

    def test_matches_are_isolated_copies(self, tmp_path: Path, monkeypatch):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], [],
        ))

        first = _sweep(tmp_path, query, db)
        first.matches[0]["locations"][0]["physicalLocation"]["region"]["startLine"] = 999
        second = _sweep(tmp_path, query, db)
        assert second.outcome == "confirmed"
        region = second.matches[0]["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 12


class TestMemoMiss:
    def test_db_manifest_change_misses(self, tmp_path: Path, monkeypatch):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], calls,
        ))

        _sweep(tmp_path, query, db)
        _make_db(tmp_path, manifest="sourceLocationPrefix: /rebuilt\n")
        _sweep(tmp_path, query, db)
        assert len(calls) == 2

    def test_query_content_change_misses(self, tmp_path: Path, monkeypatch):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], calls,
        ))

        _sweep(tmp_path, query, db)
        query.write_text("select 2", encoding="utf-8")
        _sweep(tmp_path, query, db)
        assert len(calls) == 2

    def test_different_db_paths_do_not_share(self, tmp_path: Path, monkeypatch):
        db_a = _make_db(tmp_path)
        db_b = tmp_path / "codeql-db-b"
        db_b.mkdir()
        (db_b / "codeql-database.yml").write_text(
            "sourceLocationPrefix: /src\n", encoding="utf-8",
        )
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], calls,
        ))

        _sweep(tmp_path, query, db_a)
        _sweep(tmp_path, query, db_b)
        assert len(calls) == 2


class TestErrorHandling:
    def test_analyze_failure_is_not_cached(self, tmp_path: Path, monkeypatch):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []

        def flaky(db_path, queries, output_path, **kwargs):
            calls.append({})
            if len(calls) == 1:
                msg = "codeql analyze exited 2"
                raise RuntimeError(msg)
            return _counting_analyze(
                [_sarif_result("src/a.c", 12)], [],
            )(db_path, queries, output_path, **kwargs)

        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", flaky)

        first = _sweep(tmp_path, query, db)
        assert first.outcome == "error"
        second = _sweep(tmp_path, query, db)
        assert second.outcome == "confirmed"
        assert len(calls) == 2

    def test_unreadable_sarif_keeps_error_shape_and_is_not_cached(
        self, tmp_path: Path, monkeypatch,
    ):
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []

        def truncated(db_path, queries, output_path, **kwargs):
            calls.append({})
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("{not json", encoding="utf-8")
            return SimpleNamespace(sarif_path=output_path)

        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", truncated)

        result = _sweep(tmp_path, query, db)
        assert result.outcome == "error"
        assert any("SARIF" in e for e in result.errors)
        _sweep(tmp_path, query, db)
        assert len(calls) == 2


class TestBoundedMemo:
    def test_bound_evicts_oldest(self):
        memo: BoundedMemo[int] = BoundedMemo(2)
        memo.get_or_compute(("a",), lambda: 1)
        memo.get_or_compute(("b",), lambda: 2)
        memo.get_or_compute(("c",), lambda: 3)
        assert len(memo) == 2
        # "a" evicted → recompute; "c" still cached.
        value, cached = memo.get_or_compute(("a",), lambda: 10)
        assert (value, cached) == (10, False)
        value, cached = memo.get_or_compute(("c",), lambda: 30)
        assert (value, cached) == (3, True)

    def test_none_key_always_computes(self):
        memo: BoundedMemo[int] = BoundedMemo(2)
        counter = {"n": 0}

        def compute() -> int:
            counter["n"] += 1
            return counter["n"]

        assert memo.get_or_compute(None, compute) == (1, False)
        assert memo.get_or_compute(None, compute) == (2, False)
        assert len(memo) == 0

    def test_exception_caches_nothing(self):
        memo: BoundedMemo[int] = BoundedMemo(2)

        def boom() -> int:
            msg = "tool failed"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError):
            memo.get_or_compute(("k",), boom)
        value, cached = memo.get_or_compute(("k",), lambda: 7)
        assert (value, cached) == (7, False)

    def test_concurrent_same_key_collapses_to_one_compute(self):
        memo: BoundedMemo[str] = BoundedMemo(4)
        entered = threading.Event()
        release = threading.Event()
        computes: list[int] = []

        def slow_compute() -> str:
            computes.append(1)
            entered.set()
            release.wait(timeout=10)
            return "value"

        results: list[tuple[str, bool]] = []

        def worker() -> None:
            results.append(memo.get_or_compute(("k",), slow_compute))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads[:1]:
            t.start()
        assert entered.wait(timeout=10)
        for t in threads[1:]:
            t.start()
        release.set()
        for t in threads:
            t.join(timeout=10)
        assert len(computes) == 1
        assert all(value == "value" for value, _ in results)
        assert sum(1 for _, cached in results if cached) == 3


class TestBoundedMemoBounds:
    def test_capacity_boundary_is_exact(self):
        """Two-direction cap binding: AT the cap every entry is
        retained (all hits); ONE past evicts exactly the LRU entry."""
        memo: BoundedMemo[int] = BoundedMemo(3)
        computes: list[int] = []

        def make(i: int):
            def compute() -> int:
                computes.append(i)
                return i
            return compute

        for i in range(3):
            memo.get_or_compute(("k", i), make(i))
        # At the cap: all three hit.
        for i in range(3):
            _, cached = memo.get_or_compute(("k", i), make(i))
            assert cached
        assert computes == [0, 1, 2]
        # One past: LRU (0) evicted, 1..3 retained.
        memo.get_or_compute(("k", 3), make(3))
        assert len(memo) == 3
        for i in (1, 2, 3):
            _, cached = memo.get_or_compute(("k", i), make(i))
            assert cached
        _, cached = memo.get_or_compute(("k", 0), make(0))
        assert not cached
        assert computes == [0, 1, 2, 3, 0]

    def test_min_capacity_validated(self):
        with pytest.raises(ValueError):
            BoundedMemo(0)
        assert len(BoundedMemo(1)) == 0

    def test_failed_compute_releases_its_key_lock(self):
        """Failed computes store no value, so eviction would never
        reclaim their key locks — always-failing keys must not grow
        ``_key_locks`` without bound."""
        memo: BoundedMemo[int] = BoundedMemo(4)

        def boom() -> int:
            raise RuntimeError("compute failed")

        for i in range(50):
            with pytest.raises(RuntimeError):
                memo.get_or_compute(("k", i % 5), boom)
        assert len(memo._key_locks) == 0
        # A later success on the same key still works and is cached.
        value, cached = memo.get_or_compute(("k", 0), lambda: 7)
        assert (value, cached) == (7, False)
        _, cached = memo.get_or_compute(("k", 0), lambda: 7)
        assert cached


class TestMemoLifetime:
    def test_caller_supplied_memo_is_used(
        self, tmp_path: Path, monkeypatch,
    ):
        """The orchestrator hands its run's ``config.codeql_memo``
        through the kwarg; entries must land there, hit there, and
        never leak into the module-level default instance."""
        db = _make_db(tmp_path)
        query = _make_query(tmp_path)
        calls: list[dict] = []
        import core.dataflow.codeql_augmented_run as car
        monkeypatch.setattr(car, "analyze", _counting_analyze(
            [_sarif_result("src/a.c", 12)], calls,
        ))
        memo: BoundedMemo = BoundedMemo(4)

        def sweep():
            return run_codeql_sweep(
                target_path=tmp_path,
                file_path="a.c",
                function_name="foo",
                query_path=str(query),
                database_path=str(db),
                line_start=10,
                line_end=20,
                memo=memo,
            )

        assert sweep().outcome == "confirmed"
        assert sweep().outcome == "confirmed"
        assert len(calls) == 1
        assert len(memo) == 1
        assert len(sweep_mod._codeql_memo) == 0

    def test_module_memo_binds_to_the_documented_cap(self):
        assert (
            sweep_mod._codeql_memo._max_entries
            == sweep_mod._CODEQL_MEMO_MAX_ENTRIES
        )

    def test_documented_cap_value_is_pinned(self):
        """Value pin (churn-prone-limits doctrine): the binding
        assertions above follow the constant, so mutating it would
        move both sides and stay green — the pin makes a cap change a
        deliberate, test-visible act. Both directions of the trade:
        raising it holds more whole-DB result lists (potentially many
        MB each on alert-dense targets) for the memo's lifetime;
        lowering it re-pays multi-minute ``database analyze`` runs as
        soon as an audit rotates through more (db, query) pairs than
        fit.
        """
        assert sweep_mod._CODEQL_MEMO_MAX_ENTRIES == 16

    def test_run_config_memo_binds_to_the_same_cap_per_run(self):
        from core.audit.orchestrator import OrchestratorConfig

        a = OrchestratorConfig(target_path=Path("/x"), out_dir=Path("/x"))
        b = OrchestratorConfig(target_path=Path("/x"), out_dir=Path("/x"))
        assert (
            a.codeql_memo._max_entries
            == sweep_mod._CODEQL_MEMO_MAX_ENTRIES
        )
        assert a.codeql_memo is not b.codeql_memo
        assert a.codeql_memo is not sweep_mod._codeql_memo


class TestDbStamp:
    def test_manifest_preferred(self, tmp_path: Path):
        db = _make_db(tmp_path)
        stamp = sweep_mod._codeql_db_stamp(db)
        assert stamp is not None
        assert stamp[0] == "manifest"

    def test_unreadable_manifest_returns_none(self, tmp_path: Path):
        """No manifest → no stamp → the sweep runs UNCACHED. A dirstat
        fallback here would miss nested in-place changes (a directory
        mtime only tracks direct-child churn) and serve stale
        results, so uncached is the only safe degradation."""
        db = tmp_path / "bare-db"
        db.mkdir()
        assert sweep_mod._codeql_db_stamp(db) is None

    def test_missing_db_returns_none(self, tmp_path: Path):
        assert sweep_mod._codeql_db_stamp(tmp_path / "nope") is None

    def test_unreadable_query_returns_none(self, tmp_path: Path):
        assert sweep_mod._codeql_query_stamp(tmp_path / "nope.ql") is None
