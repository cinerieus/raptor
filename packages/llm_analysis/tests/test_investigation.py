from packages.llm_analysis.investigation import (
    build_investigation_graph,
    invariant_candidates_from_context_map,
)
from packages.llm_analysis.tasks import ChainAnalysisTask


def _finding(fid, vuln_type, path):
    return {
        "finding_id": fid,
        "file_path": path,
        "start_line": 10,
        "rule_id": vuln_type,
        "code": "dangerous(value)",
        "dataflow": {
            "source": {"file": path, "line": 2},
            "sink": {"file": path, "line": 10},
        },
    }


def _result(vuln_type, *, exploitable=True, confidence="high"):
    return {
        "vuln_type": vuln_type,
        "is_true_positive": exploitable,
        "is_exploitable": exploitable,
        "confidence": confidence,
        "reasoning": f"confirmed {vuln_type}",
        "confirmed_edges": ["source->sink"] if exploitable else [],
    }


def test_builds_file_write_to_dynamic_load_chain():
    findings = [
        _finding("F-1", "arbitrary_file_write", "upload.py"),
        _finding("F-2", "dynamic_load", "plugins.py"),
    ]
    graph = build_investigation_graph(findings, {
        "F-1": _result("arbitrary_file_write"),
        "F-2": _result("dynamic_load"),
    })

    assert len(graph["evidence_packs"]) == 2
    assert len(graph["primitives"]) == 2
    assert graph["attack_chains"][0]["goal"] == "code_execution"
    assert graph["attack_chains"][0]["status"] == "candidate"


def test_high_confidence_rejection_prunes_dependent_chain():
    findings = [
        _finding("F-1", "arbitrary_file_write", "upload.py"),
        _finding("F-2", "dynamic_load", "plugins.py"),
    ]
    graph = build_investigation_graph(findings, {
        "F-1": _result("arbitrary_file_write", exploitable=False),
        "F-2": _result("dynamic_load"),
    })

    assert graph["attack_chains"][0]["status"] == "pruned"
    assert graph["attack_chains"][0]["blocked_by"]


def test_low_confidence_negative_does_not_prune():
    findings = [
        _finding("F-1", "arbitrary_file_write", "upload.py"),
        _finding("F-2", "dynamic_load", "plugins.py"),
    ]
    graph = build_investigation_graph(findings, {
        "F-1": _result(
            "arbitrary_file_write", exploitable=False, confidence="low",
        ),
        "F-2": _result("dynamic_load"),
    })

    assert graph["attack_chains"][0]["status"] == "candidate"


def test_chain_task_skips_pruned_and_caps_paid_reviews():
    task = ChainAnalysisTask()
    chains = [
        {"id": f"CHAIN-{index:04d}", "status": "candidate", "priority": 1}
        for index in range(20)
    ] + [{"id": "BLOCKED", "status": "pruned", "priority": 9}]

    selected = task.select_items(chains, {})

    assert len(selected) == 10
    assert all(item["id"] != "BLOCKED" for item in selected)


def test_invariant_seed_requires_real_in_target_source(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("\n".join(["safe = 1"] * 9 + ["run(user)"]))
    context_map = {
        "entry_points": [{"id": "EP-1", "file": "app.py", "line": 1}],
        "sink_details": [{
            "id": "SINK-1", "type": "shell_exec", "file": "app.py",
            "line": 10, "reaches_from": ["EP-1"],
        }],
    }

    candidates = invariant_candidates_from_context_map(
        context_map, tmp_path, [],
    )

    assert len(candidates) == 1
    assert candidates[0]["metadata"]["invariant_seed"] is True
    assert "run(user)" in candidates[0]["code"]


def test_invariant_seed_rejects_path_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("run(user)\n")
    context_map = {"sink_details": [{
        "id": "SINK-1", "type": "shell_exec", "file": f"../{outside.name}",
        "line": 1, "reaches_from": ["EP-1"],
    }]}

    assert invariant_candidates_from_context_map(
        context_map, tmp_path, [],
    ) == []
