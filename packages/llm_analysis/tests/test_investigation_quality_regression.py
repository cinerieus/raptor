"""Labeled regression corpus for investigation-chain selection quality."""

from packages.llm_analysis.investigation import build_investigation_graph


def _finding(fid, vuln_type, bridge):
    return {
        "finding_id": fid,
        "file_path": f"{fid}.py",
        "start_line": 10,
        "rule_id": vuln_type,
        "dataflow": {
            "source": {"variable": bridge},
            "sink": {"variable": bridge},
        },
    }


def _result(vuln_type, disposition):
    if disposition == "confirmed":
        return {
            "vuln_type": vuln_type,
            "is_true_positive": True,
            "is_exploitable": True,
            "confidence": "high",
        }
    if disposition == "rejected":
        return {
            "vuln_type": vuln_type,
            "is_true_positive": False,
            "is_exploitable": False,
            "confidence": "high",
        }
    return {
        "vuln_type": vuln_type,
        "is_true_positive": False,
        "is_exploitable": False,
        "confidence": "low",
    }


_CASES = (
    {
        "name": "confirmed write reaches plugin loader",
        "left": ("arbitrary_file_write", "plugin-path", "confirmed"),
        "right": ("dynamic_load", "plugin-path", "confirmed"),
        "chain_should_receive_review": True,
    },
    {
        "name": "confirmed auth bypass reaches command handler",
        "left": ("auth_bypass", "admin-route", "confirmed"),
        "right": ("command_execution", "admin-route", "confirmed"),
        "chain_should_receive_review": True,
    },
    {
        "name": "unrelated write and plugin loader",
        "left": ("arbitrary_file_write", "upload-path", "confirmed"),
        "right": ("dynamic_load", "plugin-path", "confirmed"),
        "chain_should_receive_review": False,
    },
    {
        "name": "rejected write reaches plugin loader",
        "left": ("arbitrary_file_write", "plugin-path", "rejected"),
        "right": ("dynamic_load", "plugin-path", "confirmed"),
        "chain_should_receive_review": False,
    },
    {
        "name": "uncertain write reaches plugin loader",
        "left": ("arbitrary_file_write", "plugin-path", "uncertain"),
        "right": ("dynamic_load", "plugin-path", "confirmed"),
        "chain_should_receive_review": True,
    },
)


def _new_selector(case):
    left_type, left_bridge, left_disposition = case["left"]
    right_type, right_bridge, right_disposition = case["right"]
    findings = [
        _finding("F-1", left_type, left_bridge),
        _finding("F-2", right_type, right_bridge),
    ]
    results = {
        "F-1": _result(left_type, left_disposition),
        "F-2": _result(right_type, right_disposition),
    }
    graph = build_investigation_graph(findings, results)
    return any(
        chain["status"] == "candidate"
        for chain in graph["attack_chains"]
    )


def _metrics(predictions):
    labels = [case["chain_should_receive_review"] for case in _CASES]
    true_positive = sum(predicted and label
                        for predicted, label in zip(predictions, labels))
    false_positive = sum(predicted and not label
                         for predicted, label in zip(predictions, labels))
    false_negative = sum(not predicted and label
                         for predicted, label in zip(predictions, labels))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return precision, recall, sum(predictions)


def test_dependency_selection_improves_labeled_candidate_precision():
    # The previous broad correlation sent every compatible capability pair to
    # review. All five corpus entries therefore model a legacy paid candidate.
    legacy = _metrics([True] * len(_CASES))
    dependency_aware = _metrics([_new_selector(case) for case in _CASES])

    assert legacy == (0.6, 1.0, 5)
    assert dependency_aware == (1.0, 1.0, 3)
    assert dependency_aware[0] > legacy[0]
    assert dependency_aware[1] == legacy[1]
    assert dependency_aware[2] < legacy[2]
