"""Deterministic evidence packs, invariants, primitives, and attack chains.

These artifacts turn per-finding verdicts into a compact investigation graph.
They are deliberately additive to the established report schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CAPABILITY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("command_execution", (
        "command_execution", "command_injection", "command execution",
        "cwe-78", "rce",
    )),
    ("file_write", ("file_write", "arbitrary file write", "path traversal write", "cwe-73")),
    ("file_read", ("file_read", "lfi", "path_traversal", "cwe-22", "cwe-23")),
    ("network_fetch", ("ssrf", "server-side request", "cwe-918")),
    ("auth_bypass", ("auth_bypass", "authorization", "idor", "cwe-285", "cwe-639")),
    ("sql_query", ("sql_injection", "sqli", "cwe-89")),
    ("dynamic_load", ("dynamic_load", "plugin", "import", "cwe-94")),
    ("unsafe_deserialization", ("deserial", "pickle", "cwe-502")),
    ("template_execution", ("template_injection", "ssti", "cwe-1336")),
    ("archive_extraction", ("archive", "zip slip", "tar slip")),
    ("memory_corruption", ("buffer_overflow", "use_after_free", "out-of-bounds", "cwe-119", "cwe-416")),
)

_CHAIN_RULES: tuple[tuple[str, str, str], ...] = (
    ("file_write", "dynamic_load", "code_execution"),
    ("file_write", "template_execution", "code_execution"),
    ("file_write", "unsafe_deserialization", "code_execution"),
    ("file_write", "command_execution", "code_execution"),
    ("archive_extraction", "dynamic_load", "code_execution"),
    ("archive_extraction", "template_execution", "code_execution"),
    ("network_fetch", "auth_bypass", "internal_privilege"),
    ("network_fetch", "file_read", "credential_or_secret_access"),
    ("auth_bypass", "command_execution", "code_execution"),
    ("auth_bypass", "file_write", "privileged_file_write"),
    ("sql_query", "auth_bypass", "account_or_tenant_takeover"),
    ("file_read", "auth_bypass", "credential_reuse"),
)


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "[nested data omitted]"
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _compact(item, depth=depth + 1)
            for key, item in list(value.items())[:30]
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def invariant_candidates_from_context_map(
    context_map: dict[str, Any] | None,
    repo_path: Path,
    existing_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert mapped source-to-sink paths into source-verified scan leads.

    Context maps are hint-tier LLM output. A seed is emitted only when its
    referenced file and line resolve inside the target and source can be read.
    The normal analysis and mechanical validation stages still decide it.
    """
    if not isinstance(context_map, dict):
        return []
    root = Path(repo_path).resolve()
    occupied = {
        (str(item.get("file_path") or ""), int(item.get("start_line") or 0))
        for item in existing_findings if isinstance(item, dict)
    }
    entries = {
        str(item.get("id")): item for item in context_map.get("entry_points") or []
        if isinstance(item, dict) and item.get("id")
    }
    unchecked_sink_ids = {
        str(item.get("sink")) for item in context_map.get("unchecked_flows") or []
        if isinstance(item, dict) and item.get("sink")
    }
    candidates: list[dict[str, Any]] = []
    for sink in context_map.get("sink_details") or []:
        if not isinstance(sink, dict):
            continue
        sink_id = str(sink.get("id") or "")
        reaches = [str(value) for value in sink.get("reaches_from") or []]
        if not reaches and sink_id not in unchecked_sink_ids:
            continue
        relative = str(sink.get("file") or "")
        try:
            line = int(sink.get("line") or 0)
            resolved = (root / relative).resolve()
            resolved.relative_to(root)
            source_lines = resolved.read_text(
                encoding="utf-8", errors="replace",
            ).splitlines()
        except (OSError, TypeError, ValueError):
            continue
        if line < 1 or line > len(source_lines) or (relative, line) in occupied:
            continue
        start = max(1, line - 12)
        end = min(len(source_lines), line + 12)
        entry = entries.get(reaches[0]) if reaches else None
        candidates.append({
            "finding_id": f"INV-SEED-{len(candidates) + 1:04d}",
            "rule_id": f"raptor.invariant.{sink.get('type') or 'capability'}",
            "rule_name": "Invariant-first source-to-sink candidate",
            "file_path": relative,
            "start_line": line,
            "end_line": line,
            "level": "warning",
            "message": (
                f"Mapped entry point may reach {sink.get('type') or 'sensitive'} "
                f"sink {sink_id}; verify attacker control and guards"
            ),
            "code": "\n".join(source_lines[start - 1:end]),
            "surrounding_context": "\n".join(source_lines[start - 1:end]),
            "has_dataflow": bool(entry),
            "dataflow": {
                "source": entry or {},
                "steps": [],
                "sink": sink,
                "sanitizers_found": [],
            },
            "metadata": {"invariant_seed": True, "sink_id": sink_id},
        })
        if len(candidates) >= 20:
            break
    return candidates


def _capabilities(finding: dict[str, Any], result: dict[str, Any]) -> list[str]:
    haystack = " ".join(str(value).lower() for value in (
        result.get("vuln_type"), result.get("cwe_id"), result.get("impact"),
        result.get("attack_scenario"), finding.get("rule_id"),
        finding.get("rule_name"), finding.get("message"),
    ) if value)
    return [name for name, terms in _CAPABILITY_TERMS
            if any(term in haystack for term in terms)]


def _decision_status(result: dict[str, Any]) -> str:
    status = str(result.get("evidence_status") or "").lower()
    if result.get("is_exploitable") is True:
        return "confirmed"
    if status == "rejected":
        return "rejected"
    confidence = str(result.get("confidence") or "").lower()
    if result.get("is_true_positive") is False and confidence == "high":
        return "rejected"
    return "needs_more_evidence"


_EDGE_KEYS = frozenset({
    "endpoint", "function", "id", "name", "path", "symbol",
    "variable",
})


def _edge_refs(value: Any) -> set[str]:
    """Extract exact structured references suitable for joining two flows."""
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _EDGE_KEYS and isinstance(item, (str, int)):
                text = str(item).strip().lower()
                if len(text) >= 3:
                    refs.add(f"{key}:{text}")
            if isinstance(item, (dict, list)):
                refs.update(_edge_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_edge_refs(item))
    return refs


def _dependency_refs(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Return concrete identifiers joining the first sink to second input."""
    produced = _edge_refs(first.get("sink")) | _edge_refs(
        first.get("confirmed_edges")
    )
    consumed = _edge_refs(second.get("source")) | _edge_refs(
        second.get("steps")
    ) | _edge_refs(second.get("confirmed_edges"))
    return sorted(produced & consumed)


def build_investigation_graph(
    findings: list[dict[str, Any]], results_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build compact evidence packs and dependency-aware chain candidates."""
    findings_by_id = {
        str(item.get("finding_id")): item for item in findings
        if item.get("finding_id")
    }
    packs: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []

    for fid in sorted(findings_by_id):
        finding = findings_by_id[fid]
        result = results_by_id.get(fid, {})
        if not isinstance(result, dict) or result.get("error"):
            continue
        capabilities = _capabilities(finding, result)
        dataflow = finding.get("dataflow") or {}
        pack_id = f"PACK-{len(packs) + 1:04d}"
        packs.append({
            "id": pack_id,
            "finding_id": fid,
            "location": {
                "file": finding.get("file_path"),
                "start_line": finding.get("start_line"),
                "end_line": finding.get("end_line"),
            },
            "claim": str(
                result.get("reasoning") or finding.get("message") or ""
            )[:4000],
            "capabilities": capabilities,
            "source": _compact(dataflow.get("source")),
            "steps": _compact(dataflow.get("steps") or []),
            "sink": _compact(dataflow.get("sink")),
            "sanitizers": _compact(dataflow.get("sanitizers_found") or []),
            "confirmed_edges": _compact(result.get("confirmed_edges") or []),
            "missing_edges": _compact(result.get("missing_edges") or []),
            "required_preconditions": _compact(
                result.get("required_preconditions")
                or result.get("prerequisites") or []
            ),
            "decision": _decision_status(result),
            "confidence": result.get("confidence"),
            "code_slice": str(
                finding.get("vulnerable_code") or finding.get("code") or ""
            )[:6000],
        })
        for capability in capabilities:
            primitive_id = f"PRIM-{len(primitives) + 1:04d}"
            primitives.append({
                "id": primitive_id,
                "finding_id": fid,
                "capability": capability,
                "evidence_pack": pack_id,
                "viability": _decision_status(result),
            })
            invariants.append({
                "id": f"INV-{len(invariants) + 1:04d}",
                "class": f"{capability}_boundary",
                "finding_id": fid,
                "statement": (
                    f"Attacker-controlled data must not reach the "
                    f"{capability} capability without an effective guard."
                ),
                "evidence_pack": pack_id,
            })

    packs_by_id = {pack["id"]: pack for pack in packs}
    chains: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for left_kind, right_kind, goal in _CHAIN_RULES:
        left = [item for item in primitives if item["capability"] == left_kind]
        right = [item for item in primitives if item["capability"] == right_kind]
        for first in left:
            for second in right:
                if first["finding_id"] == second["finding_id"]:
                    continue
                first_pack = packs_by_id[first["evidence_pack"]]
                second_pack = packs_by_id[second["evidence_pack"]]
                dependency_refs = _dependency_refs(first_pack, second_pack)
                if not dependency_refs:
                    continue
                key = (first["id"], second["id"], goal)
                if key in seen:
                    continue
                seen.add(key)
                blocked = [item["id"] for item in (first, second)
                           if item["viability"] == "rejected"]
                chains.append({
                    "id": f"CHAIN-{len(chains) + 1:04d}",
                    "goal": goal,
                    "primitive_ids": [first["id"], second["id"]],
                    "finding_ids": [first["finding_id"], second["finding_id"]],
                    "status": "pruned" if blocked else "candidate",
                    "priority": sum(
                        2 if item["viability"] == "confirmed" else 1
                        for item in (first, second)
                    ),
                    "blocked_by": blocked,
                    "dependency_evidence": dependency_refs,
                    "missing_edges": [
                        "prove the attacker can satisfy every required precondition",
                    ],
                })
                if len(chains) >= 100:
                    break
            if len(chains) >= 100:
                break
        if len(chains) >= 100:
            break

    return {
        "evidence_packs": packs,
        "invariants": invariants,
        "primitives": primitives,
        "attack_chains": chains,
    }


__all__ = ["build_investigation_graph", "invariant_candidates_from_context_map"]
