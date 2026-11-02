"""V2/V3 候选审核详情的共享投影。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.search_evaluation_evidence import SearchEvaluationEvidence

TECHNICAL_GATE_RULE_VERSION = "mining_search_approval_v1"


def evaluation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """把持久化评估证据转换为页面可展示的摘要。"""

    evidence = SearchEvaluationEvidence.from_dict(row["evidence"])
    return {
        "evaluation_id": row["evaluation_id"],
        "stage": evidence.stage,
        "status": evidence.status,
        "input_hash": evidence.input_hash,
        "evidence_hash": evidence.evidence_hash,
        "metrics": evidence.metrics,
        "warnings": list(evidence.warnings),
        "error_code": evidence.error_code,
        "error_message": evidence.error_message,
        "created_at": evidence.created_at,
        "started_at": evidence.created_at,
        "finished_at": evidence.created_at,
    }


def latest_stage(
    evaluations: list[Mapping[str, Any]],
    stage: str,
) -> Mapping[str, Any] | None:
    """返回指定阶段最后写入的一条评估证据。"""

    return next(
        (
            item
            for item in reversed(evaluations)
            if item["stage"] == stage
        ),
        None,
    )


def technical_gate_view(
    evaluations: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """按 V2 页面契约转换最新技术门禁证据。"""

    technical = latest_stage(evaluations, "technical")
    if technical is None:
        return None
    return {
        "passed": technical["status"] == "passed",
        "rule_version": TECHNICAL_GATE_RULE_VERSION,
        "checks": technical["metrics"],
    }


__all__ = [
    "TECHNICAL_GATE_RULE_VERSION",
    "evaluation_view",
    "latest_stage",
    "technical_gate_view",
]
