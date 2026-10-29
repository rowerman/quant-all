"""V2 搜索编排共用的候选、假设、证据和尝试转换。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import sha256_json
from mining.search_cards import HypothesisCard
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_evidence import EvidencePacket
from mining.search_trials import TrialRecord


def initial_hypothesis(run: Mapping[str, Any]) -> HypothesisCard:
    """把用户提交的自然语言假设转换成初始假设卡。"""

    return HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement=run["hypothesis"],
        mechanism="用户提交的研究假设",
        expected_direction="non_monotonic",
        falsification_criteria=(
            "开发期质量未达到冻结门槛",
            "连续多轮修订没有改善",
        ),
        metadata={"source": "user"},
    )


def next_hypothesis(
    *,
    run_id: str,
    turn_index: int,
    branch_id: str,
    statement: str,
    mechanism: str,
    expected_direction: str,
    falsification_criteria: tuple[str, ...],
    parent_hypothesis_id: str,
) -> HypothesisCard:
    """把 Worker 建议转换成主服务确认的下一假设卡。"""

    return HypothesisCard(
        run_id=run_id,
        turn_index=turn_index,
        branch_id=branch_id,
        statement=statement,
        mechanism=mechanism,
        expected_direction=expected_direction,
        falsification_criteria=falsification_criteria,
        parent_hypothesis_id=parent_hypothesis_id,
        metadata={"source": "worker"},
    )


def candidate_payload(candidate: CandidateDefinition) -> dict[str, Any]:
    """返回 V2 轨迹需要的完整候选定义。"""

    return {
        "name": candidate.name,
        "mechanism": candidate.mechanism,
        "description": candidate.description,
        "canonical_expression": candidate.canonical_expression,
        "direction": candidate.direction,
        "lookback": candidate.lookback,
        "dependencies": list(candidate.dependencies),
        "output_type": candidate.output_type,
        "steps": list(candidate.steps),
        "discovery_round": candidate.discovery_round,
        "discovery_index": candidate.discovery_index,
        "parameters": dict(candidate.parameters),
    }


def candidate_from_v2_record(record: Mapping[str, Any]) -> CandidateDefinition:
    """从 V2 候选轨迹载荷恢复候选定义。"""

    return CandidateDefinition(**dict(record["payload"]))


def batch_input_hash(
    *,
    request_hash: str,
    candidates: tuple[CandidateDefinition, ...],
) -> str:
    """计算一轮实验批次的稳定输入哈希。"""

    return sha256_json(
        {
            "request_hash": request_hash,
            "candidates": sorted(
                candidate.content_hash for candidate in candidates
            ),
        }
    )


def trial_record(
    *,
    run_id: str,
    trial_key: str,
    kind: str,
    turn_index: int,
    hypothesis_id: str,
    batch_id: str | None,
    candidate_id: str | None,
    reason_code: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> TrialRecord:
    """构造稳定的 V2 尝试记录。"""

    return TrialRecord(
        run_id=run_id,
        trial_key=trial_key,
        kind=kind,
        turn_index=turn_index,
        hypothesis_id=hypothesis_id,
        batch_id=batch_id,
        candidate_id=candidate_id,
        reason_code=reason_code,
        payload=dict(payload or {}),
    )


def evaluation_evidence(
    report: Any,
    *,
    stage: str | None = None,
) -> SearchEvaluationEvidence:
    """把现有评估报告转换成不可变 V2 评估证据。"""

    return SearchEvaluationEvidence.from_report(report, stage=stage)


def evaluation_report_mapping(report: Any) -> dict[str, Any]:
    """把评估报告转换成候选摘要构造器接收的映射。"""

    return asdict(report)


def candidate_score_material(evidence: SearchEvaluationEvidence) -> dict[str, Any]:
    """返回选择期排序所需的稳定摘要。"""

    metrics = evidence.metrics
    rank_ic = metrics.get("rank_ic_mean")
    icir = metrics.get("rank_ic_ir")
    coverage = (metrics.get("values") or {}).get("coverage")
    return {
        "candidate_id": evidence.candidate_id,
        "rank_ic": rank_ic,
        "icir": icir,
        "coverage": coverage,
        "evidence_hash": evidence.evidence_hash,
    }


def candidate_sort_key(material: Mapping[str, Any]) -> tuple[Any, ...]:
    """选择期按绝对 RankIC、ICIR、覆盖率、候选 ID 确定性排序。"""

    def magnitude(name: str) -> float:
        value = material.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return abs(float(value))

    return (
        -magnitude("rank_ic"),
        -magnitude("icir"),
        -magnitude("coverage"),
        str(material["candidate_id"]),
        str(material["evidence_hash"]),
    )


def merge_evidence(
    previous: EvidencePacket | None,
    current: EvidencePacket,
) -> EvidencePacket:
    """把当前 turn 反馈并入运行内开发期证据视图。"""

    if previous is None:
        return current
    return EvidencePacket(
        run_id=current.run_id,
        hypothesis_id=current.hypothesis_id,
        batch_id=current.batch_id,
        turn_index=current.turn_index,
        development_start=current.development_start,
        development_end=current.development_end,
        metrics={**previous.metrics, **current.metrics},
        failures=(*previous.failures, *current.failures)[-128:],
        correlation_clusters=(
            *previous.correlation_clusters,
            *current.correlation_clusters,
        )[-24:],
        trial_counts=current.trial_counts,
        budget_state=current.budget_state,
    )
