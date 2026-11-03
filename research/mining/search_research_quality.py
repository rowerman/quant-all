"""候选执行状态和研究质量的独立结论。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mining.contract_types import sha256_json
from mining.search_control import SearchControlPolicy
from mining.search_scorecard import CandidateScoreCard

RESEARCH_QUALITY_SCHEMA_VERSION = "mining_research_quality_v1"
RESEARCH_QUALITY_STATUSES = (
    "execution_failed",
    "quality_failed",
    "quality_passed",
    "evidence_insufficient",
)


@dataclass(frozen=True)
class ResearchQualityConclusion:
    """某个评估阶段的执行状态和研究质量结论。"""

    candidate_id: str
    stage: str
    execution_status: str | None
    status: str
    thresholds: Mapping[str, float]
    metrics: Mapping[str, float | None]
    missing_fields: tuple[str, ...]
    context: Mapping[str, Any]
    reason: str
    schema_version: str = RESEARCH_QUALITY_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "execution_status": self.execution_status,
            "status": self.status,
            "thresholds": dict(self.thresholds),
            "metrics": dict(self.metrics),
            "missing_fields": list(self.missing_fields),
            "context": dict(self.context),
            "reason": self.reason,
        }

    @property
    def conclusion_hash(self) -> str:
        """返回质量结论哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回页面使用的字典。"""

        return {**self.material(), "conclusion_hash": self.conclusion_hash}


def evaluate_research_quality(
    scorecard: CandidateScoreCard,
    policy: SearchControlPolicy,
    *,
    stage: str,
) -> ResearchQualityConclusion:
    """按冻结门槛评估执行状态、证据完整性和研究质量。"""

    thresholds = {
        "min_rank_ic": policy.min_rank_ic,
        "min_icir": policy.min_icir,
        "min_year_sign_consistency": policy.min_year_sign_consistency,
        "min_coverage": policy.min_coverage,
    }
    stage_metrics = scorecard.metrics_by_stage.get(stage)
    if not isinstance(stage_metrics, Mapping):
        return ResearchQualityConclusion(
            candidate_id=scorecard.candidate_id,
            stage=stage,
            execution_status=None,
            status="evidence_insufficient",
            thresholds=thresholds,
            metrics={},
            missing_fields=("stage",),
            context=_quality_context(scorecard),
            reason=f"{stage} 阶段没有评估证据",
        )
    execution_status = scorecard.evidence_status_by_stage.get(stage)
    if execution_status != "passed":
        return ResearchQualityConclusion(
            candidate_id=scorecard.candidate_id,
            stage=stage,
            execution_status=execution_status,
            status="execution_failed",
            thresholds=thresholds,
            metrics={},
            missing_fields=(),
            context=_quality_context(scorecard),
            reason=f"{stage} 阶段执行状态为 {execution_status or 'missing'}",
        )
    metric_names = (
        "rank_ic",
        "icir",
        "year_sign_consistency",
        "coverage",
    )
    metrics = {
        name: _finite(stage_metrics.get(name))
        for name in metric_names
    }
    missing = tuple(name for name, value in metrics.items() if value is None)
    if missing:
        return ResearchQualityConclusion(
            candidate_id=scorecard.candidate_id,
            stage=stage,
            execution_status=execution_status,
            status="evidence_insufficient",
            thresholds=thresholds,
            metrics=metrics,
            missing_fields=missing,
            context=_quality_context(scorecard),
            reason=f"研究质量指标缺失：{', '.join(missing)}",
        )
    passed = (
        abs(float(metrics["rank_ic"])) >= policy.min_rank_ic
        and abs(float(metrics["icir"])) >= policy.min_icir
        and float(metrics["year_sign_consistency"])
        >= policy.min_year_sign_consistency
        and float(metrics["coverage"]) >= policy.min_coverage
    )
    return ResearchQualityConclusion(
        candidate_id=scorecard.candidate_id,
        stage=stage,
        execution_status=execution_status,
        status="quality_passed" if passed else "quality_failed",
        thresholds=thresholds,
        metrics=metrics,
        missing_fields=(),
        context=_quality_context(scorecard),
        reason=(
            "执行成功且达到冻结质量门槛"
            if passed
            else "执行成功但未达到冻结质量门槛"
        ),
    )


def _finite(value: Any) -> float | None:
    """把有限数值转成浮点数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quality_context(scorecard: CandidateScoreCard) -> dict[str, Any]:
    """返回质量结论必须同时展示的防过拟合上下文。"""

    diagnostics = scorecard.analysis_by_stage.get("selection_diagnostics") or {}
    correlation = (
        diagnostics.get("correlation") if isinstance(diagnostics, Mapping) else {}
    ) or {}
    stability = (
        scorecard.analysis_by_stage.get("stability") or {}
    )
    return {
        "attempts": scorecard.trial_counts.to_dict(),
        "correlation_cluster": correlation.get("cluster_id"),
        "stability_status": stability.get("status"),
    }


__all__ = [
    "RESEARCH_QUALITY_SCHEMA_VERSION",
    "RESEARCH_QUALITY_STATUSES",
    "ResearchQualityConclusion",
    "evaluate_research_quality",
]
