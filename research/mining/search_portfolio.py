"""确定性的 Pareto、相关簇多样性和入围选择。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from mining.contract_types import sha256_json
from mining.search_control import SearchControlPolicy
from mining.search_research_quality import evaluate_research_quality
from mining.search_scorecard import CandidateScoreCard

PORTFOLIO_SCHEMA_VERSION = "mining_portfolio_selection_v1"


@dataclass(frozen=True)
class PortfolioCandidate:
    """进入 Pareto 选择的单候选证据。"""

    candidate_id: str
    candidate_hash: str
    cluster_id: str | None
    objectives: Mapping[str, float]
    completeness: int
    eligible: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """返回页面使用的字典。"""

        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "cluster_id": self.cluster_id,
            "objectives": dict(self.objectives),
            "completeness": self.completeness,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class PortfolioSelectionResult:
    """Pareto 层级、稳定排序和最终入围结果。"""

    candidates: tuple[PortfolioCandidate, ...]
    fronts: tuple[tuple[str, ...], ...]
    finalists: tuple[str, ...]
    schema_version: str = PORTFOLIO_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "candidates": [item.to_dict() for item in self.candidates],
            "fronts": [list(item) for item in self.fronts],
            "finalists": list(self.finalists),
        }

    @property
    def selection_hash(self) -> str:
        """返回选择结果哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化和页面字典。"""

        return {**self.material(), "selection_hash": self.selection_hash}


def _finite(value: Any) -> float | None:
    """返回有限浮点数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _candidate_material(
    scorecard: CandidateScoreCard,
    policy: SearchControlPolicy,
) -> PortfolioCandidate:
    """从选择期评分卡构造 Pareto 候选。"""

    quality = evaluate_research_quality(scorecard, policy, stage="selection")
    diagnostics = (
        scorecard.analysis_by_stage.get("selection_diagnostics") or {}
    )
    incremental = (
        diagnostics.get("incremental") if isinstance(diagnostics, Mapping) else {}
    ) or {}
    stability = (
        scorecard.analysis_by_stage.get("stability") or {}
    )
    correlation = (
        diagnostics.get("correlation") if isinstance(diagnostics, Mapping) else {}
    ) or {}
    selection_metrics = scorecard.metrics_by_stage.get("selection") or {}
    quality_value = _finite(selection_metrics.get("rank_ic"))
    incremental_value = _finite(incremental.get("incremental_rank_ic"))
    coverage = _finite(selection_metrics.get("coverage"))
    stability_status = stability.get("status")
    ratio = _finite(stability.get("rank_ic_ratio"))
    if stability_status == "stable":
        stability_value = min(1.0, ratio if ratio is not None else 1.0)
    elif stability_status == "unstable":
        stability_value = 0.0
    elif stability_status == "not_applicable":
        stability_value = 1.0
    else:
        stability_value = None
    objectives = {
        "quality": abs(quality_value) if quality_value is not None else None,
        "incremental_value": (
            abs(incremental_value) if incremental_value is not None else None
        ),
        "stability": stability_value,
        "coverage": coverage,
    }
    missing = [name for name, value in objectives.items() if value is None]
    reasons: list[str] = []
    if quality.status != "quality_passed":
        reasons.append(quality.reason)
    if stability_status == "not_computable":
        reasons.append("稳定性证据不足")
    if incremental_value is None:
        reasons.append("增量价值证据不足")
    eligible = not reasons
    return PortfolioCandidate(
        candidate_id=scorecard.candidate_id,
        candidate_hash=scorecard.candidate_hash,
        cluster_id=correlation.get("cluster_id"),
        objectives=objectives,
        completeness=len(objectives) - len(missing),
        eligible=eligible,
        exclusion_reason="；".join(reasons) if reasons else None,
    )


def _dominates(left: PortfolioCandidate, right: PortfolioCandidate) -> bool:
    """判断一个候选是否在所有目标不差且至少一项更好。"""

    names = ("quality", "incremental_value", "stability", "coverage")
    left_values = [float(left.objectives[name]) for name in names]
    right_values = [float(right.objectives[name]) for name in names]
    return all(
        first >= second for first, second in zip(left_values, right_values)
    ) and any(
        first > second for first, second in zip(left_values, right_values)
    )


def _sort_key(item: PortfolioCandidate) -> tuple[Any, ...]:
    """返回 Pareto 同层内的稳定优先级。"""

    return (
        -float(item.objectives["quality"]),
        -float(item.objectives["incremental_value"]),
        -float(item.objectives["stability"]),
        -float(item.objectives["coverage"]),
        item.candidate_hash,
        item.candidate_id,
    )


def _pareto_fronts(
    candidates: Iterable[PortfolioCandidate],
) -> tuple[tuple[str, ...], ...]:
    """按非支配层级返回稳定候选 ID。"""

    remaining = sorted(candidates, key=_sort_key)
    fronts: list[tuple[str, ...]] = []
    while remaining:
        front = [
            item
            for item in remaining
            if not any(
                other is not item and _dominates(other, item)
                for other in remaining
            )
        ]
        if not front:
            front = [remaining[0]]
        front = sorted(front, key=_sort_key)
        fronts.append(tuple(item.candidate_id for item in front))
        selected = {item.candidate_id for item in front}
        remaining = [
            item for item in remaining if item.candidate_id not in selected
        ]
    return tuple(fronts)


def select_portfolio_finalists(
    scorecards: Iterable[CandidateScoreCard],
    policy: SearchControlPolicy,
    *,
    max_finalists: int,
) -> PortfolioSelectionResult:
    """按 Pareto 层级和相关簇多样性选择最终评估候选。"""

    candidates = tuple(
        sorted(
            (
                _candidate_material(scorecard, policy)
                for scorecard in scorecards
            ),
            key=lambda item: item.candidate_id,
        )
    )
    eligible = [item for item in candidates if item.eligible]
    fronts = _pareto_fronts(eligible)
    by_id = {item.candidate_id: item for item in candidates}
    used_clusters: set[str] = set()
    finalists: list[str] = []
    for front in fronts:
        for candidate_id in front:
            item = by_id[candidate_id]
            cluster = item.cluster_id or f"singleton:{candidate_id}"
            if cluster in used_clusters:
                continue
            finalists.append(candidate_id)
            used_clusters.add(cluster)
            if len(finalists) >= max_finalists:
                break
        if len(finalists) >= max_finalists:
            break
    return PortfolioSelectionResult(
        candidates=candidates,
        fronts=fronts,
        finalists=tuple(finalists),
    )


__all__ = [
    "PORTFOLIO_SCHEMA_VERSION",
    "PortfolioCandidate",
    "PortfolioSelectionResult",
    "select_portfolio_finalists",
]
