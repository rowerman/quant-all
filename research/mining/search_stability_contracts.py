"""邻域稳定性变体和报告契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import sha256_json
from mining.search_evaluation_evidence import SearchEvaluationEvidence

STABILITY_STAGE = "stability"
STABILITY_SCHEMA_VERSION = "mining_stability_v1"
STABILITY_VARIANT_LIMIT = 2
STABILITY_RANK_IC_RATIO_FLOOR = 0.50


def optional_float(value: Any) -> float | None:
    """把有限数值转成浮点数，其余值统一返回空。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class WindowVariant:
    """基础候选主窗口的一个相邻变体。"""

    candidate: CandidateDefinition
    base_window: int
    variant_window: int
    delta: int


@dataclass(frozen=True)
class VariantPlan:
    """一次稳定性分析可评估的变体计划。"""

    base_candidate_id: str
    base_window: int | None
    variants: tuple[WindowVariant, ...]
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class WindowVariantObservation:
    """单个变体的开发期评估摘要。"""

    candidate_id: str
    candidate_hash: str
    window: int
    delta: int
    status: str
    gate_passed: bool | None
    rank_ic: float | None
    rank_ic_ir: float | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "window": self.window,
            "delta": self.delta,
            "status": self.status,
            "gate_passed": self.gate_passed,
            "rank_ic": self.rank_ic,
            "rank_ic_ir": self.rank_ic_ir,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class StabilityReport:
    """基础候选及主窗口变体的稳定性结论。"""

    run_id: str
    candidate_id: str
    base_candidate_hash: str
    direction: int
    base_rank_ic: float | None
    base_window: int | None
    variants: tuple[WindowVariantObservation, ...]
    aligned_versions: int
    metric_versions: int
    median_variant_abs_rank_ic: float | None
    rank_ic_ratio: float | None
    status: str
    reason: str | None = None
    schema_version: str = STABILITY_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "base_candidate_hash": self.base_candidate_hash,
            "direction": self.direction,
            "base_rank_ic": self.base_rank_ic,
            "base_window": self.base_window,
            "variants": [item.to_dict() for item in self.variants],
            "aligned_versions": self.aligned_versions,
            "metric_versions": self.metric_versions,
            "median_variant_abs_rank_ic": self.median_variant_abs_rank_ic,
            "rank_ic_ratio": self.rank_ic_ratio,
            "status": self.status,
            "reason": self.reason,
        }

    @property
    def evidence_hash(self) -> str:
        """返回稳定性证据哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化和页面字典。"""

        return {**self.material(), "evidence_hash": self.evidence_hash}

    def to_evaluation_evidence(self) -> SearchEvaluationEvidence:
        """转换为 V2 评估证据。"""

        return SearchEvaluationEvidence(
            run_id=self.run_id,
            candidate_id=self.candidate_id,
            stage=STABILITY_STAGE,
            input_hash=self.base_candidate_hash,
            status=self.status,
            metrics=self.to_dict(),
            warnings=((self.reason,) if self.reason else ()),
        )


def build_stability_report(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    plan: VariantPlan,
    base_rank_ic: float | None,
    variants: tuple[WindowVariantObservation, ...],
    rank_ic_ratio_floor: float = STABILITY_RANK_IC_RATIO_FLOOR,
) -> StabilityReport:
    """按方向一致率和 RankIC 保留比例生成稳定性结论。"""

    rank_values = [
        optional_float(item.rank_ic)
        for item in variants
        if optional_float(item.rank_ic) is not None
    ]
    all_rank_values = ([base_rank_ic] if base_rank_ic is not None else []) + [
        float(value) for value in rank_values
    ]
    aligned_versions = sum(
        value * candidate.direction > 0 for value in all_rank_values
    )
    metric_versions = len(all_rank_values)
    median_variant = (
        float(median(abs(float(value)) for value in rank_values))
        if rank_values
        else None
    )
    base_abs = abs(base_rank_ic) if base_rank_ic is not None else None
    ratio = (
        median_variant / base_abs
        if median_variant is not None and base_abs is not None and base_abs > 0
        else None
    )
    if plan.status == "not_applicable":
        status = "not_applicable"
        reason = plan.reason
    elif base_rank_ic is None or len(rank_values) < 2:
        status = "not_computable"
        reason = "基础候选或至少两个相邻变体缺少有效 RankIC"
    elif aligned_versions < 2:
        status = "unstable"
        reason = "基础版本和相邻变体少于两个方向一致"
    elif ratio is None:
        status = "unstable"
        reason = "基础 RankIC 为零，无法确认邻域保留比例"
    elif ratio < rank_ic_ratio_floor:
        status = "unstable"
        reason = "变体 |RankIC| 中位数低于基础版本的 50%"
    else:
        status = "stable"
        reason = None
    return StabilityReport(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        base_candidate_hash=candidate.content_hash,
        direction=candidate.direction,
        base_rank_ic=base_rank_ic,
        base_window=plan.base_window,
        variants=variants,
        aligned_versions=aligned_versions,
        metric_versions=metric_versions,
        median_variant_abs_rank_ic=median_variant,
        rank_ic_ratio=ratio,
        status=status,
        reason=reason,
    )


__all__ = [
    "STABILITY_RANK_IC_RATIO_FLOOR",
    "STABILITY_SCHEMA_VERSION",
    "STABILITY_STAGE",
    "STABILITY_VARIANT_LIMIT",
    "StabilityReport",
    "VariantPlan",
    "WindowVariant",
    "WindowVariantObservation",
    "build_stability_report",
    "optional_float",
]
