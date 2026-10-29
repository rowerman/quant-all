"""选择期诊断的版本化数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mining.contract_types import sha256_json
from mining.search_evaluation_evidence import SearchEvaluationEvidence

SELECTION_DIAGNOSTICS_STAGE = "selection_diagnostics"
SELECTION_DIAGNOSTICS_SCHEMA_VERSION = "mining_selection_diagnostics_v1"
CLUSTER_CORRELATION_THRESHOLD = 0.90


def finite_or_none(value: Any) -> float | None:
    """把有限数值转成浮点数，其余值统一返回空。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class CandidateCorrelationReport:
    """单个候选相对同批候选的相关性和相关簇。"""

    run_id: str
    candidate_id: str
    window_start: str
    window_end: str
    sample_dates: tuple[str, ...]
    peers: tuple[Mapping[str, Any], ...]
    max_abs_correlation: float | None
    cluster_id: str
    cluster_candidate_ids: tuple[str, ...]
    status: str
    reason: str | None = None
    schema_version: str = SELECTION_DIAGNOSTICS_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回不含运行时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "sample_dates": list(self.sample_dates),
            "peers": [dict(item) for item in self.peers],
            "max_abs_correlation": self.max_abs_correlation,
            "cluster_id": self.cluster_id,
            "cluster_candidate_ids": list(self.cluster_candidate_ids),
            "status": self.status,
            "reason": self.reason,
        }

    @property
    def report_hash(self) -> str:
        """返回相关性和相关簇摘要哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化和页面使用的字典。"""

        return {**self.material(), "report_hash": self.report_hash}


@dataclass(frozen=True)
class IncrementalValueReport:
    """候选对正式因子和风格暴露后的增量价值摘要。"""

    formal_factor_status: str
    formal_factor_peer: str | None
    formal_factor_correlation: float | None
    formal_factor_max_absolute: float | None
    raw_rank_ic: float | None
    incremental_rank_ic: float | None
    survivor_share: float | None
    valid_days: int | None
    status: str
    reason: str | None = None
    schema_version: str = SELECTION_DIAGNOSTICS_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回不含运行时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "formal_factor_status": self.formal_factor_status,
            "formal_factor_peer": self.formal_factor_peer,
            "formal_factor_correlation": self.formal_factor_correlation,
            "formal_factor_max_absolute": self.formal_factor_max_absolute,
            "raw_rank_ic": self.raw_rank_ic,
            "incremental_rank_ic": self.incremental_rank_ic,
            "survivor_share": self.survivor_share,
            "valid_days": self.valid_days,
            "status": self.status,
            "reason": self.reason,
        }

    @property
    def report_hash(self) -> str:
        """返回增量价值摘要哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化和页面使用的字典。"""

        return {**self.material(), "report_hash": self.report_hash}


@dataclass(frozen=True)
class SelectionDiagnosticsReport:
    """一个候选在选择期的完整脱敏诊断证据。"""

    run_id: str
    candidate_id: str
    input_hash: str
    status: str
    correlation: CandidateCorrelationReport
    incremental: IncrementalValueReport
    warnings: tuple[str, ...] = ()
    schema_version: str = SELECTION_DIAGNOSTICS_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回不含运行时间的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "input_hash": self.input_hash,
            "status": self.status,
            "correlation": self.correlation.to_dict(),
            "incremental": self.incremental.to_dict(),
            "warnings": list(self.warnings),
        }

    @property
    def evidence_hash(self) -> str:
        """返回诊断证据内容哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "evidence_hash": self.evidence_hash}

    def to_evaluation_evidence(self) -> SearchEvaluationEvidence:
        """转换为 V2 不可变评估证据。"""

        return SearchEvaluationEvidence(
            run_id=self.run_id,
            candidate_id=self.candidate_id,
            stage=SELECTION_DIAGNOSTICS_STAGE,
            input_hash=self.input_hash,
            status=self.status,
            metrics=self.to_dict(),
            warnings=self.warnings,
        )


def build_selection_diagnostics(
    *,
    run_id: str,
    candidate_id: str,
    input_material: Mapping[str, Any],
    correlation: CandidateCorrelationReport,
    incremental: IncrementalValueReport,
) -> SelectionDiagnosticsReport:
    """合并相关性与增量价值，生成可持久化证据。"""

    warnings: list[str] = []
    if correlation.status != "available":
        warnings.append(f"同批相关性不可用：{correlation.reason or correlation.status}")
    if incremental.formal_factor_status != "available":
        warnings.append(
            f"正式因子相关性不可用：{incremental.formal_factor_status}"
        )
    if incremental.status != "available":
        warnings.append(f"增量价值不可用：{incremental.reason or incremental.status}")
    correlation_available = correlation.status in {"available", "single_candidate"}
    status = (
        "available"
        if correlation_available and incremental.status == "available"
        else (
            "partial"
            if correlation_available or incremental.status == "available"
            else "not_computable"
        )
    )
    return SelectionDiagnosticsReport(
        run_id=run_id,
        candidate_id=candidate_id,
        input_hash=sha256_json(dict(input_material)),
        status=status,
        correlation=correlation,
        incremental=incremental,
        warnings=tuple(warnings),
    )


__all__ = [
    "CLUSTER_CORRELATION_THRESHOLD",
    "SELECTION_DIAGNOSTICS_SCHEMA_VERSION",
    "SELECTION_DIAGNOSTICS_STAGE",
    "CandidateCorrelationReport",
    "IncrementalValueReport",
    "SelectionDiagnosticsReport",
    "build_selection_diagnostics",
    "finite_or_none",
]
