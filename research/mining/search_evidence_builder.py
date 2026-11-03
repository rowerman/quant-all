"""把开发期评估结果转换为脱敏、稳定、可反馈的证据包。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from mining.contract_types import identifier, text
from mining.errors import MiningContractError
from mining.logging_conf import format_event, get_mining_logger
from mining.search_budget import TrialCounts
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evidence import EvidencePacket

_LOGGER = get_mining_logger("mining.search.evidence")
_PATH_PATTERN = re.compile(r"(?<![\w.-])(?:/[A-Za-z0-9._-]+){2,}")
_SECRET_PATTERN = re.compile(r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+)")
_BUDGET_FIELDS = {
    "turns_used",
    "branches_used",
    "refines_in_branch",
    "candidates_proposed",
    "candidates_evaluated",
    "full_evaluated",
    "no_improvement_turns",
    "remaining_candidates",
    "cancelled",
    "snapshot_stale",
}


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MiningContractError(f"{name} 必须是有限数值或 None")
    result = float(value)
    if not math.isfinite(result):
        raise MiningContractError(f"{name} 必须是有限数值或 None")
    return result


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MiningContractError(f"{name} 必须是非负整数或 None")
    return value


def _sanitize_reason(value: str | None) -> str | None:
    if value is None:
        return None
    result = _SECRET_PATTERN.sub("[REDACTED]", str(value))
    result = _PATH_PATTERN.sub("[PATH]", result)
    return result[:500]


@dataclass(frozen=True)
class DevelopmentCandidateEvidence:
    """单个候选在开发期产生的脱敏摘要。"""

    candidate_id: str
    technical_status: str
    quick_status: str
    gate_passed: bool | None
    rank_ic: float | None = None
    rank_ic_ir: float | None = None
    year_sign_consistency: float | None = None
    coverage: float | None = None
    valid_days: int | None = None
    sample_count: int | None = None
    turnover: float | None = None
    ic_decay: float | None = None
    failure_stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            identifier(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "technical_status",
            identifier(self.technical_status, "technical_status"),
        )
        object.__setattr__(
            self,
            "quick_status",
            identifier(self.quick_status, "quick_status"),
        )
        if self.gate_passed is not None and not isinstance(self.gate_passed, bool):
            raise MiningContractError("gate_passed 必须是布尔值或 None")
        for name in (
            "rank_ic",
            "rank_ic_ir",
            "year_sign_consistency",
            "coverage",
            "turnover",
            "ic_decay",
        ):
            object.__setattr__(
                self,
                name,
                _optional_float(getattr(self, name), name),
            )
        for name in ("valid_days", "sample_count"):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name),
            )
        for name in ("failure_stage", "error_code"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, identifier(value, name))
        if self.error_message is not None:
            object.__setattr__(
                self,
                "error_message",
                text(self.error_message, "error_message", maximum=2000),
            )

    @classmethod
    def from_reports(
        cls,
        candidate_id: str,
        *,
        gate: Mapping[str, Any],
        technical: Mapping[str, Any],
        quick: Mapping[str, Any] | None,
    ) -> DevelopmentCandidateEvidence:
        """从评估报告和安全门禁摘要构造候选证据。"""

        technical_metrics = technical.get("metrics") or {}
        values = technical_metrics.get("values") or {}
        quick_metrics = (quick or {}).get("metrics") or {}
        quick_status = str((quick or {}).get("status", "not_run"))
        technical_failed = (
            not bool(gate.get("passed"))
            or technical.get("status") != "passed"
        )
        quick_failed = quick is not None and quick.get("status") != "passed"
        failure_stage = "technical" if technical_failed else (
            "quick" if quick_failed else None
        )
        report = quick if quick is not None else technical
        return cls(
            candidate_id=candidate_id,
            technical_status=str(technical.get("status", "failed")),
            quick_status=quick_status,
            gate_passed=gate.get("passed"),
            rank_ic=quick_metrics.get("rank_ic_mean"),
            rank_ic_ir=quick_metrics.get("rank_ic_ir"),
            year_sign_consistency=quick_metrics.get("year_sign_consistency"),
            coverage=values.get("coverage"),
            valid_days=quick_metrics.get("valid_days"),
            sample_count=quick_metrics.get("sample_count"),
            failure_stage=failure_stage,
            error_code=report.get("error_code") if failure_stage else None,
            error_message=report.get("error_message") if failure_stage else None,
        )

    def metrics_payload(self) -> dict[str, Any]:
        """返回固定字段的指标摘要，缺失值保持 None。"""

        return {
            "gate_passed": self.gate_passed,
            "technical_status": self.technical_status,
            "quick_status": self.quick_status,
            "rank_ic": self.rank_ic,
            "rank_ic_ir": self.rank_ic_ir,
            "year_sign_consistency": self.year_sign_consistency,
            "coverage": self.coverage,
            "valid_days": self.valid_days,
            "sample_count": self.sample_count,
            "turnover": self.turnover,
            "ic_decay": self.ic_decay,
        }


def _budget_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MiningContractError("budget_state 必须是对象")
    unknown = set(value) - _BUDGET_FIELDS
    if unknown:
        raise MiningContractError(f"budget_state 包含未知字段：{sorted(unknown)}")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"cancelled", "snapshot_stale"}:
            if not isinstance(item, bool):
                raise MiningContractError(f"budget_state.{key} 必须是布尔值")
        elif isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise MiningContractError(f"budget_state.{key} 必须是非负整数")
        result[key] = item
    return result


def build_development_evidence(
    *,
    hypothesis: HypothesisCard,
    batch: ExperimentBatch,
    candidates: tuple[DevelopmentCandidateEvidence, ...],
    trial_counts: TrialCounts,
    budget_state: Mapping[str, Any],
    correlation_clusters: tuple[Mapping[str, Any], ...] = (),
) -> EvidencePacket:
    """汇总一轮开发期结果；不读取选择期、最终研究期或测试段。"""

    if hypothesis.run_id != batch.run_id:
        raise MiningContractError("假设和实验批次不属于同一运行")
    if batch.hypothesis_id != hypothesis.hypothesis_id:
        raise MiningContractError("实验批次没有绑定当前假设")
    if not isinstance(trial_counts, TrialCounts):
        raise MiningContractError("trial_counts 必须是 TrialCounts")
    if (
        not isinstance(candidates, (list, tuple))
        or any(
            not isinstance(item, DevelopmentCandidateEvidence)
            for item in candidates
        )
    ):
        raise MiningContractError(
            "candidates 必须是 DevelopmentCandidateEvidence 序列"
        )
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise MiningContractError("候选证据包含重复 candidate_id")
    unknown = set(candidate_ids) - set(batch.candidate_ids)
    if unknown:
        raise MiningContractError(
            f"候选证据不属于当前批次：{sorted(unknown)}"
        )
    missing = set(batch.candidate_ids) - set(candidate_ids)
    if missing:
        raise MiningContractError(
            f"当前批次缺少候选证据：{sorted(missing)}"
        )
    metrics = {
        candidate_id: next(
            item.metrics_payload()
            for item in candidates
            if item.candidate_id == candidate_id
        )
        for candidate_id in batch.candidate_ids
    }
    failures = tuple(
        {
            "candidate_id": item.candidate_id,
            "stage": item.failure_stage,
            "error_code": item.error_code or "MINING-106",
            "reason": _sanitize_reason(item.error_message)
            or f"{item.failure_stage} 阶段未通过",
        }
        for item in candidates
        if item.failure_stage is not None
    )
    evidence = EvidencePacket(
        run_id=hypothesis.run_id,
        hypothesis_id=hypothesis.hypothesis_id,
        batch_id=batch.batch_id,
        turn_index=hypothesis.turn_index,
        development_start=batch.development_start,
        development_end=batch.development_end,
        metrics=metrics,
        failures=failures,
        correlation_clusters=tuple(dict(item) for item in correlation_clusters),
        trial_counts=trial_counts,
        budget_state=_budget_state(budget_state),
    )
    _LOGGER.info(
        format_event(
            {
                "event": "development_evidence_built",
                "run_id": evidence.run_id,
                "turn_index": evidence.turn_index,
                "batch_id": evidence.batch_id,
                "evidence_hash": evidence.evidence_hash,
                "candidate_count": len(candidates),
                "failure_count": len(failures),
            }
        )
    )
    return evidence
