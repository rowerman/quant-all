"""从已登记的 V2 评估证据构建统一候选评分卡。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from mining.contract_types import sha256_json
from mining.errors import MiningContractError
from mining.search_budget import TrialCounts
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_selection_diagnostics import (
    SELECTION_DIAGNOSTICS_STAGE,
)
from mining.storage_v2 import MiningV2Store

SCORECARD_SCHEMA_VERSION = "mining_candidate_scorecard_v2"
_CORE_EVALUATION_STAGES = {
    "technical",
    "development",
    "selection",
    "full",
    "holdout",
}
SCORECARD_FIELDS = (
    "rank_ic",
    "icir",
    "year_sign_consistency",
    "coverage",
    "turnover",
    "ic_decay",
    "sample_size",
    "valid_days",
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _average(values: Any) -> float | None:
    if not isinstance(values, (list, tuple)):
        return None
    finite = [
        item
        for item in (_finite(value) for value in values)
        if item is not None
    ]
    return sum(finite) / len(finite) if finite else None


def _extract_metrics(
    stage: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    values = metrics.get("values")
    values = values if isinstance(values, Mapping) else {}
    events = metrics.get("events")
    events = events if isinstance(events, Mapping) else {}
    overview = ((events.get("overview") or {}).get("metrics") or {})
    ic_stats = overview.get("ic_stats") or {}
    rank_stats = ic_stats.get("rank_ic") or {}
    turnover_payload = (
        ((events.get("turnover") or {}).get("metrics") or {}).get("turnover")
        or {}
    )
    daily_turnover = turnover_payload.get("daily_turnover") or {}
    decay_payload = (
        ((events.get("decay") or {}).get("metrics") or {}).get("decay")
        or {}
    )
    decay_curve = decay_payload.get("curve") or []
    primary_horizon = decay_payload.get("primary_horizon")
    decay_row = next(
        (
            item
            for item in decay_curve
            if isinstance(item, Mapping)
            and item.get("horizon") == primary_horizon
        ),
        {},
    )
    rank_ic = _finite(metrics.get("rank_ic_mean"))
    if rank_ic is None:
        rank_ic = _finite(rank_stats.get("mean"))
    if rank_ic is None:
        rank_ic = _finite(decay_row.get("rank_ic_mean"))
    icir = _finite(metrics.get("rank_ic_ir"))
    if icir is None:
        icir = _finite(rank_stats.get("ir"))
    if icir is None:
        icir = _finite(decay_row.get("rank_ic_ir"))
    valid_days = _integer(metrics.get("valid_days"))
    if valid_days is None:
        valid_days = _integer(ic_stats.get("valid_days"))
    sample_size = _integer(values.get("rows"))
    if sample_size is None:
        sample_size = _integer(metrics.get("rows"))
    return {
        "stage": stage,
        "rank_ic": rank_ic,
        "icir": icir,
        "year_sign_consistency": _finite(
            metrics.get("year_sign_consistency")
        ),
        "coverage": _finite(values.get("coverage") or metrics.get("coverage")),
        "turnover": _average(daily_turnover.get("mean")),
        "ic_decay": _finite(decay_row.get("rank_ic_mean")),
        "sample_size": sample_size,
        "valid_days": valid_days,
    }


@dataclass(frozen=True)
class CandidateScoreCard:
    """只引用已登记证据的候选研究质量卡。"""

    run_id: str
    candidate_id: str
    candidate_hash: str
    semantic_hash: str
    hypothesis_id: str
    batch_id: str
    context_hash: str
    status: str
    windows: Mapping[str, str]
    decision_stage: str | None
    metrics: Mapping[str, Any]
    metrics_by_stage: Mapping[str, Any]
    analysis_by_stage: Mapping[str, Any]
    gate_passed: bool | None
    failure_reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    trial_counts: TrialCounts
    candidate_trial_counts: TrialCounts
    missing_fields: tuple[str, ...] = ()
    schema_version: str = SCORECARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCORECARD_SCHEMA_VERSION:
            raise MiningContractError("评分卡 schema_version 不匹配")
        missing = tuple(
            name for name in SCORECARD_FIELDS if self.metrics.get(name) is None
        )
        object.__setattr__(self, "missing_fields", missing)

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "semantic_hash": self.semantic_hash,
            "hypothesis_id": self.hypothesis_id,
            "batch_id": self.batch_id,
            "context_hash": self.context_hash,
            "status": self.status,
            "windows": self.windows,
            "decision_stage": self.decision_stage,
            "metrics": self.metrics,
            "metrics_by_stage": self.metrics_by_stage,
            "analysis_by_stage": self.analysis_by_stage,
            "gate_passed": self.gate_passed,
            "failure_reasons": list(self.failure_reasons),
            "evidence_ids": list(self.evidence_ids),
            "evidence_hashes": list(self.evidence_hashes),
            "trial_counts": self.trial_counts.to_dict(),
            "candidate_trial_counts": self.candidate_trial_counts.to_dict(),
            "missing_fields": list(self.missing_fields),
        }

    @property
    def scorecard_hash(self) -> str:
        """返回评分卡内容哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回页面可展示字典。"""

        return {
            **self.material(),
            "scorecard_hash": self.scorecard_hash,
            "selection_diagnostics": self.analysis_by_stage.get(
                SELECTION_DIAGNOSTICS_STAGE
            ),
        }


def _candidate_trial_counts(rows: list[dict[str, Any]]) -> TrialCounts:
    values = TrialCounts().to_dict()
    for row in rows:
        kind = row["kind"]
        if kind in values:
            values[kind] += 1
    if any(row.get("invalidated") for row in rows):
        values["invalidated"] = sum(
            bool(row.get("invalidated")) for row in rows
        )
    return TrialCounts(**values)


def build_candidate_scorecard(
    store: MiningV2Store,
    run_id: str,
    candidate_id: str,
) -> CandidateScoreCard:
    """从当前运行已登记证据构建评分卡。"""

    run = store.get_run(run_id)
    candidate = store.get_candidate(run_id, candidate_id)
    rows = store.list_evaluation_evidence(run_id, candidate_id)
    evidence = [
        SearchEvaluationEvidence.from_dict(row["evidence"])
        for row in rows
    ]
    by_stage = {
        item.stage: _extract_metrics(item.stage, item.metrics)
        for item in evidence
        if item.stage in _CORE_EVALUATION_STAGES
    }
    analysis_by_stage = {
        item.stage: item.metrics
        for item in evidence
        if item.stage not in _CORE_EVALUATION_STAGES
    }
    stage_priority = ("full", "selection", "development", "technical", "holdout")
    decision_stage = next(
        (stage for stage in stage_priority if stage in by_stage),
        None,
    )
    metrics = {}
    for name in SCORECARD_FIELDS:
        metrics[name] = next(
            (
                by_stage[stage].get(name)
                for stage in stage_priority
                if stage in by_stage
                and by_stage[stage].get(name) is not None
            ),
            None,
        )
    technical = next(
        (item for item in evidence if item.stage == "technical"),
        None,
    )
    failure_reasons = tuple(
        (
            f"{item.stage}:{item.error_code or item.status}"
            + (f":{item.error_message}" if item.error_message else "")
        )
        for item in evidence
        if item.status not in {"passed"}
    )
    candidate_lineage = candidate["lineage"]
    windows = run["context"]["windows"]
    return CandidateScoreCard(
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_hash=candidate["candidate_hash"],
        semantic_hash=candidate["semantic_hash"],
        hypothesis_id=candidate_lineage["hypothesis_id"],
        batch_id=candidate_lineage["batch_id"],
        context_hash=run["context_hash"],
        status=candidate["status"],
        windows=dict(windows),
        decision_stage=decision_stage,
        metrics=metrics,
        metrics_by_stage=by_stage,
        analysis_by_stage=analysis_by_stage,
        gate_passed=(technical.status == "passed" if technical else None),
        failure_reasons=failure_reasons,
        evidence_ids=tuple(item.evaluation_id for item in evidence),
        evidence_hashes=tuple(item.evidence_hash for item in evidence),
        trial_counts=store.trial_counts(run_id),
        candidate_trial_counts=_candidate_trial_counts(
            [
                row for row in store.list_trials(run_id)
                if row["candidate_id"] == candidate_id
            ]
        ),
    )


def build_run_scorecards(
    store: MiningV2Store,
    run_id: str,
) -> tuple[CandidateScoreCard, ...]:
    """为运行内全部候选构建评分卡。"""

    trace = store.get_search_trace(run_id)
    return tuple(
        build_candidate_scorecard(store, run_id, item["candidate_id"])
        for item in trace["candidates"]
    )
