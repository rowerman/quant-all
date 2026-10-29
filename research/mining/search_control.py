"""不依赖模型的搜索预算、停止和质量动作控制器。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mining.contract_types import identifier, sha256_json
from mining.errors import MiningContractError
from mining.search_budget import SearchBudget
from mining.search_evidence import EvidencePacket, SearchDecision
from mining.search_contract_types import (
    SEARCH_ACTION_VALUES,
    bounded_integer,
    choice,
    strict_fields,
)

SEARCH_CONTROL_SCHEMA_VERSION = "mining_search_control_v1"


@dataclass(frozen=True)
class SearchControlPolicy:
    """运行开始时冻结的研究质量与停止阈值。"""

    min_rank_ic: float = 0.02
    min_icir: float = 0.20
    min_year_sign_consistency: float = 0.60
    min_coverage: float = 0.30
    max_failure_rate: float = 0.75
    min_quality_checks_for_failure_stop: int = 4
    correlation_cluster_threshold: float = 0.90
    stability_variant_limit: int = 2
    stability_rank_ic_ratio_floor: float = 0.50
    schema_version: str = SEARCH_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_CONTROL_SCHEMA_VERSION:
            raise MiningContractError(
                "控制策略 schema_version 必须为 "
                f"{SEARCH_CONTROL_SCHEMA_VERSION}"
            )
        for name in (
            "min_rank_ic",
            "min_icir",
            "min_year_sign_consistency",
            "min_coverage",
            "max_failure_rate",
            "correlation_cluster_threshold",
            "stability_rank_ic_ratio_floor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise MiningContractError(f"{name} 必须位于 0～1")
            object.__setattr__(self, name, float(value))
        object.__setattr__(
            self,
            "min_quality_checks_for_failure_stop",
            bounded_integer(
                self.min_quality_checks_for_failure_stop,
                "min_quality_checks_for_failure_stop",
                minimum=1,
                maximum=24,
            ),
        )
        object.__setattr__(
            self,
            "stability_variant_limit",
            bounded_integer(
                self.stability_variant_limit,
                "stability_variant_limit",
                minimum=1,
                maximum=2,
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> SearchControlPolicy:
        """解析冻结控制策略，并为旧上下文补齐新增默认值。"""

        if not isinstance(value, Mapping):
            raise MiningContractError("search_control_policy 必须是对象")
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise MiningContractError(
                f"search_control_policy 包含未知字段：{sorted(unknown)}"
            )
        defaults = cls()
        payload = {
            **{
                name: getattr(defaults, name)
                for name in cls.__dataclass_fields__
            },
            **dict(value),
        }
        return cls(**strict_fields(payload, "search_control_policy", fields))

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @property
    def policy_hash(self) -> str:
        """返回控制策略哈希。"""

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class CandidateQualityScore:
    """用于确定性比较的开发期质量向量。"""

    candidate_id: str
    abs_rank_ic: float
    abs_icir: float
    year_sign_consistency: float
    coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            identifier(self.candidate_id, "candidate_id"),
        )
        for name in (
            "abs_rank_ic",
            "abs_icir",
            "year_sign_consistency",
            "coverage",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise MiningContractError(f"{name} 必须是非负有限数值")
            object.__setattr__(self, name, float(value))

    @property
    def ordering(self) -> tuple[float, float, float, float, str]:
        """返回稳定降序排序键。"""

        return (
            self.abs_rank_ic,
            self.abs_icir,
            self.year_sign_consistency,
            self.coverage,
            self.candidate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回适合证据和页面展示的字典。"""

        return {
            "candidate_id": self.candidate_id,
            "abs_rank_ic": self.abs_rank_ic,
            "abs_icir": self.abs_icir,
            "year_sign_consistency": self.year_sign_consistency,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class SearchControlState:
    """控制器从当前运行状态读取的可恢复变量。"""

    branches_used: int = 1
    refines_in_branch: int = 0
    no_improvement_turns: int = 0
    previous_best: tuple[float, float, float, float] | None = None
    proposed_action: str | None = None
    cancelled: bool = False
    snapshot_stale: bool = False

    def __post_init__(self) -> None:
        for name in ("branches_used", "refines_in_branch", "no_improvement_turns"):
            object.__setattr__(
                self,
                name,
                bounded_integer(
                    getattr(self, name),
                    name,
                    minimum=0,
                    maximum=1000,
                ),
            )
        if self.previous_best is not None:
            if len(self.previous_best) != 4:
                raise MiningContractError("previous_best 必须包含四个数值")
            values = tuple(float(item) for item in self.previous_best)
            if any(not math.isfinite(item) or item < 0 for item in values):
                raise MiningContractError("previous_best 必须是非负有限数值")
            object.__setattr__(self, "previous_best", values)
        if self.proposed_action is not None:
            object.__setattr__(
                self,
                "proposed_action",
                choice(
                    self.proposed_action,
                    "proposed_action",
                    SEARCH_ACTION_VALUES,
                ),
            )
        for name in ("cancelled", "snapshot_stale"):
            if not isinstance(getattr(self, name), bool):
                raise MiningContractError(f"{name} 必须是布尔值")


@dataclass(frozen=True)
class SearchControlResult:
    """控制器最终动作及可审计判断依据。"""

    decision: SearchDecision
    best_candidate: CandidateQualityScore | None
    no_improvement_turns: int
    failure_rate: float
    quality_candidates: tuple[str, ...]
    policy_reason: str

    def to_dict(self) -> dict[str, Any]:
        """返回页面和事件使用的摘要。"""

        return {
            "decision_id": self.decision.decision_id,
            "action": self.decision.action,
            "reason": self.decision.reason,
            "best_candidate": (
                self.best_candidate.to_dict()
                if self.best_candidate is not None
                else None
            ),
            "no_improvement_turns": self.no_improvement_turns,
            "failure_rate": self.failure_rate,
            "quality_candidates": list(self.quality_candidates),
            "policy_reason": self.policy_reason,
        }


def _metric_number(
    metrics: Mapping[str, Any],
    name: str,
) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MiningContractError(f"候选指标 {name} 必须是有限数值或 None")
    return float(value)


def _candidate_scores(
    evidence: EvidencePacket,
) -> dict[str, CandidateQualityScore]:
    result: dict[str, CandidateQualityScore] = {}
    for candidate_id, raw in evidence.metrics.items():
        if not isinstance(raw, Mapping):
            raise MiningContractError("候选证据指标必须是对象")
        rank_ic = _metric_number(raw, "rank_ic")
        icir = _metric_number(raw, "rank_ic_ir")
        year_sign = _metric_number(raw, "year_sign_consistency")
        coverage = _metric_number(raw, "coverage")
        if None in (rank_ic, icir, year_sign, coverage):
            continue
        result[str(candidate_id)] = CandidateQualityScore(
            candidate_id=str(candidate_id),
            abs_rank_ic=abs(rank_ic),
            abs_icir=abs(icir),
            year_sign_consistency=year_sign,
            coverage=coverage,
        )
    return result


def _passes_quality(
    candidate: CandidateQualityScore,
    policy: SearchControlPolicy,
) -> bool:
    return (
        candidate.abs_rank_ic >= policy.min_rank_ic
        and candidate.abs_icir >= policy.min_icir
        and candidate.year_sign_consistency >= policy.min_year_sign_consistency
        and candidate.coverage >= policy.min_coverage
    )


def _fallback_action(
    *,
    budget: SearchBudget,
    state: SearchControlState,
    quality_candidates: tuple[str, ...],
) -> str:
    if state.refines_in_branch < budget.max_refines_per_branch:
        return "refine"
    if state.branches_used < budget.max_hypothesis_branches:
        return "branch"
    if quality_candidates:
        return "promote"
    return "stop"


def decide_search_action(
    evidence: EvidencePacket,
    budget: SearchBudget,
    policy: SearchControlPolicy,
    state: SearchControlState,
) -> SearchControlResult:
    """按硬约束优先级计算本轮最终搜索动作。"""

    if not isinstance(evidence, EvidencePacket):
        raise MiningContractError("evidence 必须是 EvidencePacket")
    if not isinstance(budget, SearchBudget):
        raise MiningContractError("budget 必须是 SearchBudget")
    if not isinstance(policy, SearchControlPolicy):
        raise MiningContractError("policy 必须是 SearchControlPolicy")
    if not isinstance(state, SearchControlState):
        raise MiningContractError("state 必须是 SearchControlState")

    scores = _candidate_scores(evidence)
    best = max(scores.values(), key=lambda item: item.ordering, default=None)
    quality_candidates = tuple(
        sorted(
            (
                item.candidate_id
                for item in scores.values()
                if _passes_quality(item, policy)
            )
        )
    )
    current_best = best.ordering[:4] if best is not None else None
    improved = (
        state.previous_best is None and current_best is not None
    ) or (
        state.previous_best is not None
        and (current_best is None or current_best > state.previous_best)
    )
    no_improvement = (
        0
        if improved
        else state.no_improvement_turns + 1
    )
    counts = evidence.trial_counts
    attempted = max(counts.proposed, len(evidence.metrics), 1)
    failures = counts.rejected + counts.invalid + counts.gate_failed
    failure_rate = min(1.0, failures / attempted)
    turns_used = int(evidence.budget_state.get("turns_used", evidence.turn_index))
    budget_reason = ""
    action: str

    if state.cancelled:
        action = "stop"
        budget_reason = "运行已请求取消"
    elif state.snapshot_stale:
        action = "stop"
        budget_reason = "冻结快照已失效"
    elif not scores:
        if state.branches_used < budget.max_hypothesis_branches:
            action = "branch"
            budget_reason = "本轮没有可用于判断质量的开发期指标"
        else:
            action = "stop"
            budget_reason = "没有有效候选且分支预算已耗尽"
    elif turns_used >= budget.max_search_turns:
        if quality_candidates:
            action = "promote"
            budget_reason = "搜索轮次预算耗尽，存在质量候选"
        else:
            action = "stop"
            budget_reason = "搜索轮次预算耗尽且没有质量候选"
    elif counts.proposed >= budget.max_candidates_per_run:
        if quality_candidates:
            action = "promote"
            budget_reason = "候选预算耗尽，存在质量候选"
        else:
            action = "stop"
            budget_reason = "候选预算耗尽且没有质量候选"
    elif no_improvement >= budget.stop_after_no_improvement_turns:
        if state.branches_used < budget.max_hypothesis_branches:
            action = "branch"
            budget_reason = "连续无改善达到停止阈值，切换分支"
        else:
            action = "stop"
            budget_reason = "连续无改善且分支预算已耗尽"
    elif (
        counts.proposed >= policy.min_quality_checks_for_failure_stop
        and failure_rate >= policy.max_failure_rate
    ):
        if state.branches_used < budget.max_hypothesis_branches:
            action = "branch"
            budget_reason = "失败率达到阈值，切换分支"
        else:
            action = "stop"
            budget_reason = "失败率达到阈值且分支预算已耗尽"
    elif state.refines_in_branch >= budget.max_refines_per_branch:
        if state.branches_used < budget.max_hypothesis_branches:
            action = "branch"
            budget_reason = "当前分支修正次数耗尽"
        elif quality_candidates:
            action = "promote"
            budget_reason = "修正次数耗尽，存在质量候选"
        else:
            action = "stop"
            budget_reason = "修正次数耗尽且没有质量候选"
    else:
        proposed = state.proposed_action
        legal = {
            "refine": True,
            "branch": state.branches_used < budget.max_hypothesis_branches,
            "stop": True,
            "promote": bool(quality_candidates),
        }
        if proposed is not None and legal[proposed]:
            action = proposed
            budget_reason = f"采用模型建议 {proposed}"
        else:
            action = _fallback_action(
                budget=budget,
                state=state,
                quality_candidates=quality_candidates,
            )
            budget_reason = (
                f"模型建议 {proposed or '缺失'} 不满足约束，回退为 {action}"
            )

    reason = (
        f"{budget_reason}；失败率={failure_rate:.4f}，"
        f"连续无改善={no_improvement}，质量候选={len(quality_candidates)}"
    )
    decision = SearchDecision(
        run_id=evidence.run_id,
        hypothesis_id=evidence.hypothesis_id,
        evidence_id=evidence.evidence_id,
        turn_index=evidence.turn_index,
        action=action,
        reason=reason,
    )
    next_no_improvement = 0 if action == "branch" else no_improvement
    return SearchControlResult(
        decision=decision,
        best_candidate=best,
        no_improvement_turns=next_no_improvement,
        failure_rate=failure_rate,
        quality_candidates=quality_candidates,
        policy_reason=budget_reason,
    )
