"""模型候选、参照、评分和决策规则的通用评估计划。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from alpha.config import ALPHA_CONFIG_DIR
from alpha.errors import AlphaEvaluationError
from alpha.risk import RiskGatePolicy

EVALUATION_PLAN_SCHEMA_VERSION = "alpha_model_evaluation_plan_v3"
V2_EVALUATION_PLAN_SCHEMA_VERSION = "alpha_model_evaluation_plan_v2"
LEGACY_EVALUATION_PLAN_SCHEMA_VERSION = "alpha_model_evaluation_plan_v1"
DEFAULT_CURRENT_PLAN = (
    ALPHA_CONFIG_DIR / "final_holdout_2024_2026.json"
)
DEFAULT_SELECTION_METRICS = (
    "annual_return",
    "sharpe",
    "max_drawdown",
    "turnover_two_way_annual",
)


@dataclass(frozen=True)
class SourceSpec:
    """候选分数来源；融合来源由多个模型及其权重定义。"""

    kind: str
    id: str
    members: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"model", "baseline", "blend"}:
            raise AlphaEvaluationError(
                "source.kind 只支持 model/baseline/blend"
            )
        if not isinstance(self.id, str) or not self.id:
            raise AlphaEvaluationError("source.id 不能为空")
        members = tuple(str(item) for item in self.members)
        weights = tuple(float(item) for item in self.weights)
        if self.kind != "blend":
            if members or weights:
                raise AlphaEvaluationError("非融合来源不能设置 members/weights")
        else:
            if len(members) < 2 or len(set(members)) != len(members):
                raise AlphaEvaluationError(
                    "融合来源至少需要两个不重复成员"
                )
            if len(weights) != len(members):
                raise AlphaEvaluationError(
                    "融合来源的成员数与权重数必须一致"
                )
            if any(not np.isfinite(item) or item < 0 for item in weights):
                raise AlphaEvaluationError("融合权重必须是非负有限数")
            if sum(weights) <= 0:
                raise AlphaEvaluationError("融合权重至少一个必须为正")
            normalized = tuple(item / sum(weights) for item in weights)
            weights = normalized
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "weights", weights)

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        payload = {"kind": self.kind, "id": self.id}
        if self.kind == "blend":
            payload["members"] = list(self.members)
            payload["weights"] = list(self.weights)
        return payload


@dataclass(frozen=True)
class ScorePolicy:
    """分数变换规则。"""

    kind: str = "raw"
    smoothing_weeks: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"raw", "smooth_rank"}:
            raise AlphaEvaluationError(
                "score_policy.kind 只支持 raw/smooth_rank"
            )
        if type(self.smoothing_weeks) is not int or self.smoothing_weeks <= 0:
            raise AlphaEvaluationError("smoothing_weeks 必须是正整数")
        if self.kind == "raw" and self.smoothing_weeks != 1:
            raise AlphaEvaluationError(
                "raw 分数只能使用 smoothing_weeks=1"
            )


@dataclass(frozen=True)
class EvaluationCandidate:
    """一个评估候选及其可选择性。"""

    name: str
    source: SourceSpec
    score_policy: ScorePolicy = field(default_factory=ScorePolicy)
    holding_policy: dict = field(
        default_factory=lambda: {"kind": "top_n", "top_n": 30}
    )
    priority: int = 0
    selectable: bool = True

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise AlphaEvaluationError(f"非法候选名：{self.name!r}")
        kind = self.holding_policy.get("kind")
        if kind not in {"top_n", "rank_buffer"}:
            raise AlphaEvaluationError("holding_policy.kind 非法")
        top_n = self.holding_policy.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise AlphaEvaluationError(
                "holding_policy.top_n 必须是正整数"
            )
        if kind == "rank_buffer":
            buffer_rank = self.holding_policy.get("buffer_rank")
            if type(buffer_rank) is not int or buffer_rank < top_n:
                raise AlphaEvaluationError(
                    "rank_buffer 的 buffer_rank 必须不小于 top_n"
                )
        if type(self.priority) is not int:
            raise AlphaEvaluationError("priority 必须是整数")
        if not isinstance(self.selectable, bool):
            raise AlphaEvaluationError("selectable 必须是布尔值")
        if self.source.kind in {"baseline", "blend"} and self.selectable:
            raise AlphaEvaluationError(
                "baseline/blend 候选必须显式设置 selectable=false"
            )


@dataclass(frozen=True)
class GatePolicy:
    """P3 决策门参数。"""

    expected_years: tuple[int, ...] = ()
    required_improvements: int = 2
    turnover_multiple_limit: float = 2.0

    def __post_init__(self) -> None:
        years = tuple(int(year) for year in self.expected_years)
        if any(year < 1900 or year > 2200 for year in years):
            raise AlphaEvaluationError("expected_years 包含非法年份")
        if len(set(years)) != len(years):
            raise AlphaEvaluationError("expected_years 不能重复")
        if type(self.required_improvements) is not int or not (
            1 <= self.required_improvements <= 4
        ):
            raise AlphaEvaluationError("required_improvements 必须在 1 到 4 之间")
        if (
            not np.isfinite(float(self.turnover_multiple_limit))
            or float(self.turnover_multiple_limit) <= 0
        ):
            raise AlphaEvaluationError("turnover_multiple_limit 必须是正有限数")
        object.__setattr__(self, "expected_years", years)


@dataclass(frozen=True)
class EvaluationPlan:
    """开发、完整账本和最终留出共用的评估计划。"""

    candidates: tuple[EvaluationCandidate, ...]
    reference: str
    controls: tuple[EvaluationCandidate, ...] = ()
    selected: str | None = None
    frozen_fold_id: str | None = None
    gate: GatePolicy = field(default_factory=GatePolicy)
    risk_gate: RiskGatePolicy | None = None
    selection_metrics: tuple[str, ...] = DEFAULT_SELECTION_METRICS

    def __post_init__(self) -> None:
        if not self.candidates:
            raise AlphaEvaluationError("评估计划没有候选")
        names = [item.name for item in (*self.candidates, *self.controls)]
        if len(set(names)) != len(names):
            raise AlphaEvaluationError("评估计划候选名重复")
        if not self.reference or self.reference not in names:
            raise AlphaEvaluationError("评估计划参照不在候选中")
        if self.selected is not None and self.selected not in names:
            raise AlphaEvaluationError("评估计划选中候选不在候选中")
        control_names = {item.name for item in self.controls}
        reference = self.by_name(self.reference)
        if reference.name in control_names:
            raise AlphaEvaluationError("参照不能同时是控制候选")
        metrics = tuple(str(item) for item in self.selection_metrics)
        if not metrics or len(set(metrics)) != len(metrics):
            raise AlphaEvaluationError("selection_metrics 必须非空且不重复")
        allowed = {
            "annual_return",
            "sharpe",
            "max_drawdown",
            "turnover_two_way_annual",
        }
        unsupported = sorted(set(metrics) - allowed)
        if unsupported:
            raise AlphaEvaluationError(
                f"selection_metrics 包含不支持字段：{unsupported}"
            )
        object.__setattr__(self, "selection_metrics", metrics)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EvaluationPlan":
        """从 v1/v2 JSON 对象构造计划。"""

        schema = payload.get("schema_version")
        if schema not in {
            LEGACY_EVALUATION_PLAN_SCHEMA_VERSION,
            V2_EVALUATION_PLAN_SCHEMA_VERSION,
            EVALUATION_PLAN_SCHEMA_VERSION,
        }:
            raise AlphaEvaluationError("评估计划 schema_version 不支持")
        candidates = tuple(
            _candidate(item) for item in payload.get("candidates") or ()
        )
        controls = tuple(
            _candidate(item) for item in payload.get("controls") or ()
        )
        gate_payload = payload.get("gate") or {}
        gate = GatePolicy(
            expected_years=tuple(gate_payload.get("expected_years") or ()),
            required_improvements=int(
                gate_payload.get("required_improvements", 2)
            ),
            turnover_multiple_limit=float(
                gate_payload.get("turnover_multiple_limit", 2.0)
            ),
        )
        metrics = payload.get("selection_metrics")
        risk_payload = payload.get("risk_gate")
        risk_gate = (
            None
            if risk_payload is None
            else RiskGatePolicy(**dict(risk_payload))
        )
        return cls(
            candidates=candidates,
            reference=str(payload.get("reference") or ""),
            controls=controls,
            selected=payload.get("selected"),
            frozen_fold_id=payload.get("frozen_fold_id"),
            gate=gate,
            risk_gate=risk_gate,
            selection_metrics=(
                DEFAULT_SELECTION_METRICS
                if metrics is None
                else tuple(str(item) for item in metrics)
            ),
        )

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationPlan":
        """读取 JSON 评估计划。"""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AlphaEvaluationError(f"无法读取评估计划：{path}") from error
        if not isinstance(payload, Mapping):
            raise AlphaEvaluationError("评估计划必须是 JSON 对象")
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        """转换为稳定 v2 字典。"""

        return {
            "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
            "reference": self.reference,
            "selected": self.selected,
            "frozen_fold_id": self.frozen_fold_id,
            "gate": {
                "expected_years": list(self.gate.expected_years),
                "required_improvements": self.gate.required_improvements,
                "turnover_multiple_limit": self.gate.turnover_multiple_limit,
            },
            "risk_gate": (
                None if self.risk_gate is None else self.risk_gate.to_dict()
            ),
            "selection_metrics": list(self.selection_metrics),
            "candidates": [_candidate_dict(item) for item in self.candidates],
            "controls": [_candidate_dict(item) for item in self.controls],
        }

    def by_name(self, name: str) -> EvaluationCandidate:
        """按名称返回候选。"""

        for item in (*self.candidates, *self.controls):
            if item.name == name:
                return item
        raise AlphaEvaluationError(f"评估计划没有候选：{name}")

    def model_ids(self) -> tuple[str, ...]:
        """返回计划直接或通过融合引用的全部模型 ID。"""

        identifiers: list[str] = []
        for candidate in (*self.candidates, *self.controls):
            source = candidate.source
            if source.kind == "model":
                identifiers.append(source.id)
            elif source.kind == "blend":
                identifiers.extend(source.members)
        return tuple(dict.fromkeys(identifiers))


def _candidate(payload: Mapping) -> EvaluationCandidate:
    source = payload.get("source") or {}
    score = payload.get("score_policy") or {}
    holding = payload.get("holding_policy") or {
        "kind": "top_n",
        "top_n": 30,
    }
    kind = str(source.get("kind") or "")
    selectable = payload.get("selectable", kind == "model")
    if not isinstance(selectable, bool):
        raise AlphaEvaluationError("selectable 必须是布尔值")
    return EvaluationCandidate(
        name=str(payload.get("name") or ""),
        source=SourceSpec(
            kind=kind,
            id=str(source.get("id") or ""),
            members=tuple(source.get("members") or ()),
            weights=tuple(source.get("weights") or ()),
        ),
        score_policy=ScorePolicy(
            kind=str(score.get("kind") or "raw"),
            smoothing_weeks=int(score.get("smoothing_weeks", 1)),
        ),
        holding_policy=dict(holding),
        priority=int(payload.get("priority", 0)),
        selectable=selectable,
    )


def _candidate_dict(candidate: EvaluationCandidate) -> dict:
    return {
        "name": candidate.name,
        "source": candidate.source.to_dict(),
        "score_policy": {
            "kind": candidate.score_policy.kind,
            "smoothing_weeks": candidate.score_policy.smoothing_weeks,
        },
        "holding_policy": dict(candidate.holding_policy),
        "priority": candidate.priority,
        "selectable": candidate.selectable,
    }


def load_current_plan() -> EvaluationPlan:
    """读取当前正式评估计划。"""

    return EvaluationPlan.from_file(DEFAULT_CURRENT_PLAN)


def current_model_ids() -> tuple[str, str]:
    """返回当前计划的主模型和参照模型 ID。"""

    plan = load_current_plan()
    selected = plan.selected or plan.candidates[0].name
    selected_source = plan.by_name(selected).source
    reference_source = plan.by_name(plan.reference).source
    if selected_source.kind != "model" or reference_source.kind != "model":
        raise AlphaEvaluationError("当前计划的主模型或参照不是单一模型")
    return selected_source.id, reference_source.id


__all__ = [
    "DEFAULT_CURRENT_PLAN",
    "DEFAULT_SELECTION_METRICS",
    "EVALUATION_PLAN_SCHEMA_VERSION",
    "V2_EVALUATION_PLAN_SCHEMA_VERSION",
    "EvaluationCandidate",
    "EvaluationPlan",
    "GatePolicy",
    "LEGACY_EVALUATION_PLAN_SCHEMA_VERSION",
    "ScorePolicy",
    "SourceSpec",
    "current_model_ids",
    "load_current_plan",
]
