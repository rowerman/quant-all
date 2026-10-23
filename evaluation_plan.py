"""模型候选、参照和评分规则的通用评估计划。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from alpha.config import PROJECT_ROOT
from alpha.errors import AlphaEvaluationError

EVALUATION_PLAN_SCHEMA_VERSION = "alpha_model_evaluation_plan_v1"
DEFAULT_CURRENT_PLAN = (
    PROJECT_ROOT / "configs" / "alpha" / "final_holdout_2024_2026.json"
)


@dataclass(frozen=True)
class SourceSpec:
    """候选分数来源。"""

    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in {"model", "baseline"}:
            raise AlphaEvaluationError("source.kind 只支持 model/baseline")
        if not self.id:
            raise AlphaEvaluationError("source.id 不能为空")


@dataclass(frozen=True)
class ScorePolicy:
    """分数变换规则。"""

    kind: str = "raw"
    smoothing_weeks: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"raw", "smooth_rank"}:
            raise AlphaEvaluationError("score_policy.kind 只支持 raw/smooth_rank")
        if type(self.smoothing_weeks) is not int or self.smoothing_weeks <= 0:
            raise AlphaEvaluationError("smoothing_weeks 必须是正整数")
        if self.kind == "raw" and self.smoothing_weeks != 1:
            raise AlphaEvaluationError("raw 分数只能使用 smoothing_weeks=1")


@dataclass(frozen=True)
class EvaluationCandidate:
    """一个评估候选。"""

    name: str
    source: SourceSpec
    score_policy: ScorePolicy = field(default_factory=ScorePolicy)
    holding_policy: dict = field(
        default_factory=lambda: {"kind": "top_n", "top_n": 30}
    )

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise AlphaEvaluationError(f"非法候选名：{self.name!r}")
        kind = self.holding_policy.get("kind")
        if kind not in {"top_n", "rank_buffer"}:
            raise AlphaEvaluationError("holding_policy.kind 非法")
        top_n = self.holding_policy.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise AlphaEvaluationError("holding_policy.top_n 必须是正整数")
        if kind == "rank_buffer":
            buffer_rank = self.holding_policy.get("buffer_rank")
            if type(buffer_rank) is not int or buffer_rank < top_n:
                raise AlphaEvaluationError("rank_buffer 的 buffer_rank 必须不小于 top_n")


@dataclass(frozen=True)
class EvaluationPlan:
    """P3、完整账本和最终留出共用的评估计划。"""

    candidates: tuple[EvaluationCandidate, ...]
    reference: str
    controls: tuple[EvaluationCandidate, ...] = ()
    selected: str | None = None
    frozen_fold_id: str | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise AlphaEvaluationError("评估计划没有候选")
        names = [item.name for item in (*self.candidates, *self.controls)]
        if len(set(names)) != len(names):
            raise AlphaEvaluationError("评估计划候选名重复")
        if self.reference not in names:
            raise AlphaEvaluationError("评估计划参照不在候选中")
        if self.selected is not None and self.selected not in names:
            raise AlphaEvaluationError("评估计划选中候选不在候选中")

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EvaluationPlan":
        """从 JSON 对象构造计划。"""

        if payload.get("schema_version") != EVALUATION_PLAN_SCHEMA_VERSION:
            raise AlphaEvaluationError("评估计划 schema_version 不支持")
        candidates = tuple(
            _candidate(item) for item in payload.get("candidates") or ()
        )
        controls = tuple(
            _candidate(item) for item in payload.get("controls") or ()
        )
        return cls(
            candidates=candidates,
            reference=str(payload.get("reference") or ""),
            controls=controls,
            selected=payload.get("selected"),
            frozen_fold_id=payload.get("frozen_fold_id"),
        )

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationPlan":
        """读取 JSON 评估计划。"""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AlphaEvaluationError(f"无法读取评估计划：{path}") from error
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return {
            "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
            "reference": self.reference,
            "selected": self.selected,
            "frozen_fold_id": self.frozen_fold_id,
            "candidates": [_candidate_dict(item) for item in self.candidates],
            "controls": [_candidate_dict(item) for item in self.controls],
        }

    def by_name(self, name: str) -> EvaluationCandidate:
        """按名称返回候选。"""

        for item in (*self.candidates, *self.controls):
            if item.name == name:
                return item
        raise AlphaEvaluationError(f"评估计划没有候选：{name}")


def _candidate(payload: Mapping) -> EvaluationCandidate:
    source = payload.get("source") or {}
    score = payload.get("score_policy") or {}
    holding = payload.get("holding_policy") or {"kind": "top_n", "top_n": 30}
    return EvaluationCandidate(
        name=str(payload.get("name") or ""),
        source=SourceSpec(str(source.get("kind") or ""), str(source.get("id") or "")),
        score_policy=ScorePolicy(
            kind=str(score.get("kind") or "raw"),
            smoothing_weeks=int(score.get("smoothing_weeks", 1)),
        ),
        holding_policy=dict(holding),
    )


def _candidate_dict(candidate: EvaluationCandidate) -> dict:
    return {
        "name": candidate.name,
        "source": {"kind": candidate.source.kind, "id": candidate.source.id},
        "score_policy": {
            "kind": candidate.score_policy.kind,
            "smoothing_weeks": candidate.score_policy.smoothing_weeks,
        },
        "holding_policy": dict(candidate.holding_policy),
    }


def load_current_plan() -> EvaluationPlan:
    """读取当前正式评估计划。"""

    return EvaluationPlan.from_file(DEFAULT_CURRENT_PLAN)


def current_model_ids() -> tuple[str, str]:
    """返回当前计划的主模型和参照模型 ID。"""

    plan = load_current_plan()
    selected = plan.selected or plan.candidates[0].name
    return plan.by_name(selected).source.id, plan.by_name(plan.reference).source.id
