"""挖掘 V2 的冻结搜索预算和尝试计数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mining.contract_types import sha256_json
from mining.errors import MiningContractError
from mining.search_contract_types import (
    SEARCH_CONTEXT_SCHEMA_VERSION,
    bounded_integer,
    strict_fields,
)


@dataclass(frozen=True)
class SearchBudget:
    """一次 V2 搜索运行不可扩大的资源上限。"""

    max_search_turns: int = 6
    max_hypothesis_branches: int = 3
    max_refines_per_branch: int = 3
    max_candidates_per_turn: int = 4
    max_candidates_per_run: int = 24
    max_full_evaluations: int = 4
    stop_after_no_improvement_turns: int = 2
    max_tool_calls_per_turn: int = 4
    schema_version: str = SEARCH_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_CONTEXT_SCHEMA_VERSION:
            raise MiningContractError(
                f"搜索预算 schema_version 必须为 {SEARCH_CONTEXT_SCHEMA_VERSION}"
            )
        limits = (
            ("max_search_turns", self.max_search_turns, 1, 6),
            ("max_hypothesis_branches", self.max_hypothesis_branches, 1, 3),
            ("max_refines_per_branch", self.max_refines_per_branch, 0, 3),
            ("max_candidates_per_turn", self.max_candidates_per_turn, 1, 4),
            ("max_candidates_per_run", self.max_candidates_per_run, 1, 24),
            ("max_full_evaluations", self.max_full_evaluations, 1, 4),
            (
                "stop_after_no_improvement_turns",
                self.stop_after_no_improvement_turns,
                1,
                2,
            ),
            ("max_tool_calls_per_turn", self.max_tool_calls_per_turn, 0, 4),
        )
        for name, value, minimum, maximum in limits:
            object.__setattr__(
                self,
                name,
                bounded_integer(
                    value,
                    name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.max_candidates_per_turn > self.max_candidates_per_run:
            raise MiningContractError("max_candidates_per_turn 不得超过每运行候选上限")
        if self.max_full_evaluations > self.max_candidates_per_run:
            raise MiningContractError("max_full_evaluations 不得超过每运行候选上限")

    @classmethod
    def from_dict(cls, value: Any) -> SearchBudget:
        """严格解析搜索预算。"""

        fields = set(cls.__dataclass_fields__)
        return cls(**strict_fields(value, "search_budget", fields))

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return asdict(self)

    @property
    def budget_hash(self) -> str:
        """返回冻结预算哈希。"""

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class TrialCounts:
    """一次 V2 运行中不可被成功率掩盖的尝试分类计数。"""

    proposed: int = 0
    rejected: int = 0
    duplicate: int = 0
    invalid: int = 0
    gate_failed: int = 0
    evaluated: int = 0
    full_evaluated: int = 0
    holdout_claimed: int = 0
    invalidated: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MiningContractError(f"TrialCounts.{name} 必须是非负整数")

    @classmethod
    def from_dict(cls, value: Any) -> TrialCounts:
        """严格解析尝试计数。"""

        fields = set(cls.__dataclass_fields__)
        return cls(**strict_fields(value, "trial_counts", fields))

    def to_dict(self) -> dict[str, int]:
        """返回稳定持久化字典。"""

        return asdict(self)
