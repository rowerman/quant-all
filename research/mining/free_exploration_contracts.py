"""自由探索初始研究方向的公共契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.contract_types import identifier, sha256_json, stable_id, text
from mining.errors import MiningContractError
from mining.search_contract_types import (
    SEARCH_DIRECTION_VALUES,
    bounded_integer,
    choice,
    hash_text,
    strict_fields,
)

FREE_EXPLORATION_SCHEMA_VERSION = "mining_free_exploration_v1"
DIRECTION_SEMANTIC_SCHEMA_VERSION = "mining_free_exploration_semantic_v1"
EXPRESSION_COMPLEXITY_VALUES = ("simple", "intermediate", "compound")
EXPRESSION_DEPTH_BY_COMPLEXITY = {
    "simple": 2,
    "intermediate": 4,
    "compound": 6,
}
EXPLORATION_SEED_SOURCE_VALUES = ("request", "request_hash")
EXPLORATION_SEED_MAX = 2**64 - 1


@dataclass(frozen=True)
class ResearchDirection:
    """自由探索首轮可审计的研究方向。"""

    run_id: str
    mechanism_category: str
    statement: str
    mechanism: str
    expected_direction: str
    falsification_criteria: tuple[str, ...]
    data_dependencies: tuple[str, ...]
    expression_complexity: str
    max_expression_depth: int
    seed: int
    seed_source: str
    request_hash: str
    context_hash: str
    profile_hash: str
    schema_version: str = FREE_EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FREE_EXPLORATION_SCHEMA_VERSION:
            raise MiningContractError(
                "ResearchDirection schema_version 必须为 "
                f"{FREE_EXPLORATION_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "mechanism_category",
            identifier(self.mechanism_category, "mechanism_category"),
        )
        object.__setattr__(
            self,
            "statement",
            text(self.statement, "statement", minimum=8, maximum=4000),
        )
        object.__setattr__(
            self,
            "mechanism",
            text(self.mechanism, "mechanism", maximum=1000),
        )
        object.__setattr__(
            self,
            "expected_direction",
            choice(
                self.expected_direction,
                "expected_direction",
                SEARCH_DIRECTION_VALUES,
            ),
        )
        if (
            isinstance(self.falsification_criteria, (str, bytes))
            or not isinstance(self.falsification_criteria, (list, tuple))
            or not 1 <= len(self.falsification_criteria) <= 12
        ):
            raise MiningContractError(
                "falsification_criteria 必须包含 1～12 条字符串"
            )
        criteria: list[str] = []
        for item in self.falsification_criteria:
            criterion = text(
                item,
                "falsification_criteria",
                maximum=500,
            )
            if criterion not in criteria:
                criteria.append(criterion)
        object.__setattr__(self, "falsification_criteria", tuple(criteria))
        if (
            isinstance(self.data_dependencies, (str, bytes))
            or not isinstance(self.data_dependencies, (list, tuple))
            or not 1 <= len(self.data_dependencies) <= 8
        ):
            raise MiningContractError("data_dependencies 必须包含 1～8 个字段")
        dependencies = tuple(
            identifier(item, "data_dependencies")
            for item in self.data_dependencies
        )
        if len(dependencies) != len(set(dependencies)):
            raise MiningContractError("data_dependencies 不得重复")
        object.__setattr__(self, "data_dependencies", dependencies)
        object.__setattr__(
            self,
            "expression_complexity",
            choice(
                self.expression_complexity,
                "expression_complexity",
                EXPRESSION_COMPLEXITY_VALUES,
            ),
        )
        depth = bounded_integer(
            self.max_expression_depth,
            "max_expression_depth",
            minimum=2,
            maximum=8,
        )
        if depth != EXPRESSION_DEPTH_BY_COMPLEXITY[self.expression_complexity]:
            raise MiningContractError(
                "max_expression_depth 与 expression_complexity 不一致"
            )
        object.__setattr__(self, "max_expression_depth", depth)
        object.__setattr__(
            self,
            "seed",
            bounded_integer(
                self.seed,
                "seed",
                minimum=0,
                maximum=EXPLORATION_SEED_MAX,
            ),
        )
        object.__setattr__(
            self,
            "seed_source",
            choice(
                self.seed_source,
                "seed_source",
                EXPLORATION_SEED_SOURCE_VALUES,
            ),
        )
        object.__setattr__(
            self,
            "request_hash",
            hash_text(self.request_hash, "request_hash"),
        )
        object.__setattr__(
            self,
            "context_hash",
            hash_text(self.context_hash, "context_hash"),
        )
        object.__setattr__(
            self,
            "profile_hash",
            hash_text(self.profile_hash, "profile_hash"),
        )

    def material(self) -> dict[str, Any]:
        """返回方向内容哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mechanism_category": self.mechanism_category,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
            "data_dependencies": list(self.data_dependencies),
            "expression_complexity": self.expression_complexity,
            "max_expression_depth": self.max_expression_depth,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "request_hash": self.request_hash,
            "context_hash": self.context_hash,
            "profile_hash": self.profile_hash,
        }

    def semantic_material(self) -> dict[str, Any]:
        """返回用于判断研究方向语义重复的稳定输入。"""

        return {
            "schema_version": DIRECTION_SEMANTIC_SCHEMA_VERSION,
            "mechanism_category": self.mechanism_category,
            "data_dependencies": sorted(self.data_dependencies),
            "expected_direction": self.expected_direction,
            "expression_complexity": self.expression_complexity,
        }

    @property
    def semantic_hash(self) -> str:
        """返回忽略措辞、种子和展示哈希的方向语义哈希。"""

        return sha256_json(self.semantic_material())

    @property
    def direction_hash(self) -> str:
        """返回方向内容哈希。"""

        return sha256_json(self.material())

    @property
    def direction_id(self) -> str:
        """返回运行内稳定方向 ID。"""

        return stable_id(
            "direction",
            {"run_id": self.run_id, "direction_hash": self.direction_hash},
        )

    @classmethod
    def from_dict(cls, value: Any) -> ResearchDirection:
        """严格解析持久化方向。"""

        payload = strict_fields(
            value,
            "research_direction",
            set(cls.__dataclass_fields__),
        )
        payload["falsification_criteria"] = tuple(
            payload["falsification_criteria"]
        )
        payload["data_dependencies"] = tuple(payload["data_dependencies"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mechanism_category": self.mechanism_category,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
            "data_dependencies": list(self.data_dependencies),
            "expression_complexity": self.expression_complexity,
            "max_expression_depth": self.max_expression_depth,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "request_hash": self.request_hash,
            "context_hash": self.context_hash,
            "profile_hash": self.profile_hash,
        }


__all__ = [
    "DIRECTION_SEMANTIC_SCHEMA_VERSION",
    "EXPLORATION_SEED_MAX",
    "EXPLORATION_SEED_SOURCE_VALUES",
    "EXPRESSION_COMPLEXITY_VALUES",
    "EXPRESSION_DEPTH_BY_COMPLEXITY",
    "FREE_EXPLORATION_SCHEMA_VERSION",
    "ResearchDirection",
]
