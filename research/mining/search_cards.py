"""挖掘 V2 的假设卡和实验批次契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    ensure_string_tuple,
    identifier,
    sha256_json,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError
from mining.search_contract_types import (
    SEARCH_DIRECTION_VALUES,
    SEARCH_SCHEMA_VERSION,
    bounded_integer,
    choice,
    hash_text,
    mapping_with_size,
    nullable_identifier,
    strict_fields,
    validate_date_range,
)


@dataclass(frozen=True)
class HypothesisCard:
    """一条可证伪、可追踪来源的研究假设。"""

    run_id: str
    turn_index: int
    branch_id: str
    statement: str
    mechanism: str
    expected_direction: str
    falsification_criteria: tuple[str, ...]
    parent_hypothesis_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"HypothesisCard schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(self.turn_index, "turn_index", minimum=1, maximum=1000),
        )
        object.__setattr__(self, "branch_id", identifier(self.branch_id, "branch_id"))
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
        object.__setattr__(
            self,
            "falsification_criteria",
            ensure_string_tuple(
                self.falsification_criteria,
                "falsification_criteria",
                minimum=1,
                maximum=12,
            ),
        )
        for item in self.falsification_criteria:
            if len(item) > 500:
                raise MiningContractError("每条可证伪条件不得超过 500 个字符")
        object.__setattr__(
            self,
            "parent_hypothesis_id",
            nullable_identifier(self.parent_hypothesis_id, "parent_hypothesis_id"),
        )
        object.__setattr__(
            self,
            "metadata",
            mapping_with_size(self.metadata, "metadata", maximum_bytes=16384),
        )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "metadata": self.metadata,
        }

    @property
    def card_hash(self) -> str:
        """返回假设内容哈希。"""

        return sha256_json(self.material())

    @property
    def hypothesis_id(self) -> str:
        """返回运行内稳定假设 ID。"""

        return stable_id(
            "hyp",
            {"run_id": self.run_id, "card_hash": self.card_hash},
        )

    @classmethod
    def from_dict(cls, value: Any) -> HypothesisCard:
        """严格解析假设卡。"""

        fields = set(cls.__dataclass_fields__)
        return cls(**strict_fields(value, "hypothesis_card", fields))

    def to_dict(self) -> dict[str, Any]:
        """返回不包含派生 ID 的持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ExperimentBatch:
    """一轮搜索提交给确定性执行链的候选集合。"""

    run_id: str
    hypothesis_id: str
    turn_index: int
    branch_id: str
    candidate_ids: tuple[str, ...]
    parent_candidate_ids: tuple[str, ...]
    development_start: str
    development_end: str
    input_hash: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"ExperimentBatch schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "hypothesis_id",
            identifier(self.hypothesis_id, "hypothesis_id"),
        )
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(self.turn_index, "turn_index", minimum=1, maximum=1000),
        )
        object.__setattr__(self, "branch_id", identifier(self.branch_id, "branch_id"))
        object.__setattr__(
            self,
            "candidate_ids",
            ensure_string_tuple(
                self.candidate_ids,
                "candidate_ids",
                minimum=1,
                maximum=4,
            ),
        )
        object.__setattr__(
            self,
            "parent_candidate_ids",
            ensure_string_tuple(
                self.parent_candidate_ids,
                "parent_candidate_ids",
                maximum=4,
            ),
        )
        start, end = validate_date_range(
            self.development_start,
            self.development_end,
            "development",
        )
        object.__setattr__(self, "development_start", start)
        object.__setattr__(self, "development_end", end)
        object.__setattr__(self, "input_hash", hash_text(self.input_hash, "input_hash"))
        object.__setattr__(
            self,
            "parameters",
            mapping_with_size(self.parameters, "parameters", maximum_bytes=16384),
        )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "hypothesis_id": self.hypothesis_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "candidate_ids": list(self.candidate_ids),
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "development_start": self.development_start,
            "development_end": self.development_end,
            "input_hash": self.input_hash,
            "parameters": self.parameters,
        }

    @property
    def batch_hash(self) -> str:
        """返回实验批次内容哈希。"""

        return sha256_json(self.material())

    @property
    def batch_id(self) -> str:
        """返回运行内稳定批次 ID。"""

        return stable_id(
            "batch",
            {"run_id": self.run_id, "batch_hash": self.batch_hash},
        )

    @classmethod
    def from_dict(cls, value: Any) -> ExperimentBatch:
        """严格解析实验批次。"""

        payload = strict_fields(value, "experiment_batch", set(cls.__dataclass_fields__))
        for name in ("candidate_ids", "parent_candidate_ids"):
            payload[name] = tuple(payload[name])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "hypothesis_id": self.hypothesis_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "candidate_ids": list(self.candidate_ids),
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "development_start": self.development_start,
            "development_end": self.development_end,
            "input_hash": self.input_hash,
            "parameters": dict(self.parameters),
            "created_at": self.created_at,
        }
