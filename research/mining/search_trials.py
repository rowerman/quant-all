"""挖掘 V2 的不可删除尝试记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    identifier,
    sha256_json,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError
from mining.search_contract_types import (
    SEARCH_SCHEMA_VERSION,
    TRIAL_KIND_VALUES,
    bounded_integer,
    choice,
    mapping_with_size,
    nullable_identifier,
    strict_fields,
)


@dataclass(frozen=True)
class TrialRecord:
    """一次提出、拒绝、评估或其他搜索尝试的追加记录。"""

    run_id: str
    trial_key: str
    kind: str
    turn_index: int
    hypothesis_id: str | None = None
    batch_id: str | None = None
    candidate_id: str | None = None
    reason_code: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"TrialRecord schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(self, "trial_key", identifier(self.trial_key, "trial_key"))
        object.__setattr__(self, "kind", choice(self.kind, "kind", TRIAL_KIND_VALUES))
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(self.turn_index, "turn_index", minimum=1, maximum=1000),
        )
        for name in ("hypothesis_id", "batch_id", "candidate_id"):
            object.__setattr__(
                self,
                name,
                nullable_identifier(getattr(self, name), name),
            )
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                text(self.reason_code, "reason_code", maximum=500),
            )
        object.__setattr__(
            self,
            "payload",
            mapping_with_size(self.payload, "payload", maximum_bytes=16384),
        )
        object.__setattr__(
            self,
            "occurred_at",
            validate_utc_timestamp(self.occurred_at, "occurred_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含发生时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_key": self.trial_key,
            "kind": self.kind,
            "turn_index": self.turn_index,
            "hypothesis_id": self.hypothesis_id,
            "batch_id": self.batch_id,
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "payload": self.payload,
        }

    @property
    def trial_hash(self) -> str:
        """返回尝试内容哈希。"""

        return sha256_json(self.material())

    @property
    def trial_id(self) -> str:
        """返回由运行和调用方幂等键确定的尝试 ID。"""

        return stable_id(
            "trial",
            {"run_id": self.run_id, "trial_key": self.trial_key},
        )

    @classmethod
    def from_dict(cls, value: Any) -> TrialRecord:
        """严格解析尝试记录。"""

        return cls(
            **strict_fields(value, "trial_record", set(cls.__dataclass_fields__))
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        payload = self.material()
        payload["occurred_at"] = self.occurred_at
        return payload
