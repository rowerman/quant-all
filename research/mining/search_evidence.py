"""挖掘 V2 的脱敏证据包和搜索决策契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    canonical_json,
    identifier,
    mapping_sequence,
    sha256_json,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError
from mining.search_budget import TrialCounts
from mining.search_contract_types import (
    SEARCH_ACTION_VALUES,
    SEARCH_SCHEMA_VERSION,
    bounded_integer,
    choice,
    mapping_with_size,
    nullable_identifier,
    strict_fields,
    validate_date_range,
)


def _json_size(value: Any) -> int:
    """返回规范化 JSON 的 UTF-8 字节数。"""

    return len(canonical_json(value).encode("utf-8"))


@dataclass(frozen=True)
class EvidencePacket:
    """只包含开发期证据、可安全反馈给搜索决策链的摘要。"""

    run_id: str
    hypothesis_id: str
    batch_id: str
    turn_index: int
    development_start: str
    development_end: str
    metrics: Mapping[str, Any]
    failures: tuple[Mapping[str, Any], ...]
    correlation_clusters: tuple[Mapping[str, Any], ...]
    trial_counts: TrialCounts
    budget_state: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"EvidencePacket schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "hypothesis_id",
            identifier(self.hypothesis_id, "hypothesis_id"),
        )
        object.__setattr__(self, "batch_id", identifier(self.batch_id, "batch_id"))
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(self.turn_index, "turn_index", minimum=1, maximum=1000),
        )
        start, end = validate_date_range(
            self.development_start,
            self.development_end,
            "development",
        )
        object.__setattr__(self, "development_start", start)
        object.__setattr__(self, "development_end", end)
        metrics = mapping_with_size(self.metrics, "metrics", maximum_bytes=32768)
        if len(metrics) > 64:
            raise MiningContractError("metrics 最多包含 64 项")
        object.__setattr__(self, "metrics", metrics)
        failures = mapping_sequence(self.failures, "failures")
        if len(failures) > 128:
            raise MiningContractError("failures 最多包含 128 项")
        object.__setattr__(self, "failures", failures)
        clusters = mapping_sequence(
            self.correlation_clusters,
            "correlation_clusters",
        )
        if len(clusters) > 24:
            raise MiningContractError("correlation_clusters 最多包含 24 项")
        object.__setattr__(self, "correlation_clusters", clusters)
        if not isinstance(self.trial_counts, TrialCounts):
            raise MiningContractError("trial_counts 必须是 TrialCounts")
        object.__setattr__(
            self,
            "budget_state",
            mapping_with_size(self.budget_state, "budget_state", maximum_bytes=16384),
        )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )
        if _json_size(self.material()) > 65536:
            raise MiningContractError("EvidencePacket 规范化内容超过 65536 字节")

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "hypothesis_id": self.hypothesis_id,
            "batch_id": self.batch_id,
            "turn_index": self.turn_index,
            "development_start": self.development_start,
            "development_end": self.development_end,
            "metrics": self.metrics,
            "failures": list(self.failures),
            "correlation_clusters": list(self.correlation_clusters),
            "trial_counts": self.trial_counts.to_dict(),
            "budget_state": self.budget_state,
        }

    @property
    def evidence_hash(self) -> str:
        """返回证据内容哈希。"""

        return sha256_json(self.material())

    @property
    def evidence_id(self) -> str:
        """返回运行内稳定证据 ID。"""

        return stable_id(
            "evidence",
            {"run_id": self.run_id, "evidence_hash": self.evidence_hash},
        )

    @classmethod
    def from_dict(cls, value: Any) -> EvidencePacket:
        """严格解析证据包。"""

        payload = strict_fields(value, "evidence_packet", set(cls.__dataclass_fields__))
        payload["failures"] = tuple(payload["failures"])
        payload["correlation_clusters"] = tuple(payload["correlation_clusters"])
        payload["trial_counts"] = TrialCounts.from_dict(payload["trial_counts"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回脱敏证据的持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "hypothesis_id": self.hypothesis_id,
            "batch_id": self.batch_id,
            "turn_index": self.turn_index,
            "development_start": self.development_start,
            "development_end": self.development_end,
            "metrics": dict(self.metrics),
            "failures": [dict(item) for item in self.failures],
            "correlation_clusters": [
                dict(item) for item in self.correlation_clusters
            ],
            "trial_counts": self.trial_counts.to_dict(),
            "budget_state": dict(self.budget_state),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SearchDecision:
    """确定性控制器记录的动作和理由，LLM 建议不具有最终权威。"""

    run_id: str
    hypothesis_id: str
    evidence_id: str
    turn_index: int
    action: str
    reason: str
    target_hypothesis_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"SearchDecision schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "hypothesis_id",
            identifier(self.hypothesis_id, "hypothesis_id"),
        )
        object.__setattr__(self, "evidence_id", identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(self.turn_index, "turn_index", minimum=1, maximum=1000),
        )
        object.__setattr__(
            self,
            "action",
            choice(self.action, "action", SEARCH_ACTION_VALUES),
        )
        object.__setattr__(
            self,
            "reason",
            text(self.reason, "reason", maximum=2000),
        )
        object.__setattr__(
            self,
            "target_hypothesis_id",
            nullable_identifier(
                self.target_hypothesis_id,
                "target_hypothesis_id",
            ),
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
            "evidence_id": self.evidence_id,
            "turn_index": self.turn_index,
            "action": self.action,
            "reason": self.reason,
            "target_hypothesis_id": self.target_hypothesis_id,
        }

    @property
    def decision_hash(self) -> str:
        """返回决策内容哈希。"""

        return sha256_json(self.material())

    @property
    def decision_id(self) -> str:
        """返回运行内稳定决策 ID。"""

        return stable_id(
            "decision",
            {"run_id": self.run_id, "decision_hash": self.decision_hash},
        )

    @classmethod
    def from_dict(cls, value: Any) -> SearchDecision:
        """严格解析搜索决策。"""

        return cls(
            **strict_fields(
                value,
                "search_decision",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        payload = self.material()
        payload["created_at"] = self.created_at
        return payload
