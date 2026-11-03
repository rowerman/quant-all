"""V2 各阶段不可变评估证据的脱敏契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    canonical_json,
    ensure_string_tuple,
    identifier,
    sha256_json,
    stable_id,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError
from mining.search_contract_types import hash_text, mapping_with_size, strict_fields

SEARCH_EVALUATION_STAGES = (
    "technical",
    "development",
    "selection",
    "selection_diagnostics",
    "stability",
    "multiple_testing",
    "full",
    "holdout",
)


@dataclass(frozen=True)
class SearchEvaluationEvidence:
    """不包含行情和文件路径的 V2 单阶段评估摘要。"""

    run_id: str
    candidate_id: str
    stage: str
    input_hash: str
    status: str
    metrics: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = "mining_search_evaluation_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "mining_search_evaluation_v1":
            raise MiningContractError(
                "评估证据 schema_version 必须为 mining_search_evaluation_v1"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "candidate_id",
            identifier(self.candidate_id, "candidate_id"),
        )
        if self.stage not in SEARCH_EVALUATION_STAGES:
            raise MiningContractError(f"未知评估阶段：{self.stage!r}")
        object.__setattr__(self, "input_hash", hash_text(self.input_hash, "input_hash"))
        object.__setattr__(self, "status", identifier(self.status, "status"))
        object.__setattr__(
            self,
            "metrics",
            mapping_with_size(
                self.metrics,
                "metrics",
                maximum_bytes=120000,
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            ensure_string_tuple(self.warnings, "warnings", maximum=200),
        )
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                identifier(self.error_code, "error_code"),
            )
        if self.error_message is not None:
            if not isinstance(self.error_message, str):
                raise MiningContractError("error_message 必须是字符串")
            object.__setattr__(self, "error_message", self.error_message[:2000])
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )
        if len(canonical_json(self.to_dict()).encode("utf-8")) > 131072:
            raise MiningContractError("评估证据超过 131072 字节")

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "input_hash": self.input_hash,
            "status": self.status,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @property
    def evidence_hash(self) -> str:
        """返回评估证据内容哈希。"""

        return sha256_json(self.material())

    @property
    def evaluation_id(self) -> str:
        """返回运行内稳定评估 ID。"""

        return stable_id(
            "searcheval",
            {"run_id": self.run_id, "evidence_hash": self.evidence_hash},
        )

    @classmethod
    def from_report(
        cls,
        report: Any,
        *,
        stage: str | None = None,
    ) -> SearchEvaluationEvidence:
        """从现有评估报告提取安全字段。"""

        selected_stage = stage or str(report.stage)
        if selected_stage == "quick":
            selected_stage = "development"
        return cls(
            run_id=report.run_id,
            candidate_id=report.candidate_id,
            stage=selected_stage,
            input_hash=report.input_hash,
            status=report.status,
            metrics=dict(report.metrics),
            warnings=tuple(report.warnings),
            error_code=report.error_code,
            error_message=report.error_message,
            created_at=report.finished_at or report.started_at,
        )

    @classmethod
    def from_dict(cls, value: Any) -> SearchEvaluationEvidence:
        """严格解析持久化评估证据。"""

        payload = strict_fields(
            value,
            "search_evaluation_evidence",
            set(cls.__dataclass_fields__),
        )
        payload["warnings"] = tuple(payload["warnings"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "created_at": self.created_at}
