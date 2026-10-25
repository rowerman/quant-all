"""评估、技术门禁、审批和发布尝试契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    ensure_string_tuple,
    identifier,
    mapping,
    sha256_json,
    sha256_text,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import (
    ApprovalConflictError,
    CandidateError,
    MiningEvaluationError,
    MiningPublicationError,
)


@dataclass(frozen=True)
class EvaluationReport:
    """候选在某个评估阶段的不可变报告。"""

    run_id: str
    candidate_id: str
    stage: str
    candidate_content_hash: str
    context_hash: str
    input_hash: str
    status: str
    metrics: Mapping[str, Any]
    warnings: tuple[str, ...]
    artifact_path: str | None
    artifact_sha256: str | None
    error_code: str | None
    error_message: str | None
    started_at: str
    finished_at: str | None
    attempt: int = 1

    def __post_init__(self) -> None:
        if self.stage not in {"technical", "quick", "full", "holdout"}:
            raise MiningEvaluationError("评估 stage 必须是 technical、quick、full 或 holdout")
        if self.status not in {"running", "passed", "failed", "error"}:
            raise MiningEvaluationError("评估 status 不合法")
        for name in ("run_id", "candidate_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("candidate_content_hash", "context_hash", "input_hash"):
            object.__setattr__(self, name, sha256_text(getattr(self, name), name))
        object.__setattr__(self, "metrics", mapping(self.metrics, "metrics"))
        object.__setattr__(
            self,
            "warnings",
            ensure_string_tuple(self.warnings, "warnings", maximum=200),
        )
        if self.artifact_path is not None:
            object.__setattr__(
                self,
                "artifact_path",
                text(self.artifact_path, "artifact_path", maximum=2000),
            )
        if self.artifact_sha256 is not None:
            object.__setattr__(
                self,
                "artifact_sha256",
                sha256_text(self.artifact_sha256, "artifact_sha256"),
            )
        for name in ("error_code", "error_message"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, text(value, name, maximum=2000))
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise MiningEvaluationError("attempt 必须是正整数")
        object.__setattr__(
            self,
            "started_at",
            validate_utc_timestamp(self.started_at, "started_at"),
        )
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                validate_utc_timestamp(self.finished_at, "finished_at"),
            )

    @property
    def report_id(self) -> str:
        """返回同一候选、阶段、输入和尝试对应的稳定报告 ID。"""

        return stable_id(
            "eval",
            {
                "run_id": self.run_id,
                "candidate_id": self.candidate_id,
                "stage": self.stage,
                "candidate_content_hash": self.candidate_content_hash,
                "context_hash": self.context_hash,
                "input_hash": self.input_hash,
                "attempt": self.attempt,
            },
        )


@dataclass(frozen=True)
class TechnicalGate:
    """候选冻结前的技术门禁结论。"""

    run_id: str
    candidate_id: str
    candidate_content_hash: str
    context_hash: str
    rule_version: str
    passed: bool
    checks: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        for name in ("run_id", "candidate_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("candidate_content_hash", "context_hash"):
            object.__setattr__(self, name, sha256_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "rule_version",
            identifier(self.rule_version, "rule_version"),
        )
        if not isinstance(self.passed, bool):
            raise CandidateError("technical gate 的 passed 必须是布尔值")
        object.__setattr__(self, "checks", mapping(self.checks, "checks"))
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    @property
    def gate_id(self) -> str:
        """返回技术门禁的稳定 ID。"""

        return stable_id(
            "gate",
            {
                "run_id": self.run_id,
                "candidate_id": self.candidate_id,
                "candidate_content_hash": self.candidate_content_hash,
                "context_hash": self.context_hash,
                "rule_version": self.rule_version,
                "passed": self.passed,
                "checks": self.checks,
            },
        )

    @property
    def check_hash(self) -> str:
        """返回门禁结论及其证据的哈希。"""

        return sha256_json(
            {
                "gate_id": self.gate_id,
                "passed": self.passed,
                "checks": self.checks,
            }
        )


@dataclass(frozen=True)
class ApprovalDecision:
    """单用户人工批准或拒绝记录。"""

    run_id: str
    candidate_id: str
    decision: str
    reviewer: str
    note: str
    candidate_content_hash: str
    semantic_hash: str
    context_hash: str
    evidence_hash: str
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.decision not in {"approved", "rejected"}:
            raise ApprovalConflictError("审批结论必须是 approved 或 rejected")
        for name in ("run_id", "candidate_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "reviewer", text(self.reviewer, "reviewer", maximum=200))
        object.__setattr__(self, "note", text(self.note, "note", maximum=4000))
        for name in (
            "candidate_content_hash",
            "semantic_hash",
            "context_hash",
            "evidence_hash",
        ):
            object.__setattr__(self, name, sha256_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    @property
    def approval_id(self) -> str:
        """返回审批决定的稳定 ID。"""

        return stable_id(
            "apr",
            {
                "run_id": self.run_id,
                "candidate_id": self.candidate_id,
                "decision": self.decision,
                "reviewer": self.reviewer,
                "note": self.note,
                "candidate_content_hash": self.candidate_content_hash,
                "semantic_hash": self.semantic_hash,
                "context_hash": self.context_hash,
                "evidence_hash": self.evidence_hash,
            },
        )


@dataclass(frozen=True)
class PublicationAttempt:
    """候选正式发布的存储契约；S2 不执行真实发布。"""

    run_id: str
    candidate_id: str
    approval_id: str
    status: str
    staging_path: str
    manifest_hash: str | None
    formal_factor_name: str | None
    artifact_sha256: str | None
    error_code: str | None
    error_message: str | None
    attempt: int
    started_at: str
    finished_at: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "staging",
            "validating",
            "published",
            "failed",
            "rolled_back",
        }:
            raise MiningPublicationError("发布 status 不合法")
        for name in ("run_id", "candidate_id", "approval_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "staging_path",
            text(self.staging_path, "staging_path", maximum=2000),
        )
        for name in ("manifest_hash", "artifact_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, sha256_text(value, name))
        if self.formal_factor_name is not None:
            object.__setattr__(
                self,
                "formal_factor_name",
                identifier(self.formal_factor_name, "formal_factor_name"),
            )
        for name in ("error_code", "error_message"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, text(value, name, maximum=2000))
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise MiningPublicationError("attempt 必须是正整数")
        object.__setattr__(
            self,
            "started_at",
            validate_utc_timestamp(self.started_at, "started_at"),
        )
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                validate_utc_timestamp(self.finished_at, "finished_at"),
            )

    @property
    def publication_id(self) -> str:
        """返回发布尝试的稳定 ID。"""

        return stable_id(
            "pub",
            {
                "run_id": self.run_id,
                "candidate_id": self.candidate_id,
                "approval_id": self.approval_id,
                "attempt": self.attempt,
            },
        )
