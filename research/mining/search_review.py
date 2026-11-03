"""V2 候选审批资格、证据哈希和最终决定契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from mining.contract_evidence import ApprovalDecision
from mining.contract_types import identifier, sha256_json, validate_utc_timestamp
from mining.errors import MiningContractError
from mining.search_contract_types import hash_text
from mining.search_publication_contracts import SearchEvidenceReference

SEARCH_APPROVAL_SCHEMA_VERSION = "mining_search_approval_v1"
SEARCH_APPROVAL_REQUIRED_STAGES = (
    "technical",
    "development",
    "selection",
    "full",
    "holdout",
)


def _evidence_tuple(
    evidence: Iterable[SearchEvidenceReference],
) -> tuple[SearchEvidenceReference, ...]:
    """校验并返回确定顺序的候选级证据引用。"""

    if isinstance(evidence, (str, bytes)):
        raise MiningContractError("evidence 必须是证据引用序列")
    rows = tuple(evidence)
    if any(not isinstance(item, SearchEvidenceReference) for item in rows):
        raise MiningContractError("evidence 包含未知对象")
    evaluation_ids = tuple(item.evaluation_id for item in rows)
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise MiningContractError("evidence 不能包含重复评估 ID")
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.stage,
                item.input_hash,
                item.evaluation_id,
            ),
        )
    )


@dataclass(frozen=True)
class SearchApprovalEvidence:
    """与单个 V2 候选研究和冻结上下文绑定的审批证据。"""

    run_id: str
    candidate_id: str
    candidate_content_hash: str
    semantic_hash: str
    context_hash: str
    evidence: tuple[SearchEvidenceReference, ...]
    schema_version: str = SEARCH_APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_APPROVAL_SCHEMA_VERSION:
            raise MiningContractError("审批证据 schema_version 不匹配")
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "candidate_id",
            identifier(self.candidate_id, "candidate_id"),
        )
        for name in ("candidate_content_hash", "semantic_hash", "context_hash"):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))

    @property
    def evidence_hash(self) -> str:
        """返回可由候选级上下文和全部评估证据重新计算的哈希。"""

        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """返回稳定哈希输入和审计字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_content_hash": self.candidate_content_hash,
            "semantic_hash": self.semantic_hash,
            "context_hash": self.context_hash,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class SearchApprovalEligibility:
    """页面或发布前端用于判断候选能否审批的只读摘要。"""

    run_id: str
    candidate_id: str
    candidate_status: str
    eligible: bool
    blocking_reasons: tuple[str, ...]
    candidate_content_hash: str
    semantic_hash: str
    context_hash: str
    evidence_hash: str
    evidence: tuple[SearchEvidenceReference, ...]
    approval_id: str | None = None
    decision: str | None = None
    schema_version: str = SEARCH_APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_APPROVAL_SCHEMA_VERSION:
            raise MiningContractError("审批资格 schema_version 不匹配")
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "candidate_id",
            identifier(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_status",
            identifier(self.candidate_status, "candidate_status"),
        )
        for name in (
            "candidate_content_hash",
            "semantic_hash",
            "context_hash",
            "evidence_hash",
        ):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))
        if not isinstance(self.blocking_reasons, (list, tuple)):
            raise MiningContractError("blocking_reasons 必须是字符串序列")
        reasons = tuple(
            sorted({identifier(item, "blocking_reason") for item in self.blocking_reasons})
        )
        object.__setattr__(self, "blocking_reasons", reasons)
        if not isinstance(self.eligible, bool):
            raise MiningContractError("eligible 必须是布尔值")
        if self.eligible and (self.candidate_status != "review_ready" or reasons):
            raise MiningContractError("可审批资格与状态或阻断原因不一致")
        if self.approval_id is not None:
            object.__setattr__(
                self,
                "approval_id",
                identifier(self.approval_id, "approval_id"),
            )
        if self.decision is not None and self.decision not in {
            "approved",
            "rejected",
        }:
            raise MiningContractError("审批资格 decision 不合法")

    def to_dict(self) -> dict[str, Any]:
        """返回不包含正文的稳定审批资格字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_status": self.candidate_status,
            "eligible": self.eligible,
            "blocking_reasons": list(self.blocking_reasons),
            "candidate_content_hash": self.candidate_content_hash,
            "semantic_hash": self.semantic_hash,
            "context_hash": self.context_hash,
            "evidence_hash": self.evidence_hash,
            "evidence": [item.to_dict() for item in self.evidence],
            "approval_id": self.approval_id,
            "decision": self.decision,
        }


def search_approval_record_hash(decision: ApprovalDecision) -> str:
    """返回最终审批记录的防篡改哈希。"""

    if not isinstance(decision, ApprovalDecision):
        raise MiningContractError("decision 必须是 ApprovalDecision")
    created_at = validate_utc_timestamp(decision.created_at, "created_at")
    return sha256_json(
        {
            "run_id": decision.run_id,
            "candidate_id": decision.candidate_id,
            "decision": decision.decision,
            "reviewer": decision.reviewer,
            "note": decision.note,
            "candidate_content_hash": decision.candidate_content_hash,
            "semantic_hash": decision.semantic_hash,
            "context_hash": decision.context_hash,
            "evidence_hash": decision.evidence_hash,
            "created_at": created_at,
        }
    )


__all__ = [
    "SEARCH_APPROVAL_SCHEMA_VERSION",
    "SEARCH_APPROVAL_REQUIRED_STAGES",
    "SearchApprovalEvidence",
    "SearchApprovalEligibility",
    "search_approval_record_hash",
]
