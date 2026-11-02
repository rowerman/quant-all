"""V3 候选审核资格和最终审批视图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.contract_types import identifier
from mining.errors import MiningContractError
from mining.search_contract_types import hash_text
from mining.v3_contracts import EvidenceReference, V3_REQUIRED_EVIDENCE_STAGES

V3_REVIEW_SCHEMA_VERSION = "mining_review_v3"


@dataclass(frozen=True)
class V3CandidateReview:
    """一个候选来源的不可变审核资格摘要。"""

    run_id: str
    candidate_id: str
    source_id: str
    source_reference_hash: str
    candidate_content_hash: str
    semantic_hash: str
    context_hash: str
    candidate_envelope_hash: str
    candidate_status: str
    eligible: bool
    blocking_reasons: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    approval_evidence_hash: str | None = None
    approval_id: str | None = None
    decision: str | None = None
    schema_version: str = V3_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V3_REVIEW_SCHEMA_VERSION:
            raise MiningContractError("V3 审核 schema_version 不匹配")
        for name in ("run_id", "candidate_id", "source_id"):
            object.__setattr__(
                self,
                name,
                identifier(getattr(self, name), name),
            )
        for name in (
            "source_reference_hash",
            "candidate_content_hash",
            "semantic_hash",
            "context_hash",
            "candidate_envelope_hash",
        ):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "candidate_status",
            identifier(self.candidate_status, "candidate_status"),
        )
        if not isinstance(self.eligible, bool):
            raise MiningContractError("eligible 必须是布尔值")
        if not isinstance(self.blocking_reasons, (list, tuple)):
            raise MiningContractError("blocking_reasons 必须是字符串序列")
        object.__setattr__(
            self,
            "blocking_reasons",
            tuple(
                sorted(
                    {
                        identifier(item, "blocking_reason")
                        for item in self.blocking_reasons
                    }
                )
            ),
        )
        if isinstance(self.evidence, (str, bytes)) or not isinstance(
            self.evidence,
            (list, tuple),
        ):
            raise MiningContractError("evidence 必须是证据引用序列")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, EvidenceReference) for item in evidence):
            raise MiningContractError("evidence 包含未知对象")
        if any(item.candidate_id != self.candidate_id for item in evidence):
            raise MiningContractError("证据没有绑定当前候选")
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        V3_REQUIRED_EVIDENCE_STAGES.index(item.stage),
                        item.evaluation_id,
                    ),
                )
            ),
        )
        if self.approval_evidence_hash is not None:
            object.__setattr__(
                self,
                "approval_evidence_hash",
                hash_text(self.approval_evidence_hash, "approval_evidence_hash"),
            )
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
            raise MiningContractError("审批 decision 不合法")
        if self.eligible and self.blocking_reasons:
            raise MiningContractError("可审批资格不能同时包含阻断原因")
        if self.eligible and self.approval_evidence_hash is None:
            raise MiningContractError("可审批资格必须包含审批证据哈希")

    def to_dict(self) -> dict[str, Any]:
        """返回页面和服务层可读取的稳定摘要。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "source_reference_hash": self.source_reference_hash,
            "candidate_content_hash": self.candidate_content_hash,
            "semantic_hash": self.semantic_hash,
            "context_hash": self.context_hash,
            "candidate_envelope_hash": self.candidate_envelope_hash,
            "candidate_status": self.candidate_status,
            "eligible": self.eligible,
            "blocking_reasons": list(self.blocking_reasons),
            "approval_evidence_hash": self.approval_evidence_hash,
            "approval_id": self.approval_id,
            "decision": self.decision,
            "evidence": [item.to_dict() for item in self.evidence],
        }


__all__ = [
    "V3_REVIEW_SCHEMA_VERSION",
    "V3CandidateReview",
]
