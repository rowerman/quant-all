"""批量正式发布与已发布因子定义的领域契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.contract_requests import FrozenContext
from mining.contract_types import (
    identifier,
    sha256_json,
    sha256_text,
    stable_id,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError

PUBLICATION_BATCH_STATUSES = (
    "staging",
    "validating",
    "published",
    "failed",
    "rolled_back",
)
PUBLICATION_ACTIVE_STATUSES = ("staging", "validating")
PUBLICATION_SOURCE_KINDS = ("v1", "v2_search")


@dataclass(frozen=True)
class PublicationBatch:
    """同一冻结上下文内一次原子发布的稳定记录。"""

    run_id: str
    context_hash: str
    candidate_ids: tuple[str, ...]
    approval_ids: tuple[str, ...]
    status: str
    staging_path: str
    attempt: int
    started_at: str
    source_kind: str = "v1"
    manifest_hash: str | None = None
    artifact_sha256: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "context_hash",
            sha256_text(self.context_hash, "context_hash"),
        )
        for name in ("candidate_ids", "approval_ids"):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(values, (list, tuple)):
                raise MiningContractError(f"{name} 必须是标识序列")
            normalized = tuple(identifier(item, name) for item in values)
            if not normalized or len(set(normalized)) != len(normalized):
                raise MiningContractError(f"{name} 必须非空且不能重复")
            object.__setattr__(self, name, normalized)
        if len(self.candidate_ids) != len(self.approval_ids):
            raise MiningContractError("候选与审批数量必须一致")
        if self.status not in PUBLICATION_BATCH_STATUSES:
            raise MiningContractError(f"未知发布批次状态：{self.status}")
        if not isinstance(self.staging_path, str) or not self.staging_path.strip():
            raise MiningContractError("staging_path 不能为空")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise MiningContractError("发布批次 attempt 必须是正整数")
        object.__setattr__(
            self,
            "started_at",
            validate_utc_timestamp(self.started_at, "started_at"),
        )
        if self.source_kind not in PUBLICATION_SOURCE_KINDS:
            raise MiningContractError(
                f"未知发布来源：{self.source_kind!r}"
            )
        for name in ("manifest_hash", "artifact_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, sha256_text(value, name))
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                validate_utc_timestamp(self.finished_at, "finished_at"),
            )

    @property
    def batch_id(self) -> str:
        """返回运行、候选集合与尝试次数共同决定的稳定 ID。"""

        return stable_id(
            "pub_batch",
            {
                "attempt": self.attempt,
                "candidate_ids": list(self.candidate_ids),
                "context_hash": self.context_hash,
                "run_id": self.run_id,
            },
        )

    @property
    def batch_hash(self) -> str:
        """返回发布批次内容哈希，用于页面确认和幂等校验。"""

        return sha256_json(
            {
                "run_id": self.run_id,
                "context_hash": self.context_hash,
                "candidate_ids": list(self.candidate_ids),
                "approval_ids": list(self.approval_ids),
                "source_kind": self.source_kind,
            }
        )

    def material(self) -> dict[str, Any]:
        """返回可持久化的完整批次字段。"""

        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "context_hash": self.context_hash,
            "candidate_ids": list(self.candidate_ids),
            "approval_ids": list(self.approval_ids),
            "status": self.status,
            "staging_path": self.staging_path,
            "manifest_hash": self.manifest_hash,
            "artifact_sha256": self.artifact_sha256,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class PublicationInputs:
    """一次整批发布的只读来源输入。"""

    run_id: str
    source_kind: str
    source_context_hash: str
    builder_context: FrozenContext
    candidates: tuple[dict[str, Any], ...]
    approval_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        if self.source_kind not in PUBLICATION_SOURCE_KINDS:
            raise MiningContractError(
                f"未知发布来源：{self.source_kind!r}"
            )
        object.__setattr__(
            self,
            "source_context_hash",
            sha256_text(self.source_context_hash, "source_context_hash"),
        )
        if not isinstance(self.builder_context, FrozenContext):
            raise MiningContractError("builder_context 必须是 FrozenContext")
        if isinstance(self.candidates, str) or not isinstance(
            self.candidates,
            (list, tuple),
        ):
            raise MiningContractError("candidates 必须是候选映射序列")
        candidates = tuple(dict(item) for item in self.candidates)
        if not candidates:
            raise MiningContractError("发布输入至少需要一个候选")
        candidate_ids = tuple(
            identifier(item.get("candidate_id"), "candidate_id")
            for item in candidates
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise MiningContractError("candidates 不能包含重复候选")
        for item in candidates:
            if item.get("source_kind") != self.source_kind:
                raise MiningContractError("候选发布来源与批次不一致")
            if (
                sha256_text(
                    item.get("source_context_hash"),
                    "source_context_hash",
                )
                != self.source_context_hash
            ):
                raise MiningContractError("候选来源上下文与批次不一致")
        if isinstance(self.approval_ids, str) or not isinstance(
            self.approval_ids,
            (list, tuple),
        ):
            raise MiningContractError("approval_ids 必须是标识序列")
        approval_ids = tuple(
            identifier(item, "approval_id") for item in self.approval_ids
        )
        if len(approval_ids) != len(candidates):
            raise MiningContractError("候选与审批数量必须一致")
        ordered = tuple(
            sorted(
                zip(candidates, approval_ids, strict=True),
                key=lambda item: item[0]["candidate_id"],
            )
        )
        object.__setattr__(
            self,
            "candidates",
            tuple(item[0] for item in ordered),
        )
        object.__setattr__(
            self,
            "approval_ids",
            tuple(item[1] for item in ordered),
        )


def new_publication_batch(
    *,
    run_id: str,
    context_hash: str,
    candidate_ids: tuple[str, ...],
    approval_ids: tuple[str, ...],
    staging_path: str,
    attempt: int,
    source_kind: str = "v1",
) -> PublicationBatch:
    """构造一个新的暂存发布批次。"""

    return PublicationBatch(
        run_id=run_id,
        context_hash=context_hash,
        candidate_ids=candidate_ids,
        approval_ids=approval_ids,
        status="staging",
        staging_path=staging_path,
        attempt=attempt,
        started_at=utc_now_iso(),
        source_kind=source_kind,
    )
