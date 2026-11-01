"""沙箱知识索引、任务检索和索引级校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.contract_types import sha256_json
from mining_sandbox.errors import SandboxKnowledgeError
from mining_sandbox.knowledge_contract import (
    SANDBOX_KNOWLEDGE_SCHEMA_VERSION,
    SandboxKnowledgeEntry,
    knowledge_hash,
    knowledge_text,
    strict_knowledge_mapping,
)

_ENTRY_ORDER = {"failure": 0, "success": 1, "repair": 2}


@dataclass(frozen=True)
class SandboxKnowledgeSelection:
    """同一任务在当前运行中的检索结果。"""

    task_hash: str
    failures: tuple[SandboxKnowledgeEntry, ...]
    successes: tuple[SandboxKnowledgeEntry, ...]
    repairs: tuple[SandboxKnowledgeEntry, ...]
    selection_hash: str
    version: int = 1
    schema_version: str = SANDBOX_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_KNOWLEDGE_SCHEMA_VERSION:
            raise SandboxKnowledgeError("知识检索结果版本不匹配")
        if self.version != 1:
            raise SandboxKnowledgeError("知识检索结果版本必须为 1")
        object.__setattr__(
            self,
            "task_hash",
            knowledge_hash(self.task_hash, "task_hash"),
        )
        expected = {
            "failure": tuple(self.failures),
            "success": tuple(self.successes),
            "repair": tuple(self.repairs),
        }
        for kind, entries in expected.items():
            if any(
                not isinstance(item, SandboxKnowledgeEntry)
                or item.kind != kind
                or item.task_hash != self.task_hash
                for item in entries
            ):
                raise SandboxKnowledgeError("知识检索结果条目非法")
        if self.selection_hash != sha256_json(self.material()):
            raise SandboxKnowledgeError("知识检索结果哈希不匹配")

    def material(self) -> dict[str, Any]:
        """返回不含结果哈希的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "task_hash": self.task_hash,
            "failures": [item.to_dict() for item in self.failures],
            "successes": [item.to_dict() for item in self.successes],
            "repairs": [item.to_dict() for item in self.repairs],
        }

    def to_dict(self) -> dict[str, Any]:
        """返回稳定检索结果。"""

        return {**self.material(), "selection_hash": self.selection_hash}


@dataclass(frozen=True)
class SandboxKnowledgeBase:
    """严格绑定一个沙箱运行和一个研究任务的知识索引。"""

    sandbox_run_id: str
    request_hash: str
    task_hash: str
    entries: tuple[SandboxKnowledgeEntry, ...]
    index_hash: str
    version: int = 1
    schema_version: str = SANDBOX_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_KNOWLEDGE_SCHEMA_VERSION:
            raise SandboxKnowledgeError("知识索引版本不匹配")
        if self.version != 1:
            raise SandboxKnowledgeError("知识索引版本必须为 1")
        object.__setattr__(
            self,
            "sandbox_run_id",
            knowledge_text(
                self.sandbox_run_id,
                "sandbox_run_id",
                maximum=160,
            ),
        )
        object.__setattr__(
            self,
            "request_hash",
            knowledge_hash(self.request_hash, "request_hash"),
        )
        object.__setattr__(
            self,
            "task_hash",
            knowledge_hash(self.task_hash, "task_hash"),
        )
        if not isinstance(self.entries, (list, tuple)):
            raise SandboxKnowledgeError(
                "knowledge.entries 必须是条目序列"
            )
        entries = tuple(self.entries)
        if any(
            not isinstance(item, SandboxKnowledgeEntry)
            for item in entries
        ):
            raise SandboxKnowledgeError(
                "knowledge.entries 包含非法条目"
            )
        entries = tuple(
            sorted(
                entries,
                key=lambda item: (item.iteration, _ENTRY_ORDER[item.kind]),
            )
        )
        if any(
            item.sandbox_run_id != self.sandbox_run_id
            or item.task_hash != self.task_hash
            for item in entries
        ):
            raise SandboxKnowledgeError("知识条目与索引绑定不一致")
        if len({item.entry_id for item in entries}) != len(entries):
            raise SandboxKnowledgeError("知识条目 ID 不能重复")
        object.__setattr__(self, "entries", entries)
        if self.index_hash != sha256_json(self.material()):
            raise SandboxKnowledgeError("知识索引哈希不匹配")

    def material(self) -> dict[str, Any]:
        """返回不含索引哈希的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "sandbox_run_id": self.sandbox_run_id,
            "request_hash": self.request_hash,
            "task_hash": self.task_hash,
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SandboxKnowledgeBase:
        """严格解析知识索引。"""

        payload = strict_knowledge_mapping(
            value,
            "sandbox_knowledge",
            set(cls.__dataclass_fields__),
        )
        payload["entries"] = tuple(
            SandboxKnowledgeEntry.from_dict(item)
            for item in payload["entries"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定知识索引。"""

        return {**self.material(), "index_hash": self.index_hash}

    def failures(
        self,
        task_hash: str,
    ) -> tuple[SandboxKnowledgeEntry, ...]:
        """返回同一任务的失败轨迹。"""

        self._require_task(task_hash)
        return tuple(
            item for item in self.entries if item.kind == "failure"
        )

    def successes(
        self,
        task_hash: str,
    ) -> tuple[SandboxKnowledgeEntry, ...]:
        """返回同一任务的成功实现。"""

        self._require_task(task_hash)
        return tuple(
            item for item in self.entries if item.kind == "success"
        )

    def repair_summaries(
        self,
        task_hash: str,
    ) -> tuple[SandboxKnowledgeEntry, ...]:
        """返回同一任务的修复摘要。"""

        self._require_task(task_hash)
        return tuple(
            item for item in self.entries if item.kind == "repair"
        )

    def retrieve(self, task_hash: str) -> SandboxKnowledgeSelection:
        """检索同一任务的失败、成功和修复知识。"""

        failures = self.failures(task_hash)
        successes = self.successes(task_hash)
        repairs = self.repair_summaries(task_hash)
        material = {
            "schema_version": self.schema_version,
            "version": 1,
            "task_hash": task_hash,
            "failures": [item.to_dict() for item in failures],
            "successes": [item.to_dict() for item in successes],
            "repairs": [item.to_dict() for item in repairs],
        }
        return SandboxKnowledgeSelection(
            task_hash=task_hash,
            failures=failures,
            successes=successes,
            repairs=repairs,
            selection_hash=sha256_json(material),
        )

    def _require_task(self, task_hash: str) -> None:
        if knowledge_hash(task_hash, "task_hash") != self.task_hash:
            raise SandboxKnowledgeError(
                "知识检索任务与当前运行不一致"
            )


__all__ = [
    "SandboxKnowledgeBase",
    "SandboxKnowledgeSelection",
]
