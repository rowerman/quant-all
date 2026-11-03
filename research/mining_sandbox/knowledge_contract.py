"""沙箱运行内知识索引的数据契约和严格校验。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from mining.contract_types import canonical_json, sha256_json
from mining_sandbox.errors import SandboxKnowledgeError
from mining_sandbox.feedback import SandboxCodeFeedback

SANDBOX_KNOWLEDGE_SCHEMA_VERSION = "mining_sandbox_knowledge_v1"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_KINDS = {"failure", "success", "repair"}


def strict_knowledge_mapping(
    value: Any,
    name: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SandboxKnowledgeError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise SandboxKnowledgeError(
            f"{name} 字段不匹配：缺少 {missing}，未知 {unknown}"
        )
    return dict(value)


def knowledge_text(value: Any, name: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise SandboxKnowledgeError(f"{name} 必须是字符串")
    result = value.strip()
    if not result or len(result) > maximum:
        raise SandboxKnowledgeError(f"{name} 长度必须在 1～{maximum}")
    return result


def knowledge_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise SandboxKnowledgeError(f"{name} 必须是 SHA-256")
    return value


def knowledge_iteration(value: Any, name: str = "iteration") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 3
    ):
        raise SandboxKnowledgeError(f"{name} 必须在 1～3")
    return value


def knowledge_relative_path(value: Any, name: str) -> str:
    result = knowledge_text(value, name, maximum=500)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or result != path.as_posix():
        raise SandboxKnowledgeError(f"{name} 必须是规范相对路径")
    return result


def knowledge_json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SandboxKnowledgeError(f"{name} 必须是 JSON 对象")
    try:
        return json.loads(canonical_json(dict(value)))
    except Exception as exc:
        raise SandboxKnowledgeError(f"{name} 不是稳定 JSON") from exc

@dataclass(frozen=True)
class SandboxKnowledgeArtifact:
    """知识来源引用的一个运行内文件。"""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            knowledge_relative_path(self.path, "artifact.path"),
        )
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes < 0
        ):
            raise SandboxKnowledgeError("artifact.bytes 必须是非负整数")
        object.__setattr__(
            self,
            "sha256",
            knowledge_hash(self.sha256, "artifact.sha256"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> SandboxKnowledgeArtifact:
        """严格解析文件引用。"""

        payload = strict_knowledge_mapping(
            value,
            "sandbox_knowledge_artifact",
            set(cls.__dataclass_fields__),
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定文件引用。"""

        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SandboxKnowledgeSource:
    """一条知识引用的已完成尝试。"""

    kind: str
    sandbox_run_id: str
    iteration: int
    attempt_hash: str
    artifacts: tuple[SandboxKnowledgeArtifact, ...]
    source_hash: str
    version: int = 1
    schema_version: str = SANDBOX_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_KNOWLEDGE_SCHEMA_VERSION:
            raise SandboxKnowledgeError("知识来源版本不匹配")
        if self.version != 1:
            raise SandboxKnowledgeError("知识来源版本必须为 1")
        if self.kind != "attempt":
            raise SandboxKnowledgeError("知识来源类型必须为 attempt")
        object.__setattr__(
            self,
            "sandbox_run_id",
            knowledge_text(
                self.sandbox_run_id,
                "source.sandbox_run_id",
                maximum=160,
            ),
        )
        object.__setattr__(
            self,
            "iteration",
            knowledge_iteration(self.iteration, "source.iteration"),
        )
        object.__setattr__(
            self,
            "attempt_hash",
            knowledge_hash(self.attempt_hash, "source.attempt_hash"),
        )
        if not isinstance(self.artifacts, (list, tuple)) or not self.artifacts:
            raise SandboxKnowledgeError("source.artifacts 不能为空")
        artifacts = tuple(self.artifacts)
        if any(
            not isinstance(item, SandboxKnowledgeArtifact)
            for item in artifacts
        ):
            raise SandboxKnowledgeError(
                "source.artifacts 必须是 SandboxKnowledgeArtifact"
            )
        artifacts = tuple(sorted(artifacts, key=lambda item: item.path))
        if len({item.path for item in artifacts}) != len(artifacts):
            raise SandboxKnowledgeError("source.artifacts 路径不能重复")
        object.__setattr__(self, "artifacts", artifacts)
        expected = sha256_json(self.material())
        if self.source_hash != expected:
            raise SandboxKnowledgeError("知识来源哈希不匹配")

    def material(self) -> dict[str, Any]:
        """返回不含来源哈希的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "kind": self.kind,
            "sandbox_run_id": self.sandbox_run_id,
            "iteration": self.iteration,
            "attempt_hash": self.attempt_hash,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SandboxKnowledgeSource:
        """严格解析知识来源。"""

        payload = strict_knowledge_mapping(
            value,
            "sandbox_knowledge_source",
            set(cls.__dataclass_fields__),
        )
        payload["artifacts"] = tuple(
            SandboxKnowledgeArtifact.from_dict(item)
            for item in payload["artifacts"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定知识来源。"""

        return {**self.material(), "source_hash": self.source_hash}


def knowledge_entry_summary(
    *,
    kind: str,
    iteration: int,
    source_iterations: tuple[int, ...],
    feedback: Mapping[str, Any] | None,
) -> str:
    if kind == "success":
        return f"第 {iteration} 轮成功"
    error_code = str(feedback["error_code"]) if feedback is not None else ""
    category = str(feedback["category"]) if feedback is not None else ""
    if kind == "failure":
        return f"第 {iteration} 轮失败：{error_code}（{category}）"
    failed = ",".join(str(item) for item in source_iterations[:-1])
    return f"第 {failed} 轮失败后，第 {iteration} 轮成功；最近错误 {error_code}"


@dataclass(frozen=True)
class SandboxKnowledgeEntry:
    """一条失败、成功或修复知识。"""

    kind: str
    task_hash: str
    sandbox_run_id: str
    iteration: int
    source_iterations: tuple[int, ...]
    code_hash: str
    feedback: Mapping[str, Any] | None
    metrics: Mapping[str, Any] | None
    evaluation_hash: str | None
    summary: str
    sources: tuple[SandboxKnowledgeSource, ...]
    entry_id: str
    entry_hash: str
    version: int = 1
    schema_version: str = SANDBOX_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_KNOWLEDGE_SCHEMA_VERSION:
            raise SandboxKnowledgeError("知识条目版本不匹配")
        if self.version != 1:
            raise SandboxKnowledgeError("知识条目版本必须为 1")
        if self.kind not in _ENTRY_KINDS:
            raise SandboxKnowledgeError("知识条目 kind 非法")
        object.__setattr__(
            self,
            "task_hash",
            knowledge_hash(self.task_hash, "task_hash"),
        )
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
            "iteration",
            knowledge_iteration(self.iteration),
        )
        if not isinstance(self.source_iterations, (list, tuple)):
            raise SandboxKnowledgeError("source_iterations 必须是整数序列")
        source_iterations = tuple(
            knowledge_iteration(item, "source_iterations")
            for item in self.source_iterations
        )
        if (
            not source_iterations
            or source_iterations != tuple(sorted(set(source_iterations)))
            or source_iterations[-1] != self.iteration
        ):
            raise SandboxKnowledgeError("source_iterations 必须有序且以当前轮结束")
        object.__setattr__(self, "source_iterations", source_iterations)
        object.__setattr__(
            self,
            "code_hash",
            knowledge_hash(self.code_hash, "code_hash"),
        )
        if self.feedback is not None:
            feedback = knowledge_json_mapping(self.feedback, "feedback")
            try:
                SandboxCodeFeedback.from_dict(feedback)
            except Exception as exc:
                raise SandboxKnowledgeError("知识反馈非法") from exc
            object.__setattr__(self, "feedback", feedback)
        if self.metrics is not None:
            object.__setattr__(
                self,
                "metrics",
                knowledge_json_mapping(self.metrics, "metrics"),
            )
        if self.evaluation_hash is not None:
            object.__setattr__(
                self,
                "evaluation_hash",
                knowledge_hash(self.evaluation_hash, "evaluation_hash"),
            )
        if self.kind in {"failure", "repair"} and self.feedback is None:
            raise SandboxKnowledgeError(f"{self.kind} 条目必须包含反馈")
        if self.kind == "success" and self.feedback is not None:
            raise SandboxKnowledgeError("success 条目不能包含失败反馈")
        if self.kind in {"failure", "success"} and len(source_iterations) != 1:
            raise SandboxKnowledgeError(f"{self.kind} 条目只能引用一个来源")
        if self.kind == "repair" and len(source_iterations) < 2:
            raise SandboxKnowledgeError("repair 条目必须关联失败和成功来源")
        if self.kind == "failure":
            if self.metrics is not None and not isinstance(self.metrics, Mapping):
                raise SandboxKnowledgeError("failure.metrics 必须是 JSON 对象")
            if self.evaluation_hash is not None:
                raise SandboxKnowledgeError("failure 条目不能包含评估哈希")
        else:
            if self.metrics is None:
                raise SandboxKnowledgeError(f"{self.kind} 条目必须包含成功指标")
            if self.evaluation_hash is None:
                raise SandboxKnowledgeError(
                    f"{self.kind} 条目必须包含评估哈希"
                )
        if not isinstance(self.sources, (list, tuple)) or not self.sources:
            raise SandboxKnowledgeError("知识条目 sources 不能为空")
        sources = tuple(self.sources)
        if any(
            not isinstance(item, SandboxKnowledgeSource)
            for item in sources
        ):
            raise SandboxKnowledgeError("知识条目 sources 类型非法")
        sources = tuple(sorted(sources, key=lambda item: item.iteration))
        if tuple(item.iteration for item in sources) != source_iterations:
            raise SandboxKnowledgeError("知识来源轮次与 source_iterations 不一致")
        if any(item.sandbox_run_id != self.sandbox_run_id for item in sources):
            raise SandboxKnowledgeError("知识来源绑定了不同运行")
        object.__setattr__(self, "sources", sources)
        summary = knowledge_text(self.summary, "summary", maximum=500)
        expected_summary = knowledge_entry_summary(
            kind=self.kind,
            iteration=self.iteration,
            source_iterations=source_iterations,
            feedback=self.feedback,
        )
        if summary != expected_summary:
            raise SandboxKnowledgeError("知识摘要与条目内容不一致")
        object.__setattr__(self, "summary", summary)
        expected_hash = sha256_json(self.material())
        if self.entry_hash != expected_hash:
            raise SandboxKnowledgeError("知识条目哈希不匹配")
        expected_id = f"knowledge_{expected_hash[:24]}"
        if self.entry_id != expected_id:
            raise SandboxKnowledgeError("知识条目 ID 与哈希不一致")

    def material(self) -> dict[str, Any]:
        """返回不含条目 ID 和哈希的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "kind": self.kind,
            "task_hash": self.task_hash,
            "sandbox_run_id": self.sandbox_run_id,
            "iteration": self.iteration,
            "source_iterations": list(self.source_iterations),
            "code_hash": self.code_hash,
            "feedback": dict(self.feedback) if self.feedback is not None else None,
            "metrics": dict(self.metrics) if self.metrics is not None else None,
            "evaluation_hash": self.evaluation_hash,
            "summary": self.summary,
            "sources": [item.to_dict() for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SandboxKnowledgeEntry:
        """严格解析知识条目。"""

        payload = strict_knowledge_mapping(
            value,
            "sandbox_knowledge_entry",
            set(cls.__dataclass_fields__),
        )
        payload["source_iterations"] = tuple(payload["source_iterations"])
        payload["sources"] = tuple(
            SandboxKnowledgeSource.from_dict(item)
            for item in payload["sources"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定知识条目。"""

        return {
            **self.material(),
            "entry_id": self.entry_id,
            "entry_hash": self.entry_hash,
        }


__all__ = [
    "SANDBOX_KNOWLEDGE_SCHEMA_VERSION",
    "SandboxKnowledgeArtifact",
    "SandboxKnowledgeEntry",
    "SandboxKnowledgeSource",
    "knowledge_entry_summary",
    "knowledge_hash",
    "knowledge_iteration",
    "knowledge_json_mapping",
    "knowledge_relative_path",
    "knowledge_text",
    "strict_knowledge_mapping",
]
