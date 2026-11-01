"""Worker 代码研究提案与运行内知识证据协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_v2 import ensure_agent_payload

WORKER_CODE_SCHEMA_VERSION = "mining_worker_code_v2"
LEGACY_WORKER_CODE_SCHEMA_VERSION = "mining_worker_code_v1"
WORKER_CODE_RESPONSE_SCHEMA_VERSION = "mining_worker_code_v1"
WORKER_CODE_KNOWLEDGE_SCHEMA_VERSION = "mining_worker_code_knowledge_v1"
CODE_PROMPT_VERSION = "mining_qlib_code_prompt_v3"
LEGACY_CODE_PROMPT_VERSION = "mining_qlib_code_prompt_v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_KNOWLEDGE_KINDS = frozenset({"failure", "success", "repair"})
_FEEDBACK_RULES = {
    "MINING-138": ("compile", "compile"),
    "MINING-139": ("execution", "execution"),
    "MINING-140": ("output_format", "output"),
    "MINING-141": ("index", "evaluation"),
    "MINING-142": ("coverage", "evaluation"),
    "MINING-143": ("numeric", "evaluation"),
    "MINING-144": ("code_policy", "evaluation"),
    "MINING-145": ("resource", "execution"),
}
_METRIC_FIELDS = frozenset({"rank_ic", "icir", "coverage", "valid_days"})
_ENTRY_ORDER = {"failure": 0, "success": 1, "repair": 2}


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_mapping(
    value: Any,
    name: str,
    fields: set[str],
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        raise WorkerProtocolError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    result = dict(value)
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise WorkerProtocolError(f"{name} 超过 {maximum_bytes} 字节")
    return result


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise WorkerProtocolError(f"{name} 必须是 SHA-256")
    return value


def _text(value: Any, name: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise WorkerProtocolError(f"{name} 必须是字符串")
    result = value.strip()
    if not result or len(result) > maximum:
        raise WorkerProtocolError(f"{name} 长度必须在 1～{maximum}")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise WorkerProtocolError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def _metrics(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise WorkerProtocolError(
            f"{name} 必须恰好包含 rank_ic、icir、coverage、valid_days"
        )
    result = dict(value)
    for metric_name in ("rank_ic", "icir", "coverage"):
        metric = result[metric_name]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)
        ):
            raise WorkerProtocolError(f"{name}.{metric_name} 必须是有限数值")
    coverage = float(result["coverage"])
    if not 0.0 <= coverage <= 1.0:
        raise WorkerProtocolError(f"{name}.coverage 必须在 0～1")
    result["coverage"] = coverage
    result["valid_days"] = _integer(
        result["valid_days"],
        f"{name}.valid_days",
        1,
        1_000_000,
    )
    return result


@dataclass(frozen=True)
class WorkerCodeFeedbackEvidence:
    """Worker 可见的一条结构化沙箱失败反馈。"""

    error_code: str
    category: str
    stage: str
    summary: str
    repair_hint: str
    retryable: bool
    feedback_hash: str
    schema_version: str = "mining_sandbox_feedback_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "mining_sandbox_feedback_v1":
            raise WorkerProtocolError("知识反馈版本不匹配")
        if self.error_code not in _FEEDBACK_RULES:
            raise WorkerProtocolError("知识反馈错误码非法")
        expected_category, expected_stage = _FEEDBACK_RULES[self.error_code]
        if self.category != expected_category or self.stage != expected_stage:
            raise WorkerProtocolError("知识反馈分类、阶段或错误码不一致")
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, "feedback.summary"),
        )
        object.__setattr__(
            self,
            "repair_hint",
            _text(self.repair_hint, "feedback.repair_hint"),
        )
        if not isinstance(self.retryable, bool):
            raise WorkerProtocolError("feedback.retryable 必须是布尔值")
        object.__setattr__(
            self,
            "feedback_hash",
            _hash(self.feedback_hash, "feedback.feedback_hash"),
        )
        if self.feedback_hash != _stable_hash(self.material()):
            raise WorkerProtocolError("知识反馈哈希不匹配")

    def material(self) -> dict[str, Any]:
        """返回不包含反馈哈希的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "error_code": self.error_code,
            "category": self.category,
            "stage": self.stage,
            "summary": self.summary,
            "repair_hint": self.repair_hint,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeFeedbackEvidence:
        """严格解析结构化反馈证据。"""

        payload = _strict_mapping(
            value,
            "worker_code_feedback_evidence",
            set(cls.__dataclass_fields__),
            maximum_bytes=8192,
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定传输字典。"""

        return {**self.material(), "feedback_hash": self.feedback_hash}


@dataclass(frozen=True)
class WorkerCodeKnowledgeEntry:
    """Worker 可见的一条当前运行知识。"""

    kind: str
    iteration: int
    source_iterations: tuple[int, ...]
    code_hash: str
    feedback: WorkerCodeFeedbackEvidence | None
    metrics: Mapping[str, Any] | None
    evaluation_hash: str | None
    summary: str
    entry_hash: str

    def __post_init__(self) -> None:
        if self.kind not in _KNOWLEDGE_KINDS:
            raise WorkerProtocolError("知识条目 kind 非法")
        object.__setattr__(
            self,
            "iteration",
            _integer(self.iteration, "knowledge.iteration", 1, 3),
        )
        if not isinstance(self.source_iterations, (list, tuple)):
            raise WorkerProtocolError("knowledge.source_iterations 必须是数组")
        iterations = tuple(
            _integer(item, "knowledge.source_iterations", 1, 3)
            for item in self.source_iterations
        )
        if (
            not iterations
            or iterations != tuple(sorted(set(iterations)))
            or iterations[-1] != self.iteration
        ):
            raise WorkerProtocolError(
                "knowledge.source_iterations 必须有序且以当前轮结束"
            )
        if self.kind in {"failure", "success"} and len(iterations) != 1:
            raise WorkerProtocolError(f"{self.kind} 条目只能引用一个来源")
        if self.kind == "repair" and len(iterations) < 2:
            raise WorkerProtocolError("repair 条目必须引用失败和成功来源")
        object.__setattr__(self, "source_iterations", iterations)
        object.__setattr__(
            self,
            "code_hash",
            _hash(self.code_hash, "knowledge.code_hash"),
        )
        if self.feedback is not None and not isinstance(
            self.feedback,
            WorkerCodeFeedbackEvidence,
        ):
            raise WorkerProtocolError("knowledge.feedback 类型非法")
        if self.kind in {"failure", "repair"} and self.feedback is None:
            raise WorkerProtocolError(f"{self.kind} 条目必须包含反馈")
        if self.kind == "success" and self.feedback is not None:
            raise WorkerProtocolError("success 条目不能包含失败反馈")
        metrics = _metrics(self.metrics, "knowledge.metrics")
        object.__setattr__(self, "metrics", metrics)
        if self.evaluation_hash is not None:
            object.__setattr__(
                self,
                "evaluation_hash",
                _hash(
                    self.evaluation_hash,
                    "knowledge.evaluation_hash",
                ),
            )
        if self.kind == "failure":
            if self.evaluation_hash is not None:
                raise WorkerProtocolError("failure 条目不能包含评估哈希")
        elif metrics is None or self.evaluation_hash is None:
            raise WorkerProtocolError(
                f"{self.kind} 条目必须包含指标和评估哈希"
            )
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, "knowledge.summary"),
        )
        object.__setattr__(
            self,
            "entry_hash",
            _hash(self.entry_hash, "knowledge.entry_hash"),
        )
        ensure_agent_payload(self.to_dict(), path="knowledge_entry")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeKnowledgeEntry:
        """严格解析一条知识证据。"""

        payload = _strict_mapping(
            value,
            "worker_code_knowledge_entry",
            set(cls.__dataclass_fields__),
            maximum_bytes=16384,
        )
        payload["source_iterations"] = tuple(payload["source_iterations"])
        if payload["feedback"] is not None:
            payload["feedback"] = WorkerCodeFeedbackEvidence.from_dict(
                payload["feedback"]
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定知识证据。"""

        return {
            "kind": self.kind,
            "iteration": self.iteration,
            "source_iterations": list(self.source_iterations),
            "code_hash": self.code_hash,
            "feedback": (
                self.feedback.to_dict() if self.feedback is not None else None
            ),
            "metrics": dict(self.metrics) if self.metrics is not None else None,
            "evaluation_hash": self.evaluation_hash,
            "summary": self.summary,
            "entry_hash": self.entry_hash,
        }


@dataclass(frozen=True)
class WorkerCodeKnowledgeEvidence:
    """绑定一个当前沙箱任务和知识索引的 Worker 证据。"""

    task_hash: str
    index_hash: str
    retrieval_hash: str
    entries: tuple[WorkerCodeKnowledgeEntry, ...]
    version: int = 1
    schema_version: str = WORKER_CODE_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_CODE_KNOWLEDGE_SCHEMA_VERSION:
            raise WorkerProtocolError("Worker 知识证据版本不匹配")
        if self.version != 1:
            raise WorkerProtocolError("Worker 知识证据 version 必须为 1")
        for name in ("task_hash", "index_hash", "retrieval_hash"):
            object.__setattr__(
                self,
                name,
                _hash(getattr(self, name), f"knowledge.{name}"),
            )
        if not isinstance(self.entries, (list, tuple)):
            raise WorkerProtocolError("knowledge.entries 必须是数组")
        entries = tuple(self.entries)
        if not 1 <= len(entries) <= 4:
            raise WorkerProtocolError("knowledge.entries 必须包含 1～4 条")
        if any(
            not isinstance(item, WorkerCodeKnowledgeEntry)
            for item in entries
        ):
            raise WorkerProtocolError("knowledge.entries 包含非法条目")
        entries = tuple(
            sorted(
                entries,
                key=lambda item: (item.iteration, _ENTRY_ORDER[item.kind]),
            )
        )
        if len({item.entry_hash for item in entries}) != len(entries):
            raise WorkerProtocolError("knowledge.entries 哈希不能重复")
        object.__setattr__(self, "entries", entries)
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 65536:
            raise WorkerProtocolError("knowledge_evidence 超过 65536 字节")
        ensure_agent_payload(self.to_dict(), path="knowledge_evidence")

    def material(self) -> dict[str, Any]:
        """返回不含选择哈希的稳定证据内容。"""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "task_hash": self.task_hash,
            "index_hash": self.index_hash,
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeKnowledgeEvidence:
        """严格解析 Worker 知识证据。"""

        payload = _strict_mapping(
            value,
            "worker_code_knowledge_evidence",
            set(cls.__dataclass_fields__),
            maximum_bytes=65536,
        )
        payload["entries"] = tuple(
            WorkerCodeKnowledgeEntry.from_dict(item)
            for item in payload["entries"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定传输字典。"""

        return {
            **self.material(),
            "retrieval_hash": self.retrieval_hash,
        }


@dataclass(frozen=True)
class WorkerCodeProposalRequest:
    """一轮 Qlib 代码研究提案请求。"""

    request_id: str
    run_id: str
    iteration: int
    hypothesis: str
    snapshot: Mapping[str, Any]
    previous_attempt: Mapping[str, Any] | None
    max_code_bytes: int
    llm_model: str
    task_hash: str | None = None
    knowledge_evidence: WorkerCodeKnowledgeEvidence | None = None
    prompt_version: str = CODE_PROMPT_VERSION
    schema_version: str = WORKER_CODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_WORKER_CODE_SCHEMA_VERSION,
            WORKER_CODE_SCHEMA_VERSION,
        }:
            raise WorkerProtocolError("schema_version 不受支持")
        if self.schema_version == LEGACY_WORKER_CODE_SCHEMA_VERSION:
            if (
                self.task_hash is not None
                or self.knowledge_evidence is not None
                or self.prompt_version != LEGACY_CODE_PROMPT_VERSION
            ):
                raise WorkerProtocolError("旧版代码请求字段不兼容")
        else:
            object.__setattr__(
                self,
                "task_hash",
                _hash(self.task_hash, "task_hash"),
            )
            if self.prompt_version != CODE_PROMPT_VERSION:
                raise WorkerProtocolError("prompt_version 不匹配")
            if (
                self.knowledge_evidence is not None
                and self.knowledge_evidence.task_hash != self.task_hash
            ):
                raise WorkerProtocolError("知识证据绑定了不同任务")
            if self.iteration == 1 and self.knowledge_evidence is not None:
                raise WorkerProtocolError("第一轮代码请求不能包含历史知识")
            if self.iteration > 1 and self.knowledge_evidence is None:
                raise WorkerProtocolError("后续代码请求必须包含历史知识")
            if (
                self.knowledge_evidence is not None
                and max(
                    item.iteration
                    for item in self.knowledge_evidence.entries
                )
                >= self.iteration
            ):
                raise WorkerProtocolError("知识证据包含当前或未来轮次")
        for name in ("request_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise WorkerProtocolError(f"{name} 包含非法字符")
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or not 1 <= self.iteration <= 3
        ):
            raise WorkerProtocolError("iteration 必须在 1～3")
        if not isinstance(self.hypothesis, str) or not 8 <= len(
            self.hypothesis.strip()
        ) <= 4000:
            raise WorkerProtocolError("hypothesis 长度必须在 8～4000")
        object.__setattr__(self, "hypothesis", self.hypothesis.strip())
        object.__setattr__(
            self,
            "snapshot",
            _strict_mapping(
                self.snapshot,
                "snapshot",
                {
                    "context_hash",
                    "manifest_hash",
                    "start_date",
                    "end_date",
                    "fields",
                    "image_id",
                },
                maximum_bytes=8192,
            ),
        )
        if self.previous_attempt is not None:
            previous_fields = {
                "iteration",
                "code_hash",
                "status",
                "metrics",
                "error_code",
            }
            if self.schema_version == WORKER_CODE_SCHEMA_VERSION:
                previous_fields |= {"feedback", "feedback_hash"}
            previous = _strict_mapping(
                self.previous_attempt,
                "previous_attempt",
                previous_fields,
                maximum_bytes=32768,
            )
            if previous["status"] not in {"passed", "failed"}:
                raise WorkerProtocolError("previous_attempt.status 非法")
            current_feedback = None
            if "feedback" in previous and previous["feedback"] is not None:
                current_feedback = WorkerCodeFeedbackEvidence.from_dict(
                    previous["feedback"]
                )
                if current_feedback.feedback_hash != previous["feedback_hash"]:
                    raise WorkerProtocolError("上一轮反馈哈希不匹配")
            if previous["error_code"] != (
                current_feedback.error_code
                if current_feedback is not None
                else None
            ):
                raise WorkerProtocolError("上一轮错误码与反馈不一致")
            if (
                self.schema_version == WORKER_CODE_SCHEMA_VERSION
                and previous["status"] == "failed"
                and current_feedback is None
            ):
                raise WorkerProtocolError("失败尝试必须包含结构化反馈")
            if (
                self.schema_version == WORKER_CODE_SCHEMA_VERSION
                and previous["status"] == "passed"
                and current_feedback is not None
            ):
                raise WorkerProtocolError("成功尝试不能包含失败反馈")
            previous["metrics"] = _metrics(
                previous["metrics"],
                "previous_attempt.metrics",
            )
            object.__setattr__(self, "previous_attempt", previous)
        if (
            self.schema_version == WORKER_CODE_SCHEMA_VERSION
            and self.iteration == 1
            and self.previous_attempt is not None
        ):
            raise WorkerProtocolError("第一轮请求不能包含上一尝试")
        if (
            self.schema_version == WORKER_CODE_SCHEMA_VERSION
            and self.iteration > 1
            and self.previous_attempt is None
        ):
            raise WorkerProtocolError("后续请求必须包含上一尝试")
        if (
            isinstance(self.max_code_bytes, bool)
            or not isinstance(self.max_code_bytes, int)
            or not 1024 <= self.max_code_bytes <= 65536
        ):
            raise WorkerProtocolError("max_code_bytes 必须在 1024～65536")
        if not isinstance(self.llm_model, str) or not self.llm_model.strip():
            raise WorkerProtocolError("llm_model 不能为空")
        ensure_agent_payload(self.to_dict(), path="worker_code_request")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeProposalRequest:
        """严格解析新版请求并兼容旧版 v1 请求。"""

        if not isinstance(value, Mapping):
            raise WorkerProtocolError("worker_code_request 必须是 JSON 对象")
        schema_version = value.get("schema_version")
        if schema_version == LEGACY_WORKER_CODE_SCHEMA_VERSION:
            legacy_fields = {
                "schema_version",
                "request_id",
                "run_id",
                "iteration",
                "hypothesis",
                "snapshot",
                "previous_attempt",
                "max_code_bytes",
                "llm_model",
                "prompt_version",
            }
            payload = _strict_mapping(
                value,
                "worker_code_request",
                legacy_fields,
                maximum_bytes=131072,
            )
            payload["task_hash"] = None
            payload["knowledge_evidence"] = None
            return cls(**payload)
        payload = _strict_mapping(
            value,
            "worker_code_request",
            set(cls.__dataclass_fields__),
            maximum_bytes=196608,
        )
        if payload["knowledge_evidence"] is not None:
            payload["knowledge_evidence"] = (
                WorkerCodeKnowledgeEvidence.from_dict(
                    payload["knowledge_evidence"]
                )
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定请求字典。"""

        common = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "hypothesis": self.hypothesis,
            "snapshot": dict(self.snapshot),
            "previous_attempt": (
                dict(self.previous_attempt)
                if self.previous_attempt is not None
                else None
            ),
            "max_code_bytes": self.max_code_bytes,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
        }
        if self.schema_version == LEGACY_WORKER_CODE_SCHEMA_VERSION:
            return common
        return {
            **common,
            "task_hash": self.task_hash,
            "knowledge_evidence": (
                self.knowledge_evidence.to_dict()
                if self.knowledge_evidence is not None
                else None
            ),
        }

    @property
    def payload_hash(self) -> str:
        """返回请求稳定哈希。"""

        return _stable_hash(self.to_dict())

    @property
    def job_id(self) -> str:
        """返回稳定作业 ID。"""

        encoded = json.dumps(
            {"request_id": self.request_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"cjob_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class WorkerCodeProposalResponse:
    """RD-Agent 返回的未信任 Qlib 研究代码。"""

    proposal_id: str
    code: str
    rationale: str
    model: str
    prompt_version: str = CODE_PROMPT_VERSION
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_CODE_RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_CODE_RESPONSE_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_CODE_RESPONSE_SCHEMA_VERSION}"
            )
        if not isinstance(self.proposal_id, str) or not _IDENTIFIER.fullmatch(
            self.proposal_id
        ):
            raise WorkerProtocolError("proposal_id 包含非法字符")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 200
        ):
            raise WorkerProtocolError("model 必须是 1～200 字符")
        object.__setattr__(self, "model", self.model.strip())
        if not isinstance(self.code, str):
            raise WorkerProtocolError("code 必须是字符串")
        code = self.code.strip()
        if not 1 <= len(code.encode("utf-8")) <= 65536:
            raise WorkerProtocolError("code 长度必须在 1～65536 字节")
        object.__setattr__(self, "code", code)
        if not isinstance(self.rationale, str):
            raise WorkerProtocolError("rationale 必须是字符串")
        object.__setattr__(
            self,
            "rationale",
            self.rationale.strip()[:2000],
        )
        if self.prompt_version not in {
            LEGACY_CODE_PROMPT_VERSION,
            CODE_PROMPT_VERSION,
        }:
            raise WorkerProtocolError("prompt_version 不受支持")
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        usage = dict(self.usage)
        if len(json.dumps(usage, ensure_ascii=False).encode("utf-8")) > 16000:
            raise WorkerProtocolError("usage 超过 16000 字节")
        object.__setattr__(self, "usage", usage)
        ensure_agent_payload(self.to_dict(), path="worker_code_response")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeProposalResponse:
        """严格解析代码提案响应。"""

        return cls(
            **_strict_mapping(
                value,
                "worker_code_response",
                set(cls.__dataclass_fields__),
                maximum_bytes=131072,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回稳定响应字典。"""

        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "code": self.code,
            "rationale": self.rationale,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "usage": dict(self.usage),
        }


__all__ = [
    "CODE_PROMPT_VERSION",
    "LEGACY_CODE_PROMPT_VERSION",
    "LEGACY_WORKER_CODE_SCHEMA_VERSION",
    "WORKER_CODE_KNOWLEDGE_SCHEMA_VERSION",
    "WORKER_CODE_RESPONSE_SCHEMA_VERSION",
    "WORKER_CODE_SCHEMA_VERSION",
    "WorkerCodeFeedbackEvidence",
    "WorkerCodeKnowledgeEntry",
    "WorkerCodeKnowledgeEvidence",
    "WorkerCodeProposalRequest",
    "WorkerCodeProposalResponse",
]
