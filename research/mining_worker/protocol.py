"""Worker HTTP 请求、响应和作业状态的纯数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

WORKER_SCHEMA_VERSION = "mining_worker_v1"
JOB_STATUSES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled", "interrupted"}
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class WorkerProtocolError(ValueError):
    """Worker 请求或响应不符合固定协议。"""


def _text(value: Any, name: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    if not isinstance(value, str):
        raise WorkerProtocolError(f"{name} 必须是字符串")
    result = value.strip()
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(
            f"{name} 长度必须在 {minimum}～{maximum} 个字符"
        )
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, maximum=160)
    if not _ID_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 包含非法字符")
    return result


def _hash(value: Any, name: str) -> str:
    result = _text(value, name, minimum=64, maximum=64)
    if not _HASH_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise WorkerProtocolError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def _string_tuple(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 500,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        current = _text(item, name, maximum=500)
        if current not in seen:
            result.append(current)
            seen.add(current)
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(f"{name} 数量必须在 {minimum}～{maximum}")
    return tuple(result)


def _mapping_tuple(value: Any, name: str, *, maximum: int) -> tuple[dict, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象数组")
    if len(value) > maximum:
        raise WorkerProtocolError(f"{name} 最多包含 {maximum} 项")
    result: list[dict] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise WorkerProtocolError(f"{name} 每项必须是 JSON 对象")
        encoded = json.dumps(dict(item), ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 12000:
            raise WorkerProtocolError(f"{name} 单项超过 12000 字节")
        result.append(json.loads(encoded))
    return tuple(result)


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class WorkerCandidateDraft:
    """Worker 返回的未经信任候选；不包含权威名称、依赖或哈希。"""

    expression: str
    mechanism: str
    description: str
    direction: int
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expression", _text(self.expression, "expression", maximum=12000)
        )
        object.__setattr__(
            self, "mechanism", _text(self.mechanism, "mechanism", maximum=200)
        )
        object.__setattr__(
            self, "description", _text(self.description, "description", maximum=2000)
        )
        if self.direction not in (-1, 1):
            raise WorkerProtocolError("direction 必须是 -1 或 1")
        object.__setattr__(
            self,
            "steps",
            _string_tuple(self.steps, "steps", minimum=1, maximum=20),
        )

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCandidateDraft:
        """从不可信 JSON 构造候选草稿。"""

        if not isinstance(value, Mapping):
            raise WorkerProtocolError("candidate 必须是 JSON 对象")
        return cls(
            expression=value.get("expression"),
            mechanism=value.get("mechanism"),
            description=value.get("description"),
            direction=value.get("direction"),
            steps=value.get("steps"),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可传输字典。"""

        return {
            "expression": self.expression,
            "mechanism": self.mechanism,
            "description": self.description,
            "direction": self.direction,
            "steps": list(self.steps),
        }


@dataclass(frozen=True)
class WorkerGenerateRequest:
    """主服务发给 Worker 的一轮生成请求。"""

    request_id: str
    run_id: str
    round_index: int
    hypothesis: str
    context_hash: str
    field_catalog_hash: str
    operator_catalog_hash: str
    llm_model: str
    max_candidates: int
    max_expression_depth: int
    direction: str
    excluded_fields: tuple[str, ...]
    fields: tuple[Mapping[str, Any], ...]
    operators: tuple[Mapping[str, Any], ...]
    feedback: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = WORKER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "round_index",
            _integer(self.round_index, "round_index", 1, 10),
        )
        object.__setattr__(
            self,
            "hypothesis",
            _text(self.hypothesis, "hypothesis", minimum=8, maximum=4000),
        )
        object.__setattr__(
            self, "context_hash", _hash(self.context_hash, "context_hash")
        )
        object.__setattr__(
            self,
            "field_catalog_hash",
            _hash(self.field_catalog_hash, "field_catalog_hash"),
        )
        object.__setattr__(
            self,
            "operator_catalog_hash",
            _hash(self.operator_catalog_hash, "operator_catalog_hash"),
        )
        object.__setattr__(
            self, "llm_model", _text(self.llm_model, "llm_model", maximum=200)
        )
        object.__setattr__(
            self,
            "max_candidates",
            _integer(self.max_candidates, "max_candidates", 1, 8),
        )
        object.__setattr__(
            self,
            "max_expression_depth",
            _integer(self.max_expression_depth, "max_expression_depth", 2, 8),
        )
        if self.direction not in {"either", "positive", "negative"}:
            raise WorkerProtocolError(
                "direction 必须是 either、positive 或 negative"
            )
        object.__setattr__(
            self,
            "excluded_fields",
            _string_tuple(self.excluded_fields, "excluded_fields", maximum=300),
        )
        object.__setattr__(
            self,
            "fields",
            _mapping_tuple(self.fields, "fields", maximum=300),
        )
        object.__setattr__(
            self,
            "operators",
            _mapping_tuple(self.operators, "operators", maximum=100),
        )
        object.__setattr__(
            self,
            "feedback",
            _mapping_tuple(self.feedback, "feedback", maximum=20),
        )
        if not self.fields:
            raise WorkerProtocolError("fields 不能为空")
        if not self.operators:
            raise WorkerProtocolError("operators 不能为空")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerGenerateRequest:
        """严格解析一轮生成请求。"""

        if not isinstance(value, Mapping):
            raise WorkerProtocolError("请求根节点必须是 JSON 对象")
        expected = {
            "schema_version",
            "request_id",
            "run_id",
            "round_index",
            "hypothesis",
            "context_hash",
            "field_catalog_hash",
            "operator_catalog_hash",
            "llm_model",
            "max_candidates",
            "max_expression_depth",
            "direction",
            "excluded_fields",
            "fields",
            "operators",
            "feedback",
        }
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise WorkerProtocolError(
                f"请求字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """返回稳定可传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "round_index": self.round_index,
            "hypothesis": self.hypothesis,
            "context_hash": self.context_hash,
            "field_catalog_hash": self.field_catalog_hash,
            "operator_catalog_hash": self.operator_catalog_hash,
            "llm_model": self.llm_model,
            "max_candidates": self.max_candidates,
            "max_expression_depth": self.max_expression_depth,
            "direction": self.direction,
            "excluded_fields": list(self.excluded_fields),
            "fields": list(self.fields),
            "operators": list(self.operators),
            "feedback": list(self.feedback),
        }

    @property
    def payload_hash(self) -> str:
        """返回用于幂等冲突识别的请求哈希。"""

        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def job_id(self) -> str:
        """返回由 request_id 确定的稳定作业 ID。"""

        return _stable_id("wjob", {"request_id": self.request_id})


@dataclass(frozen=True)
class WorkerGenerationResponse:
    """Worker 成功返回的一轮候选草稿。"""

    candidates: tuple[WorkerCandidateDraft, ...]
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_SCHEMA_VERSION}"
            )
        if not self.candidates:
            raise WorkerProtocolError("Worker 没有返回候选")
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        for key, value in self.usage.items():
            if not isinstance(key, str):
                raise WorkerProtocolError("usage 键必须是字符串")
            if isinstance(value, float) and not math.isfinite(value):
                raise WorkerProtocolError("usage 不能包含非有限数值")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerGenerationResponse:
        """从 Worker JSON 严格解析候选列表。"""

        if not isinstance(value, Mapping):
            raise WorkerProtocolError("响应根节点必须是 JSON 对象")
        if set(value) != {"schema_version", "candidates", "usage"}:
            raise WorkerProtocolError("响应字段必须恰好包含 schema_version、candidates、usage")
        raw = value["candidates"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise WorkerProtocolError("candidates 必须是数组")
        return cls(
            candidates=tuple(WorkerCandidateDraft.from_dict(item) for item in raw),
            usage=dict(value["usage"]),
            schema_version=value["schema_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可传输字典。"""

        return {
            "schema_version": self.schema_version,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "usage": dict(self.usage),
        }


@dataclass(frozen=True)
class WorkerJobView:
    """作业状态查询结果。"""

    job_id: str
    request_id: str
    status: str
    result: Mapping[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _identifier(self.job_id, "job_id"))
        object.__setattr__(
            self, "request_id", _identifier(self.request_id, "request_id")
        )
        if self.status not in JOB_STATUSES:
            raise WorkerProtocolError(f"未知作业状态：{self.status}")
        if self.result is not None and not isinstance(self.result, Mapping):
            raise WorkerProtocolError("result 必须是 JSON 对象或空值")
        if self.error_code is not None:
            object.__setattr__(
                self, "error_code", _text(self.error_code, "error_code", maximum=32)
            )
        if self.error_message is not None:
            object.__setattr__(
                self,
                "error_message",
                _text(self.error_message, "error_message", maximum=500),
            )

    @classmethod
    def from_dict(cls, value: Any) -> WorkerJobView:
        """从服务响应解析作业状态。"""

        if not isinstance(value, Mapping):
            raise WorkerProtocolError("作业响应必须是 JSON 对象")
        return cls(
            job_id=value.get("job_id"),
            request_id=value.get("request_id"),
            status=value.get("status"),
            result=value.get("result"),
            error_code=value.get("error_code"),
            error_message=value.get("error_message"),
            created_at=value.get("created_at"),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回 API JSON。"""

        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "status": self.status,
            "result": dict(self.result) if self.result is not None else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
