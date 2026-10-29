"""Worker v3：受限分析工具补充与最终搜索决策协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_v2 import (
    WorkerSearchRequest,
    WorkerSearchResponse,
    ensure_agent_payload,
)

WORKER_TOOL_SEARCH_SCHEMA_VERSION = "mining_worker_v3"
WORKER_SEARCH_PHASES = frozenset({"discover", "finalize"})
_CALL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")


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
    payload = dict(value)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise WorkerProtocolError(f"{name} 超过 {maximum_bytes} 字节")
    return payload


@dataclass(frozen=True)
class WorkerAnalysisToolCall:
    """Worker 请求主服务执行的固定工具调用。"""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not _CALL_ID.fullmatch(
            self.call_id
        ):
            raise WorkerProtocolError("call_id 包含非法字符")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise WorkerProtocolError("tool_name 不能为空")
        if not isinstance(self.arguments, Mapping):
            raise WorkerProtocolError("arguments 必须是 JSON 对象")
        arguments = dict(self.arguments)
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 4096:
            raise WorkerProtocolError("arguments 超过 4096 字节")
        object.__setattr__(self, "arguments", arguments)
        ensure_agent_payload(self.to_dict(), path="tool_call")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerAnalysisToolCall:
        """严格解析工具调用。"""

        payload = _strict_mapping(
            value,
            "tool_call",
            set(cls.__dataclass_fields__),
            maximum_bytes=8192,
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class WorkerSearchTurnRequest:
    """一次 discover 或 finalize 工具补充请求。"""

    request_id: str
    phase: str
    continuation_index: int
    base_request: WorkerSearchRequest
    tool_catalog: tuple[Mapping[str, Any], ...]
    tool_results: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = WORKER_TOOL_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_TOOL_SEARCH_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_TOOL_SEARCH_SCHEMA_VERSION}"
            )
        if not isinstance(self.request_id, str) or not _CALL_ID.fullmatch(
            self.request_id
        ):
            raise WorkerProtocolError("request_id 包含非法字符")
        if self.phase not in WORKER_SEARCH_PHASES:
            raise WorkerProtocolError("phase 必须是 discover 或 finalize")
        if (
            isinstance(self.continuation_index, bool)
            or not isinstance(self.continuation_index, int)
            or self.continuation_index not in {0, 1}
        ):
            raise WorkerProtocolError("continuation_index 必须是 0 或 1")
        if not isinstance(self.base_request, WorkerSearchRequest):
            raise WorkerProtocolError("base_request 必须是 WorkerSearchRequest")
        catalog = _mapping_tuple(
            self.tool_catalog,
            "tool_catalog",
            maximum=7,
            item_maximum=4096,
        )
        results = _mapping_tuple(
            self.tool_results,
            "tool_results",
            maximum=4,
            item_maximum=65536,
        )
        if not catalog:
            raise WorkerProtocolError("tool_catalog 不能为空")
        if self.phase == "discover":
            if self.continuation_index != 0 or results:
                raise WorkerProtocolError(
                    "discover 必须是首次调用且不包含工具结果"
                )
        else:
            if self.continuation_index != 1 or not results:
                raise WorkerProtocolError(
                    "finalize 必须包含一次工具补充和结果"
                )
        object.__setattr__(self, "tool_catalog", catalog)
        object.__setattr__(self, "tool_results", results)
        ensure_agent_payload(self.to_dict(), path="worker_search_turn_request")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchTurnRequest:
        """严格解析 Worker v3 请求。"""

        payload = _strict_mapping(
            value,
            "worker_search_turn_request",
            set(cls.__dataclass_fields__),
            maximum_bytes=327680,
        )
        payload["base_request"] = WorkerSearchRequest.from_dict(
            payload["base_request"]
        )
        payload["tool_catalog"] = tuple(payload["tool_catalog"])
        payload["tool_results"] = tuple(payload["tool_results"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "phase": self.phase,
            "continuation_index": self.continuation_index,
            "base_request": self.base_request.to_dict(),
            "tool_catalog": [dict(item) for item in self.tool_catalog],
            "tool_results": [dict(item) for item in self.tool_results],
        }

    @property
    def payload_hash(self) -> str:
        """返回请求稳定哈希。"""

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
        """返回请求 ID 对应的稳定作业 ID。"""

        encoded = json.dumps(
            {"request_id": self.request_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"tjob_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class WorkerSearchTurnResponse:
    """Worker 返回工具调用批次或最终搜索决策。"""

    request_id: str
    kind: str
    tool_calls: tuple[WorkerAnalysisToolCall, ...] = ()
    response: WorkerSearchResponse | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_TOOL_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_TOOL_SEARCH_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_TOOL_SEARCH_SCHEMA_VERSION}"
            )
        if not isinstance(self.request_id, str) or not _CALL_ID.fullmatch(
            self.request_id
        ):
            raise WorkerProtocolError("request_id 包含非法字符")
        if self.kind not in {"tool_batch", "decision"}:
            raise WorkerProtocolError("kind 必须是 tool_batch 或 decision")
        if isinstance(self.tool_calls, (str, bytes)) or not isinstance(
            self.tool_calls,
            (list, tuple),
        ):
            raise WorkerProtocolError("tool_calls 必须是数组")
        calls = tuple(self.tool_calls)
        if any(not isinstance(item, WorkerAnalysisToolCall) for item in calls):
            raise WorkerProtocolError(
                "tool_calls 每项必须是 WorkerAnalysisToolCall"
            )
        object.__setattr__(self, "tool_calls", calls)
        if self.kind == "tool_batch":
            if not 1 <= len(calls) <= 4 or self.response is not None:
                raise WorkerProtocolError(
                    "tool_batch 必须包含 1～4 次调用且不能包含最终响应"
                )
        elif calls or not isinstance(self.response, WorkerSearchResponse):
            raise WorkerProtocolError(
                "decision 不得包含工具调用且必须提供 WorkerSearchResponse"
            )
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        usage = dict(self.usage)
        if len(json.dumps(usage, ensure_ascii=False).encode("utf-8")) > 16000:
            raise WorkerProtocolError("usage 超过 16000 字节")
        object.__setattr__(self, "usage", usage)
        ensure_agent_payload(self.to_dict(), path="worker_search_turn_response")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchTurnResponse:
        """严格解析 Worker v3 响应。"""

        payload = _strict_mapping(
            value,
            "worker_search_turn_response",
            set(cls.__dataclass_fields__),
            maximum_bytes=327680,
        )
        raw_calls = payload["tool_calls"]
        if not isinstance(raw_calls, list):
            raise WorkerProtocolError("tool_calls 必须是数组")
        payload["tool_calls"] = tuple(
            WorkerAnalysisToolCall.from_dict(item) for item in raw_calls
        )
        if payload["response"] is not None:
            payload["response"] = WorkerSearchResponse.from_dict(
                payload["response"]
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "kind": self.kind,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "response": (
                self.response.to_dict() if self.response is not None else None
            ),
            "usage": dict(self.usage),
        }


def _mapping_tuple(
    value: Any,
    name: str,
    *,
    maximum: int,
    item_maximum: int,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象数组")
    if len(value) > maximum:
        raise WorkerProtocolError(f"{name} 最多包含 {maximum} 项")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise WorkerProtocolError(f"{name} 每项必须是 JSON 对象")
        encoded = json.dumps(
            dict(item),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > item_maximum:
            raise WorkerProtocolError(f"{name} 单项超过 {item_maximum} 字节")
        result.append(json.loads(encoded))
    return tuple(result)


__all__ = [
    "WORKER_SEARCH_PHASES",
    "WORKER_TOOL_SEARCH_SCHEMA_VERSION",
    "WorkerAnalysisToolCall",
    "WorkerSearchTurnRequest",
    "WorkerSearchTurnResponse",
]
