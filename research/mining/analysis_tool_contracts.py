"""受控分析工具的固定目录、请求和结果契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping

from mining.contract_types import (
    canonical_json,
    identifier,
    sha256_json,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import (
    AnalysisToolArgumentError,
    AnalysisToolNotFoundError,
    AnalysisToolResultError,
)
from mining.search_contract_types import bounded_integer, mapping_with_size, strict_fields

ANALYSIS_TOOL_SCHEMA_VERSION = "mining_analysis_tool_v1"
ANALYSIS_TOOL_CATALOG_VERSION = "mining_analysis_catalog_v1"
ANALYSIS_TOOL_TIMEOUT_SECONDS = 5.0
ANALYSIS_TOOL_MAX_RESULT_BYTES = 65536
ANALYSIS_TOOL_MAX_CALLS_PER_TURN = 4
ANALYSIS_TOOL_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")


@dataclass(frozen=True)
class AnalysisToolDefinition:
    """一个只读分析工具的主服务目录项。"""

    name: str
    version: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    maximum_result_bytes: int = ANALYSIS_TOOL_MAX_RESULT_BYTES

    def to_dict(self) -> dict[str, Any]:
        """返回 Worker 可见的稳定目录项。"""

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "required_arguments": list(self.required_arguments),
            "optional_arguments": list(self.optional_arguments),
            "maximum_result_bytes": self.maximum_result_bytes,
        }


TOOL_DEFINITIONS = (
    AnalysisToolDefinition(
        name="lookup_fields",
        version="1",
        description="按名称或标签查询当前冻结字段目录",
        required_arguments=("query",),
        optional_arguments=("limit",),
    ),
    AnalysisToolDefinition(
        name="lookup_operators",
        version="1",
        description="按名称或说明查询当前冻结算子目录",
        required_arguments=("query",),
        optional_arguments=("limit",),
    ),
    AnalysisToolDefinition(
        name="run_summary",
        version="1",
        description="读取当前运行的假设、候选和尝试分类计数",
        required_arguments=(),
    ),
    AnalysisToolDefinition(
        name="candidate_evidence",
        version="1",
        description="读取当前运行候选中已登记的开发期证据摘要",
        required_arguments=("candidate_id",),
    ),
    AnalysisToolDefinition(
        name="candidate_correlation",
        version="1",
        description="读取当前运行批次中已有的开发期相关摘要",
        required_arguments=("candidate_id",),
    ),
    AnalysisToolDefinition(
        name="candidate_ablation",
        version="1",
        description="对候选固定顶层组成项执行开发期留一消融",
        required_arguments=("candidate_id",),
    ),
    AnalysisToolDefinition(
        name="window_neighborhood",
        version="1",
        description="返回候选首个整数窗口的确定性相邻变体摘要",
        required_arguments=("candidate_id",),
    ),
)
TOOL_BY_NAME = {item.name: item for item in TOOL_DEFINITIONS}
TOOL_NAMES = tuple(item.name for item in TOOL_DEFINITIONS)
TOOL_CATALOG_HASH = sha256_json(
    {
        "schema_version": ANALYSIS_TOOL_CATALOG_VERSION,
        "tools": [item.to_dict() for item in TOOL_DEFINITIONS],
    }
)


def tool_catalog() -> tuple[Mapping[str, Any], ...]:
    """返回不可变的工具目录快照。"""

    return tuple(item.to_dict() for item in TOOL_DEFINITIONS)


def _validate_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    definition = TOOL_BY_NAME.get(tool_name)
    if definition is None:
        raise AnalysisToolNotFoundError(f"未知分析工具：{tool_name!r}")
    if not isinstance(arguments, Mapping):
        raise AnalysisToolArgumentError("工具 arguments 必须是 JSON 对象")
    actual = set(arguments)
    allowed = set(definition.required_arguments) | set(definition.optional_arguments)
    missing = set(definition.required_arguments) - actual
    unknown = actual - allowed
    if missing or unknown:
        raise AnalysisToolArgumentError(
            f"工具参数不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    normalized: dict[str, Any] = {}
    if tool_name in {"lookup_fields", "lookup_operators"}:
        normalized["query"] = text(
            arguments["query"],
            "query",
            maximum=80,
        )
        normalized["limit"] = bounded_integer(
            arguments.get("limit", 10),
            "limit",
            minimum=1,
            maximum=20,
        )
    elif tool_name in {
        "candidate_evidence",
        "candidate_correlation",
        "candidate_ablation",
        "window_neighborhood",
    }:
        normalized["candidate_id"] = identifier(
            arguments["candidate_id"],
            "candidate_id",
        )
    if len(canonical_json(normalized).encode("utf-8")) > 4096:
        raise AnalysisToolArgumentError("工具参数超过 4096 字节")
    return normalized


@dataclass(frozen=True)
class AnalysisToolRequest:
    """Worker 通过工具补充协议发起的单个固定工具请求。"""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ANALYSIS_TOOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_TOOL_SCHEMA_VERSION:
            raise AnalysisToolArgumentError(
                f"工具请求 schema_version 必须为 {ANALYSIS_TOOL_SCHEMA_VERSION}"
            )
        call_id = text(self.call_id, "call_id", maximum=80)
        if not ANALYSIS_TOOL_CALL_ID_PATTERN.fullmatch(call_id):
            raise AnalysisToolArgumentError("call_id 包含非法字符")
        object.__setattr__(self, "call_id", call_id)
        tool_name = text(self.tool_name, "tool_name", maximum=80)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(
            self,
            "arguments",
            _validate_arguments(tool_name, self.arguments),
        )

    @classmethod
    def from_dict(cls, value: Any) -> AnalysisToolRequest:
        """严格解析 Worker 工具请求。"""

        return cls(
            **strict_fields(
                value,
                "analysis_tool_request",
                set(cls.__dataclass_fields__),
            )
        )

    def material(self) -> dict[str, Any]:
        """返回不包含传输调用名的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "tool_version": TOOL_BY_NAME[self.tool_name].version,
            "arguments": dict(self.arguments),
        }

    @property
    def request_hash(self) -> str:
        """返回工具与参数的稳定哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回可传输字典。"""

        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class AnalysisToolResult:
    """主服务返回给 Worker 的脱敏工具结果。"""

    call_id: str
    tool_name: str
    tool_version: str
    payload: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = ANALYSIS_TOOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_TOOL_SCHEMA_VERSION:
            raise AnalysisToolArgumentError(
                f"工具结果 schema_version 必须为 {ANALYSIS_TOOL_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "call_id", text(self.call_id, "call_id", maximum=80))
        object.__setattr__(
            self,
            "tool_name",
            text(self.tool_name, "tool_name", maximum=80),
        )
        if self.tool_name not in TOOL_BY_NAME:
            raise AnalysisToolNotFoundError(f"未知分析工具：{self.tool_name!r}")
        expected_version = TOOL_BY_NAME[self.tool_name].version
        if self.tool_version != expected_version:
            raise AnalysisToolArgumentError("工具版本与当前目录不一致")
        object.__setattr__(
            self,
            "tool_version",
            text(self.tool_version, "tool_version", maximum=20),
        )
        try:
            payload = mapping_with_size(
                self.payload,
                "payload",
                maximum_bytes=ANALYSIS_TOOL_MAX_RESULT_BYTES,
            )
        except Exception as exc:
            raise AnalysisToolResultError("工具结果超过 65536 字节") from exc
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )
        if len(canonical_json(self.material()).encode("utf-8")) > ANALYSIS_TOOL_MAX_RESULT_BYTES:
            raise AnalysisToolResultError("工具结果超过 65536 字节")

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定内容。"""

        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "payload": self.payload,
        }

    @property
    def result_hash(self) -> str:
        """返回工具结果稳定哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> AnalysisToolResult:
        """严格解析持久化或传输工具结果。"""

        return cls(
            **strict_fields(
                value,
                "analysis_tool_result",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "created_at": self.created_at}


def finite_float(value: Any) -> float | None:
    """把有限数值转换为浮点，缺失值和非有限值返回 None。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


__all__ = [
    "ANALYSIS_TOOL_CATALOG_VERSION",
    "ANALYSIS_TOOL_MAX_CALLS_PER_TURN",
    "ANALYSIS_TOOL_MAX_RESULT_BYTES",
    "ANALYSIS_TOOL_SCHEMA_VERSION",
    "ANALYSIS_TOOL_TIMEOUT_SECONDS",
    "AnalysisToolDefinition",
    "AnalysisToolRequest",
    "AnalysisToolResult",
    "TOOL_CATALOG_HASH",
    "TOOL_DEFINITIONS",
    "TOOL_NAMES",
    "finite_float",
    "tool_catalog",
]
