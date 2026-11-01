"""Worker v6 研报字段到受控 DSL 的转换协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_research import (
    WORKER_RESEARCH_FACTOR_FIELDS,
    WorkerResearchFieldDraft,
)

WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION = (
    "mining_worker_research_conversion_v1"
)
WORKER_RESEARCH_CONVERSION_PROMPT_VERSION = (
    "mining_research_dsl_conversion_prompt_v1"
)
WORKER_RESEARCH_CONVERSION_STATUSES = ("converted", "rejected")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, name: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    """校验字符串并去除两端空白。"""

    if not isinstance(value, str):
        raise WorkerProtocolError(f"{name} 必须是字符串")
    result = value.strip()
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(
            f"{name} 长度必须在 {minimum}～{maximum} 个字符"
        )
    return result


def _identifier(value: Any, name: str) -> str:
    """校验稳定标识。"""

    result = _text(value, name, maximum=160)
    if not _IDENTIFIER.fullmatch(result):
        raise WorkerProtocolError(f"{name} 包含非法字符")
    return result


def _hash(value: Any, name: str) -> str:
    """校验 SHA-256。"""

    result = _text(value, name, minimum=64, maximum=64)
    if not _HASH.fullmatch(result):
        raise WorkerProtocolError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """校验整数范围。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise WorkerProtocolError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def _strict_fields(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    """校验 JSON 对象字段集合。"""

    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise WorkerProtocolError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    return dict(value)


@dataclass(frozen=True)
class WorkerResearchCatalogField:
    """Worker 可见的允许字段元数据。"""

    name: str
    label: str
    dimension: str
    unit: str

    def __post_init__(self) -> None:
        name = _text(self.name, "field name", maximum=64)
        if not _FIELD_NAME.fullmatch(name):
            raise WorkerProtocolError("field name 不合法")
        object.__setattr__(self, "name", name)
        for attribute in ("label", "dimension", "unit"):
            object.__setattr__(
                self,
                attribute,
                _text(getattr(self, attribute), attribute, maximum=120),
            )

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchCatalogField:
        """严格解析字段元数据。"""

        return cls(
            **_strict_fields(
                value,
                "worker_research_catalog_field",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, str]:
        """返回传输字典。"""

        return {
            "name": self.name,
            "label": self.label,
            "dimension": self.dimension,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class WorkerResearchCatalogOperator:
    """Worker 可见的允许算子元数据。"""

    name: str
    kind: str
    min_args: int
    max_args: int
    min_window: int | None = None
    max_window: int | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, "operator name", maximum=64)
        if not _FIELD_NAME.fullmatch(name):
            raise WorkerProtocolError("operator name 不合法")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "kind",
            _text(self.kind, "operator kind", maximum=32),
        )
        object.__setattr__(
            self,
            "min_args",
            _integer(self.min_args, "min_args", minimum=1, maximum=8),
        )
        object.__setattr__(
            self,
            "max_args",
            _integer(self.max_args, "max_args", minimum=1, maximum=8),
        )
        if self.min_args > self.max_args:
            raise WorkerProtocolError("min_args 不得超过 max_args")
        if (self.min_window is None) != (self.max_window is None):
            raise WorkerProtocolError("窗口最小值和最大值必须同时提供")
        if self.min_window is not None:
            object.__setattr__(
                self,
                "min_window",
                _integer(
                    self.min_window,
                    "min_window",
                    minimum=1,
                    maximum=252,
                ),
            )
            object.__setattr__(
                self,
                "max_window",
                _integer(
                    self.max_window,
                    "max_window",
                    minimum=1,
                    maximum=252,
                ),
            )
            if self.min_window > self.max_window:
                raise WorkerProtocolError("min_window 不得超过 max_window")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchCatalogOperator:
        """严格解析算子元数据。"""

        return cls(
            **_strict_fields(
                value,
                "worker_research_catalog_operator",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "name": self.name,
            "kind": self.kind,
            "min_args": self.min_args,
            "max_args": self.max_args,
            "min_window": self.min_window,
            "max_window": self.max_window,
        }


@dataclass(frozen=True)
class WorkerResearchConversionDraft:
    """一份研报中等待转换的一个因子草稿。"""

    draft_id: str
    factor_hash: str
    fields: tuple[WorkerResearchFieldDraft, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_id", _identifier(self.draft_id, "draft_id"))
        object.__setattr__(
            self,
            "factor_hash",
            _hash(self.factor_hash, "factor_hash"),
        )
        if isinstance(self.fields, (str, bytes)) or not isinstance(
            self.fields,
            (list, tuple),
        ):
            raise WorkerProtocolError("fields 必须是字段数组")
        fields = tuple(self.fields)
        if tuple(item.field_name for item in fields) != WORKER_RESEARCH_FACTOR_FIELDS:
            raise WorkerProtocolError(
                "fields 必须按固定顺序完整包含六类因子字段"
            )
        if any(not isinstance(item, WorkerResearchFieldDraft) for item in fields):
            raise WorkerProtocolError("fields 包含非法字段")
        object.__setattr__(self, "fields", fields)

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchConversionDraft:
        """严格解析转换草稿。"""

        payload = _strict_fields(
            value,
            "worker_research_conversion_draft",
            set(cls.__dataclass_fields__),
        )
        if isinstance(payload["fields"], (str, bytes)) or not isinstance(
            payload["fields"],
            (list, tuple),
        ):
            raise WorkerProtocolError("fields 必须是数组")
        payload["fields"] = tuple(
            WorkerResearchFieldDraft.from_dict(item)
            for item in payload["fields"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "draft_id": self.draft_id,
            "factor_hash": self.factor_hash,
            "fields": [item.to_dict() for item in self.fields],
        }


@dataclass(frozen=True)
class WorkerResearchConversionRequest:
    """主服务提交给 Worker v6 的一个研报转换分块。"""

    request_id: str
    document_id: str
    extraction_id: str
    chunk_index: int
    chunk_count: int
    catalog_hash: str
    drafts: tuple[WorkerResearchConversionDraft, ...]
    fields: tuple[WorkerResearchCatalogField, ...]
    operators: tuple[WorkerResearchCatalogOperator, ...]
    llm_model: str
    prompt_version: str = WORKER_RESEARCH_CONVERSION_PROMPT_VERSION
    schema_version: str = WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 "
                f"{WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _identifier(self.document_id, "document_id"),
        )
        object.__setattr__(
            self,
            "extraction_id",
            _identifier(self.extraction_id, "extraction_id"),
        )
        object.__setattr__(
            self,
            "catalog_hash",
            _hash(self.catalog_hash, "catalog_hash"),
        )
        chunk_count = _integer(
            self.chunk_count,
            "chunk_count",
            minimum=1,
            maximum=64,
        )
        object.__setattr__(
            self,
            "chunk_index",
            _integer(
                self.chunk_index,
                "chunk_index",
                minimum=0,
                maximum=chunk_count - 1,
            ),
        )
        object.__setattr__(self, "chunk_count", chunk_count)
        for name, item_type, maximum in (
            ("drafts", WorkerResearchConversionDraft, 8),
            ("fields", WorkerResearchCatalogField, 500),
            ("operators", WorkerResearchCatalogOperator, 100),
        ):
            items = getattr(self, name)
            if isinstance(items, (str, bytes)) or not isinstance(
                items,
                (list, tuple),
            ):
                raise WorkerProtocolError(f"{name} 必须是数组")
            values = tuple(items)
            if not 1 <= len(values) <= maximum:
                raise WorkerProtocolError(
                    f"{name} 数量必须在 1～{maximum}"
                )
            if any(not isinstance(item, item_type) for item in values):
                raise WorkerProtocolError(f"{name} 包含非法元素")
            object.__setattr__(self, name, values)
        if len({item.draft_id for item in self.drafts}) != len(self.drafts):
            raise WorkerProtocolError("drafts 不能包含重复 draft_id")
        if len({item.name for item in self.fields}) != len(self.fields):
            raise WorkerProtocolError("fields 不能包含重复字段")
        if len({item.name for item in self.operators}) != len(self.operators):
            raise WorkerProtocolError("operators 不能包含重复算子")
        object.__setattr__(
            self,
            "llm_model",
            _text(self.llm_model, "llm_model", maximum=200),
        )
        if self.prompt_version != WORKER_RESEARCH_CONVERSION_PROMPT_VERSION:
            raise WorkerProtocolError("prompt_version 不受支持")

    @property
    def payload_hash(self) -> str:
        """返回请求内容哈希。"""

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
        """返回稳定作业 ID。"""

        digest = hashlib.sha256(
            f"{self.request_id}:{self.payload_hash}".encode("utf-8")
        ).hexdigest()
        return f"wjob_{digest[:24]}"

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchConversionRequest:
        """严格解析请求。"""

        payload = _strict_fields(
            value,
            "worker_research_conversion_request",
            set(cls.__dataclass_fields__),
        )
        for name, item_type in (
            ("drafts", WorkerResearchConversionDraft),
            ("fields", WorkerResearchCatalogField),
            ("operators", WorkerResearchCatalogOperator),
        ):
            if isinstance(payload[name], (str, bytes)) or not isinstance(
                payload[name],
                (list, tuple),
            ):
                raise WorkerProtocolError(f"{name} 必须是数组")
            payload[name] = tuple(
                item_type.from_dict(item) for item in payload[name]
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "extraction_id": self.extraction_id,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "catalog_hash": self.catalog_hash,
            "drafts": [item.to_dict() for item in self.drafts],
            "fields": [item.to_dict() for item in self.fields],
            "operators": [item.to_dict() for item in self.operators],
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class WorkerResearchConversionOutcome:
    """Worker 对一个草稿给出的转换或拒绝结论。"""

    draft_id: str
    status: str
    expression: str | None
    field_mapping: Mapping[str, str] = field(default_factory=dict)
    reason_code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_id", _identifier(self.draft_id, "draft_id"))
        if self.status not in WORKER_RESEARCH_CONVERSION_STATUSES:
            raise WorkerProtocolError(
                "status 必须是 converted 或 rejected"
            )
        if not isinstance(self.field_mapping, Mapping):
            raise WorkerProtocolError("field_mapping 必须是 JSON 对象")
        mapping = dict(self.field_mapping)
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not _FIELD_NAME.fullmatch(value)
            for key, value in mapping.items()
        ):
            raise WorkerProtocolError(
                "field_mapping 的键和值必须是非空字符串"
            )
        object.__setattr__(self, "field_mapping", mapping)
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _identifier(self.reason_code, "reason_code"),
            )
        if self.message is not None:
            object.__setattr__(
                self,
                "message",
                _text(self.message, "message", maximum=500),
            )
        if self.status == "converted":
            if self.expression is None:
                raise WorkerProtocolError("converted 必须提供 expression")
            object.__setattr__(
                self,
                "expression",
                _text(self.expression, "expression", maximum=12000),
            )
            if not mapping:
                raise WorkerProtocolError("converted 必须提供 field_mapping")
            if self.reason_code is not None:
                raise WorkerProtocolError("converted 不能提供 reason_code")
        else:
            if self.expression is not None or mapping:
                raise WorkerProtocolError(
                    "rejected 不能提供 expression 或 field_mapping"
                )
            if self.reason_code is None:
                raise WorkerProtocolError("rejected 必须提供 reason_code")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchConversionOutcome:
        """严格解析转换结论。"""

        payload = _strict_fields(
            value,
            "worker_research_conversion_outcome",
            set(cls.__dataclass_fields__),
        )
        payload["field_mapping"] = dict(payload["field_mapping"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "draft_id": self.draft_id,
            "status": self.status,
            "expression": self.expression,
            "field_mapping": dict(self.field_mapping),
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class WorkerResearchConversionResponse:
    """Worker v6 返回的逐草稿转换结论。"""

    request_id: str
    outcomes: tuple[WorkerResearchConversionOutcome, ...]
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 "
                f"{WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, "request_id"),
        )
        if isinstance(self.outcomes, (str, bytes)) or not isinstance(
            self.outcomes,
            (list, tuple),
        ):
            raise WorkerProtocolError("outcomes 必须是数组")
        outcomes = tuple(self.outcomes)
        if len(outcomes) > 8:
            raise WorkerProtocolError("outcomes 最多包含 8 项")
        if any(
            not isinstance(item, WorkerResearchConversionOutcome)
            for item in outcomes
        ):
            raise WorkerProtocolError("outcomes 包含非法结论")
        if len({item.draft_id for item in outcomes}) != len(outcomes):
            raise WorkerProtocolError("outcomes 不能包含重复 draft_id")
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        for key, value in self.usage.items():
            if not isinstance(key, str):
                raise WorkerProtocolError("usage 键必须是字符串")
            if isinstance(value, float) and not math.isfinite(value):
                raise WorkerProtocolError("usage 不能包含非有限数值")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "usage", dict(self.usage))

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchConversionResponse:
        """严格解析响应。"""

        payload = _strict_fields(
            value,
            "worker_research_conversion_response",
            set(cls.__dataclass_fields__),
        )
        if isinstance(payload["outcomes"], (str, bytes)) or not isinstance(
            payload["outcomes"],
            (list, tuple),
        ):
            raise WorkerProtocolError("outcomes 必须是数组")
        payload["outcomes"] = tuple(
            WorkerResearchConversionOutcome.from_dict(item)
            for item in payload["outcomes"]
        )
        if not isinstance(payload["usage"], Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        payload["usage"] = dict(payload["usage"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "usage": dict(self.usage),
        }


__all__ = [
    "WORKER_RESEARCH_CONVERSION_PROMPT_VERSION",
    "WORKER_RESEARCH_CONVERSION_SCHEMA_VERSION",
    "WORKER_RESEARCH_CONVERSION_STATUSES",
    "WorkerResearchCatalogField",
    "WorkerResearchCatalogOperator",
    "WorkerResearchConversionDraft",
    "WorkerResearchConversionOutcome",
    "WorkerResearchConversionRequest",
    "WorkerResearchConversionResponse",
]
