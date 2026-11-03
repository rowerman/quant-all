"""Worker v5 研报字段抽取请求和响应契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerProtocolError

WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION = "mining_worker_research_extraction_v1"
WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION = "mining_research_extraction_prompt_v1"
WORKER_RESEARCH_FACTOR_FIELDS = (
    "factor_name",
    "economic_mechanism",
    "formula",
    "variables",
    "expected_direction",
    "falsification_conditions",
)
WORKER_RESEARCH_FIELD_STATUSES = ("explicit", "inferred", "missing")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, name: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    """校验并去除字符串两端空白。"""

    if not isinstance(value, str):
        raise WorkerProtocolError(f"{name} 必须是字符串")
    result = value.strip()
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(
            f"{name} 长度必须在 {minimum}～{maximum} 个字符"
        )
    return result


def _identifier(value: Any, name: str) -> str:
    """校验稳定标识字符集。"""

    result = _text(value, name, maximum=160)
    if not _ID_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 包含非法字符")
    return result


def _hash(value: Any, name: str) -> str:
    """校验小写 SHA-256。"""

    result = _text(value, name, minimum=64, maximum=64)
    if not _HASH_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def _integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """校验整数范围并拒绝布尔值。"""

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
    maximum: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    """校验字符串数组并保留模型返回顺序。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是字符串数组")
    if not minimum <= len(value) <= maximum:
        raise WorkerProtocolError(f"{name} 数量必须在 {minimum}～{maximum}")
    return tuple(
        _text(item, name, maximum=maximum_chars) for item in value
    )


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
class WorkerResearchPageSegment:
    """Worker 可见的一页或页内片段。"""

    page_number: int
    start_offset: int
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "page_number",
            _integer(self.page_number, "page_number", 1, 100000),
        )
        object.__setattr__(
            self,
            "start_offset",
            _integer(self.start_offset, "start_offset", 0, 20 * 1024 * 1024),
        )
        object.__setattr__(
            self,
            "text",
            _text(self.text, "page segment text", maximum=30000),
        )

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchPageSegment:
        """严格解析页片段。"""

        return cls(
            **_strict_fields(
                value,
                "worker_research_page_segment",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回字典。"""

        return {
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "text": self.text,
        }


@dataclass(frozen=True)
class WorkerResearchCitationDraft:
    """模型返回的待校验原文引用。"""

    page_number: int
    quote: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "page_number",
            _integer(self.page_number, "page_number", 1, 100000),
        )
        object.__setattr__(
            self,
            "quote",
            _text(self.quote, "quote", maximum=2000),
        )

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchCitationDraft:
        """严格解析引用草稿。"""

        return cls(
            **_strict_fields(
                value,
                "worker_research_citation",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回字典。"""

        return {"page_number": self.page_number, "quote": self.quote}


@dataclass(frozen=True)
class WorkerResearchFieldDraft:
    """模型返回的因子字段及证据状态。"""

    field_name: str
    values: tuple[str, ...]
    status: str
    citations: tuple[WorkerResearchCitationDraft, ...]

    def __post_init__(self) -> None:
        if self.field_name not in WORKER_RESEARCH_FACTOR_FIELDS:
            raise WorkerProtocolError(
                f"field_name 必须是 {', '.join(WORKER_RESEARCH_FACTOR_FIELDS)} 之一"
            )
        if self.status not in WORKER_RESEARCH_FIELD_STATUSES:
            raise WorkerProtocolError(
                "status 必须是 explicit、inferred 或 missing 之一"
            )
        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values,
            (list, tuple),
        ):
            raise WorkerProtocolError("values 必须是字符串数组")
        values = _string_tuple(
            self.values,
            "values",
            maximum=20,
            maximum_chars=2000,
        )
        if isinstance(self.citations, (str, bytes)) or not isinstance(
            self.citations,
            (list, tuple),
        ):
            raise WorkerProtocolError("citations 必须是引用数组")
        citations = tuple(self.citations)
        if any(
            not isinstance(item, WorkerResearchCitationDraft)
            for item in citations
        ):
            raise WorkerProtocolError("citations 包含非法引用")
        if len(citations) > 8:
            raise WorkerProtocolError("citations 最多包含 8 条")
        if self.status == "missing":
            if values or citations:
                raise WorkerProtocolError("missing 字段不能包含值或引用")
        else:
            if not values:
                raise WorkerProtocolError(
                    f"{self.status} 字段必须包含至少一个值"
                )
            if not citations:
                raise WorkerProtocolError(
                    f"{self.status} 字段必须包含至少一条引用"
                )
        if self.field_name in {
            "factor_name",
            "economic_mechanism",
            "formula",
            "expected_direction",
        } and len(values) > 1:
            raise WorkerProtocolError(
                f"{self.field_name} 最多包含一个值"
            )
        if (
            self.field_name == "expected_direction"
            and values
            and values[0] not in {"positive", "negative", "non_monotonic"}
        ):
            raise WorkerProtocolError(
                "expected_direction 值必须是 positive、negative 或 non_monotonic"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "citations", citations)

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchFieldDraft:
        """严格解析字段草稿。"""

        payload = _strict_fields(
            value,
            "worker_research_field",
            set(cls.__dataclass_fields__),
        )
        for field_name in ("values", "citations"):
            items = payload[field_name]
            if isinstance(items, (str, bytes)) or not isinstance(
                items,
                (list, tuple),
            ):
                raise WorkerProtocolError(f"{field_name} 必须是数组")
        payload["values"] = tuple(payload["values"])
        payload["citations"] = tuple(
            WorkerResearchCitationDraft.from_dict(item)
            for item in payload["citations"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回字典。"""

        return {
            "field_name": self.field_name,
            "values": list(self.values),
            "status": self.status,
            "citations": [item.to_dict() for item in self.citations],
        }


@dataclass(frozen=True)
class WorkerResearchFactorDraft:
    """模型返回的一个未信任因子草稿。"""

    fields: tuple[WorkerResearchFieldDraft, ...]

    def __post_init__(self) -> None:
        if isinstance(self.fields, (str, bytes)) or not isinstance(
            self.fields,
            (list, tuple),
        ):
            raise WorkerProtocolError("fields 必须是字段数组")
        fields = tuple(self.fields)
        if any(not isinstance(item, WorkerResearchFieldDraft) for item in fields):
            raise WorkerProtocolError("fields 包含非法字段")
        if tuple(item.field_name for item in fields) != WORKER_RESEARCH_FACTOR_FIELDS:
            raise WorkerProtocolError(
                "fields 必须按固定顺序完整包含六类因子字段"
            )
        if all(item.status == "missing" for item in fields):
            raise WorkerProtocolError("因子草稿至少需要一个非 missing 字段")
        object.__setattr__(self, "fields", fields)

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchFactorDraft:
        """严格解析因子草稿。"""

        payload = _strict_fields(
            value,
            "worker_research_factor",
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
        """返回字典。"""

        return {"fields": [item.to_dict() for item in self.fields]}


@dataclass(frozen=True)
class WorkerResearchExtractionRequest:
    """主服务提交给 Worker v5 的单个窗口抽取请求。"""

    request_id: str
    document_id: str
    text_sha256: str
    window_index: int
    window_count: int
    pages: tuple[WorkerResearchPageSegment, ...]
    llm_model: str
    max_factors: int
    prompt_version: str = WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION
    schema_version: str = WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 "
                f"{WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self, "request_id", _identifier(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "document_id", _identifier(self.document_id, "document_id")
        )
        object.__setattr__(
            self, "text_sha256", _hash(self.text_sha256, "text_sha256")
        )
        object.__setattr__(
            self,
            "window_index",
            _integer(self.window_index, "window_index", 0, 7),
        )
        window_count = _integer(self.window_count, "window_count", 1, 8)
        if self.window_index >= window_count:
            raise WorkerProtocolError("window_index 必须小于 window_count")
        if isinstance(self.pages, (str, bytes)) or not isinstance(
            self.pages,
            (list, tuple),
        ):
            raise WorkerProtocolError("pages 必须是页片段数组")
        pages = tuple(self.pages)
        if not 1 <= len(pages) <= 64:
            raise WorkerProtocolError("pages 数量必须在 1～64")
        if any(
            not isinstance(item, WorkerResearchPageSegment) for item in pages
        ):
            raise WorkerProtocolError("pages 包含非法片段")
        if sum(len(item.text) for item in pages) > 30000:
            raise WorkerProtocolError("单个窗口正文不得超过 30000 个字符")
        object.__setattr__(
            self, "llm_model", _text(self.llm_model, "llm_model", maximum=200)
        )
        object.__setattr__(
            self,
            "max_factors",
            _integer(self.max_factors, "max_factors", 1, 8),
        )
        if self.prompt_version != WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION:
            raise WorkerProtocolError("prompt_version 不受支持")
        object.__setattr__(self, "window_count", window_count)
        object.__setattr__(self, "pages", pages)

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
        """返回当前请求的稳定作业 ID。"""

        digest = hashlib.sha256(
            f"{self.request_id}:{self.payload_hash}".encode("utf-8")
        ).hexdigest()
        return f"wjob_{digest[:24]}"

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchExtractionRequest:
        """严格解析请求。"""

        payload = _strict_fields(
            value,
            "worker_research_extraction_request",
            set(cls.__dataclass_fields__),
        )
        if isinstance(payload["pages"], (str, bytes)) or not isinstance(
            payload["pages"],
            (list, tuple),
        ):
            raise WorkerProtocolError("pages 必须是数组")
        payload["pages"] = tuple(
            WorkerResearchPageSegment.from_dict(item)
            for item in payload["pages"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "text_sha256": self.text_sha256,
            "window_index": self.window_index,
            "window_count": self.window_count,
            "pages": [item.to_dict() for item in self.pages],
            "llm_model": self.llm_model,
            "max_factors": self.max_factors,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class WorkerResearchExtractionResponse:
    """Worker v5 返回的未信任因子草稿。"""

    request_id: str
    factors: tuple[WorkerResearchFactorDraft, ...]
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 "
                f"{WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self, "request_id", _identifier(self.request_id, "request_id")
        )
        if isinstance(self.factors, (str, bytes)) or not isinstance(
            self.factors,
            (list, tuple),
        ):
            raise WorkerProtocolError("factors 必须是数组")
        factors = tuple(self.factors)
        if len(factors) > 8:
            raise WorkerProtocolError("factors 最多包含 8 项")
        if any(
            not isinstance(item, WorkerResearchFactorDraft) for item in factors
        ):
            raise WorkerProtocolError("factors 包含非法草稿")
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        for key, value in self.usage.items():
            if not isinstance(key, str):
                raise WorkerProtocolError("usage 键必须是字符串")
            if isinstance(value, float) and not math.isfinite(value):
                raise WorkerProtocolError("usage 不能包含非有限数值")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "usage", dict(self.usage))

    @classmethod
    def from_dict(cls, value: Any) -> WorkerResearchExtractionResponse:
        """严格解析响应。"""

        payload = _strict_fields(
            value,
            "worker_research_extraction_response",
            set(cls.__dataclass_fields__),
        )
        if isinstance(payload["factors"], (str, bytes)) or not isinstance(
            payload["factors"],
            (list, tuple),
        ):
            raise WorkerProtocolError("factors 必须是数组")
        payload["factors"] = tuple(
            WorkerResearchFactorDraft.from_dict(item)
            for item in payload["factors"]
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
            "factors": [item.to_dict() for item in self.factors],
            "usage": dict(self.usage),
        }


__all__ = [
    "WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION",
    "WORKER_RESEARCH_EXTRACTION_SCHEMA_VERSION",
    "WORKER_RESEARCH_FACTOR_FIELDS",
    "WORKER_RESEARCH_FIELD_STATUSES",
    "WorkerResearchCitationDraft",
    "WorkerResearchExtractionRequest",
    "WorkerResearchExtractionResponse",
    "WorkerResearchFactorDraft",
    "WorkerResearchFieldDraft",
    "WorkerResearchPageSegment",
]
