"""研报因子字段抽取、来源引用和结果契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping

from mining.contract_types import (
    identifier,
    sha256_json,
    sha256_text,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError, ResearchMaterialContractError

RESEARCH_EXTRACTION_SCHEMA_VERSION = "mining_research_extraction_v1"
RESEARCH_EXTRACTION_RULE_VERSION = "mining_research_extraction_v1"
RESEARCH_EXTRACTION_PROMPT_VERSION = "mining_research_extraction_prompt_v1"
RESEARCH_FIELD_EVIDENCE_STATUSES = ("explicit", "inferred", "missing")
RESEARCH_FACTOR_FIELDS = (
    "factor_name",
    "economic_mechanism",
    "formula",
    "variables",
    "expected_direction",
    "falsification_conditions",
)
RESEARCH_EXPECTED_DIRECTIONS = (
    "positive",
    "negative",
    "non_monotonic",
)
_DOCUMENT_ID_PATTERN = re.compile(r"^doc_[0-9a-f]{24}$")


def _strict_fields(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    """校验 JSON 对象字段集合恰好等于契约字段集合。"""

    if not isinstance(value, Mapping):
        raise ResearchMaterialContractError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ResearchMaterialContractError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    return dict(value)


def _document_id(value: Any) -> str:
    """校验资料内容 ID。"""

    try:
        result = identifier(value, "document_id")
    except MiningContractError as error:
        raise ResearchMaterialContractError(str(error)) from error
    if not _DOCUMENT_ID_PATTERN.fullmatch(result):
        raise ResearchMaterialContractError(
            "document_id 必须是 doc_ 和 24 位内容哈希"
        )
    return result


def _hash(value: Any, name: str) -> str:
    """校验 SHA-256 并统一契约错误。"""

    try:
        return sha256_text(value, name)
    except MiningContractError as error:
        raise ResearchMaterialContractError(str(error)) from error


def _bounded_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """校验整数范围并明确拒绝布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchMaterialContractError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise ResearchMaterialContractError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def _string_tuple(
    value: Any,
    name: str,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    """校验不重复、稳定排序的字符串序列。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ResearchMaterialContractError(f"{name} 必须是字符串数组")
    if len(value) > maximum_items:
        raise ResearchMaterialContractError(
            f"{name} 最多包含 {maximum_items} 项"
        )
    items: list[str] = []
    for item in value:
        try:
            current = text(item, name, maximum=maximum_chars)
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        items.append(current)
    if items != sorted(set(items)):
        raise ResearchMaterialContractError(f"{name} 必须去重并按字典序排序")
    return tuple(items)


@dataclass(frozen=True)
class ResearchExtractionCitation:
    """一条已绑定页码、偏移和原文哈希的来源引用。"""

    document_id: str
    page_number: int
    start_offset: int
    end_offset: int
    quote: str
    schema_version: str = RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchExtractionCitation schema_version 必须为 "
                f"{RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        page_number = _bounded_integer(
            self.page_number,
            "page_number",
            minimum=1,
            maximum=100000,
        )
        start_offset = _bounded_integer(
            self.start_offset,
            "start_offset",
            minimum=0,
            maximum=20 * 1024 * 1024,
        )
        end_offset = _bounded_integer(
            self.end_offset,
            "end_offset",
            minimum=1,
            maximum=20 * 1024 * 1024,
        )
        if not isinstance(self.quote, str):
            raise ResearchMaterialContractError("quote 必须是字符串")
        if not 1 <= len(self.quote) <= 2000:
            raise ResearchMaterialContractError("quote 长度必须在 1～2000")
        if not self.quote.strip():
            raise ResearchMaterialContractError("quote 不能只有空白")
        if end_offset - start_offset != len(self.quote):
            raise ResearchMaterialContractError(
                "引用起止偏移必须与 quote 长度一致"
            )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "page_number", page_number)
        object.__setattr__(self, "start_offset", start_offset)
        object.__setattr__(self, "end_offset", end_offset)

    @property
    def quote_sha256(self) -> str:
        """返回原文引用 SHA-256。"""

        return hashlib.sha256(self.quote.encode("utf-8")).hexdigest()

    @property
    def locator(self) -> str:
        """返回页内字符偏移定位。"""

        return f"page:{self.page_number}:offset:{self.start_offset}-{self.end_offset}"

    def material(self) -> dict[str, Any]:
        """返回稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "quote": self.quote,
        }

    @property
    def citation_id(self) -> str:
        """返回引用稳定 ID。"""

        return stable_id("cite", {"citation_hash": self.citation_hash})

    @property
    def citation_hash(self) -> str:
        """返回引用内容哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchExtractionCitation:
        """严格解析引用。"""

        return cls(
            **_strict_fields(
                value,
                "research_extraction_citation",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return self.material()


@dataclass(frozen=True)
class ResearchFieldClaim:
    """因子一个字段的内容、证据状态和来源。"""

    field_name: str
    values: tuple[str, ...]
    status: str
    citations: tuple[ResearchExtractionCitation, ...]
    schema_version: str = RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchFieldClaim schema_version 必须为 "
                f"{RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        if self.field_name not in RESEARCH_FACTOR_FIELDS:
            raise ResearchMaterialContractError(
                f"field_name 必须是 {', '.join(RESEARCH_FACTOR_FIELDS)} 之一"
            )
        if self.status not in RESEARCH_FIELD_EVIDENCE_STATUSES:
            raise ResearchMaterialContractError(
                "status 必须是 explicit、inferred 或 missing 之一"
            )
        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("values 必须是字符串数组")
        if isinstance(self.citations, (str, bytes)) or not isinstance(
            self.citations,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("citations 必须是引用数组")
        if any(
            not isinstance(item, ResearchExtractionCitation)
            for item in self.citations
        ):
            raise ResearchMaterialContractError(
                "citations 只能包含 ResearchExtractionCitation"
            )
        values = _string_tuple(
            self.values,
            "values",
            maximum_items=20,
            maximum_chars=2000,
        )
        citations = tuple(self.citations)
        if len(citations) > 8:
            raise ResearchMaterialContractError("每个字段最多包含 8 条引用")
        if len({item.citation_hash for item in citations}) != len(citations):
            raise ResearchMaterialContractError("同一字段不能包含重复引用")
        if len({item.document_id for item in citations}) > 1:
            raise ResearchMaterialContractError("同一字段引用必须属于同一资料")
        if self.status == "missing":
            if values or citations:
                raise ResearchMaterialContractError(
                    "missing 字段不能包含值或引用"
                )
        else:
            if not values:
                raise ResearchMaterialContractError(
                    f"{self.status} 字段必须包含至少一个值"
                )
            if not citations:
                raise ResearchMaterialContractError(
                    f"{self.status} 字段必须包含至少一条引用"
                )
        if self.field_name in {
            "factor_name",
            "economic_mechanism",
            "formula",
            "expected_direction",
        }:
            if len(values) > 1:
                raise ResearchMaterialContractError(
                    f"{self.field_name} 最多包含一个值"
                )
        if self.field_name == "expected_direction" and values:
            if values[0] not in RESEARCH_EXPECTED_DIRECTIONS:
                raise ResearchMaterialContractError(
                    "expected_direction 值必须是 positive、negative 或 "
                    "non_monotonic"
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "citations", citations)

    def material(self) -> dict[str, Any]:
        """返回稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "field_name": self.field_name,
            "values": list(self.values),
            "status": self.status,
            "citations": [item.material() for item in self.citations],
        }

    @property
    def claim_hash(self) -> str:
        """返回字段声明稳定哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchFieldClaim:
        """严格解析字段声明。"""

        payload = _strict_fields(
            value,
            "research_field_claim",
            set(cls.__dataclass_fields__),
        )
        for field_name in ("values", "citations"):
            items = payload[field_name]
            if isinstance(items, (str, bytes)) or not isinstance(
                items,
                (list, tuple),
            ):
                raise ResearchMaterialContractError(
                    f"{field_name} 必须是数组"
                )
        payload["values"] = tuple(payload["values"])
        payload["citations"] = tuple(
            ResearchExtractionCitation.from_dict(item)
            for item in payload["citations"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            **self.material(),
            "citations": [item.to_dict() for item in self.citations],
        }


@dataclass(frozen=True)
class ResearchExtractedFactor:
    """一份研报中的一个待筛选因子字段草稿。"""

    document_id: str
    text_sha256: str
    fields: tuple[ResearchFieldClaim, ...]
    schema_version: str = RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchExtractedFactor schema_version 必须为 "
                f"{RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        text_sha256 = _hash(self.text_sha256, "text_sha256")
        if isinstance(self.fields, (str, bytes)) or not isinstance(
            self.fields,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("fields 必须是字段声明数组")
        fields = tuple(self.fields)
        if any(not isinstance(item, ResearchFieldClaim) for item in fields):
            raise ResearchMaterialContractError(
                "fields 只能包含 ResearchFieldClaim"
            )
        if tuple(item.field_name for item in fields) != RESEARCH_FACTOR_FIELDS:
            raise ResearchMaterialContractError(
                "fields 必须按固定顺序完整包含六类因子字段"
            )
        if any(
            citation.document_id != document_id
            for item in fields
            for citation in item.citations
        ):
            raise ResearchMaterialContractError(
                "字段引用必须属于当前资料"
            )
        if all(item.status == "missing" for item in fields):
            raise ResearchMaterialContractError(
                "因子草稿至少需要一个非 missing 字段"
            )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "text_sha256", text_sha256)
        object.__setattr__(self, "fields", fields)

    def field(self, name: str) -> ResearchFieldClaim:
        """按字段名返回声明。"""

        for item in self.fields:
            if item.field_name == name:
                return item
        raise ResearchMaterialContractError(f"缺少字段 {name}")

    def material(self) -> dict[str, Any]:
        """返回稳定草稿哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "text_sha256": self.text_sha256,
            "fields": [item.material() for item in self.fields],
        }

    @property
    def factor_hash(self) -> str:
        """返回因子草稿内容哈希。"""

        return sha256_json(self.material())

    @property
    def draft_id(self) -> str:
        """返回因子草稿稳定 ID。"""

        return stable_id("factordraft", {"factor_hash": self.factor_hash})

    @classmethod
    def from_dict(cls, value: Any) -> ResearchExtractedFactor:
        """严格解析因子草稿。"""

        payload = _strict_fields(
            value,
            "research_extracted_factor",
            set(cls.__dataclass_fields__),
        )
        if isinstance(payload["fields"], (str, bytes)) or not isinstance(
            payload["fields"],
            (list, tuple),
        ):
            raise ResearchMaterialContractError("fields 必须是数组")
        payload["fields"] = tuple(
            ResearchFieldClaim.from_dict(item) for item in payload["fields"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            **self.material(),
            "fields": [item.to_dict() for item in self.fields],
        }


@dataclass(frozen=True)
class ResearchExtractionResult:
    """一份研报完成全部抽取窗口后的可审计结果。"""

    document_id: str
    text_sha256: str
    model: str
    prompt_version: str
    extraction_version: str
    window_count: int
    factors: tuple[ResearchExtractedFactor, ...]
    worker_job_ids: tuple[str, ...]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_EXTRACTION_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchExtractionResult schema_version 必须为 "
                f"{RESEARCH_EXTRACTION_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        text_sha256 = _hash(self.text_sha256, "text_sha256")
        try:
            model = text(self.model, "model", maximum=200)
            prompt_version = text(
                self.prompt_version,
                "prompt_version",
                maximum=128,
            )
            extraction_version = text(
                self.extraction_version,
                "extraction_version",
                maximum=128,
            )
            created_at = validate_utc_timestamp(self.created_at, "created_at")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        if prompt_version != RESEARCH_EXTRACTION_PROMPT_VERSION:
            raise ResearchMaterialContractError(
                f"prompt_version 必须为 {RESEARCH_EXTRACTION_PROMPT_VERSION}"
            )
        if extraction_version != RESEARCH_EXTRACTION_RULE_VERSION:
            raise ResearchMaterialContractError(
                f"extraction_version 必须为 {RESEARCH_EXTRACTION_RULE_VERSION}"
            )
        window_count = _bounded_integer(
            self.window_count,
            "window_count",
            minimum=1,
            maximum=8,
        )
        if isinstance(self.factors, (str, bytes)) or not isinstance(
            self.factors,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("factors 必须是草稿数组")
        factors = tuple(self.factors)
        if len(factors) > 64:
            raise ResearchMaterialContractError("factors 不得超过 64 个")
        if any(not isinstance(item, ResearchExtractedFactor) for item in factors):
            raise ResearchMaterialContractError(
                "factors 只能包含 ResearchExtractedFactor"
            )
        if any(
            item.document_id != document_id
            or item.text_sha256 != text_sha256
            for item in factors
        ):
            raise ResearchMaterialContractError(
                "因子草稿必须绑定当前资料 ID 和正文哈希"
            )
        if len({item.factor_hash for item in factors}) != len(factors):
            raise ResearchMaterialContractError("结果不能包含完全重复的因子草稿")
        if isinstance(self.worker_job_ids, (str, bytes)) or not isinstance(
            self.worker_job_ids,
            (list, tuple),
        ):
            raise ResearchMaterialContractError(
                "worker_job_ids 必须是字符串数组"
            )
        worker_job_ids = tuple(self.worker_job_ids)
        if len(worker_job_ids) != window_count:
            raise ResearchMaterialContractError(
                "worker_job_ids 数量必须等于 window_count"
            )
        if len(set(worker_job_ids)) != len(worker_job_ids):
            raise ResearchMaterialContractError("worker_job_ids 不能重复")
        for job_id in worker_job_ids:
            try:
                identifier(job_id, "worker_job_id")
            except MiningContractError as error:
                raise ResearchMaterialContractError(str(error)) from error
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "text_sha256", text_sha256)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(self, "extraction_version", extraction_version)
        object.__setattr__(self, "window_count", window_count)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "worker_job_ids", worker_job_ids)
        object.__setattr__(self, "created_at", created_at)

    @property
    def status(self) -> str:
        """返回结果是否发现因子草稿。"""

        return "extracted" if self.factors else "no_factor_evidence"

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "text_sha256": self.text_sha256,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "extraction_version": self.extraction_version,
            "window_count": self.window_count,
            "factors": [item.material() for item in self.factors],
            "worker_job_ids": list(self.worker_job_ids),
        }

    @property
    def extraction_hash(self) -> str:
        """返回抽取结果稳定哈希。"""

        return sha256_json(self.material())

    @property
    def extraction_id(self) -> str:
        """返回抽取结果稳定 ID。"""

        return stable_id("extraction", {"extraction_hash": self.extraction_hash})

    @classmethod
    def from_dict(cls, value: Any) -> ResearchExtractionResult:
        """严格解析抽取结果。"""

        payload = _strict_fields(
            value,
            "research_extraction_result",
            set(cls.__dataclass_fields__),
        )
        for field_name in ("factors", "worker_job_ids"):
            items = payload[field_name]
            if isinstance(items, (str, bytes)) or not isinstance(
                items,
                (list, tuple),
            ):
                raise ResearchMaterialContractError(
                    f"{field_name} 必须是数组"
                )
        payload["factors"] = tuple(
            ResearchExtractedFactor.from_dict(item)
            for item in payload["factors"]
        )
        payload["worker_job_ids"] = tuple(payload["worker_job_ids"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            **self.material(),
            "factors": [item.to_dict() for item in self.factors],
            "worker_job_ids": list(self.worker_job_ids),
            "created_at": self.created_at,
        }


__all__ = [
    "RESEARCH_EXPECTED_DIRECTIONS",
    "RESEARCH_EXTRACTION_PROMPT_VERSION",
    "RESEARCH_EXTRACTION_RULE_VERSION",
    "RESEARCH_EXTRACTION_SCHEMA_VERSION",
    "RESEARCH_FACTOR_FIELDS",
    "RESEARCH_FIELD_EVIDENCE_STATUSES",
    "ResearchExtractedFactor",
    "ResearchExtractionCitation",
    "ResearchExtractionResult",
    "ResearchFieldClaim",
]
