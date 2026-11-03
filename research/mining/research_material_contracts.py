"""资料驱动的输入、来源引用和解析结果契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from mining.contract_types import (
    canonical_json,
    identifier,
    sha256_json,
    sha256_text,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import (
    MiningContractError,
    ResearchMaterialContractError,
)
from mining.research_material_safety import (
    redact_research_material_text,
    validate_relative_material_path,
)

RESEARCH_MATERIAL_SCHEMA_VERSION = "mining_research_materials_v1"
RESEARCH_DOCUMENT_STATUSES = ("parsed", "failed")
RESEARCH_FAILURE_CATEGORIES = (
    "unreadable",
    "input_changed",
    "unsupported_format",
    "corrupt_document",
    "encrypted_document",
    "no_extractable_text",
    "resource_limit",
    "unsafe_input",
    "parser_error",
)

_FAILURE_CODE_BY_CATEGORY = {
    "unreadable": "MINING-149",
    "input_changed": "MINING-149",
    "unsupported_format": "MINING-150",
    "corrupt_document": "MINING-150",
    "encrypted_document": "MINING-150",
    "no_extractable_text": "MINING-150",
    "resource_limit": "MINING-152",
    "unsafe_input": "MINING-148",
    "parser_error": "MINING-151",
}
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_PARSER_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")


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


def _choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    """校验字符串枚举。"""

    if not isinstance(value, str) or value not in allowed:
        raise ResearchMaterialContractError(f"{name} 必须是 {', '.join(allowed)} 之一")
    return value


def _media_type(value: Any) -> str:
    """校验并规范化媒体类型。"""

    try:
        result = text(value, "media_type", maximum=120).lower()
    except MiningContractError as error:
        raise ResearchMaterialContractError(str(error)) from error
    if not _MEDIA_TYPE_PATTERN.fullmatch(result):
        raise ResearchMaterialContractError("media_type 必须是合法的 MIME 类型")
    return result


def _parser_version(value: Any) -> str:
    """校验冻结解析器版本。"""

    try:
        result = text(value, "parser_version", maximum=128)
    except MiningContractError as error:
        raise ResearchMaterialContractError(str(error)) from error
    if not _PARSER_VERSION_PATTERN.fullmatch(result):
        raise ResearchMaterialContractError("parser_version 格式不合法")
    return result


def _document_id(value: Any) -> str:
    """校验资料内容 ID。"""

    try:
        result = identifier(value, "document_id")
    except MiningContractError as error:
        raise ResearchMaterialContractError(str(error)) from error
    if not re.fullmatch(r"doc_[0-9a-f]{24}", result):
        raise ResearchMaterialContractError("document_id 必须是 doc_ 和 24 位内容哈希")
    return result


@dataclass(frozen=True)
class ResearchMaterialInput:
    """一份已指纹化、可交给解析器处理的资料。"""

    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    parser_version: str
    schema_version: str = RESEARCH_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialInput schema_version 必须为 "
                f"{RESEARCH_MATERIAL_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "relative_path",
            validate_relative_material_path(self.relative_path),
        )
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        object.__setattr__(
            self,
            "size_bytes",
            _bounded_integer(
                self.size_bytes,
                "size_bytes",
                minimum=0,
                maximum=2**40,
            ),
        )
        try:
            object.__setattr__(
                self,
                "sha256",
                sha256_text(self.sha256, "sha256"),
            )
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        object.__setattr__(self, "parser_version", _parser_version(self.parser_version))

    def material(self) -> dict[str, Any]:
        """返回稳定指纹和路径相关字段。"""

        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "parser_version": self.parser_version,
        }

    @property
    def document_id(self) -> str:
        """返回只由文件内容决定的稳定资料 ID。"""

        return stable_id("doc", {"sha256": self.sha256})

    @property
    def material_hash(self) -> str:
        """返回包含来源路径和解析版本的输入哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchMaterialInput:
        """严格解析资料输入。"""

        return cls(
            **_strict_fields(
                value,
                "research_material_input",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典，不复制派生 ID。"""

        return self.material()


@dataclass(frozen=True)
class ResearchMaterialManifest:
    """一次资料解析使用的有序、无重复输入清单。"""

    materials: tuple[ResearchMaterialInput, ...]
    schema_version: str = RESEARCH_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialManifest schema_version 必须为 "
                f"{RESEARCH_MATERIAL_SCHEMA_VERSION}"
            )
        if isinstance(self.materials, (str, bytes)) or not isinstance(
            self.materials, (list, tuple)
        ):
            raise ResearchMaterialContractError("materials 必须是资料输入序列")
        materials = tuple(self.materials)
        if not 1 <= len(materials) <= 1000:
            raise ResearchMaterialContractError("materials 数量必须在 1～1000")
        if any(not isinstance(item, ResearchMaterialInput) for item in materials):
            raise ResearchMaterialContractError(
                "materials 只能包含 ResearchMaterialInput"
            )
        document_ids = [item.document_id for item in materials]
        if len(document_ids) != len(set(document_ids)):
            raise ResearchMaterialContractError("同一资料清单不能包含重复文件内容")
        object.__setattr__(self, "materials", materials)

    def material(self) -> dict[str, Any]:
        """返回与输入顺序无关的清单哈希字段。"""

        items = sorted(
            (item.material() for item in self.materials),
            key=canonical_json,
        )
        return {
            "schema_version": self.schema_version,
            "materials": items,
        }

    @property
    def manifest_hash(self) -> str:
        """返回资料清单哈希。"""

        return sha256_json(self.material())

    @property
    def manifest_id(self) -> str:
        """返回资料清单稳定 ID。"""

        return stable_id("materials", {"manifest_hash": self.manifest_hash})

    @classmethod
    def from_dict(cls, value: Any) -> ResearchMaterialManifest:
        """严格解析资料清单。"""

        payload = _strict_fields(
            value,
            "research_material_manifest",
            set(cls.__dataclass_fields__),
        )
        materials = payload["materials"]
        if isinstance(materials, (str, bytes)) or not isinstance(
            materials, (list, tuple)
        ):
            raise ResearchMaterialContractError("materials 必须是 JSON 数组")
        payload["materials"] = tuple(
            ResearchMaterialInput.from_dict(item) for item in materials
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典，不复制派生 ID。"""

        return {
            "schema_version": self.schema_version,
            "materials": [item.to_dict() for item in self.materials],
        }


@dataclass(frozen=True)
class ResearchMaterialFailure:
    """一次不影响整批其他资料的解析失败。"""

    category: str
    message: str
    retryable: bool
    error_code: str | None = None
    occurred_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialFailure schema_version 必须为 "
                f"{RESEARCH_MATERIAL_SCHEMA_VERSION}"
            )
        category = _choice(self.category, "category", RESEARCH_FAILURE_CATEGORIES)
        expected_code = _FAILURE_CODE_BY_CATEGORY[category]
        if self.error_code is not None and self.error_code != expected_code:
            raise ResearchMaterialContractError(
                f"{category} 必须使用错误码 {expected_code}"
            )
        try:
            message = text(self.message, "message", maximum=1000)
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        if not isinstance(self.retryable, bool):
            raise ResearchMaterialContractError("retryable 必须是布尔值")
        try:
            occurred_at = validate_utc_timestamp(self.occurred_at, "occurred_at")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "error_code", expected_code)
        object.__setattr__(
            self,
            "message",
            redact_research_material_text(message),
        )
        object.__setattr__(self, "occurred_at", occurred_at)

    def material(self) -> dict[str, Any]:
        """返回失败记录哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "error_code": self.error_code,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
        }

    @property
    def failure_hash(self) -> str:
        """返回失败记录稳定哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchMaterialFailure:
        """严格解析失败记录。"""

        return cls(
            **_strict_fields(
                value,
                "research_material_failure",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "occurred_at": self.occurred_at}


@dataclass(frozen=True)
class SourceReference:
    """资料内一条可审计的页码和摘录定位。"""

    document_id: str
    page_number: int
    excerpt_sha256: str
    locator: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"SourceReference schema_version 必须为 "
                f"{RESEARCH_MATERIAL_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        page_number = _bounded_integer(
            self.page_number,
            "page_number",
            minimum=1,
            maximum=100000,
        )
        try:
            excerpt_sha256 = sha256_text(self.excerpt_sha256, "excerpt_sha256")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        locator = self.locator or f"page:{page_number}"
        try:
            locator = redact_research_material_text(
                text(locator, "locator", maximum=240)
            )
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        try:
            created_at = validate_utc_timestamp(self.created_at, "created_at")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "page_number", page_number)
        object.__setattr__(self, "excerpt_sha256", excerpt_sha256)
        object.__setattr__(self, "locator", locator)
        object.__setattr__(self, "created_at", created_at)

    def material(self) -> dict[str, Any]:
        """返回来源引用哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "excerpt_sha256": self.excerpt_sha256,
            "locator": self.locator,
        }

    @property
    def source_reference_id(self) -> str:
        """返回来源引用稳定 ID。"""

        return stable_id("source", self.material())

    @property
    def reference_hash(self) -> str:
        """返回来源引用内容哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> SourceReference:
        """严格解析来源引用。"""

        return cls(
            **_strict_fields(
                value,
                "source_reference",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "created_at": self.created_at}


@dataclass(frozen=True)
class ResearchDocumentRecord:
    """一份资料在指定解析器版本下的成功或失败记录。"""

    document_id: str
    material_hash: str
    parser_version: str
    status: str
    page_count: int | None = None
    text_sha256: str | None = None
    source_references: tuple[SourceReference, ...] = ()
    failure: ResearchMaterialFailure | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchDocumentRecord schema_version 必须为 "
                f"{RESEARCH_MATERIAL_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        try:
            material_hash = sha256_text(self.material_hash, "material_hash")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        parser_version = _parser_version(self.parser_version)
        status = _choice(self.status, "status", RESEARCH_DOCUMENT_STATUSES)
        if isinstance(self.source_references, (str, bytes)) or not isinstance(
            self.source_references, (list, tuple)
        ):
            raise ResearchMaterialContractError("source_references 必须是引用序列")
        references = tuple(self.source_references)
        if len(references) > 10000:
            raise ResearchMaterialContractError("source_references 不得超过 10000 条")
        if any(not isinstance(item, SourceReference) for item in references):
            raise ResearchMaterialContractError(
                "source_references 只能包含 SourceReference"
            )
        if any(item.document_id != document_id for item in references):
            raise ResearchMaterialContractError("来源引用必须属于当前资料")
        if status == "parsed":
            if self.failure is not None:
                raise ResearchMaterialContractError("parsed 记录不能包含 failure")
            page_count = _bounded_integer(
                self.page_count,
                "page_count",
                minimum=1,
                maximum=100000,
            )
            try:
                text_sha256 = sha256_text(self.text_sha256, "text_sha256")
            except MiningContractError as error:
                raise ResearchMaterialContractError(str(error)) from error
        else:
            if not isinstance(self.failure, ResearchMaterialFailure):
                raise ResearchMaterialContractError("failed 记录必须包含 failure")
            if self.page_count is not None or self.text_sha256 is not None:
                raise ResearchMaterialContractError(
                    "failed 记录不能包含成功的页数或文本哈希"
                )
            if references:
                raise ResearchMaterialContractError(
                    "failed 记录不能包含来源引用"
                )
            page_count = None
            text_sha256 = None
        try:
            created_at = validate_utc_timestamp(self.created_at, "created_at")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "material_hash", material_hash)
        object.__setattr__(self, "parser_version", parser_version)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "page_count", page_count)
        object.__setattr__(self, "text_sha256", text_sha256)
        object.__setattr__(self, "source_references", references)
        object.__setattr__(self, "created_at", created_at)

    def material(self) -> dict[str, Any]:
        """返回解析记录哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "material_hash": self.material_hash,
            "parser_version": self.parser_version,
            "status": self.status,
            "page_count": self.page_count,
            "text_sha256": self.text_sha256,
            "source_references": [
                item.material() for item in self.source_references
            ],
            "failure": self.failure.material() if self.failure else None,
        }

    @property
    def record_id(self) -> str:
        """返回解析记录稳定 ID。"""

        return stable_id("documentrecord", {"record_hash": self.record_hash})

    @property
    def record_hash(self) -> str:
        """返回解析记录稳定哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchDocumentRecord:
        """严格解析资料解析记录。"""

        payload = _strict_fields(
            value,
            "research_document_record",
            set(cls.__dataclass_fields__),
        )
        references = payload["source_references"]
        if isinstance(references, (str, bytes)) or not isinstance(
            references, (list, tuple)
        ):
            raise ResearchMaterialContractError(
                "source_references 必须是 JSON 数组"
            )
        payload["source_references"] = tuple(
            SourceReference.from_dict(item) for item in references
        )
        failure = payload["failure"]
        payload["failure"] = (
            None if failure is None else ResearchMaterialFailure.from_dict(failure)
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典，不复制派生 ID。"""

        return {
            **self.material(),
            "source_references": [
                item.to_dict() for item in self.source_references
            ],
            "failure": self.failure.to_dict() if self.failure else None,
            "created_at": self.created_at,
        }

__all__ = [
    "RESEARCH_DOCUMENT_STATUSES",
    "RESEARCH_FAILURE_CATEGORIES",
    "RESEARCH_MATERIAL_SCHEMA_VERSION",
    "ResearchDocumentRecord",
    "ResearchMaterialFailure",
    "ResearchMaterialInput",
    "ResearchMaterialManifest",
    "SourceReference",
]
