"""文本型 PDF 解析、页正文和研报分类契约。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

from mining.contract_types import (
    identifier,
    sha256_json,
    sha256_text,
    stable_id,
    text,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError, ResearchMaterialContractError
from mining.research_material_contracts import (
    ResearchDocumentRecord,
    ResearchMaterialInput,
    ResearchMaterialManifest,
)

RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION = "mining_research_material_parse_v1"
RESEARCH_PAGE_SCHEMA_VERSION = "mining_research_page_v1"
RESEARCH_CLASSIFICATION_SCHEMA_VERSION = "mining_research_classification_v1"
RESEARCH_CLASSIFICATION_RULE_VERSION = "mining_research_report_v1"
RESEARCH_CLASSIFICATION_KINDS = ("financial_research", "not_target")
RESEARCH_REPORT_SIGNALS = (
    "A股研究报告",
    "上市公司研究",
    "公司研究报告",
    "投资策略",
    "投资组合研究",
    "行业研究报告",
    "证券研究报告",
    "金融工程",
    "量化研究",
    "equity research",
    "investment research",
    "investment strategy",
    "quantitative research",
    "research report",
)
RESEARCH_FINANCE_SIGNALS = (
    "A股",
    "alpha",
    "earnings",
    "equity",
    "factor",
    "momentum",
    "portfolio",
    "stock",
    "valuation",
    "volatility",
    "基本面",
    "估值",
    "动量",
    "因子",
    "回测",
    "投资组合",
    "波动率",
    "盈利",
    "股票",
    "行业",
    "证券",
    "超额收益",
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


def _signal_tuple(
    value: Any,
    name: str,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    """校验固定短语信号并保持稳定排序。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ResearchMaterialContractError(f"{name} 必须是字符串序列")
    signals: list[str] = []
    for item in value:
        try:
            signal = text(item, name, maximum=160)
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        if signal not in allowed:
            raise ResearchMaterialContractError(
                f"{name} 包含未知固定信号 {signal!r}"
            )
        signals.append(signal)
    if signals != sorted(set(signals)):
        raise ResearchMaterialContractError(f"{name} 必须去重并按字典序排序")
    return tuple(signals)


def page_text_material(pages: tuple["ResearchPageText", ...]) -> dict[str, Any]:
    """返回页正文稳定哈希使用的结构。"""

    return {
        "schema_version": RESEARCH_PAGE_SCHEMA_VERSION,
        "pages": [page.material() for page in pages],
    }


def research_pages_text_sha256(pages: tuple["ResearchPageText", ...]) -> str:
    """计算完整页结构 SHA-256，保留分页边界。"""

    return sha256_json(page_text_material(pages))


@dataclass(frozen=True)
class ResearchPageText:
    """一页从文本型 PDF 提取的规范化正文。"""

    page_number: int
    text: str
    schema_version: str = RESEARCH_PAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_PAGE_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchPageText schema_version 必须为 "
                f"{RESEARCH_PAGE_SCHEMA_VERSION}"
            )
        page_number = _bounded_integer(
            self.page_number,
            "page_number",
            minimum=1,
            maximum=100000,
        )
        if not isinstance(self.text, str):
            raise ResearchMaterialContractError("page text 必须是字符串")
        if len(self.text) > 2 * 1024 * 1024:
            raise ResearchMaterialContractError("page text 超过 2 MiB")
        object.__setattr__(self, "page_number", page_number)

    @property
    def has_text(self) -> bool:
        """返回该页是否有非空正文。"""

        return bool(self.text.strip())

    @property
    def text_sha256(self) -> str:
        """返回页正文 SHA-256。"""

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def material(self) -> dict[str, Any]:
        """返回页正文稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "page_number": self.page_number,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchPageText":
        """严格解析页正文。"""

        return cls(
            **_strict_fields(
                value,
                "research_page",
                set(cls.__dataclass_fields__),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回页正文字典。"""

        return self.material()


@dataclass(frozen=True)
class ResearchMaterialClassification:
    """金融研报确定性分类结果及命中依据。"""

    document_id: str
    text_sha256: str
    kind: str
    rule_version: str
    report_signals: tuple[str, ...]
    finance_signals: tuple[str, ...]
    created_at: str
    schema_version: str = RESEARCH_CLASSIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_CLASSIFICATION_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialClassification schema_version 必须为 "
                f"{RESEARCH_CLASSIFICATION_SCHEMA_VERSION}"
            )
        document_id = _document_id(self.document_id)
        text_sha256 = _hash(self.text_sha256, "text_sha256")
        if self.kind not in RESEARCH_CLASSIFICATION_KINDS:
            raise ResearchMaterialContractError(
                f"kind 必须是 {', '.join(RESEARCH_CLASSIFICATION_KINDS)} 之一"
            )
        try:
            rule_version = text(self.rule_version, "rule_version", maximum=128)
            created_at = validate_utc_timestamp(self.created_at, "created_at")
        except MiningContractError as error:
            raise ResearchMaterialContractError(str(error)) from error
        if rule_version != RESEARCH_CLASSIFICATION_RULE_VERSION:
            raise ResearchMaterialContractError(
                "rule_version 必须为 "
                f"{RESEARCH_CLASSIFICATION_RULE_VERSION}"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}", rule_version):
            raise ResearchMaterialContractError("rule_version 格式不合法")
        report_signals = _signal_tuple(
            self.report_signals,
            "report_signals",
            RESEARCH_REPORT_SIGNALS,
        )
        finance_signals = _signal_tuple(
            self.finance_signals,
            "finance_signals",
            RESEARCH_FINANCE_SIGNALS,
        )
        expected_kind = (
            "financial_research"
            if report_signals and len(finance_signals) >= 2
            else "not_target"
        )
        if self.kind != expected_kind:
            raise ResearchMaterialContractError(
                "kind 与报告标题信号和金融领域词命中数不一致"
            )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "text_sha256", text_sha256)
        object.__setattr__(self, "rule_version", rule_version)
        object.__setattr__(self, "report_signals", report_signals)
        object.__setattr__(self, "finance_signals", finance_signals)
        object.__setattr__(self, "created_at", created_at)

    @property
    def is_target(self) -> bool:
        """返回资料是否属于目标金融研报。"""

        return self.kind == "financial_research"

    def material(self) -> dict[str, Any]:
        """返回分类稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "text_sha256": self.text_sha256,
            "kind": self.kind,
            "rule_version": self.rule_version,
            "report_signals": list(self.report_signals),
            "finance_signals": list(self.finance_signals),
        }

    @property
    def classification_hash(self) -> str:
        """返回分类结果稳定哈希。"""

        return sha256_json(self.material())

    @property
    def classification_id(self) -> str:
        """返回分类结果稳定 ID。"""

        return stable_id("classification", {"classification_hash": self.classification_hash})

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchMaterialClassification":
        """严格解析分类结果。"""

        payload = _strict_fields(
            value,
            "research_classification",
            set(cls.__dataclass_fields__),
        )
        for field_name in ("report_signals", "finance_signals"):
            signals = payload[field_name]
            if isinstance(signals, (str, bytes)) or not isinstance(
                signals,
                (list, tuple),
            ):
                raise ResearchMaterialContractError(
                    f"{field_name} 必须是字符串数组"
                )
        payload["report_signals"] = tuple(payload["report_signals"])
        payload["finance_signals"] = tuple(payload["finance_signals"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回分类结果字典。"""

        return {**self.material(), "created_at": self.created_at}


@dataclass(frozen=True)
class ResearchMaterialParseLimits:
    """单个 PDF 的确定性资源上限。"""

    max_file_bytes: int = 64 * 1024 * 1024
    max_pages: int = 2000
    max_page_chars: int = 2 * 1024 * 1024
    max_total_text_chars: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_file_bytes",
            _bounded_integer(
                self.max_file_bytes,
                "max_file_bytes",
                minimum=1,
                maximum=2**40,
            ),
        )
        object.__setattr__(
            self,
            "max_pages",
            _bounded_integer(
                self.max_pages,
                "max_pages",
                minimum=1,
                maximum=100000,
            ),
        )
        object.__setattr__(
            self,
            "max_page_chars",
            _bounded_integer(
                self.max_page_chars,
                "max_page_chars",
                minimum=1,
                maximum=2 * 1024 * 1024,
            ),
        )
        object.__setattr__(
            self,
            "max_total_text_chars",
            _bounded_integer(
                self.max_total_text_chars,
                "max_total_text_chars",
                minimum=1,
                maximum=20 * 1024 * 1024,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回资源上限字典。"""

        return {
            "max_file_bytes": self.max_file_bytes,
            "max_pages": self.max_pages,
            "max_page_chars": self.max_page_chars,
            "max_total_text_chars": self.max_total_text_chars,
        }


@dataclass(frozen=True)
class ResearchMaterialParseResult:
    """一份资料的成功解析内容或隔离失败记录。"""

    material: ResearchMaterialInput
    record: ResearchDocumentRecord
    pages: tuple[ResearchPageText, ...] = ()
    classification: ResearchMaterialClassification | None = None
    schema_version: str = RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialParseResult schema_version 必须为 "
                f"{RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION}"
            )
        if not isinstance(self.material, ResearchMaterialInput):
            raise ResearchMaterialContractError(
                "material 必须是 ResearchMaterialInput"
            )
        if not isinstance(self.record, ResearchDocumentRecord):
            raise ResearchMaterialContractError(
                "record 必须是 ResearchDocumentRecord"
            )
        if (
            self.record.document_id != self.material.document_id
            or self.record.material_hash != self.material.material_hash
            or self.record.parser_version != self.material.parser_version
        ):
            raise ResearchMaterialContractError(
                "解析记录与资料输入的 ID、哈希或解析器版本不一致"
            )
        if isinstance(self.pages, (str, bytes)) or not isinstance(
            self.pages,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("pages 必须是页正文序列")
        pages = tuple(self.pages)
        if any(not isinstance(page, ResearchPageText) for page in pages):
            raise ResearchMaterialContractError(
                "pages 只能包含 ResearchPageText"
            )
        if self.record.status == "failed":
            if pages or self.classification is not None:
                raise ResearchMaterialContractError(
                    "failed 解析结果不能包含页正文或分类"
                )
        else:
            if not pages:
                raise ResearchMaterialContractError("parsed 解析结果必须包含页正文")
            if len(pages) != self.record.page_count:
                raise ResearchMaterialContractError(
                    "页正文数量必须等于解析记录 page_count"
                )
            if tuple(page.page_number for page in pages) != tuple(
                range(1, len(pages) + 1)
            ):
                raise ResearchMaterialContractError(
                    "页正文必须从 1 开始连续编号"
                )
            if research_pages_text_sha256(pages) != self.record.text_sha256:
                raise ResearchMaterialContractError(
                    "页正文哈希与解析记录 text_sha256 不一致"
                )
            if not isinstance(
                self.classification,
                ResearchMaterialClassification,
            ):
                raise ResearchMaterialContractError(
                    "parsed 解析结果必须包含分类"
                )
            if (
                self.classification.document_id != self.material.document_id
                or self.classification.text_sha256 != self.record.text_sha256
            ):
                raise ResearchMaterialContractError(
                    "分类结果与解析记录的文档 ID 或正文哈希不一致"
                )
        object.__setattr__(self, "pages", pages)

    @property
    def is_financial_research(self) -> bool:
        """返回成功解析结果是否属于目标金融研报。"""

        return bool(
            self.classification is not None and self.classification.is_target
        )

    @property
    def is_filtered(self) -> bool:
        """返回成功解析但不是目标研报的资料。"""

        return bool(
            self.classification is not None
            and not self.classification.is_target
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchMaterialParseResult":
        """严格解析单份资料结果。"""

        payload = _strict_fields(
            value,
            "research_material_parse_result",
            set(cls.__dataclass_fields__),
        )
        payload["material"] = ResearchMaterialInput.from_dict(payload["material"])
        payload["record"] = ResearchDocumentRecord.from_dict(payload["record"])
        payload["pages"] = tuple(
            ResearchPageText.from_dict(page) for page in payload["pages"]
        )
        classification = payload["classification"]
        payload["classification"] = (
            None
            if classification is None
            else ResearchMaterialClassification.from_dict(classification)
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回单份资料结果字典，包含页正文。"""

        return {
            "schema_version": self.schema_version,
            "material": self.material.to_dict(),
            "record": self.record.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "classification": (
                None if self.classification is None else self.classification.to_dict()
            ),
        }


@dataclass(frozen=True)
class ResearchMaterialBatchResult:
    """一次有序资料批次中所有成功项和隔离失败项。"""

    manifest: ResearchMaterialManifest
    documents: tuple[ResearchMaterialParseResult, ...]
    schema_version: str = RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION:
            raise ResearchMaterialContractError(
                f"ResearchMaterialBatchResult schema_version 必须为 "
                f"{RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION}"
            )
        if not isinstance(self.manifest, ResearchMaterialManifest):
            raise ResearchMaterialContractError(
                "manifest 必须是 ResearchMaterialManifest"
            )
        if isinstance(self.documents, (str, bytes)) or not isinstance(
            self.documents,
            (list, tuple),
        ):
            raise ResearchMaterialContractError("documents 必须是解析结果序列")
        documents = tuple(self.documents)
        if any(
            not isinstance(item, ResearchMaterialParseResult)
            for item in documents
        ):
            raise ResearchMaterialContractError(
                "documents 只能包含 ResearchMaterialParseResult"
            )
        if len(documents) != len(self.manifest.materials):
            raise ResearchMaterialContractError(
                "documents 数量必须等于资料清单数量"
            )
        for material, document in zip(
            self.manifest.materials,
            documents,
            strict=True,
        ):
            if material.material_hash != document.material.material_hash:
                raise ResearchMaterialContractError(
                    "批次结果顺序与资料清单不一致"
                )
        object.__setattr__(self, "documents", documents)

    @property
    def parsed(self) -> tuple[ResearchMaterialParseResult, ...]:
        """返回成功解析的资料。"""

        return tuple(
            item for item in self.documents if item.record.status == "parsed"
        )

    @property
    def financial_research(self) -> tuple[ResearchMaterialParseResult, ...]:
        """返回成功解析的目标金融研报。"""

        return tuple(item for item in self.parsed if item.is_financial_research)

    @property
    def filtered(self) -> tuple[ResearchMaterialParseResult, ...]:
        """返回成功解析但被分类过滤的资料。"""

        return tuple(item for item in self.parsed if item.is_filtered)

    @property
    def failures(self) -> tuple[ResearchMaterialParseResult, ...]:
        """返回逐文件隔离的失败资料。"""

        return tuple(
            item for item in self.documents if item.record.status == "failed"
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchMaterialBatchResult":
        """严格解析资料批次结果。"""

        payload = _strict_fields(
            value,
            "research_material_batch_result",
            set(cls.__dataclass_fields__),
        )
        payload["manifest"] = ResearchMaterialManifest.from_dict(
            payload["manifest"]
        )
        payload["documents"] = tuple(
            ResearchMaterialParseResult.from_dict(item)
            for item in payload["documents"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回批次结果字典。"""

        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "documents": [item.to_dict() for item in self.documents],
        }


__all__ = [
    "RESEARCH_CLASSIFICATION_KINDS",
    "RESEARCH_CLASSIFICATION_RULE_VERSION",
    "RESEARCH_CLASSIFICATION_SCHEMA_VERSION",
    "RESEARCH_FINANCE_SIGNALS",
    "RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION",
    "RESEARCH_PAGE_SCHEMA_VERSION",
    "RESEARCH_REPORT_SIGNALS",
    "ResearchMaterialBatchResult",
    "ResearchMaterialClassification",
    "ResearchMaterialParseLimits",
    "ResearchMaterialParseResult",
    "ResearchPageText",
    "page_text_material",
    "research_pages_text_sha256",
]
