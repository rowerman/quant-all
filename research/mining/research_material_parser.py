"""文本型 PDF 的逐文件解析、分类和批次失败隔离。"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path
import time
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PyPdfError

from mining.contract_types import utc_now_iso
from mining.errors import (
    ResearchMaterialContractError,
    ResearchMaterialParseError,
    ResearchMaterialReadError,
    ResearchMaterialResourceError,
    ResearchMaterialSecurityError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.research_material_classification import (
    classify_research_material,
)
from mining.research_material_contracts import (
    ResearchDocumentRecord,
    ResearchMaterialFailure,
    ResearchMaterialInput,
    SourceReference,
)
from mining.research_material_io import (
    build_research_material_manifest,
    discover_research_material_paths,
    resolve_research_file,
)
from mining.research_material_parse_contracts import (
    ResearchMaterialBatchResult,
    ResearchMaterialParseLimits,
    ResearchMaterialParseResult,
    ResearchPageText,
    research_pages_text_sha256,
)
from mining.research_material_safety import safe_research_material_log_fields

RESEARCH_PDF_PARSER_VERSION = "pypdf-6.21.2.page-v1"
_READ_CHUNK_BYTES = 1024 * 1024
_LOGGER = get_mining_logger("mining.research_materials")


class _DocumentFailure(Exception):
    """携带稳定失败类别与可重试标记的单文件解析异常。"""

    def __init__(self, category: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


def _failure_result(
    material: ResearchMaterialInput,
    *,
    category: str,
    message: str,
    retryable: bool,
) -> ResearchMaterialParseResult:
    """构造不影响同批其他文件的失败结果。"""

    failure = ResearchMaterialFailure(
        category=category,
        message=message,
        retryable=retryable,
    )
    record = ResearchDocumentRecord(
        document_id=material.document_id,
        material_hash=material.material_hash,
        parser_version=material.parser_version,
        status="failed",
        failure=failure,
    )
    return ResearchMaterialParseResult(material=material, record=record)


def _read_and_verify_stream(
    stream: BinaryIO,
    material: ResearchMaterialInput,
    limits: ResearchMaterialParseLimits,
) -> None:
    """流式重算文件指纹并确认输入未变化。"""

    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_READ_CHUNK_BYTES):
        size += len(chunk)
        if size > limits.max_file_bytes:
            raise ResearchMaterialResourceError("PDF 文件超过解析大小上限")
        digest.update(chunk)
    stream.seek(0)
    if size != material.size_bytes or digest.hexdigest() != material.sha256:
        raise _DocumentFailure(
            "input_changed",
            "资料文件在解析前发生变化",
            retryable=True,
        )


def _metadata_title(reader: PdfReader) -> str | None:
    """尽力读取 PDF 元数据标题，不因异常元数据阻断正文解析。"""

    try:
        value = reader.metadata.title
    except Exception:
        return None
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _extract_pages(
    reader: PdfReader,
    limits: ResearchMaterialParseLimits,
) -> tuple[tuple[ResearchPageText, ...], str | None]:
    """提取全部页面正文并应用页数和字符上限。"""

    if reader.is_encrypted:
        raise _DocumentFailure(
            "encrypted_document",
            "PDF 已加密，未尝试破解密码",
            retryable=False,
        )
    page_count = len(reader.pages)
    if page_count > limits.max_pages:
        raise ResearchMaterialResourceError("PDF 页数超过解析上限")
    pages: list[ResearchPageText] = []
    total_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        if not isinstance(extracted, str):
            raise _DocumentFailure(
                "corrupt_document",
                "PDF 页面正文不是文本",
                retryable=False,
            )
        normalized = (
            extracted.replace("\r\n", "\n").replace("\r", "\n").strip()
        )
        if len(normalized) > limits.max_page_chars:
            raise ResearchMaterialResourceError("PDF 单页正文超过解析上限")
        total_chars += len(normalized)
        if total_chars > limits.max_total_text_chars:
            raise ResearchMaterialResourceError("PDF 总正文超过解析上限")
        pages.append(
            ResearchPageText(
                page_number=page_number,
                text=normalized,
            )
        )
    if not any(page.has_text for page in pages):
        raise _DocumentFailure(
            "no_extractable_text",
            "PDF 没有可提取正文，扫描件 OCR 未启用",
            retryable=False,
        )
    return tuple(pages), _metadata_title(reader)


def _success_result(
    material: ResearchMaterialInput,
    pages: tuple[ResearchPageText, ...],
    metadata_title: str | None,
) -> ResearchMaterialParseResult:
    """构造成功解析记录、分类和页级来源引用。"""

    text_sha256 = research_pages_text_sha256(pages)
    classification = classify_research_material(
        document_id=material.document_id,
        text_sha256=text_sha256,
        text="\n".join(page.text for page in pages[:2]),
        metadata_title=metadata_title,
    )
    references = tuple(
        SourceReference(
            document_id=material.document_id,
            page_number=page.page_number,
            excerpt_sha256=page.text_sha256,
        )
        for page in pages
        if page.has_text
    )
    record = ResearchDocumentRecord(
        document_id=material.document_id,
        material_hash=material.material_hash,
        parser_version=material.parser_version,
        status="parsed",
        page_count=len(pages),
        text_sha256=text_sha256,
        source_references=references,
    )
    return ResearchMaterialParseResult(
        material=material,
        record=record,
        pages=pages,
        classification=classification,
    )


def _parse_research_material_checked(
    root: Path,
    material: ResearchMaterialInput,
    limits: ResearchMaterialParseLimits,
) -> ResearchMaterialParseResult:
    """执行单文件读取；预期失败通过项目异常表达。"""

    if Path(material.relative_path).suffix.casefold() != ".pdf":
        return _failure_result(
            material,
            category="unsupported_format",
            message="只支持扩展名为 .pdf 的文本型资料",
            retryable=False,
        )
    path = resolve_research_file(root, material.relative_path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ResearchMaterialReadError("PDF 文件不可读") from error
    if size > limits.max_file_bytes:
        raise ResearchMaterialResourceError("PDF 文件超过解析大小上限")
    try:
        with path.open("rb") as stream:
            _read_and_verify_stream(stream, material, limits)
            reader = PdfReader(stream, strict=False)
            pages, metadata_title = _extract_pages(reader, limits)
    except OSError as error:
        raise ResearchMaterialReadError("PDF 文件不可读") from error
    return _success_result(material, pages, metadata_title)


def _log_document_result(
    result: ResearchMaterialParseResult,
    *,
    elapsed_ms: float,
) -> None:
    """记录不含正文、相对路径和原始异常的单文件结果。"""

    failure = result.record.failure
    classification = result.classification
    fields = safe_research_material_log_fields(
        {
            "operation": "parse_research_material",
            "document_id": result.material.document_id,
            "parser_version": result.material.parser_version,
            "status": result.record.status,
            "classification": (
                None if classification is None else classification.kind
            ),
            "error_code": None if failure is None else failure.error_code,
            "category": None if failure is None else failure.category,
            "retryable": None if failure is None else failure.retryable,
            "page_count": result.record.page_count,
            "size_bytes": result.material.size_bytes,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    )
    _LOGGER.info(format_event(fields))


def parse_research_material(
    root: Path,
    material: ResearchMaterialInput,
    *,
    limits: ResearchMaterialParseLimits | None = None,
) -> ResearchMaterialParseResult:
    """解析一份文本型 PDF，并把文件级错误转换为隔离记录。

    参数:
        root: 资料根目录。
        material: P2-01 指纹化资料输入。
        limits: 文件、页数和正文字符上限。
    """

    if not isinstance(material, ResearchMaterialInput):
        raise ResearchMaterialContractError(
            "material 必须是 ResearchMaterialInput"
        )
    if material.parser_version != RESEARCH_PDF_PARSER_VERSION:
        raise ResearchMaterialContractError(
            f"当前解析器版本必须为 {RESEARCH_PDF_PARSER_VERSION}"
        )
    resolved_limits = limits or ResearchMaterialParseLimits()
    if not isinstance(resolved_limits, ResearchMaterialParseLimits):
        raise ResearchMaterialContractError(
            "limits 必须是 ResearchMaterialParseLimits"
        )
    started = time.perf_counter()
    try:
        result = _parse_research_material_checked(
            root,
            material,
            resolved_limits,
        )
    except _DocumentFailure as error:
        result = _failure_result(
            material,
            category=error.category,
            message=str(error),
            retryable=error.retryable,
        )
    except ResearchMaterialSecurityError as error:
        result = _failure_result(
            material,
            category="unsafe_input",
            message=str(error),
            retryable=False,
        )
    except ResearchMaterialResourceError as error:
        result = _failure_result(
            material,
            category="resource_limit",
            message=str(error),
            retryable=False,
        )
    except ResearchMaterialReadError as error:
        result = _failure_result(
            material,
            category="unreadable",
            message=str(error),
            retryable=True,
        )
    except ResearchMaterialParseError as error:
        result = _failure_result(
            material,
            category="corrupt_document",
            message=str(error),
            retryable=False,
        )
    except (FileNotDecryptedError, PyPdfError) as error:
        result = _failure_result(
            material,
            category="corrupt_document",
            message="PDF 已损坏或无法解析",
            retryable=False,
        )
        _LOGGER.info(
            format_event(
                safe_research_material_log_fields(
                    {
                        "operation": "parse_research_material",
                        "document_id": material.document_id,
                        "status": "failed",
                        "category": "corrupt_document",
                        "failure_type": type(error).__name__,
                    }
                )
            )
        )
    except Exception as error:
        result = _failure_result(
            material,
            category="parser_error",
            message="PDF 解析器发生未预期错误",
            retryable=True,
        )
        _LOGGER.info(
            format_event(
                safe_research_material_log_fields(
                    {
                        "operation": "parse_research_material",
                        "document_id": material.document_id,
                        "status": "failed",
                        "category": "parser_error",
                        "failure_type": type(error).__name__,
                    }
                )
            )
        )
    _log_document_result(
        result,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    return result


def parse_research_materials(
    root: Path,
    relative_paths: Iterable[str],
    *,
    limits: ResearchMaterialParseLimits | None = None,
) -> ResearchMaterialBatchResult:
    """解析有序资料批次，逐文件隔离解析失败。

    参数:
        root: 资料根目录。
        relative_paths: 根目录内相对 POSIX 路径序列。
        limits: 文件、页数和正文字符上限。
    """

    if isinstance(relative_paths, (str, bytes)):
        raise ResearchMaterialContractError("relative_paths 必须是路径序列")
    paths = tuple(relative_paths)
    manifest = build_research_material_manifest(
        root,
        paths,
        parser_version=RESEARCH_PDF_PARSER_VERSION,
    )
    documents = tuple(
        parse_research_material(
            root,
            material,
            limits=limits,
        )
        for material in manifest.materials
    )
    result = ResearchMaterialBatchResult(
        manifest=manifest,
        documents=documents,
    )
    _LOGGER.info(
        format_event(
            safe_research_material_log_fields(
                {
                    "operation": "parse_research_materials",
                    "manifest_id": manifest.manifest_id,
                    "document_count": len(result.documents),
                    "financial_research_count": len(result.financial_research),
                    "filtered_count": len(result.filtered),
                    "failed_count": len(result.failures),
                }
            )
        )
    )
    return result


def parse_research_material_target(
    root: Path,
    target: str,
    *,
    recursive: bool = True,
    limits: ResearchMaterialParseLimits | None = None,
) -> ResearchMaterialBatchResult:
    """发现并解析单文件或目录资料。

    参数:
        root: 资料根目录。
        target: 根目录内的相对文件或目录。
        recursive: 目录是否递归扫描。
        limits: 文件、页数和正文字符上限。
    """

    paths = discover_research_material_paths(
        root,
        target,
        recursive=recursive,
    )
    return parse_research_materials(root, paths, limits=limits)


__all__ = [
    "RESEARCH_PDF_PARSER_VERSION",
    "parse_research_material",
    "parse_research_material_target",
    "parse_research_materials",
]
