"""研报页窗口编排、Worker 调用和来源引用二次校验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Protocol

from mining.contract_types import text
from mining.errors import (
    MiningContractError,
    ResearchExtractionContractError,
    ResearchExtractionEvidenceError,
    ResearchExtractionResourceError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.research_extraction_contracts import (
    RESEARCH_EXTRACTION_PROMPT_VERSION,
    RESEARCH_EXTRACTION_RULE_VERSION,
    RESEARCH_FACTOR_FIELDS,
    ResearchExtractedFactor,
    ResearchExtractionCitation,
    ResearchExtractionResult,
    ResearchFieldClaim,
)
from mining.research_material_parse_contracts import (
    ResearchMaterialParseResult,
    ResearchPageText,
)
from mining.research_material_safety import (
    redact_research_material_text,
    safe_research_material_log_fields,
)
from mining.worker_client import HttpWorkerClient
from mining_worker.protocol_research import (
    WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION,
    WorkerResearchExtractionRequest,
    WorkerResearchExtractionResponse,
    WorkerResearchFactorDraft,
    WorkerResearchPageSegment,
)
from mining_worker.worker_config import load_worker_config

_LOGGER = get_mining_logger("mining.research_extraction")


def _bounded_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """校验抽取资源上限。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchExtractionContractError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise ResearchExtractionContractError(
            f"{name} 必须在 {minimum}～{maximum}"
        )
    return value


@dataclass(frozen=True)
class ResearchExtractionLimits:
    """单份研报的固定抽取窗口和输出上限。"""

    max_window_chars: int = 24000
    overlap_chars: int = 512
    max_windows: int = 8
    max_factors_per_window: int = 8
    max_quote_chars: int = 2000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_window_chars",
            _bounded_integer(
                self.max_window_chars,
                "max_window_chars",
                minimum=513,
                maximum=30000,
            ),
        )
        object.__setattr__(
            self,
            "overlap_chars",
            _bounded_integer(
                self.overlap_chars,
                "overlap_chars",
                minimum=0,
                maximum=2000,
            ),
        )
        if self.overlap_chars >= self.max_window_chars:
            raise ResearchExtractionContractError(
                "overlap_chars 必须小于 max_window_chars"
            )
        object.__setattr__(
            self,
            "max_windows",
            _bounded_integer(
                self.max_windows,
                "max_windows",
                minimum=1,
                maximum=8,
            ),
        )
        object.__setattr__(
            self,
            "max_factors_per_window",
            _bounded_integer(
                self.max_factors_per_window,
                "max_factors_per_window",
                minimum=1,
                maximum=8,
            ),
        )
        object.__setattr__(
            self,
            "max_quote_chars",
            _bounded_integer(
                self.max_quote_chars,
                "max_quote_chars",
                minimum=1,
                maximum=2000,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        """返回资源上限字典。"""

        return {
            "max_window_chars": self.max_window_chars,
            "overlap_chars": self.overlap_chars,
            "max_windows": self.max_windows,
            "max_factors_per_window": self.max_factors_per_window,
            "max_quote_chars": self.max_quote_chars,
        }


class ResearchFactorExtractor(Protocol):
    """研报抽取编排依赖的 Worker 窗口接口。"""

    def generate(
        self,
        request: WorkerResearchExtractionRequest,
    ) -> WorkerResearchExtractionResponse:
        """提交并等待一个窗口抽取完成。"""


class HttpResearchFactorExtractor(HttpWorkerClient):
    """通过 Worker v5 作业接口执行研报字段抽取。"""

    def submit(self, request: WorkerResearchExtractionRequest):
        """提交一个研报抽取窗口。"""

        from mining.errors import WorkerResponseError
        from mining_worker.protocol import WorkerJobView, WorkerProtocolError

        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v5/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v5 作业响应非法：{exc}") from exc

    def get_job(self, job_id: str):
        """读取 Worker 作业状态。"""

        from mining.errors import WorkerResponseError
        from mining_worker.protocol import WorkerJobView, WorkerProtocolError

        try:
            return WorkerJobView.from_dict(
                self._request("GET", f"/v1/jobs/{job_id}")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v5 作业响应非法：{exc}") from exc

    def cancel(self, job_id: str):
        """取消 Worker v5 作业。"""

        from mining.errors import WorkerResponseError
        from mining_worker.protocol import WorkerJobView, WorkerProtocolError

        try:
            return WorkerJobView.from_dict(
                self._request("POST", f"/v1/jobs/{job_id}/cancel")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v5 作业响应非法：{exc}") from exc

    def wait(self, job_id: str) -> WorkerResearchExtractionResponse:
        """轮询到研报抽取作业终态并解析严格响应。"""

        from mining.errors import (
            MiningWorkerError,
            WorkerCancelledError,
            WorkerResponseError,
            WorkerTimeoutError,
        )
        from mining.worker_client import _ERROR_TYPES
        from mining_worker.protocol import WorkerProtocolError

        deadline = time.monotonic() + self.request_timeout
        while True:
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("Worker v5 成功状态缺少结果")
                try:
                    response = WorkerResearchExtractionResponse.from_dict(
                        job.result
                    )
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(
                        f"Worker v5 研报抽取响应非法：{exc}"
                    ) from exc
                if response.request_id != job.request_id:
                    raise WorkerResponseError(
                        "Worker v5 响应 request_id 与作业请求不一致"
                    )
                return response
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"Worker v5 作业状态为 {job.status}"
                )
            if job.status == "failed":
                code = job.error_code or "MINING-103"
                error_type = _ERROR_TYPES.get(code, MiningWorkerError)
                raise error_type(
                    (job.error_message or "Worker v5 研报抽取失败")[:500]
                )
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待 Worker v5 研报抽取超时")
            time.sleep(self.poll_interval)

    def generate(
        self,
        request: WorkerResearchExtractionRequest,
    ) -> WorkerResearchExtractionResponse:
        """提交并等待一个窗口抽取。"""

        job = self.submit(request)
        return self.wait(job.job_id)


@dataclass(frozen=True)
class _PageWindow:
    """一次 Worker 调用可见的有序页片段。"""

    segments: tuple[WorkerResearchPageSegment, ...]
    page_numbers: tuple[int, ...]


def _split_page_segments(
    pages: tuple[ResearchPageText, ...],
    limits: ResearchExtractionLimits,
) -> tuple[WorkerResearchPageSegment, ...]:
    """把长页拆成带重叠的字符片段。"""

    segments: list[WorkerResearchPageSegment] = []
    for page in pages:
        if not page.has_text:
            continue
        start = 0
        while start < len(page.text):
            end = min(start + limits.max_window_chars, len(page.text))
            if end < len(page.text):
                minimum_break = start + limits.max_window_chars // 2
                newline = page.text.rfind("\n", minimum_break, end)
                if newline >= minimum_break:
                    end = newline + 1
            segment_text = page.text[start:end]
            segments.append(
                WorkerResearchPageSegment(
                    page_number=page.page_number,
                    start_offset=start,
                    text=segment_text,
                )
            )
            if end >= len(page.text):
                break
            next_start = max(end - limits.overlap_chars, start + 1)
            if next_start <= start:
                raise ResearchExtractionResourceError("研报页窗口切分未前进")
            start = next_start
    if not segments:
        raise ResearchExtractionContractError("研报没有可抽取的非空页正文")
    return tuple(segments)


def _build_windows(
    pages: tuple[ResearchPageText, ...],
    limits: ResearchExtractionLimits,
) -> tuple[_PageWindow, ...]:
    """把页片段打包成固定大小的 Worker 窗口。"""

    segments = _split_page_segments(pages, limits)
    windows: list[_PageWindow] = []
    current: list[WorkerResearchPageSegment] = []
    current_chars = 0
    for segment in segments:
        segment_chars = len(segment.text)
        if current and current_chars + segment_chars > limits.max_window_chars:
            windows.append(
                _PageWindow(
                    segments=tuple(current),
                    page_numbers=tuple(
                        sorted({item.page_number for item in current})
                    ),
                )
            )
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += segment_chars
    if current:
        windows.append(
            _PageWindow(
                segments=tuple(current),
                page_numbers=tuple(sorted({item.page_number for item in current})),
            )
        )
    if len(windows) > limits.max_windows:
        raise ResearchExtractionResourceError(
            f"研报需要 {len(windows)} 个窗口，超过上限 {limits.max_windows}"
        )
    return tuple(windows)


def _request_id(
    document_id: str,
    text_sha256: str,
    window_index: int,
    window_count: int,
) -> str:
    """生成与当前正文和窗口位置绑定的稳定请求 ID。"""

    digest = hashlib.sha256(
        json.dumps(
            {
                "document_id": document_id,
                "text_sha256": text_sha256,
                "extraction_version": RESEARCH_EXTRACTION_RULE_VERSION,
                "prompt_version": RESEARCH_EXTRACTION_PROMPT_VERSION,
                "window_index": window_index,
                "window_count": window_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"research_window_{digest[:24]}"


def _validated_citation(
    document_id: str,
    page: ResearchPageText,
    page_number: int,
    quote: str,
    *,
    max_quote_chars: int,
) -> ResearchExtractionCitation:
    """在完整页正文中重新定位模型引用。"""

    if page.page_number != page_number:
        raise ResearchExtractionEvidenceError("模型引用页码与当前页不一致")
    if len(quote) > max_quote_chars:
        raise ResearchExtractionEvidenceError(
            f"模型引用超过 {max_quote_chars} 个字符"
        )
    start = page.text.find(quote)
    if start < 0:
        raise ResearchExtractionEvidenceError(
            f"第 {page_number} 页找不到模型返回的逐字原文"
        )
    return ResearchExtractionCitation(
        document_id=document_id,
        page_number=page_number,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
    )


def _validated_factor(
    document: ResearchMaterialParseResult,
    draft: WorkerResearchFactorDraft,
    page_numbers: tuple[int, ...],
    *,
    max_quote_chars: int,
) -> ResearchExtractedFactor:
    """把模型草稿转换为主服务权威契约。"""

    pages = {
        item.page_number: item
        for item in document.pages
        if item.page_number in page_numbers
    }
    fields: list[ResearchFieldClaim] = []
    for draft_field in draft.fields:
        citations: list[ResearchExtractionCitation] = []
        for citation in draft_field.citations:
            page = pages.get(citation.page_number)
            if page is None:
                raise ResearchExtractionEvidenceError(
                    f"模型引用了当前窗口不可见的第 {citation.page_number} 页"
                )
            citations.append(
                _validated_citation(
                    document.material.document_id,
                    page,
                    citation.page_number,
                    citation.quote,
                    max_quote_chars=max_quote_chars,
                )
            )
        unique_citations = tuple(
            {
                item.citation_hash: item for item in citations
            }.values()
        )
        fields.append(
            ResearchFieldClaim(
                field_name=draft_field.field_name,
                values=tuple(sorted(set(draft_field.values))),
                status=draft_field.status,
                citations=unique_citations,
            )
        )
    if tuple(item.field_name for item in fields) != RESEARCH_FACTOR_FIELDS:
        raise ResearchExtractionEvidenceError(
            "模型返回的因子字段顺序或字段集合不合法"
        )
    return ResearchExtractedFactor(
        document_id=document.material.document_id,
        text_sha256=document.record.text_sha256,
        fields=tuple(fields),
    )


def extract_research_factor_drafts(
    document: ResearchMaterialParseResult,
    *,
    extractor: ResearchFactorExtractor,
    limits: ResearchExtractionLimits | None = None,
    llm_model: str | None = None,
) -> ResearchExtractionResult:
    """从一份已解析金融研报中抽取因子字段草稿。

    参数:
        document: P2-02 成功解析的金融研报。
        extractor: 实现 Worker v5 窗口调用的同步抽取器。
        limits: 窗口、因子数、引用长度上限。
        llm_model: 冻结模型名；省略时读取 Worker 默认模型。
    """

    if not isinstance(document, ResearchMaterialParseResult):
        raise ResearchExtractionContractError(
            "document 必须是 ResearchMaterialParseResult"
        )
    if document.record.status != "parsed" or not document.is_financial_research:
        raise ResearchExtractionContractError(
            "只有成功解析的金融研报可以执行字段抽取"
        )
    if document.record.text_sha256 is None:
        raise ResearchExtractionContractError("金融研报缺少正文哈希")
    resolved_limits = limits or ResearchExtractionLimits()
    if not isinstance(resolved_limits, ResearchExtractionLimits):
        raise ResearchExtractionContractError(
            "limits 必须是 ResearchExtractionLimits"
        )
    try:
        model = text(
            llm_model or load_worker_config().default_model,
            "llm_model",
            maximum=200,
        )
    except MiningContractError as error:
        raise ResearchExtractionContractError(str(error)) from error
    if WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION != RESEARCH_EXTRACTION_PROMPT_VERSION:
        raise ResearchExtractionContractError("Worker 与主服务提示版本不一致")

    raw_text = "\n".join(page.text for page in document.pages)
    if redact_research_material_text(raw_text) != raw_text:
        raise ResearchExtractionEvidenceError(
            "研报正文包含疑似密钥或主机绝对路径，未发送给 Worker"
        )
    windows = _build_windows(document.pages, resolved_limits)
    started = time.perf_counter()
    factors_by_hash: dict[str, ResearchExtractedFactor] = {}
    job_ids: list[str] = []
    for index, window in enumerate(windows):
        request = WorkerResearchExtractionRequest(
            request_id=_request_id(
                document.material.document_id,
                document.record.text_sha256,
                index,
                len(windows),
            ),
            document_id=document.material.document_id,
            text_sha256=document.record.text_sha256,
            window_index=index,
            window_count=len(windows),
            pages=window.segments,
            llm_model=model,
            max_factors=resolved_limits.max_factors_per_window,
        )
        response = extractor.generate(request)
        if response.request_id != request.request_id:
            raise ResearchExtractionEvidenceError(
                "Worker 响应 request_id 与抽取请求不一致"
            )
        if len(response.factors) > resolved_limits.max_factors_per_window:
            raise ResearchExtractionEvidenceError(
                "Worker 返回因子数超过当前窗口上限"
            )
        job_ids.append(request.job_id)
        for draft in response.factors:
            factor = _validated_factor(
                document,
                draft,
                window.page_numbers,
                max_quote_chars=resolved_limits.max_quote_chars,
            )
            factors_by_hash.setdefault(factor.factor_hash, factor)
    factors = tuple(
        sorted(factors_by_hash.values(), key=lambda item: item.factor_hash)
    )
    result = ResearchExtractionResult(
        document_id=document.material.document_id,
        text_sha256=document.record.text_sha256,
        model=model,
        prompt_version=RESEARCH_EXTRACTION_PROMPT_VERSION,
        extraction_version=RESEARCH_EXTRACTION_RULE_VERSION,
        window_count=len(windows),
        factors=factors,
        worker_job_ids=tuple(job_ids),
    )
    missing_fields = sum(
        item.status == "missing"
        for factor in factors
        for item in factor.fields
    )
    _LOGGER.info(
        format_event(
            safe_research_material_log_fields(
                {
                    "operation": "extract_research_factor_drafts",
                    "document_id": document.material.document_id,
                    "record_hash": document.record.record_hash,
                    "extraction_id": result.extraction_id,
                    "prompt_version": result.prompt_version,
                    "model": result.model,
                    "status": result.status,
                    "window_count": result.window_count,
                    "factor_count": len(result.factors),
                    "missing_field_count": missing_fields,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000,
                        1,
                    ),
                }
            )
        )
    )
    return result


__all__ = [
    "HttpResearchFactorExtractor",
    "ResearchExtractionLimits",
    "ResearchFactorExtractor",
    "extract_research_factor_drafts",
]
