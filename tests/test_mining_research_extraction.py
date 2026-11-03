"""研报字段抽取契约、窗口编排和引用校验测试。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import logging

import pytest

from mining import (
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    RESEARCH_PDF_PARSER_VERSION,
    ResearchExtractedFactor,
    ResearchExtractionContractError,
    ResearchExtractionCitation,
    ResearchExtractionEvidenceError,
    ResearchExtractionLimits,
    ResearchExtractionResourceError,
    ResearchExtractionResult,
    ResearchFieldClaim,
    ResearchMaterialClassification,
    ResearchMaterialContractError,
    ResearchMaterialInput,
    ResearchMaterialParseResult,
    ResearchPageText,
    ResearchDocumentRecord,
    ResearchMaterialManifest,
    extract_research_factor_drafts,
    research_pages_text_sha256,
)
from mining_worker.protocol_research import (
    WorkerResearchCitationDraft,
    WorkerResearchExtractionRequest,
    WorkerResearchExtractionResponse,
    WorkerResearchFactorDraft,
    WorkerResearchFieldDraft,
)

FIELDS = (
    "factor_name",
    "economic_mechanism",
    "formula",
    "variables",
    "expected_direction",
    "falsification_conditions",
)


def _document(
    *,
    text: str = (
        "证券研究报告\n"
        "短期反转因子：过去 5 日收益较低的股票未来收益可能更高。\n"
        "公式为 rank(-ret_5)，预期正向。若 RankIC 低于 0.02 则证伪。"
    ),
    kind: str = "financial_research",
) -> ResearchMaterialParseResult:
    sha = hashlib.sha256(b"report-" + text.encode("utf-8")).hexdigest()
    material = ResearchMaterialInput(
        relative_path="reports/example.pdf",
        media_type="application/pdf",
        size_bytes=len(text.encode("utf-8")),
        sha256=sha,
        parser_version=RESEARCH_PDF_PARSER_VERSION,
    )
    pages = (ResearchPageText(page_number=1, text=text),)
    text_sha256 = research_pages_text_sha256(pages)
    classification = ResearchMaterialClassification(
        document_id=material.document_id,
        text_sha256=text_sha256,
        kind=kind,
        rule_version=RESEARCH_CLASSIFICATION_RULE_VERSION,
        report_signals=(
            ("证券研究报告",) if kind == "financial_research" else ()
        ),
        finance_signals=(
            ("因子", "股票") if kind == "financial_research" else ()
        ),
        created_at="2026-11-01T00:00:00+00:00",
    )
    record = ResearchDocumentRecord(
        document_id=material.document_id,
        material_hash=material.material_hash,
        parser_version=material.parser_version,
        status="parsed",
        page_count=1,
        text_sha256=text_sha256,
    )
    return ResearchMaterialParseResult(
        material=material,
        record=record,
        pages=pages,
        classification=classification,
    )


def _citation(document, quote: str, page_number: int = 1):
    page = next(item for item in document.pages if item.page_number == page_number)
    start = page.text.find(quote)
    assert start >= 0
    return ResearchExtractionCitation(
        document_id=document.material.document_id,
        page_number=page_number,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
    )


def _claim(
    document,
    field_name: str,
    values: tuple[str, ...],
    *,
    status: str,
    quote: str | None = None,
    page_number: int = 1,
) -> ResearchFieldClaim:
    citations = (
        ()
        if status == "missing"
        else (_citation(document, quote or document.pages[0].text[:20], page_number),)
    )
    return ResearchFieldClaim(
        field_name=field_name,
        values=tuple(sorted(values)),
        status=status,
        citations=citations,
    )


def _factor(document) -> ResearchExtractedFactor:
    return ResearchExtractedFactor(
        document_id=document.material.document_id,
        text_sha256=document.record.text_sha256,
        fields=(
            _claim(
                document,
                "factor_name",
                ("短期反转因子",),
                status="explicit",
                quote="短期反转因子",
            ),
            _claim(
                document,
                "economic_mechanism",
                ("低收益股票后续修复",),
                status="inferred",
                quote="过去 5 日收益较低的股票未来收益可能更高",
            ),
            _claim(
                document,
                "formula",
                ("rank(-ret_5)",),
                status="explicit",
                quote="rank(-ret_5)",
            ),
            _claim(
                document,
                "variables",
                ("ret_5",),
                status="explicit",
                quote="ret_5",
            ),
            _claim(
                document,
                "expected_direction",
                ("positive",),
                status="explicit",
                quote="预期正向",
            ),
            _claim(
                document,
                "falsification_conditions",
                ("RankIC 低于 0.02",),
                status="inferred",
                quote="若 RankIC 低于 0.02 则证伪",
            ),
        ),
    )


class FakeExtractor:
    def __init__(self, *, factors: bool = True, quote: str | None = None) -> None:
        self.factors = factors
        self.quote = quote
        self.requests: list[WorkerResearchExtractionRequest] = []

    def generate(self, request):
        self.requests.append(request)
        if not self.factors:
            return WorkerResearchExtractionResponse(
                request_id=request.request_id,
                factors=(),
                usage={"calls": 1},
            )
        citations = lambda page, text: (
            WorkerResearchCitationDraft(page_number=page, quote=self.quote or text),
        )
        fields = (
            WorkerResearchFieldDraft(
                field_name="factor_name",
                values=("短期反转因子",),
                status="explicit",
                citations=citations(1, "短期反转因子"),
            ),
            WorkerResearchFieldDraft(
                field_name="economic_mechanism",
                values=("低收益股票后续修复",),
                status="inferred",
                citations=citations(1, "过去 5 日收益较低的股票未来收益可能更高"),
            ),
            WorkerResearchFieldDraft(
                field_name="formula",
                values=("rank(-ret_5)",),
                status="explicit",
                citations=citations(1, "rank(-ret_5)"),
            ),
            WorkerResearchFieldDraft(
                field_name="variables",
                values=("ret_5",),
                status="explicit",
                citations=citations(1, "ret_5"),
            ),
            WorkerResearchFieldDraft(
                field_name="expected_direction",
                values=("positive",),
                status="explicit",
                citations=citations(1, "预期正向"),
            ),
            WorkerResearchFieldDraft(
                field_name="falsification_conditions",
                values=("RankIC 低于 0.02",),
                status="inferred",
                citations=citations(1, "若 RankIC 低于 0.02 则证伪"),
            ),
        )
        return WorkerResearchExtractionResponse(
            request_id=request.request_id,
            factors=(WorkerResearchFactorDraft(fields=fields),),
            usage={"calls": 1},
        )


def test_extraction_contract_round_trip_and_stable_hashes():
    document = _document()
    factor = _factor(document)
    result = ResearchExtractionResult(
        document_id=document.material.document_id,
        text_sha256=document.record.text_sha256,
        model="deepseek/deepseek-chat",
        prompt_version="mining_research_extraction_prompt_v1",
        extraction_version="mining_research_extraction_v1",
        window_count=1,
        factors=(factor,),
        worker_job_ids=("wjob_12345678",),
    )
    later = replace(result, created_at="2026-11-02T00:00:00+00:00")
    assert result.extraction_hash == later.extraction_hash
    assert result.extraction_id == later.extraction_id
    assert result.status == "extracted"
    assert ResearchExtractionResult.from_dict(result.to_dict()) == result
    assert ResearchExtractedFactor.from_dict(factor.to_dict()) == factor


def test_contract_rejects_missing_evidence_and_tampered_offsets():
    document = _document()
    factor = _factor(document)
    explicit_name = factor.field("factor_name")
    with pytest.raises(ResearchMaterialContractError, match="至少一条引用"):
        ResearchFieldClaim(
            field_name="factor_name",
            values=("短期反转因子",),
            status="explicit",
            citations=(),
        )
    with pytest.raises(ResearchMaterialContractError, match="不能包含值"):
        ResearchFieldClaim(
            field_name="formula",
            values=("rank(-ret_5)",),
            status="missing",
            citations=(),
        )
    citation = explicit_name.citations[0]
    with pytest.raises(ResearchMaterialContractError, match="偏移"):
        ResearchExtractionCitation(
            document_id=citation.document_id,
            page_number=1,
            start_offset=0,
            end_offset=1,
            quote="短期反转因子",
        )
    duplicate_fields = tuple(
        item for item in factor.fields if item.field_name != "formula"
    ) + (factor.field("factor_name"),)
    with pytest.raises(ResearchMaterialContractError, match="固定顺序"):
        ResearchExtractedFactor(
            document_id=factor.document_id,
            text_sha256=factor.text_sha256,
            fields=duplicate_fields,
        )


def test_extract_research_factor_drafts_validates_and_locates_quotes():
    document = _document()
    extractor = FakeExtractor()
    result = extract_research_factor_drafts(document, extractor=extractor)
    assert result.status == "extracted"
    assert len(result.factors) == 1
    factor = result.factors[0]
    assert [item.field_name for item in factor.fields] == list(FIELDS)
    assert factor.field("expected_direction").values == ("positive",)
    assert factor.field("economic_mechanism").status == "inferred"
    citation = factor.field("formula").citations[0]
    page = document.pages[0]
    assert page.text[citation.start_offset : citation.end_offset] == citation.quote
    assert extractor.requests[0].document_id == document.material.document_id


def test_extract_rejects_hallucinated_quotes_and_hidden_pages():
    document = _document()
    bad_quote = extract_research_factor_drafts
    with pytest.raises(ResearchExtractionEvidenceError, match="逐字原文"):
        bad_quote(
            document,
            extractor=FakeExtractor(quote="报告中不存在的原句"),
        )

    class HiddenPageExtractor(FakeExtractor):
        def generate(self, request):
            response = super().generate(request)
            factor = response.factors[0]
            fields = tuple(
                replace(
                    field,
                    citations=(
                        WorkerResearchCitationDraft(
                            page_number=2,
                            quote=field.citations[0].quote,
                        ),
                    ),
                )
                for field in factor.fields
            )
            return WorkerResearchExtractionResponse(
                request_id=response.request_id,
                factors=(WorkerResearchFactorDraft(fields=fields),),
            )

    with pytest.raises(ResearchExtractionEvidenceError, match="不可见"):
        extract_research_factor_drafts(
            document,
            extractor=HiddenPageExtractor(),
        )


def test_extract_splits_windows_and_allows_empty_result():
    long_text = "证券研究报告因子 " + ("有效正文\n" * 180)
    document = _document(text=long_text)
    extractor = FakeExtractor(factors=False)
    result = extract_research_factor_drafts(
        document,
        extractor=extractor,
        limits=ResearchExtractionLimits(
            max_window_chars=600,
            overlap_chars=100,
            max_windows=8,
        ),
    )
    assert result.status == "no_factor_evidence"
    assert result.window_count == len(extractor.requests)
    assert result.window_count > 1
    assert len({item.request_id for item in extractor.requests}) == len(
        extractor.requests
    )
    with pytest.raises(ResearchExtractionResourceError, match="超过上限"):
        extract_research_factor_drafts(
            document,
            extractor=extractor,
            limits=ResearchExtractionLimits(
                max_window_chars=513,
                overlap_chars=0,
                max_windows=1,
            ),
        )


def test_identical_overlap_drafts_are_deduplicated():
    long_text = (
        "证券研究报告\n"
        "短期反转因子：过去 5 日收益较低的股票未来收益可能更高。\n"
        "公式为 rank(-ret_5)，预期正向。若 RankIC 低于 0.02 则证伪。\n"
        + ("有效正文\n" * 180)
    )
    document = _document(text=long_text)
    result = extract_research_factor_drafts(
        document,
        extractor=FakeExtractor(),
        limits=ResearchExtractionLimits(
            max_window_chars=600,
            overlap_chars=100,
            max_windows=8,
        ),
    )
    assert result.window_count > 1
    assert len(result.factors) == 1
    assert result.worker_job_ids


def test_extract_rejects_non_target_and_sensitive_page_before_worker():
    extractor = FakeExtractor()
    with pytest.raises(ResearchExtractionContractError, match="金融研报"):
        extract_research_factor_drafts(
            _document(kind="not_target"),
            extractor=extractor,
        )
    with pytest.raises(ResearchExtractionEvidenceError, match="密钥"):
        extract_research_factor_drafts(
            _document(text="证券研究报告 factor stock api_key=secret-value"),
            extractor=extractor,
        )
    assert extractor.requests == []


def test_worker_protocol_rejects_unknown_fields_and_incomplete_citations():
    with pytest.raises(Exception, match="字段"):
        WorkerResearchCitationDraft.from_dict(
            {"page_number": 1, "quote": "原句", "offset": 0}
        )
    with pytest.raises(Exception, match="至少一条引用"):
        WorkerResearchFieldDraft(
            field_name="factor_name",
            values=("因子",),
            status="explicit",
            citations=(),
        )


def test_manifest_export_remains_available():
    document = _document()
    manifest = ResearchMaterialManifest((document.material,))
    assert manifest.materials == (document.material,)


def test_extraction_logs_only_counts_and_identifiers(
    caplog: pytest.LogCaptureFixture,
):
    document = _document()
    with caplog.at_level(logging.INFO):
        result = extract_research_factor_drafts(
            document,
            extractor=FakeExtractor(),
        )
    assert result.factors
    assert document.material.document_id in caplog.text
    assert "短期反转因子" not in caplog.text
    assert "rank(-ret_5)" not in caplog.text
    assert document.pages[0].text[:20] not in caplog.text
