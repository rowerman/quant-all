"""资料驱动 V2 运行的文件解析、抽取、转换和候选准入。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any

from mining.catalog import load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.errors import ResearchRunFailure
from mining.logging_conf import format_event, get_mining_logger
from mining.research_extraction_contracts import ResearchExtractionResult
from mining.research_factor_conversion_client import ResearchFactorConverter
from mining.research_factor_screening import convert_research_factor_drafts
from mining.research_factor_screening_contracts import (
    RESEARCH_CONVERSION_PROMPT_VERSION,
    ResearchConversionContext,
    ResearchConversionResult,
    ResearchConvertedCandidate,
)
from mining.research_material_extractor import (
    ResearchFactorExtractor,
    extract_research_factor_drafts,
)
from mining.research_material_parser import parse_research_materials
from mining.research_material_safety import safe_research_material_log_fields
from mining.research_run_contracts import (
    ResearchCandidateProvenance,
    ResearchMaterialView,
    ResearchRunDecision,
    ResearchRunInput,
    ResearchRunSummary,
    material_view_from_result,
)
from mining.search_context import FrozenSearchContext

_LOGGER = get_mining_logger("mining.research.run")


@dataclass(frozen=True)
class ResearchRunPreparation:
    """资料运行完成转换后可落库的摘要和准入来源。"""

    summary: ResearchRunSummary
    provenances: tuple[ResearchCandidateProvenance, ...]

    @property
    def candidates(self) -> tuple[CandidateDefinition, ...]:
        """返回按来源稳定顺序排列的准入候选。"""

        return tuple(item.candidate for item in self.provenances)


def _material_view(document: Any) -> ResearchMaterialView:
    """把单文件解析结果转换成运行摘要。"""

    record = document.record
    if record.status == "failed":
        return material_view_from_result(
            material=document.material,
            status="failed",
            classification=None,
            page_count=0,
            failure=record.failure,
        )
    classification = document.classification
    return material_view_from_result(
        material=document.material,
        status="parsed" if document.is_financial_research else "filtered",
        classification=None if classification is None else classification.kind,
        page_count=record.page_count,
        failure=None,
    )


def _admission_decisions(
    conversion: ResearchConversionResult | None,
    *,
    admitted_ids: dict[str, str],
) -> tuple[ResearchRunDecision, ...]:
    """把转换结论映射为资料运行摘要。"""

    if conversion is None:
        return ()
    result: list[ResearchRunDecision] = []
    for decision in conversion.decisions:
        candidate_id = (
            None
            if decision.candidate_id is None
            else admitted_ids.get(decision.candidate_id)
        )
        admitted = decision.status == "accepted" and candidate_id is not None
        reason = decision.reason_code
        if decision.status == "accepted" and not admitted:
            reason = "admission_budget_exhausted"
        result.append(
            ResearchRunDecision.from_screening(
                decision,
                candidate_id=candidate_id,
                admitted=admitted,
            )
        )
        if reason != decision.reason_code:
            result[-1] = ResearchRunDecision(
                document_id=result[-1].document_id,
                draft_id=result[-1].draft_id,
                factor_hash=result[-1].factor_hash,
                status=result[-1].status,
                reason_code=reason,
                checks=result[-1].checks,
                candidate_id=result[-1].candidate_id,
                admitted=result[-1].admitted,
            )
    return tuple(result)


def _provenance(
    converted: ResearchConvertedCandidate,
    *,
    extraction: ResearchExtractionResult,
    document: Any,
    candidate: CandidateDefinition,
) -> ResearchCandidateProvenance:
    """构造一个候选的完整来源记录。"""

    factor = next(
        item
        for item in extraction.factors
        if item.draft_id == converted.draft_id
    )
    return ResearchCandidateProvenance(
        run_id=converted.hypothesis.run_id,
        candidate_id=candidate.candidate_id,
        document_id=converted.document_id,
        extraction_id=converted.extraction_id,
        extraction_hash=extraction.extraction_hash,
        draft_id=converted.draft_id,
        factor_hash=converted.factor_hash,
        material=document.material,
        classification=document.classification,
        fields=factor.fields,
        checks=converted.checks,
        field_mapping=converted.field_mapping,
        catalog_hash=converted.hypothesis.metadata["catalog_hash"],
        converter_model=converted.hypothesis.metadata["converter_model"],
        conversion_prompt_version=RESEARCH_CONVERSION_PROMPT_VERSION,
        extraction_model=extraction.model,
        extraction_prompt_version=extraction.prompt_version,
        hypothesis=converted.hypothesis,
        candidate=candidate,
    )


def _enrich_candidate(
    candidate: CandidateDefinition,
    provenance: ResearchCandidateProvenance,
) -> CandidateDefinition:
    """把来源哈希写入候选参数，使候选身份绑定来源。"""

    return replace(
        candidate,
        parameters={
            **dict(candidate.parameters),
            "source_kind": "research_report",
            "research_provenance_hash": provenance.provenance_hash,
        },
    )


def prepare_research_run(
    research_input: ResearchRunInput,
    *,
    material_root: Any,
    context: FrozenSearchContext,
    extractor: ResearchFactorExtractor,
    converter: ResearchFactorConverter,
    admission_limit: int,
    enrich_candidates: bool = True,
) -> ResearchRunPreparation:
    """执行资料解析、字段抽取、候选转换和来源构造。

    参数:
        research_input: 创建运行时冻结的资料输入。
        material_root: 固定资料根目录。
        context: V2 冻结上下文，提供模型与开发期边界之外的配置。
        extractor: Worker v5 抽取客户端。
        converter: Worker v6 转换客户端。
        admission_limit: 本次运行最多自动准入的候选数。
        enrich_candidates: V2 是否把来源哈希写入候选参数；V3 固定为否。
    """

    if not isinstance(research_input, ResearchRunInput):
        raise ResearchRunFailure("research_input 必须是 ResearchRunInput")
    if isinstance(admission_limit, bool) or not isinstance(admission_limit, int):
        raise ResearchRunFailure("admission_limit 必须是整数")
    if admission_limit < 1:
        raise ResearchRunFailure("admission_limit 必须大于 0")
    started = time.perf_counter()
    paths = tuple(
        item.relative_path for item in research_input.manifest.materials
    )
    parsed_batch = parse_research_materials(material_root, paths)
    materials = tuple(_material_view(item) for item in parsed_batch.documents)
    financial_documents = parsed_batch.financial_research
    extractions: list[ResearchExtractionResult] = []
    for document in financial_documents:
        extracted = extract_research_factor_drafts(
            document,
            extractor=extractor,
            llm_model=context.llm_model,
        )
        extractions.append(extracted)
    extraction_tuple = tuple(extractions)
    extracted_factor_count = sum(
        len(item.factors) for item in extraction_tuple
    )

    conversion: ResearchConversionResult | None = None
    non_empty = tuple(item for item in extraction_tuple if item.factors)
    if non_empty:
        conversion = convert_research_factor_drafts(
            non_empty,
            context=ResearchConversionContext(
                run_id=research_input.run_id,
                turn_index=1,
                branch_id="branch-0001",
            ),
            converter=converter,
            llm_model=context.llm_model,
        )
    accepted = () if conversion is None else conversion.candidates
    admitted = accepted[:admission_limit]
    document_by_id = {
        item.material.document_id: item for item in financial_documents
    }
    extraction_by_id = {
        item.document_id: item for item in extraction_tuple
    }
    admitted_ids: dict[str, str] = {}
    provenances: list[ResearchCandidateProvenance] = []
    for converted in admitted:
        extraction = extraction_by_id[converted.document_id]
        document = document_by_id[converted.document_id]
        provisional = _provenance(
            converted,
            extraction=extraction,
            document=document,
            candidate=converted.candidate,
        )
        if enrich_candidates:
            enriched = _enrich_candidate(converted.candidate, provisional)
            provenance = _provenance(
                converted,
                extraction=extraction,
                document=document,
                candidate=enriched,
            )
            stored_candidate_id = enriched.candidate_id
        else:
            provenance = provisional
            stored_candidate_id = converted.candidate.candidate_id
        admitted_ids[converted.candidate.candidate_id] = stored_candidate_id
        provenances.append(provenance)

    decisions = _admission_decisions(
        conversion,
        admitted_ids=admitted_ids,
    )
    accepted_count = len(accepted)
    catalog_hash = context.context_hash
    if conversion is not None:
        catalog_hash = conversion.catalog_hash
    else:
        catalog_hash = load_catalog().catalog_hash
    status = (
        "ready"
        if provenances
        else "failed"
    )
    summary = ResearchRunSummary(
        input=research_input,
        status=status,
        extraction_ids=tuple(item.extraction_id for item in extraction_tuple),
        conversion_id=(
            None if conversion is None else conversion.conversion_id
        ),
        conversion_hash=(
            None if conversion is None else conversion.conversion_hash
        ),
        model=context.llm_model,
        catalog_hash=catalog_hash,
        materials=materials,
        decisions=decisions,
        extracted_factor_count=extracted_factor_count,
        accepted_count=accepted_count,
        admitted_count=len(provenances),
    )
    _LOGGER.info(
        format_event(
            safe_research_material_log_fields(
                {
                    "operation": "prepare_research_run",
                    "manifest_hash": research_input.manifest.manifest_hash,
                    "document_count": len(materials),
                    "financial_research_count": len(financial_documents),
                    "filtered_count": len(parsed_batch.filtered),
                    "failed_count": len(parsed_batch.failures),
                    "factor_count": extracted_factor_count,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000,
                        1,
                    ),
                }
            )
        )
    )
    return ResearchRunPreparation(
        summary=summary,
        provenances=tuple(provenances),
    )


def ensure_research_candidates(preparation: ResearchRunPreparation) -> None:
    """拒绝没有准入候选的资料运行。"""

    if not isinstance(preparation, ResearchRunPreparation):
        raise ResearchRunFailure("preparation 必须是 ResearchRunPreparation")
    if preparation.summary.status != "ready" or not preparation.provenances:
        raise ResearchRunFailure(
            "资料批次没有产生可进入 V2 验证的候选"
        )


__all__ = [
    "ResearchRunPreparation",
    "ensure_research_candidates",
    "prepare_research_run",
]
