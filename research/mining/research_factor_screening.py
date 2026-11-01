"""研报因子草稿的批量筛选、DSL 转换和语义去重编排。"""

from __future__ import annotations

from typing import Iterable
import time

from mining.catalog import MiningCatalog, load_catalog
from mining.contract_types import text
from mining.errors import (
    ResearchScreeningContractError,
    ResearchScreeningResourceError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.research_extraction_contracts import ResearchExtractionResult
from mining.research_factor_conversion_client import ResearchFactorConverter
from mining.research_factor_conversion_support import (
    DecisionDraft,
    PreparedDraft,
    ValidatedDraft,
    build_trial,
    candidate_and_hypothesis,
    conversion_request,
    direction,
    outcome_map,
    pre_screen,
    public_decision,
    validate_outcome,
)
from mining.research_factor_screening_contracts import (
    ResearchConversionContext,
    ResearchConversionLimits,
    ResearchConversionResult,
    ResearchConvertedCandidate,
)
from mining_worker.worker_config import load_worker_config

_LOGGER = get_mining_logger("mining.research_screening")


def convert_research_factor_drafts(
    extractions: Iterable[ResearchExtractionResult],
    *,
    context: ResearchConversionContext,
    converter: ResearchFactorConverter,
    catalog: MiningCatalog | None = None,
    limits: ResearchConversionLimits | None = None,
    llm_model: str | None = None,
) -> ResearchConversionResult:
    """筛选研报因子草稿并转换为未落库的 HypothesisCard 和 DSL 候选。

    参数:
        extractions: 已通过 P2-03 抽取的一份或多份研报结果。
        context: 绑定 V2 运行、轮次和分支的转换上下文。
        converter: 实现 Worker v6 分块转换的同步客户端。
        catalog: 可注入的冻结字段与算子目录。
        limits: 文档、草稿和表达式资源上限。
        llm_model: 冻结模型名；省略时读取 Worker 默认模型。
    """

    if not isinstance(context, ResearchConversionContext):
        raise ResearchScreeningContractError(
            "context 必须是 ResearchConversionContext"
        )
    if not hasattr(converter, "generate"):
        raise ResearchScreeningContractError("converter 必须实现 generate")
    resolved_limits = limits or ResearchConversionLimits()
    if not isinstance(resolved_limits, ResearchConversionLimits):
        raise ResearchScreeningContractError(
            "limits 必须是 ResearchConversionLimits"
        )
    resolved_catalog = catalog or load_catalog()
    try:
        model = text(
            llm_model or load_worker_config().default_model,
            "llm_model",
            maximum=200,
        )
    except Exception as error:
        raise ResearchScreeningContractError(str(error)) from error

    raw_extractions = tuple(extractions)
    if not raw_extractions:
        raise ResearchScreeningContractError("至少需要一份研报抽取结果")
    if any(
        not isinstance(item, ResearchExtractionResult)
        for item in raw_extractions
    ):
        raise ResearchScreeningContractError(
            "extractions 只能包含 ResearchExtractionResult"
        )
    ordered_extractions = tuple(
        sorted(raw_extractions, key=lambda item: (item.document_id, item.extraction_id))
    )
    if len(ordered_extractions) > resolved_limits.max_documents:
        raise ResearchScreeningResourceError(
            f"研报数量 {len(ordered_extractions)} 超过上限 "
            f"{resolved_limits.max_documents}"
        )
    if len({item.document_id for item in ordered_extractions}) != len(
        ordered_extractions
    ):
        raise ResearchScreeningContractError("同一批次不能重复提交相同资料")
    all_factors = [
        (extraction, factor)
        for extraction in ordered_extractions
        for factor in extraction.factors
    ]
    if len(all_factors) > resolved_limits.max_factors:
        raise ResearchScreeningResourceError(
            f"因子草稿数量 {len(all_factors)} 超过上限 "
            f"{resolved_limits.max_factors}"
        )

    started = time.perf_counter()
    prepared_by_document: dict[str, list[PreparedDraft]] = {
        item.document_id: [] for item in ordered_extractions
    }
    decision_drafts: list[DecisionDraft] = []
    for extraction, factor in all_factors:
        reason, checks = pre_screen(extraction, factor)
        if reason is None:
            prepared_by_document[extraction.document_id].append(
                PreparedDraft(
                    extraction=extraction,
                    factor=factor,
                    direction=direction(factor),
                )
            )
        else:
            decision_drafts.append(
                DecisionDraft(
                    document_id=extraction.document_id,
                    draft_id=factor.draft_id,
                    factor_hash=factor.factor_hash,
                    status="rejected",
                    reason_code=reason,
                    checks=checks,
                )
            )

    validated_drafts: list[ValidatedDraft] = []
    for extraction in ordered_extractions:
        prepared = sorted(
            prepared_by_document[extraction.document_id],
            key=lambda item: item.factor.draft_id,
        )
        chunks = tuple(
            tuple(prepared[index : index + resolved_limits.max_factors_per_job])
            for index in range(0, len(prepared), resolved_limits.max_factors_per_job)
        )
        for chunk_index, chunk in enumerate(chunks):
            request = conversion_request(
                extraction,
                chunk,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                catalog=resolved_catalog,
                llm_model=model,
            )
            response = converter.generate(request)
            outcomes = outcome_map(request, response)
            for item in chunk:
                converted, reason = validate_outcome(
                    item,
                    outcomes[item.factor.draft_id],
                    catalog=resolved_catalog,
                    excluded_fields=context.excluded_fields,
                    max_expression_depth=resolved_limits.max_expression_depth,
                )
                if converted is None:
                    decision_drafts.append(
                        DecisionDraft(
                            document_id=extraction.document_id,
                            draft_id=item.factor.draft_id,
                            factor_hash=item.factor.factor_hash,
                            status="invalid",
                            reason_code=reason or "invalid_conversion",
                            checks=(
                                "document_bound",
                                "evidence_complete",
                                "safe_evidence",
                                "supported_direction",
                            ),
                        )
                    )
                else:
                    validated_drafts.append(converted)

    validated_drafts.sort(
        key=lambda item: (
            item.prepared.extraction.document_id,
            item.prepared.factor.draft_id,
        )
    )
    candidates: list[ResearchConvertedCandidate] = []
    seen_semantic: dict[str, ResearchConvertedCandidate] = {}
    checks = (
        "catalog_available",
        "direction_consistent",
        "document_bound",
        "dsl_valid",
        "evidence_complete",
        "mapping_consistent",
        "safe_evidence",
        "supported_direction",
    )
    for discovery_index, item in enumerate(validated_drafts, start=1):
        candidate, hypothesis = candidate_and_hypothesis(
            item,
            context=context,
            discovery_index=discovery_index,
            catalog=resolved_catalog,
            llm_model=model,
        )
        converted = ResearchConvertedCandidate(
            document_id=item.prepared.extraction.document_id,
            extraction_id=item.prepared.extraction.extraction_id,
            draft_id=item.prepared.factor.draft_id,
            factor_hash=item.prepared.factor.factor_hash,
            checks=checks,
            field_mapping=item.field_mapping,
            hypothesis=hypothesis,
            candidate=candidate,
        )
        winner = seen_semantic.get(candidate.semantic_hash)
        if winner is not None:
            decision_drafts.append(
                DecisionDraft(
                    document_id=converted.document_id,
                    draft_id=converted.draft_id,
                    factor_hash=converted.factor_hash,
                    status="duplicate",
                    reason_code="semantic_duplicate",
                    checks=checks,
                    candidate_id=winner.candidate.candidate_id,
                    hypothesis_id=winner.hypothesis.hypothesis_id,
                )
            )
            continue
        seen_semantic[candidate.semantic_hash] = converted
        candidates.append(converted)
        decision_drafts.append(
            DecisionDraft(
                document_id=converted.document_id,
                draft_id=converted.draft_id,
                factor_hash=converted.factor_hash,
                status="accepted",
                reason_code=None,
                checks=checks,
                candidate_id=candidate.candidate_id,
                hypothesis_id=hypothesis.hypothesis_id,
            )
        )

    decision_drafts.sort(key=lambda item: (item.document_id, item.draft_id))
    trials = tuple(
        build_trial(context=context, decision=item) for item in decision_drafts
    )
    decisions = tuple(
        public_decision(item, trial)
        for item, trial in zip(decision_drafts, trials, strict=True)
    )
    result = ResearchConversionResult(
        extraction_ids=tuple(item.extraction_id for item in ordered_extractions),
        model=model,
        catalog_hash=resolved_catalog.catalog_hash,
        context=context,
        limits=resolved_limits,
        candidates=tuple(candidates),
        decisions=decisions,
        trials=trials,
    )
    _LOGGER.info(
        format_event(
            {
                "operation": "convert_research_factor_drafts",
                "conversion_id": result.conversion_id,
                "model": model,
                "document_count": len(ordered_extractions),
                "source_factor_count": len(all_factors),
                "accepted": len(result.candidates),
                "rejected": sum(item.status == "rejected" for item in decisions),
                "invalid": sum(item.status == "invalid" for item in decisions),
                "duplicate": sum(item.status == "duplicate" for item in decisions),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    )
    return result


__all__ = ["convert_research_factor_drafts"]
