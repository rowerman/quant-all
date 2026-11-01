"""研报因子草稿的筛查、Worker 请求和候选转换辅助逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from mining.catalog import MiningCatalog
from mining.contract_candidates import CandidateDefinition
from mining.errors import (
    ExpressionValidationError,
    ResearchScreeningEvidenceError,
)
from mining.expression_validation import ValidatedExpression, validate_expression
from mining.research_extraction_contracts import (
    ResearchExtractedFactor,
    ResearchExtractionResult,
    ResearchFieldClaim,
)
from mining.research_factor_screening_contracts import (
    RESEARCH_CONVERSION_PROMPT_VERSION,
    ResearchConversionContext,
    ResearchScreeningDecision,
)
from mining.research_material_safety import redact_research_material_text
from mining.search_cards import HypothesisCard
from mining.search_trials import TrialRecord
from mining_worker.protocol_research import (
    WorkerResearchCitationDraft,
    WorkerResearchFieldDraft,
)
from mining_worker.protocol_research_conversion import (
    WorkerResearchCatalogField,
    WorkerResearchCatalogOperator,
    WorkerResearchConversionDraft,
    WorkerResearchConversionOutcome,
    WorkerResearchConversionRequest,
    WorkerResearchConversionResponse,
)

REQUIRED_FIELDS = (
    "factor_name",
    "economic_mechanism",
    "formula",
    "variables",
    "expected_direction",
    "falsification_conditions",
)


@dataclass(frozen=True)
class PreparedDraft:
    """已通过来源筛查、等待模型转换的草稿。"""

    extraction: ResearchExtractionResult
    factor: ResearchExtractedFactor
    direction: int


@dataclass(frozen=True)
class ValidatedDraft:
    """模型转换后已通过主服务确定性校验的草稿。"""

    prepared: PreparedDraft
    validated: ValidatedExpression
    field_mapping: dict[str, str]


@dataclass(frozen=True)
class DecisionDraft:
    """尚未分配尝试 ID 的筛选结论。"""

    document_id: str
    draft_id: str
    factor_hash: str
    status: str
    reason_code: str | None
    checks: tuple[str, ...]
    candidate_id: str | None = None
    hypothesis_id: str | None = None


def field_claim(factor: ResearchExtractedFactor, name: str) -> ResearchFieldClaim:
    """按固定字段名读取字段声明。"""

    return factor.field(name)


def direction(factor: ResearchExtractedFactor) -> int:
    """把研报方向转换为候选方向。"""

    value = field_claim(factor, "expected_direction").values[0]
    if value == "positive":
        return 1
    if value == "negative":
        return -1
    raise ResearchScreeningEvidenceError(
        "non_monotonic 方向无法映射为受控 DSL 候选"
    )


def _unsafe_evidence(factor: ResearchExtractedFactor) -> bool:
    """检查发送给 Worker 的字段和短引用是否已脱敏。"""

    material = "\n".join(
        [
            *(value for claim in factor.fields for value in claim.values),
            *(
                citation.quote
                for claim in factor.fields
                for citation in claim.citations
            ),
        ]
    )
    return redact_research_material_text(material) != material


def pre_screen(
    extraction: ResearchExtractionResult,
    factor: ResearchExtractedFactor,
) -> tuple[str | None, tuple[str, ...]]:
    """在模型调用前执行字段完整性和安全筛查。"""

    claim_map = {claim.field_name: claim for claim in factor.fields}
    complete = all(
        name in claim_map
        and claim_map[name].status != "missing"
        and bool(claim_map[name].values)
        and bool(claim_map[name].citations)
        for name in REQUIRED_FIELDS
    )
    if not complete:
        return "incomplete_evidence", ("evidence_complete",)
    if _unsafe_evidence(factor):
        return "unsafe_evidence", ("evidence_complete", "safe_evidence")
    expected_direction = claim_map["expected_direction"].values[0]
    if expected_direction not in {"positive", "negative"}:
        return "unsupported_direction", (
            "evidence_complete",
            "safe_evidence",
            "supported_direction",
        )
    if (
        extraction.document_id != factor.document_id
        or extraction.text_sha256 != factor.text_sha256
    ):
        return "document_mismatch", ("document_bound",)
    return None, (
        "document_bound",
        "evidence_complete",
        "safe_evidence",
        "supported_direction",
    )


def _worker_field(claim: ResearchFieldClaim) -> WorkerResearchFieldDraft:
    """把主服务字段声明转换为 Worker 白名单字段。"""

    return WorkerResearchFieldDraft(
        field_name=claim.field_name,
        values=claim.values,
        status=claim.status,
        citations=tuple(
            WorkerResearchCitationDraft(
                page_number=citation.page_number,
                quote=citation.quote,
            )
            for citation in claim.citations
        ),
    )


def _catalog_fields(catalog: MiningCatalog) -> tuple[WorkerResearchCatalogField, ...]:
    """返回不含表和连接信息的允许字段。"""

    return tuple(
        WorkerResearchCatalogField(
            name=field.name,
            label=field.label,
            dimension=field.dimension,
            unit=field.unit,
        )
        for field in catalog.allowed_fields
    )


def _catalog_operators(
    catalog: MiningCatalog,
) -> tuple[WorkerResearchCatalogOperator, ...]:
    """返回 Worker 可见的算子目录。"""

    return tuple(
        WorkerResearchCatalogOperator(
            name=operator.name,
            kind=operator.kind,
            min_args=operator.min_args,
            max_args=operator.max_args,
            min_window=operator.min_window,
            max_window=operator.max_window,
        )
        for operator in catalog.operators
    )


def _request_id(
    extraction: ResearchExtractionResult,
    drafts: Iterable[PreparedDraft],
    *,
    chunk_index: int,
    chunk_count: int,
    catalog_hash: str,
) -> str:
    """生成与正文、目录和分块内容绑定的稳定请求 ID。"""

    digest = hashlib.sha256(
        json.dumps(
            {
                "document_id": extraction.document_id,
                "extraction_hash": extraction.extraction_hash,
                "draft_ids": [item.factor.draft_id for item in drafts],
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "catalog_hash": catalog_hash,
                "prompt_version": RESEARCH_CONVERSION_PROMPT_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"research_convert_{digest[:24]}"


def conversion_request(
    extraction: ResearchExtractionResult,
    drafts: tuple[PreparedDraft, ...],
    *,
    chunk_index: int,
    chunk_count: int,
    catalog: MiningCatalog,
    llm_model: str,
) -> WorkerResearchConversionRequest:
    """构造一个受控转换请求。"""

    return WorkerResearchConversionRequest(
        request_id=_request_id(
            extraction,
            drafts,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            catalog_hash=catalog.catalog_hash,
        ),
        document_id=extraction.document_id,
        extraction_id=extraction.extraction_id,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        catalog_hash=catalog.catalog_hash,
        drafts=tuple(
            WorkerResearchConversionDraft(
                draft_id=item.factor.draft_id,
                factor_hash=item.factor.factor_hash,
                fields=tuple(_worker_field(claim) for claim in item.factor.fields),
            )
            for item in drafts
        ),
        fields=_catalog_fields(catalog),
        operators=_catalog_operators(catalog),
        llm_model=llm_model,
    )


def outcome_map(
    request: WorkerResearchConversionRequest,
    response: WorkerResearchConversionResponse,
) -> dict[str, WorkerResearchConversionOutcome]:
    """确认 Worker 对每个请求草稿恰好返回一个结论。"""

    if response.request_id != request.request_id:
        raise ResearchScreeningEvidenceError(
            "Worker v6 响应 request_id 与转换请求不一致"
        )
    expected = {item.draft_id for item in request.drafts}
    actual = {item.draft_id for item in response.outcomes}
    if expected != actual:
        raise ResearchScreeningEvidenceError(
            "Worker v6 响应没有完整覆盖当前转换分块"
        )
    return {item.draft_id: item for item in response.outcomes}


def validate_outcome(
    prepared: PreparedDraft,
    outcome: WorkerResearchConversionOutcome,
    *,
    catalog: MiningCatalog,
    excluded_fields: tuple[str, ...],
    max_expression_depth: int,
) -> tuple[ValidatedDraft | None, str | None]:
    """在主服务重新校验模型表达式和变量映射。"""

    if outcome.status == "rejected":
        return None, outcome.reason_code or "converter_rejected"
    assert outcome.expression is not None
    source_variables = set(field_claim(prepared.factor, "variables").values)
    mapping = dict(outcome.field_mapping)
    if not set(mapping).issubset(source_variables):
        return None, "unsupported_variable_mapping"
    try:
        for target in mapping.values():
            if target in excluded_fields:
                raise ExpressionValidationError(
                    f"字段 {target!r} 已被本次运行排除",
                    reason="excluded_field",
                )
            catalog.field(target, require_allowed=True)
    except ExpressionValidationError as error:
        return None, error.reason or "invalid_conversion"
    except Exception:
        return None, "unavailable_field"
    try:
        validated = validate_expression(
            outcome.expression,
            direction=prepared.direction,
            max_depth=max_expression_depth,
            excluded_fields=excluded_fields,
            catalog=catalog,
        )
    except ExpressionValidationError as error:
        return None, getattr(error, "reason", None) or "invalid_conversion"
    except Exception:
        return None, "invalid_conversion"
    if not set(validated.dependencies).issubset(set(mapping.values())):
        return None, "unsupported_expression_dependency"
    return (
        ValidatedDraft(
            prepared=prepared,
            validated=validated,
            field_mapping=dict(sorted(mapping.items())),
        ),
        None,
    )


def _hypothesis_statement(factor: ResearchExtractedFactor) -> str:
    """从因子名称和公式形成可追踪假设陈述。"""

    name = field_claim(factor, "factor_name").values[0]
    formula = field_claim(factor, "formula").values[0]
    statement = f"{name}：{formula}"[:4000]
    if len(statement) < 8:
        statement = f"{statement}的研报因子假设"[:4000]
    return statement


def candidate_and_hypothesis(
    validated: ValidatedDraft,
    *,
    context: ResearchConversionContext,
    discovery_index: int,
    catalog: MiningCatalog,
    llm_model: str,
) -> tuple[CandidateDefinition, HypothesisCard]:
    """把已验证 DSL 包装成候选和研报假设卡。"""

    factor = validated.prepared.factor
    extraction = validated.prepared.extraction
    factor_name = field_claim(factor, "factor_name").values[0]
    formula = field_claim(factor, "formula").values[0]
    mechanism = field_claim(factor, "economic_mechanism").values[0]
    falsification = field_claim(factor, "falsification_conditions").values
    expected_direction = field_claim(factor, "expected_direction").values[0]
    candidate = validated.validated.to_candidate(
        mechanism=factor_name[:200],
        description=f"{factor_name}：{formula}"[:2000],
        steps=(
            "读取研报因子字段和页码证据",
            "把研报变量映射到冻结字段",
            "由主服务校验受控 DSL 表达式",
        ),
        discovery_round=context.turn_index,
        discovery_index=discovery_index,
    )
    claim_hashes = {
        claim.field_name: claim.claim_hash for claim in factor.fields
    }
    citation_ids = {
        claim.field_name: [item.citation_id for item in claim.citations]
        for claim in factor.fields
    }
    hypothesis = HypothesisCard(
        run_id=context.run_id,
        turn_index=context.turn_index,
        branch_id=context.branch_id,
        statement=_hypothesis_statement(factor),
        mechanism=mechanism,
        expected_direction=expected_direction,
        falsification_criteria=falsification,
        parent_hypothesis_id=context.parent_hypothesis_id,
        metadata={
            "source_kind": "research_report",
            "document_id": extraction.document_id,
            "extraction_id": extraction.extraction_id,
            "extraction_hash": extraction.extraction_hash,
            "draft_id": factor.draft_id,
            "factor_hash": factor.factor_hash,
            "claim_hashes": claim_hashes,
            "citation_ids": citation_ids,
            "field_mapping": validated.field_mapping,
            "catalog_hash": catalog.catalog_hash,
            "converter_model": llm_model,
            "prompt_version": RESEARCH_CONVERSION_PROMPT_VERSION,
        },
    )
    return candidate, hypothesis


def build_trial(
    *,
    context: ResearchConversionContext,
    decision: DecisionDraft,
) -> TrialRecord:
    """把筛选结论转换为现有 V2 尝试台账记录。"""

    kind = {
        "accepted": "proposed",
        "rejected": "rejected",
        "invalid": "invalid",
        "duplicate": "duplicate",
    }[decision.status]
    return TrialRecord(
        run_id=context.run_id,
        trial_key=f"research-{decision.draft_id}",
        kind=kind,
        turn_index=context.turn_index,
        hypothesis_id=decision.hypothesis_id,
        candidate_id=decision.candidate_id,
        reason_code=decision.reason_code,
        payload={
            "source_kind": "research_report",
            "document_id": decision.document_id,
            "draft_id": decision.draft_id,
            "factor_hash": decision.factor_hash,
            "screening_status": decision.status,
        },
    )


def public_decision(
    draft: DecisionDraft,
    trial: TrialRecord,
) -> ResearchScreeningDecision:
    """构造公开筛选结论。"""

    return ResearchScreeningDecision(
        document_id=draft.document_id,
        draft_id=draft.draft_id,
        factor_hash=draft.factor_hash,
        status=draft.status,
        reason_code=draft.reason_code,
        checks=draft.checks,
        candidate_id=draft.candidate_id,
        trial_id=trial.trial_id,
    )


__all__ = [
    "DecisionDraft",
    "PreparedDraft",
    "ValidatedDraft",
    "build_trial",
    "candidate_and_hypothesis",
    "conversion_request",
    "direction",
    "field_claim",
    "outcome_map",
    "pre_screen",
    "public_decision",
    "validate_outcome",
]
