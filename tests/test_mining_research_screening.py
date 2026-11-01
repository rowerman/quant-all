"""研报因子草稿筛选、DSL 转换和尝试台账测试。"""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from mining import (
    RESEARCH_EXTRACTION_PROMPT_VERSION,
    RESEARCH_EXTRACTION_RULE_VERSION,
    ResearchConversionContext,
    ResearchConversionLimits,
    ResearchExtractedFactor,
    ResearchExtractionCitation,
    ResearchExtractionResult,
    ResearchFieldClaim,
    ResearchScreeningContractError,
    ResearchScreeningEvidenceError,
    ResearchScreeningResourceError,
    convert_research_factor_drafts,
)
from mining_worker.protocol_research_conversion import (
    WorkerResearchConversionOutcome,
    WorkerResearchConversionResponse,
)
from mining.contract_types import canonical_json

FIELDS = (
    "factor_name",
    "economic_mechanism",
    "formula",
    "variables",
    "expected_direction",
    "falsification_conditions",
)


def _document_id(index: int) -> str:
    return "doc_" + f"{index:024x}"


def _citation(document_id: str, quote: str = "因子") -> ResearchExtractionCitation:
    return ResearchExtractionCitation(
        document_id=document_id,
        page_number=1,
        start_offset=0,
        end_offset=len(quote),
        quote=quote,
    )


def _claim(
    document_id: str,
    field_name: str,
    values: tuple[str, ...],
    *,
    status: str = "explicit",
) -> ResearchFieldClaim:
    return ResearchFieldClaim(
        field_name=field_name,
        values=values,
        status=status,
        citations=(
            ()
            if status == "missing"
            else (_citation(document_id),)
        ),
    )


def _factor(
    document_id: str,
    *,
    name: str = "短期反转因子",
    formula: str = "rank(ret)",
    variables: tuple[str, ...] = ("ret_5",),
    direction: str = "positive",
    missing_field: str | None = None,
) -> ResearchExtractedFactor:
    values = {
        "factor_name": (name,),
        "economic_mechanism": ("低收益股票后续修复",),
        "formula": (formula,),
        "variables": variables,
        "expected_direction": (direction,),
        "falsification_conditions": ("RankIC 低于 0.02",),
    }
    return ResearchExtractedFactor(
        document_id=document_id,
        text_sha256=hashlib.sha256(document_id.encode("utf-8")).hexdigest(),
        fields=tuple(
            _claim(
                document_id,
                field_name,
                () if field_name == missing_field else values[field_name],
                status="missing" if field_name == missing_field else "explicit",
            )
            for field_name in FIELDS
        ),
    )


def _extraction(
    index: int,
    factors: tuple[ResearchExtractedFactor, ...],
) -> ResearchExtractionResult:
    document_id = _document_id(index)
    return ResearchExtractionResult(
        document_id=document_id,
        text_sha256=hashlib.sha256(document_id.encode("utf-8")).hexdigest(),
        model="deepseek/deepseek-chat",
        prompt_version=RESEARCH_EXTRACTION_PROMPT_VERSION,
        extraction_version=RESEARCH_EXTRACTION_RULE_VERSION,
        window_count=1,
        factors=factors,
        worker_job_ids=(f"wjob_{index:024x}",),
    )


class FakeConverter:
    """按 draft_id 返回预置转换结论。"""

    def __init__(self, outcomes: dict[str, WorkerResearchConversionOutcome]) -> None:
        self.outcomes = outcomes
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return WorkerResearchConversionResponse(
            request_id=request.request_id,
            outcomes=tuple(
                self.outcomes[item.draft_id] for item in request.drafts
            ),
            usage={"calls": 1},
        )


def _converted(draft_id: str, *, mapping: dict[str, str] | None = None):
    return WorkerResearchConversionOutcome(
        draft_id=draft_id,
        status="converted",
        expression="rank(ret)",
        field_mapping=mapping or {"ret_5": "ret"},
    )


def _context() -> ResearchConversionContext:
    return ResearchConversionContext(
        run_id="searchrun_12345678",
        turn_index=1,
        branch_id="branch-0001",
    )


def test_conversion_builds_auditable_hypothesis_candidate_and_trial():
    document_id = _document_id(1)
    factor = _factor(document_id)
    extraction = _extraction(1, (factor,))
    converter = FakeConverter({factor.draft_id: _converted(factor.draft_id)})

    result = convert_research_factor_drafts(
        (extraction,),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert len(converter.requests) == 1
    assert len(result.candidates) == 1
    converted = result.candidates[0]
    assert converted.candidate.canonical_expression == "rank(ret)"
    assert converted.candidate.direction == 1
    assert converted.hypothesis.run_id == "searchrun_12345678"
    assert converted.hypothesis.metadata["document_id"] == document_id
    assert converted.hypothesis.metadata["draft_id"] == factor.draft_id
    assert converted.hypothesis.metadata["field_mapping"] == {"ret_5": "ret"}
    assert result.decisions[0].status == "accepted"
    assert result.decisions[0].candidate_id == converted.candidate.candidate_id
    assert len(result.trials) == 1
    assert result.trials[0].kind == "proposed"
    assert result.trials[0].candidate_id == converted.candidate.candidate_id

    later = replace(result, created_at="2026-11-02T00:00:00+00:00")
    assert later.conversion_hash == result.conversion_hash
    assert later.conversion_id == result.conversion_id
    assert canonical_json(result.to_dict())

    repeated = convert_research_factor_drafts(
        (extraction,),
        context=_context(),
        converter=FakeConverter({factor.draft_id: _converted(factor.draft_id)}),
        llm_model="deepseek/deepseek-chat",
    )
    assert repeated.conversion_hash == result.conversion_hash


def test_pre_screen_rejects_missing_evidence_and_non_monotonic_direction():
    first_document = _document_id(1)
    second_document = _document_id(2)
    incomplete = _factor(first_document, missing_field="formula")
    non_monotonic = _factor(
        second_document,
        name="非线性估值因子",
        formula="rank(pb)",
        variables=("pb",),
        direction="non_monotonic",
    )
    converter = FakeConverter({})

    result = convert_research_factor_drafts(
        (
            _extraction(1, (incomplete,)),
            _extraction(2, (non_monotonic,)),
        ),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert converter.requests == []
    assert [item.reason_code for item in result.decisions] == [
        "incomplete_evidence",
        "unsupported_direction",
    ]
    assert [item.status for item in result.decisions] == ["rejected", "rejected"]
    assert [item.kind for item in result.trials] == ["rejected", "rejected"]
    assert result.candidates == ()


def test_model_outputs_are_revalidated_against_mapping_and_catalog():
    document_id = _document_id(1)
    unknown_mapping_factor = _factor(
        document_id,
        name="未知映射因子",
        formula="rank(unknown)",
        variables=("ret_5",),
    )
    invented_key_factor = _factor(
        document_id,
        name="伪造变量因子",
        formula="rank(ret)",
        variables=("ret_5",),
    )
    unknown_field_factor = _factor(
        document_id,
        name="未知字段因子",
        formula="rank(missing_field)",
        variables=("missing_field",),
    )
    extraction = _extraction(
        1,
        (
            unknown_mapping_factor,
            invented_key_factor,
            unknown_field_factor,
        ),
    )
    converter = FakeConverter(
        {
            unknown_mapping_factor.draft_id: _converted(
                unknown_mapping_factor.draft_id,
                mapping={"ret_5": "unknown"},
            ),
            invented_key_factor.draft_id: _converted(
                invented_key_factor.draft_id,
                mapping={"invented": "ret"},
            ),
            unknown_field_factor.draft_id: WorkerResearchConversionOutcome(
                draft_id=unknown_field_factor.draft_id,
                status="converted",
                expression="rank(missing_field)",
                field_mapping={"missing_field": "ret"},
            ),
        }
    )

    result = convert_research_factor_drafts(
        (extraction,),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert result.candidates == ()
    assert [item.status for item in result.decisions] == ["invalid"] * 3
    assert sorted(item.reason_code for item in result.decisions) == [
        "unavailable_field",
        "unknown_field",
        "unsupported_variable_mapping",
    ]
    assert [item.kind for item in result.trials] == ["invalid"] * 3


def test_cross_document_semantic_duplicates_keep_stable_winner():
    first_document = _document_id(1)
    second_document = _document_id(2)
    first = _factor(first_document, name="反转甲", formula="rank(ret)")
    second = _factor(second_document, name="反转乙", formula="rank(ret)")
    converter = FakeConverter(
        {
            first.draft_id: _converted(first.draft_id),
            second.draft_id: _converted(second.draft_id),
        }
    )

    result = convert_research_factor_drafts(
        (
            _extraction(2, (second,)),
            _extraction(1, (first,)),
        ),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert len(result.candidates) == 1
    assert [item.status for item in result.decisions] == ["accepted", "duplicate"]
    assert result.decisions[1].candidate_id == result.candidates[0].candidate.candidate_id
    assert [item.kind for item in result.trials] == ["proposed", "duplicate"]
    assert result.trials[1].candidate_id == result.candidates[0].candidate.candidate_id


def test_large_report_is_split_into_eight_item_worker_jobs():
    document_id = _document_id(1)
    factors = tuple(
        _factor(document_id, name=f"候选因子{i:02d}", formula="rank(ret)")
        for i in range(10)
    )
    converter = FakeConverter(
        {factor.draft_id: _converted(factor.draft_id) for factor in factors}
    )

    result = convert_research_factor_drafts(
        (_extraction(1, factors),),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert [len(request.drafts) for request in converter.requests] == [8, 2]
    assert len(result.candidates) == 1
    assert sum(item.status == "duplicate" for item in result.decisions) == 9
    assert len(result.trials) == 10


def test_batch_limits_and_protocol_errors_are_explicit():
    document_id = _document_id(1)
    factor = _factor(document_id)
    second = _factor(document_id, name="第二因子")
    converter = FakeConverter(
        {
            factor.draft_id: _converted(factor.draft_id),
            second.draft_id: _converted(second.draft_id),
        }
    )

    with pytest.raises(ResearchScreeningResourceError, match="因子草稿数量"):
        convert_research_factor_drafts(
            (_extraction(1, (factor, second)),),
            context=_context(),
            converter=converter,
            limits=ResearchConversionLimits(
                max_factors=1,
                max_factors_per_job=1,
            ),
        )

    with pytest.raises(ResearchScreeningContractError, match="extractions"):
        convert_research_factor_drafts(
            ([],),
            context=_context(),
            converter=converter,
        )


def test_excluded_field_and_converter_rejection_are_recorded():
    document_id = _document_id(1)
    factor = _factor(document_id)
    extraction = _extraction(1, (factor,))
    converter = FakeConverter({factor.draft_id: _converted(factor.draft_id)})

    excluded = convert_research_factor_drafts(
        (extraction,),
        context=ResearchConversionContext(
            run_id="searchrun_12345678",
            turn_index=1,
            branch_id="branch-0001",
            excluded_fields=("ret",),
        ),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )
    assert excluded.decisions[0].status == "invalid"
    assert excluded.decisions[0].reason_code == "excluded_field"
    assert excluded.trials[0].kind == "invalid"

    rejecting = FakeConverter(
        {
            factor.draft_id: WorkerResearchConversionOutcome(
                draft_id=factor.draft_id,
                status="rejected",
                expression=None,
                reason_code="unmapped_formula",
                message="公式无法确定映射",
            )
        }
    )
    rejected = convert_research_factor_drafts(
        (extraction,),
        context=_context(),
        converter=rejecting,
        llm_model="deepseek/deepseek-chat",
    )
    assert rejected.decisions[0].status == "invalid"
    assert rejected.decisions[0].reason_code == "unmapped_formula"
    assert rejected.trials[0].kind == "invalid"


def test_unsafe_evidence_is_not_sent_to_worker():
    document_id = _document_id(1)
    factor = _factor(document_id, name="sk-1234567890泄露因子")
    converter = FakeConverter({})

    result = convert_research_factor_drafts(
        (_extraction(1, (factor,)),),
        context=_context(),
        converter=converter,
        llm_model="deepseek/deepseek-chat",
    )

    assert converter.requests == []
    assert result.decisions[0].reason_code == "unsafe_evidence"
    assert result.trials[0].kind == "rejected"


class _MismatchedConverter:
    def generate(self, request):
        return WorkerResearchConversionResponse(
            request_id="different-request",
            outcomes=(),
        )


def test_worker_response_must_bind_exact_request():
    document_id = _document_id(1)
    factor = _factor(document_id)

    with pytest.raises(ResearchScreeningEvidenceError, match="request_id"):
        convert_research_factor_drafts(
            (_extraction(1, (factor,)),),
            context=_context(),
            converter=_MismatchedConverter(),
            llm_model="deepseek/deepseek-chat",
        )
