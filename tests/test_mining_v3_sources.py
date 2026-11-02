"""V3 四类研究来源、候选身份和证据适配测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import sha256_json
from mining.errors import MiningContractError
from mining.free_exploration_contracts import ResearchDirection
from mining.free_exploration_profiles import FREE_EXPLORATION_PROFILE_HASH
from mining.research_extraction_contracts import (
    RESEARCH_FACTOR_FIELDS,
    ResearchExtractionCitation,
    ResearchFieldClaim,
)
from mining.research_factor_screening_contracts import ResearchConversionLimits
from mining.research_material_contracts import (
    ResearchMaterialInput,
    ResearchMaterialManifest,
)
from mining.research_material_parse_contracts import (
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    ResearchMaterialClassification,
)
from mining.research_run_contracts import (
    ResearchCandidateProvenance,
    ResearchRunInput,
)
from mining.search_cards import HypothesisCard
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_run import SearchRunRequest
from mining.v3_contracts import (
    V3_REQUIRED_EVIDENCE_STAGES,
    CandidateEnvelope,
    ResearchSourceReference,
)
from mining.v3_sources import (
    costeer_candidate_source,
    costeer_input_envelope,
    evidence_reference_from_search,
    free_exploration_candidate_source,
    free_exploration_input_envelope,
    hypothesis_input_envelope,
    research_report_candidate_source,
    research_report_input_envelope,
    source_independent_research_candidate,
    validated_research_report_provenance,
)
from mining_sandbox.artifacts import PromotionRequest


CREATED_AT = "2026-11-02T00:00:00+00:00"


def _candidate(
    *,
    parameters: dict[str, object] | None = None,
) -> CandidateDefinition:
    return CandidateDefinition(
        name="V3_SHARED_FACTOR",
        mechanism="价格趋势",
        description="来源无关的 V3 联合候选",
        canonical_expression="rank(mean(close, 20))",
        direction=1,
        lookback=20,
        dependencies=("close",),
        output_type="score",
        steps=("计算二十日价格均值", "按交易日做截面排名"),
        discovery_round=1,
        discovery_index=1,
        parameters=parameters or {},
    )


def _evidence(
    candidate_id: str,
    *,
    digest: str = "d",
) -> tuple:
    return tuple(
        evidence_reference_from_search(
            SearchEvaluationEvidence(
                run_id="searchrun_111111111111111111111111",
                candidate_id=candidate_id,
                stage=stage,
                input_hash=f"{index + 1:x}" * 64,
                status="passed",
                metrics={"stage": stage},
                created_at=CREATED_AT,
            )
        )
        for index, stage in enumerate(V3_REQUIRED_EVIDENCE_STAGES)
    )


def _material() -> ResearchMaterialInput:
    return ResearchMaterialInput(
        relative_path="reports/report.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="a" * 64,
        parser_version="pypdf-6.21.2.page-v1",
    )


def _research_input(material: ResearchMaterialInput) -> ResearchRunInput:
    return ResearchRunInput(
        run_id="searchrun_222222222222222222222222",
        idempotency_key="v3-report-input-0001",
        material_path=".",
        manifest=ResearchMaterialManifest(materials=(material,)),
        limits=ResearchConversionLimits(max_documents=1),
    )


def _provenance(candidate: CandidateDefinition) -> ResearchCandidateProvenance:
    material = _material()
    document_id = material.document_id
    classification = ResearchMaterialClassification(
        document_id=document_id,
        text_sha256=material.sha256,
        kind="financial_research",
        rule_version=RESEARCH_CLASSIFICATION_RULE_VERSION,
        report_signals=("equity research",),
        finance_signals=("factor", "stock"),
        created_at=CREATED_AT,
    )
    quote = "价格趋势因子"
    citation = ResearchExtractionCitation(
        document_id=document_id,
        page_number=1,
        start_offset=0,
        end_offset=len(quote),
        quote=quote,
    )
    values = {
        "factor_name": ("价格趋势",),
        "economic_mechanism": ("价格延续",),
        "formula": ("rank(mean(close, 20))",),
        "variables": ("close",),
        "expected_direction": ("positive",),
        "falsification_conditions": ("RankIC 低于门槛",),
    }
    fields = tuple(
        ResearchFieldClaim(
            field_name=name,
            values=values[name],
            status="explicit",
            citations=(citation,),
        )
        for name in RESEARCH_FACTOR_FIELDS
    )
    hypothesis = HypothesisCard(
        run_id="searchrun_222222222222222222222222",
        turn_index=1,
        branch_id="branch-0001",
        statement="价格趋势可能带来未来超额收益",
        mechanism="价格延续",
        expected_direction="positive",
        falsification_criteria=("RankIC 低于门槛",),
        metadata={"source": "research_report"},
    )
    return ResearchCandidateProvenance(
        run_id=hypothesis.run_id,
        candidate_id=candidate.candidate_id,
        document_id=document_id,
        extraction_id="extract_000000000000000000000001",
        extraction_hash="b" * 64,
        draft_id="draft_000000000000000000000001",
        factor_hash="c" * 64,
        material=material,
        classification=classification,
        fields=fields,
        checks=("complete",),
        field_mapping={"close": "close"},
        catalog_hash="d" * 64,
        converter_model="fixture-model",
        conversion_prompt_version="fixture-conversion-v1",
        extraction_model="fixture-model",
        extraction_prompt_version="fixture-extraction-v1",
        hypothesis=hypothesis,
        candidate=candidate,
    )


def _free_direction(request: SearchRunRequest) -> ResearchDirection:
    return ResearchDirection(
        run_id=request.run_id,
        mechanism_category="price",
        statement="价格趋势可能带来未来超额收益",
        mechanism="价格延续",
        expected_direction="positive",
        falsification_criteria=("RankIC 低于门槛",),
        data_dependencies=("close",),
        expression_complexity="simple",
        max_expression_depth=2,
        seed=request.effective_exploration_seed,
        seed_source="request_hash",
        request_hash=request.request_hash,
        context_hash="e" * 64,
        profile_hash=FREE_EXPLORATION_PROFILE_HASH,
    )


def _promotion() -> PromotionRequest:
    return PromotionRequest(
        sandbox_run_id="sandbox_000000000000000000000001",
        artifact_manifest_hash="1" * 64,
        reviewer="reviewer-a",
        reason="人工转写为受控 DSL 并重新评估",
        target_type="dsl",
        target_reference="rank(mean(close, 20))",
    )


def test_source_envelopes_are_stable_and_mode_specific():
    hypothesis = SearchRunRequest(
        "v3-hypothesis-0001",
        "价格趋势可能带来未来超额收益",
    )
    free = SearchRunRequest(
        "v3-free-0001",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
        exploration_seed=7,
    )

    first = hypothesis_input_envelope(hypothesis, context_hash="e" * 64)
    second = hypothesis_input_envelope(hypothesis, context_hash="e" * 64)
    free_envelope = free_exploration_input_envelope(
        free,
        context_hash="e" * 64,
    )
    direction = _free_direction(free)
    direction_source = free_exploration_candidate_source(free, direction)

    assert first == second
    assert first.source.source_kind == "hypothesis"
    assert free_envelope.source.source_kind == "free_exploration"
    assert direction_source.source_hash != free_envelope.source.source_hash
    assert direction_source.source_kind == "free_exploration"

    with pytest.raises(MiningContractError, match="只有 hypothesis"):
        hypothesis_input_envelope(free, context_hash="e" * 64)
    with pytest.raises(MiningContractError, match="只有 free_exploration"):
        free_exploration_input_envelope(
            hypothesis,
            context_hash="e" * 64,
        )


def test_research_and_costeer_sources_keep_required_hashes_only():
    material = _material()
    research_input = _research_input(material)
    report = research_report_input_envelope(
        research_input,
        context_hash="e" * 64,
    )
    promotion = _promotion()
    costeer = costeer_input_envelope(
        promotion,
        context_hash="e" * 64,
    )

    assert report.source.source_kind == "research_report"
    assert report.source.artifact_manifest_hash is None
    assert report.source.promotion_hash is None
    assert report.request_hash == research_input.input_hash
    assert costeer.source.source_kind == "costeer"
    assert costeer.source.source_id == promotion.promotion_id
    assert costeer.source.artifact_manifest_hash == promotion.artifact_manifest_hash
    assert costeer.source.promotion_hash is not None


def test_research_candidate_strips_v2_source_parameters_without_changing_identity():
    base = _candidate()
    provisional = _provenance(base)
    enriched = replace(
        base,
        parameters={
            "source_kind": "research_report",
            "research_provenance_hash": provisional.provenance_hash,
        },
    )
    provenance = replace(provisional, candidate=enriched)

    restored = source_independent_research_candidate(provenance)
    source = research_report_candidate_source(provenance)

    assert restored == base
    assert restored.candidate_id == base.candidate_id
    assert provenance.candidate.candidate_id != restored.candidate_id
    assert source.source_hash == provenance.provenance_hash
    assert source.source_kind == "research_report"
    envelope = CandidateEnvelope(candidate=restored, source=source)
    assert (
        validated_research_report_provenance(provenance, envelope)
        == provenance
    )
    clean = replace(provenance, candidate=base)
    assert (
        validated_research_report_provenance(
            clean,
            CandidateEnvelope(candidate=base, source=source),
        )
        == clean
    )

    mismatched = replace(
        provenance,
        candidate=replace(
            enriched,
            parameters={
                "source_kind": "research_report",
                "research_provenance_hash": "f" * 64,
            },
        ),
    )
    with pytest.raises(MiningContractError, match="provenance"):
        source_independent_research_candidate(mismatched)
    with pytest.raises(MiningContractError, match="内容哈希"):
        validated_research_report_provenance(
            provenance,
            CandidateEnvelope(
                candidate=replace(restored, description="被替换的候选"),
                source=source,
            ),
        )
    other_source = ResearchSourceReference(
        source_kind="research_report",
        source_id="v3src_" + "9" * 24,
        source_hash="9" * 64,
    )
    with pytest.raises(MiningContractError, match="来源引用"):
        validated_research_report_provenance(
            provenance,
            CandidateEnvelope(candidate=restored, source=other_source),
        )


def test_evidence_adapter_only_accepts_five_v3_stages():
    candidate_id = "cand_" + "1" * 24
    for index, stage in enumerate(V3_REQUIRED_EVIDENCE_STAGES):
        reference = evidence_reference_from_search(
            SearchEvaluationEvidence(
                run_id="searchrun_111111111111111111111111",
                candidate_id=candidate_id,
                stage=stage,
                input_hash=f"{index + 1:x}" * 64,
                status="passed",
                metrics={},
                created_at=CREATED_AT,
            )
        )
        assert reference.stage == stage

    with pytest.raises(MiningContractError, match="五阶段"):
        evidence_reference_from_search(
            SearchEvaluationEvidence(
                run_id="searchrun_111111111111111111111111",
                candidate_id=candidate_id,
                stage="selection_diagnostics",
                input_hash="f" * 64,
                status="passed",
                metrics={},
                created_at=CREATED_AT,
            )
        )


def test_same_candidate_changes_source_and_approval_hash_only():
    candidate = _candidate()
    hypothesis_source = ResearchSourceReference(
        source_kind="hypothesis",
        source_id="source_hypothesis",
        source_hash="a" * 64,
    )
    report_source = ResearchSourceReference(
        source_kind="research_report",
        source_id="source_report",
        source_hash="b" * 64,
    )
    evidence = _evidence(candidate.candidate_id)

    hypothesis = CandidateEnvelope(
        candidate=candidate,
        source=hypothesis_source,
        evidence=evidence,
    )
    report = CandidateEnvelope(
        candidate=candidate,
        source=report_source,
        evidence=evidence,
    )

    assert hypothesis.candidate.candidate_id == report.candidate.candidate_id
    assert hypothesis.candidate.content_hash == report.candidate.content_hash
    assert hypothesis.envelope_hash != report.envelope_hash
    assert hypothesis.approval_evidence_hash(
        "e" * 64
    ) != report.approval_evidence_hash("e" * 64)
    assert sha256_json(candidate.content_material()) == candidate.content_hash
