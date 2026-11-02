"""V3 统一研究来源、候选、证据和状态门面契约测试。"""

from __future__ import annotations

import pytest

from mining import (
    MINING_V3_SCHEMA_VERSION,
    V3_CANDIDATE_STATUSES,
    V3_PUBLICATION_SOURCE_KIND,
    V3_REQUIRED_EVIDENCE_STAGES,
    V3_RESEARCH_INPUT_MODES,
    V3_RESEARCH_RUN_STATUSES,
    V3_RUN_ID_PREFIX,
    CandidateDefinition,
    CandidateEnvelope,
    EvidenceReference,
    MiningContractError,
    MiningStateError,
    ResearchInputEnvelope,
    ResearchSourceReference,
    V3ResearchStateMachine,
)
from mining.search_states import SEARCH_RUN_STATUSES
from mining.states import CANDIDATE_STATUSES, PUBLICATION_STATUSES


def _candidate() -> CandidateDefinition:
    return CandidateDefinition(
        name="V3_CONTRACT_FACTOR",
        mechanism="价格趋势",
        description="用于验证 V3 候选包络的来源无关身份",
        canonical_expression="rank(mean(close, 20))",
        direction=1,
        lookback=20,
        dependencies=("close",),
        output_type="score",
        steps=("计算二十日价格均值", "按交易日做截面排名"),
        discovery_round=1,
        discovery_index=1,
    )


def _source(
    kind: str,
    *,
    digest: str = "a",
    source_id: str = "source_0001",
) -> ResearchSourceReference:
    sandbox = kind == "costeer"
    return ResearchSourceReference(
        source_kind=kind,
        source_id=source_id,
        source_hash=digest * 64,
        artifact_manifest_hash="b" * 64 if sandbox else None,
        promotion_hash="c" * 64 if sandbox else None,
    )


def _evidence(
    candidate_id: str,
    *,
    digest: str = "d",
) -> tuple[EvidenceReference, ...]:
    return tuple(
        EvidenceReference(
            candidate_id=candidate_id,
            stage=stage,
            evaluation_id=f"eval_{stage}",
            input_hash=f"{index + 1:x}" * 64,
            evidence_hash=digest * 64,
            status="passed",
        )
        for index, stage in enumerate(V3_REQUIRED_EVIDENCE_STAGES)
    )


def _input_envelope(kind: str) -> ResearchInputEnvelope:
    return ResearchInputEnvelope(
        run_id=f"{V3_RUN_ID_PREFIX}_{'1' * 24}",
        idempotency_key="v3-contract-0001",
        input_mode=kind,
        context_hash="e" * 64,
        request_hash="f" * 64,
        source=_source(kind),
    )


def test_v3_versions_and_public_modes_are_frozen():
    assert MINING_V3_SCHEMA_VERSION == "mining_v3"
    assert V3_PUBLICATION_SOURCE_KIND == "v3_research"
    assert V3_RESEARCH_INPUT_MODES == (
        "hypothesis",
        "free_exploration",
        "research_report",
        "costeer",
    )
    assert V3_RESEARCH_RUN_STATUSES == SEARCH_RUN_STATUSES
    assert V3_CANDIDATE_STATUSES == CANDIDATE_STATUSES


@pytest.mark.parametrize("kind", V3_RESEARCH_INPUT_MODES)
def test_research_input_envelopes_round_trip_for_all_sources(kind):
    envelope = _input_envelope(kind)

    restored = ResearchInputEnvelope.from_dict(envelope.to_dict())

    assert restored == envelope
    assert restored.input_hash == envelope.input_hash
    assert restored.source.reference_hash == envelope.source.reference_hash


def test_research_input_rejects_mode_source_mismatch_and_unknown_fields():
    envelope = _input_envelope("hypothesis")
    mismatched = envelope.to_dict()
    mismatched["input_mode"] = "free_exploration"
    with pytest.raises(MiningContractError, match="来源类型一致"):
        ResearchInputEnvelope.from_dict(mismatched)

    unknown = _input_envelope("costeer").to_dict()
    unknown["raw_prompt"] = "不能持久化"
    with pytest.raises(MiningContractError, match="未知"):
        ResearchInputEnvelope.from_dict(unknown)


def test_costeer_source_requires_artifact_and_promotion_hashes():
    with pytest.raises(MiningContractError, match="产物清单"):
        ResearchSourceReference(
            source_kind="costeer",
            source_id="sandbox_0001",
            source_hash="a" * 64,
        )

    with pytest.raises(MiningContractError, match="非 costeer"):
        ResearchSourceReference(
            source_kind="research_report",
            source_id="report_0001",
            source_hash="a" * 64,
            artifact_manifest_hash="b" * 64,
        )


def test_source_changes_envelope_hash_but_not_candidate_identity():
    candidate = _candidate()
    first = CandidateEnvelope(
        candidate=candidate,
        source=_source("hypothesis", digest="a"),
        evidence=_evidence(candidate.candidate_id),
    )
    second = CandidateEnvelope(
        candidate=candidate,
        source=_source("research_report", digest="b"),
        evidence=_evidence(candidate.candidate_id),
    )

    assert first.candidate.candidate_id == second.candidate.candidate_id
    assert first.candidate.content_hash == second.candidate.content_hash
    assert first.source.reference_hash != second.source.reference_hash
    assert first.envelope_hash != second.envelope_hash
    assert first.approval_evidence_hash("e" * 64) != second.approval_evidence_hash(
        "e" * 64
    )


def test_candidate_envelope_round_trip_and_evidence_tamper_detection():
    candidate = _candidate()
    envelope = CandidateEnvelope(
        candidate=candidate,
        source=_source("costeer", source_id="sandbox_0001"),
        evidence=_evidence(candidate.candidate_id),
    )

    restored = CandidateEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope
    assert restored.approval_evidence_hash("e" * 64) == envelope.approval_evidence_hash(
        "e" * 64
    )

    tampered = envelope.to_dict()
    tampered["evidence"][-1]["evidence_hash"] = "0" * 64
    restored_tampered = CandidateEnvelope.from_dict(tampered)
    assert restored_tampered.approval_evidence_hash(
        "e" * 64
    ) != envelope.approval_evidence_hash("e" * 64)


def test_candidate_envelope_rejects_non_sequence_evidence():
    payload = CandidateEnvelope(
        candidate=_candidate(),
        source=_source("hypothesis"),
        evidence=_evidence(_candidate().candidate_id),
    ).to_dict()
    payload["evidence"] = "not-a-list"

    with pytest.raises(MiningContractError, match="证据引用序列"):
        CandidateEnvelope.from_dict(payload)


def test_candidate_envelope_requires_complete_passed_evidence_for_approval():
    candidate = _candidate()
    evidence = _evidence(candidate.candidate_id)[:-1]
    envelope = CandidateEnvelope(
        candidate=candidate,
        source=_source("hypothesis"),
        evidence=evidence,
    )

    with pytest.raises(MiningContractError, match="holdout"):
        envelope.approval_evidence_hash("e" * 64)


def test_candidate_envelope_rejects_evidence_for_another_candidate():
    candidate = _candidate()
    with pytest.raises(MiningContractError, match="当前候选"):
        CandidateEnvelope(
            candidate=candidate,
            source=_source("hypothesis"),
            evidence=_evidence("cand_another"),
        )


def test_v3_state_facade_delegates_to_existing_state_machines():
    V3ResearchStateMachine.ensure_run("queued", "searching")
    V3ResearchStateMachine.ensure_run(
        "interrupted",
        "evaluating",
        interrupted_from="evaluating",
    )
    V3ResearchStateMachine.ensure_candidate("review_ready", "approved")
    V3ResearchStateMachine.ensure_publication("staging", "validating")

    with pytest.raises(MiningStateError):
        V3ResearchStateMachine.ensure_run(
            "interrupted",
            "selecting",
            interrupted_from="evaluating",
        )
    with pytest.raises(MiningStateError):
        V3ResearchStateMachine.ensure_candidate("published", "approved")
    with pytest.raises(MiningStateError):
        V3ResearchStateMachine.ensure_publication("published", "failed")

    assert PUBLICATION_STATUSES == (
        "staging",
        "validating",
        "published",
        "failed",
        "rolled_back",
    )
