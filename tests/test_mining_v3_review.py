"""V3 冻结上下文、审核资格和单一来源审批测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mining.contract_evidence import ApprovalDecision
from mining.errors import ApprovalConflictError, MiningContractError
from mining.storage_v3 import MiningV3Store
from mining.v3_contracts import CandidateEnvelope, EvidenceReference
from tests.mining_v3_fixtures import (
    CREATED_AT,
    approved_v3_run,
    make_v3_candidate_review_ready,
    seed_v3_runtime_state,
    v3_candidate_envelope,
    v3_context,
    v3_input,
    v3_source,
)


def _store(tmp_path: Path) -> MiningV3Store:
    return MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )


def _decision(
    review,
    *,
    decision: str = "approved",
    evidence_hash: str | None = None,
) -> ApprovalDecision:
    return ApprovalDecision(
        run_id=review.run_id,
        candidate_id=review.candidate_id,
        decision=decision,
        reviewer="reviewer-a",
        note="统一 V3 人工审核",
        candidate_content_hash=review.candidate_content_hash,
        semantic_hash=review.semantic_hash,
        context_hash=review.context_hash,
        evidence_hash=evidence_hash or review.approval_evidence_hash,
        created_at=CREATED_AT,
    )


@pytest.mark.parametrize(
    "source_kind",
    ("hypothesis", "free_exploration", "research_report", "costeer"),
)
def test_every_v3_source_uses_same_review_and_approval(
    tmp_path: Path,
    source_kind: str,
):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input(source_kind, context)
    envelope = v3_candidate_envelope(input_envelope)
    store.register_input(input_envelope)
    store.register_context(input_envelope.run_id, context)
    store.record_candidate_envelope(input_envelope.run_id, envelope)
    seed_v3_runtime_state(store, input_envelope, context)
    make_v3_candidate_review_ready(
        store,
        input_envelope.run_id,
        envelope,
    )

    review = store.candidate_review(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )
    assert review.eligible is True
    assert review.blocking_reasons == ()
    assert review.approval_evidence_hash is not None
    assert review.source_reference_hash == envelope.source.reference_hash
    assert store.list_candidate_reviews(input_envelope.run_id) == (review,)

    approval = store.record_approval(
        _decision(review),
        envelope.source.source_id,
    )

    assert approval["decision"] == "approved"
    assert approval["source_id"] == envelope.source.source_id
    assert approval["candidate_envelope_hash"] == envelope.envelope_hash
    assert store.get_candidate(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )["status"] == "approved"
    assert store.approval_is_current(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )
    event_types = {
        item["event_type"] for item in store.list_events(input_envelope.run_id)
    }
    assert {
        "v3_context_registered",
        "v3_candidate_approved",
    } <= event_types


def test_review_blocks_missing_context_and_failed_stage(tmp_path: Path):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input("hypothesis", context)
    missing_context = v3_candidate_envelope(
        input_envelope,
        name="V3_REVIEW_MISSING_CONTEXT",
    )
    failed = v3_candidate_envelope(
        input_envelope,
        name="V3_REVIEW_FAILED_STAGE",
        failed_stage="selection",
    )
    store.register_input(input_envelope)
    store.record_candidate_envelope(input_envelope.run_id, missing_context)
    store.record_candidate_envelope(input_envelope.run_id, failed)

    missing_review = store.candidate_review(
        input_envelope.run_id,
        missing_context.candidate.candidate_id,
        missing_context.source.source_id,
    )
    failed_review = store.candidate_review(
        input_envelope.run_id,
        failed.candidate.candidate_id,
        failed.source.source_id,
    )

    assert missing_review.eligible is False
    assert "CONTEXT_NOT_REGISTERED" in missing_review.blocking_reasons
    assert missing_review.approval_evidence_hash is None
    assert failed_review.eligible is False
    assert "EVIDENCE_NOT_PASSED:selection" in failed_review.blocking_reasons

    with pytest.raises(ApprovalConflictError, match="CONTEXT_NOT_REGISTERED"):
        store.record_approval(
            _decision(missing_review, evidence_hash="a" * 64),
            missing_context.source.source_id,
        )


def test_context_registration_is_idempotent_and_rejects_different_value(
    tmp_path: Path,
):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input("hypothesis", context)
    store.register_input(input_envelope)

    first = store.register_context(input_envelope.run_id, context)
    second = store.register_context(input_envelope.run_id, context)

    assert first["reused"] is False
    assert second["reused"] is True
    assert store.load_context(input_envelope.run_id) == context
    with pytest.raises(MiningContractError, match="上下文哈希不一致"):
        store.register_context(
            input_envelope.run_id,
            replace(context, data_fingerprint="b" * 64),
        )


def test_approval_is_idempotent_and_evidence_change_invalidates_it(
    tmp_path: Path,
):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    stored = store.get_approval(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )
    decision = ApprovalDecision(
        run_id=stored["run_id"],
        candidate_id=stored["candidate_id"],
        decision=stored["decision"],
        reviewer=stored["reviewer"],
        note=stored["note"],
        candidate_content_hash=stored["candidate_content_hash"],
        semantic_hash=stored["semantic_hash"],
        context_hash=stored["context_hash"],
        evidence_hash=stored["evidence_hash"],
        created_at=stored["created_at"],
    )

    repeated = store.record_approval(decision, envelope.source.source_id)
    assert repeated["reused"] is True

    extra = EvidenceReference(
        candidate_id=envelope.candidate.candidate_id,
        stage="full",
        evaluation_id="eval_full_recheck",
        input_hash="f" * 64,
        evidence_hash="e" * 64,
        status="passed",
    )
    store.record_candidate_envelope(
        input_envelope.run_id,
        CandidateEnvelope(
            candidate=envelope.candidate,
            source=envelope.source,
            evidence=(*envelope.evidence, extra),
        ),
    )

    assert store.approval_is_current(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    ) is False
    changed_review = store.candidate_review(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )
    assert "FINAL_DECISION_EXISTS" in changed_review.blocking_reasons


def test_one_final_approval_selects_one_of_multiple_sources(tmp_path: Path):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input("hypothesis", context)
    primary = v3_candidate_envelope(input_envelope)
    second_source = v3_source("hypothesis", suffix="secondary")
    secondary = CandidateEnvelope(
        candidate=primary.candidate,
        source=second_source,
        evidence=primary.evidence,
    )
    store.register_input(input_envelope)
    store.register_context(input_envelope.run_id, context)
    store.record_candidate_envelope(input_envelope.run_id, primary)
    store.record_candidate_envelope(input_envelope.run_id, secondary)
    seed_v3_runtime_state(store, input_envelope, context)
    make_v3_candidate_review_ready(
        store,
        input_envelope.run_id,
        primary,
    )
    first_review = store.candidate_review(
        input_envelope.run_id,
        primary.candidate.candidate_id,
        primary.source.source_id,
    )
    store.record_approval(
        _decision(first_review),
        primary.source.source_id,
    )

    second_review = store.candidate_review(
        input_envelope.run_id,
        primary.candidate.candidate_id,
        second_source.source_id,
    )
    assert "FINAL_DECISION_EXISTS" in second_review.blocking_reasons
    with pytest.raises(ApprovalConflictError, match="不同"):
        store.record_approval(
            _decision(second_review),
            second_source.source_id,
        )


@pytest.mark.parametrize("run_status", ("failed", "interrupted"))
def test_review_and_approval_require_review_ready_run(
    tmp_path: Path,
    run_status: str,
):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input("hypothesis", context)
    envelope = v3_candidate_envelope(input_envelope)
    store.register_input(input_envelope)
    store.register_context(input_envelope.run_id, context)
    store.record_candidate_envelope(input_envelope.run_id, envelope)
    seed_v3_runtime_state(
        store,
        input_envelope,
        context,
        run_status=run_status,
    )
    make_v3_candidate_review_ready(
        store,
        input_envelope.run_id,
        envelope,
    )

    review = store.candidate_review(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )

    assert review.eligible is False
    assert f"RUN_NOT_REVIEW_READY:{run_status}" in review.blocking_reasons
    with pytest.raises(
        ApprovalConflictError,
        match=f"RUN_NOT_REVIEW_READY:{run_status}",
    ):
        store.record_approval(
            _decision(review, evidence_hash="a" * 64),
            envelope.source.source_id,
        )


def test_review_and_approval_require_review_ready_candidate(tmp_path: Path):
    store = _store(tmp_path)
    context = v3_context()
    input_envelope = v3_input("hypothesis", context)
    envelope = v3_candidate_envelope(input_envelope)
    store.register_input(input_envelope)
    store.register_context(input_envelope.run_id, context)
    store.record_candidate_envelope(input_envelope.run_id, envelope)
    seed_v3_runtime_state(store, input_envelope, context)
    store.initialize_candidate_state(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
        status="proposed",
    )

    review = store.candidate_review(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )

    assert review.eligible is False
    assert "CANDIDATE_NOT_REVIEW_READY:proposed" in review.blocking_reasons
    with pytest.raises(
        ApprovalConflictError,
        match="CANDIDATE_NOT_REVIEW_READY:proposed",
    ):
        store.record_approval(
            _decision(review, evidence_hash="a" * 64),
            envelope.source.source_id,
        )
