"""开发期脱敏证据包构建测试。"""

from __future__ import annotations

import pytest

from mining.errors import MiningContractError
from mining.search_budget import TrialCounts
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evidence_builder import (
    DevelopmentCandidateEvidence,
    build_development_evidence,
)


def _card() -> HypothesisCard:
    return HypothesisCard(
        run_id="searchrun_12345678",
        turn_index=1,
        branch_id="branch-0001",
        statement="短期价格反转可能带来开发期超额收益",
        mechanism="短期过度反应后的价格修复",
        expected_direction="positive",
        falsification_criteria=("年度同号率低于 0.6",),
    )


def _batch(card: HypothesisCard, candidate_ids=("cand_0001",)) -> ExperimentBatch:
    return ExperimentBatch(
        run_id=card.run_id,
        hypothesis_id=card.hypothesis_id,
        turn_index=card.turn_index,
        branch_id=card.branch_id,
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )


def _candidate(**overrides) -> DevelopmentCandidateEvidence:
    payload = {
        "candidate_id": "cand_0001",
        "technical_status": "passed",
        "quick_status": "passed",
        "gate_passed": True,
        "rank_ic": 0.03,
        "rank_ic_ir": 0.4,
        "year_sign_consistency": 0.75,
        "coverage": 0.8,
        "valid_days": 100,
    }
    payload.update(overrides)
    return DevelopmentCandidateEvidence(**payload)


def _budget() -> dict:
    return {
        "turns_used": 1,
        "branches_used": 1,
        "refines_in_branch": 0,
        "candidates_proposed": 1,
        "candidates_evaluated": 1,
        "full_evaluated": 0,
        "no_improvement_turns": 0,
        "remaining_candidates": 23,
        "cancelled": False,
        "snapshot_stale": False,
    }


def test_builder_preserves_missing_metrics_and_stable_hash():
    card = _card()
    batch = _batch(card)
    candidate = _candidate(turnover=None, ic_decay=None, valid_days=None)

    first = build_development_evidence(
        hypothesis=card,
        batch=batch,
        candidates=(candidate,),
        trial_counts=TrialCounts(proposed=1, evaluated=1),
        budget_state=_budget(),
    )
    second = build_development_evidence(
        hypothesis=card,
        batch=batch,
        candidates=(candidate,),
        trial_counts=TrialCounts(proposed=1, evaluated=1),
        budget_state=_budget(),
    )

    assert first.metrics["cand_0001"]["rank_ic"] == 0.03
    assert first.metrics["cand_0001"]["turnover"] is None
    assert first.metrics["cand_0001"]["ic_decay"] is None
    assert first.metrics["cand_0001"]["valid_days"] is None
    assert first.evidence_hash == second.evidence_hash
    assert first.evidence_id == second.evidence_id


def test_builder_input_changes_hash_and_failures_are_sanitized(caplog):
    card = _card()
    batch = _batch(card)
    failed = _candidate(
        quick_status="failed",
        rank_ic=None,
        rank_ic_ir=None,
        year_sign_consistency=None,
        failure_stage="quick",
        error_code="MINING-106",
        error_message="读取 /home/private/data.parquet 失败 sk-abcdefghijkl",
    )
    evidence = build_development_evidence(
        hypothesis=card,
        batch=batch,
        candidates=(failed,),
        trial_counts=TrialCounts(proposed=1, rejected=1),
        budget_state=_budget(),
    )

    assert evidence.failures[0]["stage"] == "quick"
    assert evidence.failures[0]["error_code"] == "MINING-106"
    assert "/home/private" not in evidence.failures[0]["reason"]
    assert "sk-abcdefghijkl" not in evidence.failures[0]["reason"]
    assert "development_evidence_built" in caplog.text
    assert "sk-abcdefghijkl" not in caplog.text
    assert "rank_ic=" not in caplog.text


def test_builder_rejects_candidates_outside_or_missing_from_batch():
    card = _card()
    batch = _batch(card, ("cand_0001", "cand_0002"))

    with pytest.raises(MiningContractError, match="缺少候选证据"):
        build_development_evidence(
            hypothesis=card,
            batch=batch,
            candidates=(_candidate(),),
            trial_counts=TrialCounts(),
            budget_state=_budget(),
        )

    with pytest.raises(MiningContractError, match="不属于当前批次"):
        build_development_evidence(
            hypothesis=card,
            batch=_batch(card),
            candidates=(_candidate(candidate_id="cand_9999"),),
            trial_counts=TrialCounts(),
            budget_state=_budget(),
        )


def test_from_reports_marks_missing_and_failed_stages():
    evidence = DevelopmentCandidateEvidence.from_reports(
        "cand_0001",
        gate={"passed": False},
        technical={
            "status": "failed",
            "metrics": {"values": {"coverage": 0.2}},
            "error_code": "MINING-106",
            "error_message": "技术门禁失败",
        },
        quick=None,
    )

    assert evidence.gate_passed is False
    assert evidence.failure_stage == "technical"
    assert evidence.quick_status == "not_run"
    assert evidence.rank_ic is None
