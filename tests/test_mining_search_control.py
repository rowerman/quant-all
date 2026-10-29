"""确定性搜索控制器测试。"""

from __future__ import annotations

import pytest

from mining.search_budget import SearchBudget, TrialCounts
from mining.search_control import (
    SearchControlPolicy,
    SearchControlState,
    decide_search_action,
)
from mining.search_evidence import EvidencePacket


def _metrics(**overrides):
    payload = {
        "gate_passed": True,
        "technical_status": "passed",
        "quick_status": "passed",
        "rank_ic": 0.03,
        "rank_ic_ir": 0.4,
        "year_sign_consistency": 0.75,
        "coverage": 0.8,
        "valid_days": 100,
        "sample_count": 1000,
        "turnover": 0.2,
        "ic_decay": 0.7,
    }
    payload.update(overrides)
    return payload


def _evidence(
    *,
    metrics=None,
    counts=None,
    turns_used=1,
) -> EvidencePacket:
    return EvidencePacket(
        run_id="searchrun_12345678",
        hypothesis_id="hyp_12345678",
        batch_id="batch_12345678",
        turn_index=1,
        development_start="2012-01-01",
        development_end="2019-12-31",
        metrics=metrics if metrics is not None else {"cand_0001": _metrics()},
        failures=(),
        correlation_clusters=(),
        trial_counts=counts or TrialCounts(proposed=1, evaluated=1),
        budget_state={"turns_used": turns_used},
    )


def test_controller_accepts_legal_proposal_and_is_deterministic():
    first = decide_search_action(
        _evidence(),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(proposed_action="refine"),
    )
    second = decide_search_action(
        _evidence(),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(proposed_action="refine"),
    )

    assert first.decision.action == "refine"
    assert first.quality_candidates == ("cand_0001",)
    assert first.decision.decision_id == second.decision.decision_id
    assert first.decision.reason == second.decision.reason


def test_controller_prioritizes_budget_over_worker_suggestion():
    result = decide_search_action(
        _evidence(turns_used=6),
        SearchBudget(max_search_turns=6),
        SearchControlPolicy(),
        SearchControlState(proposed_action="refine"),
    )

    assert result.decision.action == "promote"
    assert "预算耗尽" in result.decision.reason


def test_controller_branches_after_no_improvement_or_failure_rate():
    no_improvement = decide_search_action(
        _evidence(),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(
            no_improvement_turns=1,
            previous_best=(0.05, 0.5, 0.8, 0.9),
            proposed_action="refine",
        ),
    )
    assert no_improvement.decision.action == "branch"
    assert no_improvement.no_improvement_turns == 0
    assert "连续无改善=2" in no_improvement.decision.reason

    failures = decide_search_action(
        _evidence(
            metrics={
                "cand_0001": _metrics(gate_passed=False, rank_ic=None),
                "cand_0002": _metrics(gate_passed=False, rank_ic=None),
                "cand_0003": _metrics(gate_passed=False, rank_ic=None),
                "cand_0004": _metrics(gate_passed=False, rank_ic=None),
            },
            counts=TrialCounts(proposed=4, gate_failed=3),
        ),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(proposed_action="refine"),
    )
    assert failures.decision.action == "branch"
    assert failures.failure_rate == 0.75


def test_controller_stops_for_cancel_stale_or_exhausted_family():
    cancelled = decide_search_action(
        _evidence(),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(cancelled=True, proposed_action="refine"),
    )
    stale = decide_search_action(
        _evidence(),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(snapshot_stale=True, proposed_action="refine"),
    )
    no_family = decide_search_action(
        _evidence(metrics={"cand_0001": _metrics(rank_ic=None)}),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(
            branches_used=3,
            proposed_action="branch",
        ),
    )

    assert cancelled.decision.action == "stop"
    assert stale.decision.action == "stop"
    assert no_family.decision.action == "stop"


def test_controller_falls_back_when_proposal_is_illegal():
    result = decide_search_action(
        _evidence(metrics={"cand_0001": _metrics(rank_ic=0.01)}),
        SearchBudget(),
        SearchControlPolicy(),
        SearchControlState(proposed_action="promote"),
    )

    assert result.decision.action == "refine"
    assert result.quality_candidates == ()
    assert "不满足约束" in result.decision.reason


def test_control_policy_validation_and_hash():
    policy = SearchControlPolicy()

    assert SearchControlPolicy.from_dict(policy.to_dict()) == policy
    assert len(policy.policy_hash) == 64
    legacy = policy.to_dict()
    legacy.pop("correlation_cluster_threshold")
    assert (
        SearchControlPolicy.from_dict(legacy).correlation_cluster_threshold
        == 0.90
    )
    with pytest.raises(Exception, match="min_rank_ic"):
        SearchControlPolicy(min_rank_ic=-0.1)
