"""Worker v2 搜索协议契约测试。"""

from __future__ import annotations

import pytest

from mining_worker.protocol import WorkerCandidateDraft, WorkerProtocolError
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchCandidate,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)


def _hypothesis(**overrides):
    payload = {
        "statement": "短期价格反转可能带来开发期超额收益",
        "mechanism": "短期过度反应后的价格修复",
        "expected_direction": "positive",
        "falsification_criteria": ["年度同号率低于 0.6"],
        "parent_hypothesis_id": None,
    }
    payload.update(overrides)
    return payload


def _lineage(**overrides):
    payload = {
        "candidate_id": "cand_0001",
        "name": "MINE_REVERSAL_0001",
        "expression": "rank(mean(ret,5))",
        "status": "screening",
        "parent_candidate_id": None,
    }
    payload.update(overrides)
    return payload


def _evidence(**overrides):
    payload = {
        "schema_version": "mining_v2",
        "run_id": "searchrun_12345678",
        "hypothesis_id": "hyp_12345678",
        "batch_id": "batch_12345678",
        "turn_index": 1,
        "development_start": "2012-01-01",
        "development_end": "2019-12-31",
        "metrics": {"cand_0001": {"rank_ic": 0.03}},
        "failures": [],
        "correlation_clusters": [],
        "trial_counts": {
            "proposed": 1,
            "rejected": 0,
            "duplicate": 0,
            "invalid": 0,
            "gate_failed": 0,
            "evaluated": 1,
            "full_evaluated": 0,
            "holdout_claimed": 0,
            "invalidated": 0,
        },
        "budget_state": {"turns_used": 1},
        "created_at": "2026-10-29T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _request_payload(**overrides):
    payload = {
        "schema_version": "mining_worker_v2",
        "request_id": "search-turn-1",
        "run_id": "searchrun_12345678",
        "turn_index": 1,
        "branch_id": "branch-0001",
        "context_hash": "a" * 64,
        "hypothesis": _hypothesis(),
        "lineage": [_lineage()],
        "evidence": _evidence(),
        "budget": {
            "max_search_turns": 6,
            "max_hypothesis_branches": 3,
            "max_refines_per_branch": 3,
            "max_candidates_per_turn": 4,
            "max_candidates_per_run": 24,
            "max_full_evaluations": 4,
            "stop_after_no_improvement_turns": 2,
            "max_tool_calls_per_turn": 4,
            "schema_version": "mining_search_v2",
        },
        "allowed_actions": ["refine", "branch", "stop", "promote"],
        "llm_model": "deepseek/deepseek-chat",
        "max_candidates": 2,
        "max_expression_depth": 6,
        "excluded_fields": [],
        "fields": [{"name": "ret", "dtype": "number"}],
        "operators": [{"name": "rank", "arity": 1}],
    }
    payload.update(overrides)
    return payload


def _response_payload(**overrides):
    payload = {
        "schema_version": "mining_worker_v2",
        "decision": {
            "action": "refine",
            "reason": "反转信号方向一致，但覆盖率仍需改善",
        },
        "next_hypothesis": _hypothesis(
            statement="更短窗口的短期反转可能改善覆盖率",
        ),
        "candidates": [
            {
                "expression": "rank(mean(ret,3))",
                "mechanism": "更短窗口的过度反应",
                "description": "三日收益排名",
                "direction": 1,
                "steps": ["计算三日收益", "按交易日排名"],
            }
        ],
        "usage": {"calls": 1, "candidates": 1},
    }
    payload.update(overrides)
    return payload


def test_worker_search_request_round_trip_and_stable_identity():
    request = WorkerSearchRequest.from_dict(_request_payload())

    assert request.hypothesis.statement.startswith("短期价格反转")
    assert request.lineage[0].candidate_id == "cand_0001"
    assert WorkerSearchRequest.from_dict(request.to_dict()) == request
    assert request.job_id == WorkerSearchRequest.from_dict(request.to_dict()).job_id
    assert request.payload_hash == WorkerSearchRequest.from_dict(
        request.to_dict()
    ).payload_hash


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"unknown": True}, "字段不匹配"),
        ({"turn_index": 0}, "turn_index"),
        ({"allowed_actions": ["search"]}, "allowed_actions"),
        ({"max_candidates": 5}, "max_candidates"),
        ({"lineage": [_lineage(status="selection")]}, "受控标识"),
        (
            {
                "evidence": _evidence(
                    development_start="2020-01-01",
                )
            },
            "非开发期日期",
        ),
    ],
)
def test_worker_search_request_rejects_unsafe_or_unbounded_input(overrides, message):
    with pytest.raises(WorkerProtocolError, match=message):
        WorkerSearchRequest.from_dict(_request_payload(**overrides))


def test_worker_search_response_round_trip_and_action_rules():
    response = WorkerSearchResponse.from_dict(_response_payload())

    assert response.decision.action == "refine"
    assert response.next_hypothesis is not None
    assert response.candidates[0].expression == "rank(mean(ret,3))"
    assert WorkerSearchResponse.from_dict(response.to_dict()) == response

    with pytest.raises(WorkerProtocolError, match="必须提供下一假设"):
        WorkerSearchResponse.from_dict(
            _response_payload(next_hypothesis=None)
        )
    with pytest.raises(WorkerProtocolError, match="不得提供"):
        WorkerSearchResponse.from_dict(
            _response_payload(
                decision={"action": "stop", "reason": "没有可继续验证的候选"},
            )
        )


def test_nested_models_reject_unknown_fields_and_private_windows():
    with pytest.raises(WorkerProtocolError, match="字段不匹配"):
        WorkerSearchHypothesis.from_dict({**_hypothesis(), "extra": 1})

    with pytest.raises(WorkerProtocolError, match="受控标识"):
        WorkerSearchCandidate.from_dict(_lineage(parent_candidate_id="holdout_1"))

    with pytest.raises(WorkerProtocolError, match="未知动作"):
        WorkerDecisionProposal.from_dict(
            {"action": "publish", "reason": "不能自动发布"}
        )
