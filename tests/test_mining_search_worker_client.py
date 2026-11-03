"""Worker v2 客户端和主服务二次校验测试。"""

from __future__ import annotations

from mining.search_worker_client import (
    MockSearchWorkerClient,
    generate_search_candidates,
)
from mining_worker.protocol import WorkerCandidateDraft
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)


def _request() -> WorkerSearchRequest:
    return WorkerSearchRequest(
        request_id="search-round-candidate",
        run_id="searchrun_12345678",
        turn_index=1,
        branch_id="branch-0001",
        context_hash="a" * 64,
        hypothesis=WorkerSearchHypothesis(
            statement="短期价格反转可能带来开发期超额收益",
            mechanism="短期过度反应后的价格修复",
            expected_direction="positive",
            falsification_criteria=("年度同号率低于 0.6",),
        ),
        lineage=(),
        evidence={
            "schema_version": "mining_v2",
            "run_id": "searchrun_12345678",
            "hypothesis_id": "hyp_12345678",
            "batch_id": "batch_12345678",
            "turn_index": 1,
            "development_start": "2012-01-01",
            "development_end": "2019-12-31",
            "metrics": {},
            "failures": [],
            "correlation_clusters": [],
            "trial_counts": {
                "proposed": 0,
                "rejected": 0,
                "duplicate": 0,
                "invalid": 0,
                "gate_failed": 0,
                "evaluated": 0,
                "full_evaluated": 0,
                "holdout_claimed": 0,
                "invalidated": 0,
            },
            "budget_state": {},
            "created_at": "2026-10-29T00:00:00+00:00",
        },
        budget={
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
        allowed_actions=("refine", "branch", "stop", "promote"),
        llm_model="deepseek/deepseek-chat",
        max_candidates=2,
        max_expression_depth=6,
        excluded_fields=(),
        fields=({"name": "close"},),
        operators=({"name": "mean"},),
    )


def _response(*expressions: str) -> WorkerSearchResponse:
    return WorkerSearchResponse(
        decision=WorkerDecisionProposal(
            action="refine",
            reason="开发期反馈仍需继续修订",
        ),
        next_hypothesis=WorkerSearchHypothesis(
            statement="修正窗口后可能改善短期反转信号",
            mechanism="降低过慢窗口造成的信号衰减",
            expected_direction="positive",
            falsification_criteria=("连续两轮没有改善",),
        ),
        candidates=tuple(
            WorkerCandidateDraft(
                expression=expression,
                mechanism="短期反转",
                description="短期价格统计",
                direction=1,
                steps=("读取价格", "计算受控表达式"),
            )
            for expression in expressions
        ),
        usage={"calls": 1},
    )


def test_main_service_revalidates_worker_v2_candidates():
    client = MockSearchWorkerClient(
        _response("mean(close,5)", "SELECT * FROM daily")
    )
    request = _request()

    result = generate_search_candidates(client, request)

    assert len(result.accepted) == 1
    assert result.accepted[0].canonical_expression == "mean(close, 5)"
    assert result.accepted[0].discovery_round == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].error_code == "MINING-114"
    assert result.decision.action == "refine"
    assert result.next_hypothesis is not None
    assert client.requests == [request]


def test_main_service_counts_duplicate_drafts_as_rejected():
    response = _response("mean(close,5)", "mean(close,5)")
    client = MockSearchWorkerClient(response)

    result = generate_search_candidates(client, _request())

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].error_code == "MINING-114"
