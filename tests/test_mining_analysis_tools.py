"""受控分析工具契约、网关与 Worker v3 补充流程测试。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from mining.analysis_tool_contracts import (
    AnalysisToolRequest,
    AnalysisToolResult,
    tool_catalog,
)
from mining.analysis_tool_gateway import AnalysisToolGateway
from mining.contract_candidates import CandidateDefinition
from mining.errors import (
    AnalysisToolArgumentError,
    AnalysisToolNotFoundError,
    AnalysisToolResultError,
    AnalysisToolTimeoutError,
)
from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_orchestrator_support import candidate_payload
from mining.search_run import SearchRunRequest
from mining.search_worker_client_v3 import (
    MockSearchTurnWorkerClient,
    generate_search_candidates_with_tools,
)
from mining.storage_v2 import MiningV2Store
from mining_worker.protocol import WorkerCandidateDraft
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchCandidate,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)
from mining_worker.protocol_v3 import (
    WorkerAnalysisToolCall,
    WorkerSearchTurnResponse,
)


def _fixture(tmp_path: Path):
    store = MiningV2Store(tmp_path / "mining.duckdb", tmp_path / "data")
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close", "label": "收盘价"},
            {"name": "volume_share", "label": "成交量"},
        ),
        operator_catalog=(
            {"name": "mean", "description": "窗口均值"},
            {"name": "rank", "description": "截面排名"},
        ),
    )
    run = store.create_run(
        SearchRunRequest(
            "analysis-tool-run-0001",
            "短期均值反转可能提供开发期信息",
        ),
        context,
    )
    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期均值反转可能提供开发期信息",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("开发期 RankIC 不足",),
    )
    validated = validate_expression(
        "mean(close, 5) + rank(volume_share)",
        direction=1,
    )
    candidate: CandidateDefinition = validated.to_candidate(
        mechanism="价格与成交量组合",
        description="测试候选",
        steps=("计算均值", "计算排名"),
        discovery_round=1,
        discovery_index=1,
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], hypothesis, batch)
    store.record_candidate(
        run["run_id"],
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        semantic_hash=candidate.semantic_hash,
        name=candidate.name,
        status="proposed",
        hypothesis_id=hypothesis.hypothesis_id,
        batch_id=batch.batch_id,
        payload=candidate_payload(candidate),
    )
    return store, context, run, hypothesis, batch, candidate


def test_analysis_tool_catalog_and_request_are_strict():
    assert len(tool_catalog()) == 7
    request = AnalysisToolRequest(
        call_id="call-1",
        tool_name="lookup_fields",
        arguments={"query": "close"},
    )
    assert request.arguments == {"query": "close", "limit": 10}
    assert AnalysisToolRequest.from_dict(request.to_dict()) == request

    with pytest.raises(AnalysisToolNotFoundError):
        AnalysisToolRequest(
            call_id="call-2",
            tool_name="run_sql",
            arguments={"query": "select *"},
        )
    with pytest.raises(AnalysisToolArgumentError):
        AnalysisToolRequest(
            call_id="call-3",
            tool_name="lookup_fields",
            arguments={"query": "close", "sql": "select 1"},
        )
    with pytest.raises(AnalysisToolResultError):
        AnalysisToolResult(
            call_id="call-4",
            tool_name="run_summary",
            tool_version="1",
            payload={"value": "x" * 70000},
        )


def test_gateway_reuses_only_the_same_run_and_turn(tmp_path: Path):
    store, context, run, *_ = _fixture(tmp_path)
    gateway = AnalysisToolGateway(store=store)
    request = AnalysisToolRequest(
        call_id="call-1",
        tool_name="run_summary",
    )

    first = gateway.execute_batch(
        run_id=run["run_id"],
        context=context,
        turn_index=1,
        turn_request_hash="c" * 64,
        requests=(request,),
    )[0]
    second = gateway.execute_batch(
        run_id=run["run_id"],
        context=context,
        turn_index=2,
        turn_request_hash="c" * 64,
        requests=(
            AnalysisToolRequest(
                call_id="call-2",
                tool_name="run_summary",
            ),
        ),
    )[0]
    shifted = gateway.execute_batch(
        run_id=run["run_id"],
        context=context,
        turn_index=3,
        turn_request_hash="d" * 64,
        requests=(request,),
    )[0]

    assert first.result_hash != second.result_hash
    assert first.payload == second.payload
    assert shifted.payload == first.payload
    audits = store.list_analysis_tool_calls(run["run_id"])
    assert [item["cache_hit"] for item in audits] == [False, True, False]


def test_gateway_rejects_more_than_four_calls(tmp_path: Path):
    store, context, run, *_ = _fixture(tmp_path)
    gateway = AnalysisToolGateway(store=store)
    requests = tuple(
        AnalysisToolRequest(
            call_id=f"call-{index}",
            tool_name="run_summary",
        )
        for index in range(5)
    )
    with pytest.raises(AnalysisToolArgumentError):
        gateway.execute_batch(
            run_id=run["run_id"],
            context=context,
            turn_index=1,
            turn_request_hash="e" * 64,
            requests=requests,
        )


def test_gateway_times_out_and_records_failure(tmp_path: Path):
    store, context, run, *_ = _fixture(tmp_path)
    ticks = iter((10.0, 16.0))
    gateway = AnalysisToolGateway(
        store=store,
        timeout_seconds=5.0,
        clock=lambda: next(ticks),
    )
    with pytest.raises(AnalysisToolTimeoutError):
        gateway.execute_batch(
            run_id=run["run_id"],
            context=context,
            turn_index=1,
            turn_request_hash="e" * 64,
            requests=(
                AnalysisToolRequest(
                    call_id="slow",
                    tool_name="run_summary",
                ),
            ),
        )
    audit = store.list_analysis_tool_calls(run["run_id"])[0]
    assert audit["status"] == "failed"
    assert audit["error_code"] == "MINING-130"


def test_gateway_context_drift_invalidates_cache(tmp_path: Path):
    store, context, run, *_ = _fixture(tmp_path)
    gateway = AnalysisToolGateway(store=store)
    request = AnalysisToolRequest(call_id="call", tool_name="run_summary")
    gateway.execute_batch(
        run_id=run["run_id"],
        context=context,
        turn_index=1,
        turn_request_hash="1" * 64,
        requests=(request,),
    )
    changed = replace(context, data_fingerprint="2" * 64)
    gateway.execute_batch(
        run_id=run["run_id"],
        context=changed,
        turn_index=2,
        turn_request_hash="1" * 64,
        requests=(request,),
    )
    assert [
        item["cache_hit"]
        for item in store.list_analysis_tool_calls(run["run_id"])
    ] == [False, False]


def test_gateway_returns_only_development_safe_candidate_evidence(tmp_path: Path):
    store, context, run, _hypothesis, _batch, candidate = _fixture(tmp_path)
    gateway = AnalysisToolGateway(store=store)
    results = gateway.execute_batch(
        run_id=run["run_id"],
        context=context,
        turn_index=1,
        turn_request_hash="f" * 64,
        requests=(
            AnalysisToolRequest(
                call_id="evidence",
                tool_name="candidate_evidence",
                arguments={"candidate_id": candidate.candidate_id},
            ),
            AnalysisToolRequest(
                call_id="ablation",
                tool_name="candidate_ablation",
                arguments={"candidate_id": candidate.candidate_id},
            ),
        ),
    )

    assert results[0].payload["scope"] == "candidate_development_evidence"
    assert results[1].payload["status"] == "available"
    assert len(results[1].payload["components"]) == 2
    assert candidate.canonical_expression not in str(
        [item.payload for item in results]
    )


def test_worker_v3_requests_tools_once_then_returns_final_candidate(tmp_path: Path):
    store, context, run, hypothesis, batch, candidate = _fixture(tmp_path)
    base = WorkerSearchRequest(
        request_id=f"{run['run_id']}:search:1",
        run_id=run["run_id"],
        turn_index=1,
        branch_id=hypothesis.branch_id,
        context_hash=context.context_hash,
        hypothesis=WorkerSearchHypothesis(
            statement=hypothesis.statement,
            mechanism=hypothesis.mechanism,
            expected_direction=hypothesis.expected_direction,
            falsification_criteria=hypothesis.falsification_criteria,
            parent_hypothesis_id=None,
        ),
        lineage=(
            WorkerSearchCandidate(
                candidate_id=candidate.candidate_id,
                name=candidate.name,
                expression=candidate.canonical_expression,
                status="proposed",
            ),
        ),
        evidence=None,
        budget=context.budget.to_dict(),
        allowed_actions=("refine", "branch", "stop", "promote"),
        llm_model=context.llm_model,
        max_candidates=1,
        max_expression_depth=6,
        excluded_fields=(),
        fields=context.field_catalog,
        operators=context.operator_catalog,
    )
    discover = WorkerSearchTurnResponse(
        request_id=f"{base.request_id}.tools",
        kind="tool_batch",
        tool_calls=(
            WorkerAnalysisToolCall(
                call_id="c1",
                tool_name="lookup_fields",
                arguments={"query": "close"},
            ),
        ),
    )
    final_search = WorkerSearchResponse(
        decision=WorkerDecisionProposal(action="refine", reason="继续验证"),
        next_hypothesis=WorkerSearchHypothesis(
            statement="价格均值与成交量排名组合值得修订",
            mechanism="短期价格修复",
            expected_direction="positive",
            falsification_criteria=("开发期 RankIC 不足",),
        ),
        candidates=(
            WorkerCandidateDraft(
                expression="mean(close, 10)",
                mechanism="短期价格修复",
                description="窗口修订",
                direction=1,
                steps=("平滑收盘价",),
            ),
        ),
    )
    finalize = WorkerSearchTurnResponse(
        request_id=f"{base.request_id}.final",
        kind="decision",
        response=final_search,
    )
    worker = MockSearchTurnWorkerClient(
        tool_response=discover,
        final_response=finalize,
    )

    normalized = generate_search_candidates_with_tools(
        worker,
        base,
        AnalysisToolGateway(store=store),
        context=context,
    )

    assert [item.phase for item in worker.requests] == ["discover", "finalize"]
    assert worker.requests[1].tool_results[0]["tool_name"] == "lookup_fields"
    assert normalized.accepted[0].canonical_expression == "mean(close, 10)"
    assert store.list_analysis_tool_calls(run["run_id"])[0]["status"] == "succeeded"
