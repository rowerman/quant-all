"""V2 搜索实验 API 分流和脱敏测试。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.search_context import FrozenSearchContext
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_progress import progress_for_search
from mining.search_run import SearchRunRequest
from mining.search_statistical_diagnostics import (
    build_multiple_testing_report,
)
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


class StubSearchOrchestrator:
    def __init__(self, store: MiningV2Store) -> None:
        self.store = store
        self.closed = False

    def create_run(self, request, context, *, background=True):
        return self.store.create_run(request, context)

    def progress(self, run_id):
        return progress_for_search(self.store, run_id)

    def cancel(self, run_id):
        return self.store.transition_search_run(
            run_id,
            "cancelled",
            event_key="run_cancelled",
            stage="queued",
            stop_reason="用户取消",
        )

    def resume(self, run_id, *, background=True):
        return self.store.search_control_state(run_id)

    def close(self):
        self.closed = True


def _services(tmp_path: Path):
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    v2_store = MiningV2Store(
        tmp_path / "factor_mining_v2.duckdb",
        tmp_path / "v2",
    )
    search = MiningSearchWebService(
        store=v2_store,
        orchestrator=StubSearchOrchestrator(v2_store),
    )
    service = MiningWebService(
        store=v1_store,
        defaults=None,
        snapshot_check=lambda *args, **kwargs: None,
        search_service=search,
    )
    return service, v2_store


def test_v2_api_creates_lists_and_reads_safe_detail(tmp_path: Path):
    service, _store = _services(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()
        assert options["engine"] == "v2"
        assert options["search_v2"]["enabled"] is True

        created = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期反转可能带来开发期超额收益",
                "idempotency_key": "web-v2-search-0001",
                "budget": {
                    **options["search_v2"]["defaults"],
                    "max_search_turns": 2,
                },
            },
        )
        assert created.status_code == 200
        detail = created.json()
        assert detail["run"]["engine"] == "v2"
        assert detail["progress"]["status"] == "queued"
        assert detail["hypotheses"] == []
        assert detail["turns"] == []
        assert "constraints" not in detail["run"]

        listed = client.get("/api/mining/runs").json()
        assert listed[0]["run_id"] == detail["run"]["run_id"]
        assert listed[0]["engine"] == "v2"

        run_id = detail["run"]["run_id"]
        events = client.get(f"/api/mining/runs/{run_id}/events").json()
        assert events["events"][0]["event_type"] == "run_created"
        status = client.post(f"/api/mining/runs/{run_id}/cancel").json()
        assert status["status"] == "cancelled"

        candidate = client.get(
            f"/api/mining/runs/{run_id}/candidates/unknown"
        )
        assert candidate.status_code == 400


def test_v2_detail_does_not_require_v1_run(tmp_path: Path):
    service, store = _services(tmp_path)
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    run = store.create_run(
        SearchRunRequest(
            "v2-detail-only-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )

    detail = service.run_detail(run["run_id"])

    assert detail["run"]["run_id"] == run["run_id"]
    assert "payload" not in detail["run"]["context"]
    assert detail["attempts"]["proposed"] == 0


def test_v2_detail_exposes_safe_m3_quality_evidence(tmp_path: Path):
    service, store = _services(tmp_path)
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    run = store.create_run(
        SearchRunRequest(
            "v2-m3-detail-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )
    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("年度同号率低于 0.6",),
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=("cand_api_0001",),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], hypothesis, batch)
    store.record_candidate(
        run["run_id"],
        candidate_id="cand_api_0001",
        candidate_hash="c" * 64,
        semantic_hash="d" * 64,
        name="MINE_API_0001",
        status="screening",
        hypothesis_id=hypothesis.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "mean(close, 5)"},
    )
    store.record_evaluation_evidence(
        SearchEvaluationEvidence(
            run_id=run["run_id"],
            candidate_id="cand_api_0001",
            stage="selection",
            input_hash="e" * 64,
            status="passed",
            metrics={
                "rank_ic_mean": 0.04,
                "rank_ic_ir": 0.5,
                "year_sign_consistency": 0.8,
                "values": {"coverage": 0.7},
                "valid_days": 100,
            },
        )
    )
    store.record_evaluation_evidence(
        SearchEvaluationEvidence(
            run_id=run["run_id"],
            candidate_id="cand_api_0001",
            stage="selection_diagnostics",
            input_hash="f" * 64,
            status="available",
            metrics={
                "correlation": {
                    "cluster_id": "cluster-0001",
                    "cluster_candidate_ids": ["cand_api_0001"],
                },
                "incremental": {"incremental_rank_ic": 0.03},
            },
        )
    )
    store.record_evaluation_evidence(
        SearchEvaluationEvidence(
            run_id=run["run_id"],
            candidate_id="cand_api_0001",
            stage="stability",
            input_hash="1" * 64,
            status="stable",
            metrics={
                "status": "stable",
                "rank_ic_ratio": 0.8,
                "reason": None,
            },
        )
    )
    days = 25
    dates = tuple(f"2020-01-{index + 1:02d}" for index in range(days))
    values = tuple(0.01 * ((index % 5) - 2) for index in range(days))
    statistics = build_multiple_testing_report(
        run_id=run["run_id"],
        series={"cand_api_0001": (dates, values)},
        attempt_count=3,
        evaluated_count=1,
    )
    store.record_evaluation_evidence(
        statistics.to_evaluation_evidence("cand_api_0001")
    )

    detail = service.run_detail(run["run_id"])
    candidate = detail["candidates"][0]
    encoded = json.dumps(detail, ensure_ascii=False)

    assert detail["branch_count"] == 1
    assert candidate["selection_quality"]["status"] == "quality_passed"
    assert candidate["research_quality"]["status"] == "quality_passed"
    assert candidate["selection_diagnostics"]["correlation"]["cluster_id"] == (
        "cluster-0001"
    )
    assert candidate["stability"]["status"] == "stable"
    assert candidate["multiple_testing"]["bh_q_value"] is not None
    assert detail["multiple_testing"]["denominator"] == 3
    assert "mean(close, 5)" not in encoded


def test_v2_detail_separates_tool_supplements_and_failures(tmp_path: Path):
    service, store = _services(tmp_path)
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    run = store.create_run(
        SearchRunRequest(
            "v2-tool-audit-0001",
            "工具补充证据需要接受审核",
        ),
        context,
    )
    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="工具补充证据需要接受审核",
        mechanism="研究证据核对",
        expected_direction="positive",
        falsification_criteria=("工具证据不可复现",),
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=("cand_tool_0001",),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], hypothesis, batch)
    store.record_candidate(
        run["run_id"],
        candidate_id="cand_tool_0001",
        candidate_hash="c" * 64,
        semantic_hash="d" * 64,
        name="MINE_TOOL_0001",
        status="proposed",
        hypothesis_id=hypothesis.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "mean(close, 5)"},
    )
    common = {
        "run_id": run["run_id"],
        "turn_index": 1,
        "context_hash": context.context_hash,
        "turn_request_hash": "f" * 64,
        "candidate_hash": "c" * 64,
        "tool_name": "candidate_evidence",
        "tool_version": "1",
        "candidate_id": "cand_tool_0001",
        "request_hash": "1" * 64,
        "arguments_summary": {"candidate_id": "cand_tool_0001"},
        "cache_hit": False,
        "duration_ms": 2.5,
    }
    store.record_analysis_tool_call(
        **common,
        cache_key="e" * 64,
        call_index=1,
        status="succeeded",
        payload={
            "schema_version": "mining_analysis_tool_v1",
            "call_id": "c1",
            "tool_name": "candidate_evidence",
            "tool_version": "1",
            "payload": {"status": "available"},
        },
        result_hash="2" * 64,
    )
    store.record_analysis_tool_call(
        **common,
        cache_key="3" * 64,
        call_index=2,
        status="failed",
        payload=None,
        result_hash=None,
        error_code="MINING-131",
        error_message="工具结果过大",
    )

    detail = service.run_detail(run["run_id"])
    calls = detail["turns"][0]["tool_calls"]
    encoded = json.dumps(detail, ensure_ascii=False)

    assert [item["evidence_class"] for item in calls] == [
        "supplemental",
        "failure",
    ]
    assert detail["candidates"][0]["analysis_count"] == 2
    assert detail["analysis_tools"]["total_calls"] == 2
    assert detail["analysis_tools"]["failed_calls"] == 1
    assert calls[0]["arguments_summary"] == {
        "candidate_id": "cand_tool_0001"
    }
    assert "mean(close, 5)" not in encoded
