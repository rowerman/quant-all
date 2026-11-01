"""V2 搜索实验 API 分流和脱敏测试。"""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.errors import SnapshotStaleError
from mining.expression_validation import validate_expression
from mining.search_context import FrozenSearchContext
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_progress import progress_for_search
from mining.search_review import SEARCH_APPROVAL_REQUIRED_STAGES
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


def _services(tmp_path: Path, *, snapshot_check=None):
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
        snapshot_check=snapshot_check or (lambda *_args, **_kwargs: None),
    )
    service = MiningWebService(
        store=v1_store,
        defaults=None,
        snapshot_check=lambda *args, **kwargs: None,
        search_service=search,
    )
    return service, v2_store


def _seed_v2_review_ready(
    store: MiningV2Store,
    key: str,
    *,
    candidate_count: int = 2,
) -> tuple[str, tuple[str, ...]]:
    """创建带完整五阶段证据的 V2 审核态运行。"""

    run = store.create_run(
        SearchRunRequest(
            key,
            "短期趋势可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )
    card = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期趋势可能带来开发期超额收益",
        mechanism="价格趋势延续",
        expected_direction="positive",
        falsification_criteria=("完整评估未通过",),
    )
    definitions = tuple(
        validate_expression(
            f"mean(close, {5 + index * 5})",
            direction=1,
        ).to_candidate(
            mechanism="短期趋势",
            description=f"{5 + index * 5}日收盘均值",
            steps=("读取收盘价", "计算移动均值"),
            discovery_round=1,
            discovery_index=index + 1,
        )
        for index in range(candidate_count)
    )
    candidate_ids = tuple(item.candidate_id for item in definitions)
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id=card.branch_id,
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], card, batch)
    for candidate_index, candidate in enumerate(definitions, start=1):
        store.record_candidate(
            run["run_id"],
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.content_hash,
            semantic_hash=candidate.semantic_hash,
            name=candidate.name,
            status="proposed",
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            payload=candidate_payload(candidate),
        )
        for target in (
            "validating",
            "screening",
            "evaluating",
            "evaluated",
            "holdout_running",
            "review_ready",
        ):
            store.transition_candidate(
                run["run_id"],
                candidate.candidate_id,
                target,
            )
        for stage_index, stage in enumerate(SEARCH_APPROVAL_REQUIRED_STAGES):
            store.record_evaluation_evidence(
                SearchEvaluationEvidence(
                    run_id=run["run_id"],
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    input_hash=(
                        f"{candidate_index:x}{stage_index + 1:x}" * 32
                    ),
                    status="passed",
                    metrics={
                        "rank_ic_mean": 0.03,
                        "rank_ic_ir": 0.4,
                        "year_sign_consistency": 0.75,
                        "valid_days": 100,
                        "values": {"coverage": 0.8, "rows": 10000},
                    },
                )
            )
    for target in (
        "searching",
        "selecting",
        "evaluating",
        "holding_out",
        "review_ready",
    ):
        store.transition_search_run(
            run["run_id"],
            target,
            event_key=f"run_{target}",
            stage=target,
        )
    return run["run_id"], candidate_ids


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
        assert candidate.status_code == 404


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


def test_v2_candidate_detail_exposes_approval_evidence(tmp_path: Path):
    service, store = _services(tmp_path)
    run_id, candidate_ids = _seed_v2_review_ready(
        store,
        "v2-review-detail-0001",
    )
    app = create_app(mining_service=service)

    with TestClient(app) as client:
        detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        )

    assert detail.status_code == 200
    payload = detail.json()
    assert payload["run"]["engine"] == "v2"
    assert payload["candidate"]["definition"]["canonical_expression"] == (
        "mean(close, 5)"
    )
    assert {
        item["stage"]
        for item in payload["evaluations"]
        if item["stage"] in SEARCH_APPROVAL_REQUIRED_STAGES
    } == set(SEARCH_APPROVAL_REQUIRED_STAGES)
    assert payload["review"]["eligible"] is True
    assert payload["review"]["blocking_reasons"] == []
    assert payload["technical_gate"]["passed"] is True
    assert payload["scorecard"]["scorecard_hash"]
    assert payload["publication"]["eligible"] is False


def test_v2_decision_api_is_hash_bound_and_idempotent(tmp_path: Path):
    service, store = _services(tmp_path)
    run_id, candidate_ids = _seed_v2_review_ready(
        store,
        "v2-review-decision-0001",
    )
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        ).json()
        review = detail["review"]
        payload = {
            "decision": "approved",
            "reviewer": "reviewer-2",
            "note": "五阶段证据完整，允许正式构建",
            "candidate_content_hash": review["candidate_content_hash"],
            "evidence_hash": review["evidence_hash"],
        }
        conflict = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}/decision",
            json={**payload, "evidence_hash": "0" * 64},
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "MINING-108"

        approved = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}/decision",
            json=payload,
        )
        repeated = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}/decision",
            json=payload,
        )
        second = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[1]}"
        ).json()
        rejected = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[1]}/decision",
            json={
                **payload,
                "decision": "rejected",
                "note": "机制解释不足",
                "candidate_content_hash": (
                    second["review"]["candidate_content_hash"]
                ),
                "evidence_hash": second["review"]["evidence_hash"],
            },
        )
        after = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        ).json()

    assert approved.status_code == 200
    assert approved.json()["candidate"]["status"] == "approved"
    assert repeated.status_code == 200
    assert repeated.json()["approval"]["approval_id"] == (
        approved.json()["approval"]["approval_id"]
    )
    assert rejected.status_code == 200
    assert rejected.json()["candidate"]["status"] == "rejected"
    assert after["approval"]["decision"] == "approved"
    assert after["publication"]["approved_candidate_ids"] == [candidate_ids[0]]


def test_v2_snapshot_change_blocks_decision(tmp_path: Path):
    stale = {"value": False}

    def snapshot_check(*_args, **_kwargs):
        if stale["value"]:
            raise SnapshotStaleError("V2 数据快照已变化")

    service, store = _services(tmp_path, snapshot_check=snapshot_check)
    run_id, candidate_ids = _seed_v2_review_ready(
        store,
        "v2-review-stale-0001",
    )
    detail = service.candidate_detail(run_id, candidate_ids[0])
    review = detail["review"]
    stale["value"] = True
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        current = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        )
        blocked = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-3",
                "note": "快照变化前提交",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )

    assert current.json()["review"]["snapshot_current"] is False
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "MINING-120"
    assert store.get_run(run_id)["status"] == "stale"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"stale"}


def test_v2_snapshot_change_blocks_publication_and_closes_run_stale(
    tmp_path: Path,
):
    stale = {"value": False}

    def snapshot_check(*_args, **_kwargs):
        if stale["value"]:
            raise SnapshotStaleError("V2 发布快照已变化")

    service, store = _services(tmp_path, snapshot_check=snapshot_check)
    run_id, candidate_ids = _seed_v2_review_ready(
        store,
        "v2-publication-stale-0001",
    )
    for candidate_id in candidate_ids:
        detail = service.candidate_detail(run_id, candidate_id)
        review = detail["review"]
        service.decide(
            run_id,
            candidate_id,
            {
                "decision": "approved",
                "reviewer": "reviewer-stale",
                "note": "批准后发生快照变化",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
    stale["value"] = True
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        blocked = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": list(candidate_ids),
                "expected_context_hash": store.get_run(run_id)["context_hash"],
            },
        )

    assert blocked.status_code == 409
    assert blocked.json()["code"] == "MINING-120"
    assert store.get_run(run_id)["status"] == "stale"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"stale"}


def test_v2_publication_api_requires_batch_and_supports_retry(
    tmp_path: Path,
):
    service, store = _services(tmp_path)
    run_id, candidate_ids = _seed_v2_review_ready(
        store,
        "v2-review-publication-0001",
    )
    for candidate_id in candidate_ids:
        detail = service.candidate_detail(run_id, candidate_id)
        review = detail["review"]
        service.decide(
            run_id,
            candidate_id,
            {
                "decision": "approved",
                "reviewer": "reviewer-4",
                "note": "批准整批发布",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
    context_hash = store.get_run(run_id)["context_hash"]
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        partial = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": [candidate_ids[0]],
                "expected_context_hash": context_hash,
            },
        )
        assert partial.status_code == 409
        assert partial.json()["code"] == "MINING-121"

        def fail_prepare(**_kwargs):
            raise RuntimeError("injected publication failure")

        service.search_service.publication_service._prepare = fail_prepare
        started = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": list(candidate_ids),
                "expected_context_hash": context_hash,
            },
        )
        assert started.status_code == 202
        failed = _wait_for_batch(
            client,
            run_id,
            started.json()["batch_id"],
            "failed",
        )
        assert failed["error_code"]
        failed_detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        ).json()
        assert failed_detail["candidate"]["status"] == "build_failed"
        assert failed_detail["publication"]["eligible"] is True

        service.search_service.publication_service._prepare = (
            lambda **kwargs: SimpleNamespace(
                manifest_hash="a" * 64,
                artifact_sha256="b" * 64,
                factor_names=tuple(
                    item["name"] for item in kwargs["candidates"]
                ),
            )
        )
        service.search_service.publication_service._commit = (
            lambda **_kwargs: None
        )
        service.search_service.publication_service._rollback = (
            lambda **_kwargs: None
        )
        retried = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": list(candidate_ids),
                "expected_context_hash": context_hash,
            },
        )
        assert retried.status_code == 202
        published = _wait_for_batch(
            client,
            run_id,
            retried.json()["batch_id"],
            "published",
        )
        detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_ids[0]}"
        ).json()

    assert published["attempt"] == 2
    assert "staging_path" not in published
    assert all(
        "staging_path" not in item
        for item in published["publications"]
    )
    assert detail["candidate"]["status"] == "published"
    assert detail["publication"]["latest_batch"]["status"] == "published"


def _wait_for_batch(
    client: TestClient,
    run_id: str,
    batch_id: str,
    expected: str,
) -> dict:
    for _ in range(100):
        response = client.get(
            f"/api/mining/runs/{run_id}/publication-batches/{batch_id}"
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"发布批次未进入 {expected} 状态")
