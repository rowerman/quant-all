"""V3 统一入口和监控 API 测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.app import create_app
from backend.mining_service import MiningWebService
from backend.mining_v3_service import MiningV3WebService
from mining.contract_candidates import CandidateDefinition
from mining.expression_validation import validate_expression
from mining.errors import MiningNotFoundError
from mining.free_exploration import (
    direction_to_hypothesis,
    generate_initial_direction,
)
from mining.publication_contracts import new_publication_batch
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_progress import progress_for_search
from mining.storage import MiningStore
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from tests.mining_v3_recovery_support import (
    build_research_fixture,
    research_orchestrator,
)


class StubV3Orchestrator:
    def __init__(self, store: MiningV3RuntimeStore) -> None:
        self.store = store
        self.closed = False

    def create_run(self, request, context, *, background=True, research_input=None):
        return self.store.create_run(
            request,
            context,
            research_input=research_input,
        )

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


def _service(tmp_path: Path):
    contract = MiningV3Store(tmp_path / "factor_mining.duckdb", tmp_path / "v3")
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "v3",
        contract_store=contract,
    )
    v3 = MiningV3WebService(
        contract_store=contract,
        runtime_store=runtime,
        orchestrator=StubV3Orchestrator(runtime),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    v1_store = MiningStore(tmp_path / "factor_mining_v1.duckdb", tmp_path / "v1")
    return MiningWebService(store=v1_store, v3_service=v3), runtime, contract


def _prepare_v3_review_candidate(
    runtime: MiningV3RuntimeStore,
    run_id: str,
    *,
    run_status: str = "review_ready",
) -> tuple[HypothesisCard, CandidateDefinition]:
    """把一个 V3 运行候选推进到可审批状态。"""

    hypothesis = HypothesisCard(
        run_id=run_id,
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("五阶段证据未通过",),
    )
    candidate = validate_expression(
        "mean(close, 5)",
        direction=1,
    ).to_candidate(
        mechanism="短期价格修复",
        description="五日均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    batch = ExperimentBatch(
        run_id=run_id,
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    runtime.begin_turn(run_id, hypothesis, batch)
    runtime.record_candidate(
        run_id,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        semantic_hash=candidate.semantic_hash,
        name=candidate.name,
        status="proposed",
        hypothesis_id=hypothesis.hypothesis_id,
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
        runtime.transition_candidate(run_id, candidate.candidate_id, target)
    for index, stage in enumerate(
        ("technical", "development", "selection", "full", "holdout")
    ):
        runtime.record_evaluation_evidence(
            SearchEvaluationEvidence(
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                stage=stage,
                input_hash=f"{index + 1:x}" * 64,
                status="passed",
                metrics={"rank_ic_mean": 0.03, "rank_ic_ir": 0.4},
            )
        )
    turn = runtime.get_search_trace(run_id)["turns"][0]
    runtime.record_analysis_tool_call(
        run_id=run_id,
        turn_index=1,
        call_index=1,
        cache_key="a" * 64,
        context_hash=runtime.load_run_context(run_id).context_hash,
        turn_request_hash=turn["request_hash"],
        candidate_hash=candidate.content_hash,
        tool_name="ic_summary",
        tool_version="v1",
        candidate_id=candidate.candidate_id,
        request_hash="b" * 64,
        arguments_summary={"window": "development"},
        status="succeeded",
        cache_hit=False,
        duration_ms=1.5,
        payload={"rank_ic": 0.03},
        result_hash="c" * 64,
    )
    targets = (
        (
            "searching",
            "selecting",
            "evaluating",
            "holding_out",
            "review_ready",
        )
        if run_status == "review_ready"
        else (run_status,)
    )
    for target in targets:
        runtime.transition_search_run(
            run_id,
            target,
            event_key=f"run_{target}_{len(targets)}",
            stage=target,
        )
    return hypothesis, candidate


def test_v3_service_defaults_to_tool_aware_search(tmp_path: Path):
    contract = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "v3",
    )
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "v3",
        contract_store=contract,
    )
    service = MiningV3WebService(
        contract_store=contract,
        runtime_store=runtime,
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    try:
        assert service.orchestrator.analysis_tool_gateway is not None
    finally:
        service.close()


def test_v3_api_options_create_list_and_monitoring(tmp_path: Path):
    service, _runtime, contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()
        assert options["engine"] == "v3"
        assert options["search_v3"]["enabled"] is True
        assert [item["value"] for item in options["search_v3"]["input_modes"]] == [
            "hypothesis",
            "free_exploration",
            "research_report",
            "costeer",
        ]

        created = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期反转可能带来开发期超额收益",
                "idempotency_key": "web-v3-create-0001",
                "budget": options["search_v3"]["defaults"],
            },
        )
        assert created.status_code == 200
        detail = created.json()
        run_id = detail["run"]["run_id"]
        assert run_id.startswith("researchrun_")
        assert detail["run"]["engine"] == "v3"
        assert contract.load_input(run_id).input_mode == "hypothesis"

        listed = client.get("/api/mining/runs").json()
        assert any(item["run_id"] == run_id and item["engine"] == "v3" for item in listed)
        events = client.get(f"/api/mining/runs/{run_id}/events").json()
        assert events["events"][0]["event_type"] == "run_created"
        cancelled = client.post(f"/api/mining/runs/{run_id}/cancel").json()
        assert cancelled["status"] == "cancelled"

        monitor = client.get("/api/mining/v3/status").json()
        assert monitor["enabled"] is True
        assert monitor["source_counts"]["hypothesis"] == 1
        assert monitor["legacy_runs"] == 0


def test_v3_candidate_exposes_five_stage_review_and_accepts_decision(
    tmp_path: Path,
):
    service, runtime, _contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()["search_v3"]
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期反转可能带来开发期超额收益",
                "idempotency_key": "web-v3-review-0001",
                "budget": options["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        hypothesis, candidate = _prepare_v3_review_candidate(
            runtime,
            run_id,
        )

        candidate_detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}"
        ).json()
        assert candidate_detail["run"]["engine"] == "v3"
        assert candidate_detail["candidate"]["status"] == "review_ready"
        assert candidate_detail["review"]["eligible"] is True
        assert candidate_detail["review"]["candidate_status"] == "review_ready"
        assert len(candidate_detail["review"]["evidence"]) == 5
        assert candidate_detail["technical_gate"] == {
            "passed": True,
            "rule_version": "mining_search_approval_v1",
            "checks": {"rank_ic_mean": 0.03, "rank_ic_ir": 0.4},
        }
        decided = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-v3",
                "note": "五阶段证据完整",
                "candidate_content_hash": candidate_detail["review"][
                    "candidate_content_hash"
                ],
                "evidence_hash": candidate_detail["review"][
                    "approval_evidence_hash"
                ],
            },
        )
        assert decided.status_code == 200
        decided_payload = decided.json()
        assert decided_payload["approval"]["decision"] == "approved"
        assert decided_payload["candidate"]["status"] == "approved"
        approved_detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}"
        ).json()
        assert approved_detail["candidate"]["status"] == "approved"
        assert approved_detail["review"]["candidate_status"] == "approved"
        run_detail = client.get(f"/api/mining/runs/{run_id}").json()
        assert len(run_detail["hypotheses"]) == 1
        assert len(run_detail["turns"]) == 1
        assert run_detail["branch_count"] == 1
        assert run_detail["turns"][0]["tool_calls"][0]["tool_name"] == "ic_summary"
        assert run_detail["analysis_tools"]["total_calls"] == 1
        assert run_detail["candidates"][0]["candidate_id"] == candidate.candidate_id
        assert run_detail["candidates"][0]["hypothesis_id"] == hypothesis.hypothesis_id
        assert run_detail["candidates"][0]["analysis_count"] == 1
        assert run_detail["candidates"][0]["status"] == "approved"
        assert run_detail["progress"]["candidates"] == len(run_detail["candidates"])


@pytest.mark.parametrize("run_status", ("failed", "interrupted"))
def test_v3_api_rejects_review_for_non_review_ready_run(
    tmp_path: Path,
    run_status: str,
):
    service, runtime, _contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "非审核态运行不能提交人工决定",
                "idempotency_key": f"web-v3-{run_status}-0001",
                "budget": service.v3_service.options()["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        _hypothesis, candidate = _prepare_v3_review_candidate(
            runtime,
            run_id,
            run_status=run_status,
        )
        candidate_detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}"
        ).json()

        assert candidate_detail["review"]["eligible"] is False
        assert (
            f"RUN_NOT_REVIEW_READY:{run_status}"
            in candidate_detail["review"]["blocking_reasons"]
        )
        response = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-v3",
                "note": "尝试绕过运行状态门禁",
                "candidate_content_hash": candidate_detail["review"][
                    "candidate_content_hash"
                ],
                "evidence_hash": candidate_detail["review"][
                    "approval_evidence_hash"
                ],
            },
        )
        assert response.status_code == 400
        assert f"RUN_NOT_REVIEW_READY:{run_status}" in response.json()["message"]


def test_v3_api_rejects_approval_and_publication_after_stage_change(
    tmp_path: Path,
):
    service, runtime, contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "页面读取后运行状态变化必须阻止审批",
                "idempotency_key": "web-v3-stage-race-0001",
                "budget": service.v3_service.options()["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        _hypothesis, candidate = _prepare_v3_review_candidate(runtime, run_id)
        candidate_detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}"
        ).json()
        runtime.transition_search_run(
            run_id,
            "stale",
            event_key="run_stale_after_review_preview",
            stage="review_ready",
        )

        rejected = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-v3",
                "note": "页面打开后状态变化",
                "candidate_content_hash": candidate_detail["review"][
                    "candidate_content_hash"
                ],
                "evidence_hash": candidate_detail["review"][
                    "approval_evidence_hash"
                ],
            },
        )

        assert rejected.status_code == 400
        assert "RUN_NOT_REVIEW_READY:stale" in rejected.json()["message"]
        with pytest.raises(MiningNotFoundError, match="尚无审批"):
            contract.get_approval(run_id, candidate.candidate_id)


def test_v3_api_rejects_publication_after_run_becomes_stale(tmp_path: Path):
    service, runtime, contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "批准后运行过期时不能启动正式发布",
                "idempotency_key": "web-v3-publish-race-0001",
                "budget": service.v3_service.options()["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        _hypothesis, candidate = _prepare_v3_review_candidate(runtime, run_id)
        review = contract.candidate_review(
            run_id,
            candidate.candidate_id,
            contract.candidate_source_id(run_id, candidate.candidate_id),
        )
        approved = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-v3",
                "note": "先正常批准，再模拟运行过期",
                "candidate_content_hash": review.candidate_content_hash,
                "evidence_hash": review.approval_evidence_hash,
            },
        )
        assert approved.status_code == 200
        runtime.transition_search_run(
            run_id,
            "stale",
            event_key="run_stale_before_publication",
            stage="review_ready",
        )

        rejected = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": [candidate.candidate_id],
                "expected_context_hash": contract.load_input(run_id).context_hash,
            },
        )

        assert rejected.status_code == 409
        assert "stale" in rejected.json()["message"]
        assert contract.list_publication_batches(run_id) == []


def test_v3_contract_status_is_projected_through_lifecycle(tmp_path: Path):
    service, runtime, contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "状态投影必须统一读取候选 contract",
                "idempotency_key": "web-v3-projection-0001",
                "budget": service.v3_service.options()["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        _hypothesis, candidate = _prepare_v3_review_candidate(runtime, run_id)
        review = contract.candidate_review(
            run_id,
            candidate.candidate_id,
            contract.candidate_source_id(run_id, candidate.candidate_id),
        )
        client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate.candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "reviewer-v3",
                "note": "状态投影测试",
                "candidate_content_hash": review.candidate_content_hash,
                "evidence_hash": review.approval_evidence_hash,
            },
        ).raise_for_status()
        candidate_ids = (candidate.candidate_id,)
        approval_id = contract.get_approval(
            run_id,
            candidate.candidate_id,
        )["approval_id"]
        context_hash = contract.load_input(run_id).context_hash

        first_batch = new_publication_batch(
            run_id=run_id,
            context_hash=context_hash,
            candidate_ids=candidate_ids,
            approval_ids=(approval_id,),
            staging_path=str(tmp_path / "staging-1"),
            attempt=1,
            source_kind="v3_research",
        )
        contract.start_publication_batch(first_batch)
        assert _candidate_projection(client, run_id, candidate.candidate_id) == (
            "building",
            "building",
            "building",
        )

        contract.fail_publication_batch(
            first_batch.batch_id,
            error_code="MINING-109",
            error_message="模拟构建失败",
        )
        assert _candidate_projection(client, run_id, candidate.candidate_id) == (
            "build_failed",
            "build_failed",
            "build_failed",
        )

        second_batch = new_publication_batch(
            run_id=run_id,
            context_hash=context_hash,
            candidate_ids=candidate_ids,
            approval_ids=(approval_id,),
            staging_path=str(tmp_path / "staging-2"),
            attempt=2,
            source_kind="v3_research",
        )
        contract.start_publication_batch(second_batch)
        contract.mark_publication_validating(second_batch.batch_id)
        contract.complete_publication_batch(
            second_batch.batch_id,
            manifest_hash="a" * 64,
            artifact_sha256="b" * 64,
            formal_factor_names=(candidate.name,),
        )
        assert _candidate_projection(client, run_id, candidate.candidate_id) == (
            "published",
            "published",
            "published",
        )


def _candidate_projection(
    client: TestClient,
    run_id: str,
    candidate_id: str,
) -> tuple[str, str, str]:
    run_detail = client.get(f"/api/mining/runs/{run_id}").json()
    candidate_detail = client.get(
        f"/api/mining/runs/{run_id}/candidates/{candidate_id}"
    ).json()
    return (
        run_detail["candidates"][0]["status"],
        candidate_detail["candidate"]["status"],
        candidate_detail["review"]["candidate_status"],
    )


def test_v3_free_exploration_run_detail_uses_persisted_direction(
    tmp_path: Path,
):
    service, runtime, contract = _service(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()["search_v3"]
        detail = client.post(
            "/api/mining/runs",
            json={
                "input_mode": "free_exploration",
                "exploration_seed": "20261102",
                "idempotency_key": "web-v3-free-detail-0001",
                "budget": options["defaults"],
            },
        ).json()
        run_id = detail["run"]["run_id"]
        request = runtime.load_run_request(run_id)
        context = contract.load_context(run_id)
        direction = generate_initial_direction(request, context)
        hypothesis = direction_to_hypothesis(direction)
        batch = ExperimentBatch(
            run_id=run_id,
            hypothesis_id=hypothesis.hypothesis_id,
            turn_index=1,
            branch_id=hypothesis.branch_id,
            candidate_ids=(),
            parent_candidate_ids=(),
            development_start=context.windows.development_start,
            development_end=context.windows.development_end,
            input_hash="d" * 64,
        )
        runtime.begin_turn(run_id, hypothesis, batch)

        run_detail = client.get(f"/api/mining/runs/{run_id}").json()

    assert run_detail["run"]["exploration_seed"] == 20261102
    assert run_detail["run"]["effective_exploration_seed"] == 20261102
    assert run_detail["run"]["seed_source"] == "request"
    assert run_detail["hypotheses"][0]["statement"] == direction.statement
    assert run_detail["turns"][0]["turn_index"] == 1
    assert run_detail["branch_count"] == 1
    assert (
        run_detail["exploration"]["initial_direction"]["direction_hash"]
        == direction.direction_hash
    )
    assert run_detail["progress"]["hypotheses"] == 1
    assert run_detail["progress"]["candidates"] == 0


def test_v3_research_candidate_returns_validated_provenance(
    tmp_path: Path,
):
    (
        runtime,
        contract,
        request,
        context,
        research_input,
        material_root,
    ) = build_research_fixture(tmp_path)
    orchestrator = research_orchestrator(runtime, material_root)
    run = orchestrator.create_run(
        request,
        context,
        background=False,
        research_input=research_input,
    )
    try:
        assert orchestrator.execute(run["run_id"])["status"] == "review_ready"
    finally:
        orchestrator.close()

    service = MiningV3WebService(
        contract_store=contract,
        runtime_store=runtime,
        orchestrator=StubV3Orchestrator(runtime),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    web = MiningWebService(store=v1_store, v3_service=service)
    app = create_app(mining_service=web)
    review = next(
        item for item in contract.list_candidate_reviews(run["run_id"])
        if item.eligible
    )
    with TestClient(app) as client:
        candidate_detail = client.get(
            f"/api/mining/runs/{run['run_id']}/candidates/"
            f"{review.candidate_id}"
        ).json()

    assert candidate_detail["technical_gate"]["passed"] is True
    assert candidate_detail["technical_gate"]["rule_version"] == (
        "mining_search_approval_v1"
    )
    source = candidate_detail["research_source"]
    assert source["material"]["relative_path"] == "report.pdf"
    assert source["fields"][0]["citations"][0]["page_number"] == 1
    assert source["provenance_hash"] == candidate_detail["v3_source"]["source_hash"]
