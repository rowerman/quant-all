"""S7 挖掘 Web API、SSE 和人工审批联合测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from backend.app import create_app
from backend.mining_service import MiningWebService
from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_requests import FrozenContext, MiningConstraints, MiningRunRequest
from mining.contract_types import sha256_json, utc_now_iso
from mining.errors import SnapshotStaleError
from mining.expression_validation import validate_expression
from mining.orchestration_support import context_from_run, progress_for_run
from mining.orchestrator import MiningOrchestrator
from mining.storage import MiningStore


class StubOrchestrator:
    """不调用 Worker 的隔离编排器，保留运行状态和存储契约。"""

    def __init__(self, store: MiningStore) -> None:
        self.store = store
        self.closed = False

    def create_run(self, request: MiningRunRequest, *, background: bool = True):
        context = FrozenContext(
            data_fingerprint="a" * 64,
            field_catalog=({"name": "close"},),
            operator_catalog=({"name": "mean"},),
            budget=request.constraints,
            llm_profile=request.llm_profile,
        )
        return self.store.create_run(request, context, worker_mode="rdagent")

    def progress(self, run_id: str):
        return progress_for_run(self.store, run_id)

    def cancel(self, run_id: str):
        return self.store.terminate_run(
            run_id,
            "cancelled",
            event_key="test_run_cancelled",
        )

    def resume(self, run_id: str, *, background: bool = True):
        return self.store.resume_run(run_id)

    def close(self) -> None:
        self.closed = True


def _constraints() -> MiningConstraints:
    return MiningConstraints(
        max_rounds=1,
        max_candidates=2,
        full_eval_top_k=1,
        max_llm_calls=2,
        min_quality_checks_for_failure_stop=2,
    )


def _service(
    tmp_path: Path,
    *,
    snapshot_check=None,
) -> tuple[MiningWebService, StubOrchestrator]:
    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    orchestrator = StubOrchestrator(store)
    service = MiningWebService(
        store=store,
        orchestrator=orchestrator,
        defaults=_constraints(),
        reviewer="local-reviewer",
        snapshot_check=snapshot_check or (lambda _context, **_kwargs: None),
    )
    return service, orchestrator


def _create_run(service: MiningWebService, key: str = "web-test-run-0001") -> dict:
    return service.create_run(
        {
            "hypothesis": "短期量价变化可能预测未来五日收益",
            "idempotency_key": key,
            "constraints": {"max_candidates": 2, "full_eval_top_k": 1},
        }
    )


def _report(
    context: FrozenContext,
    candidate,
    stage: str,
    metrics: dict[str, Any],
) -> EvaluationReport:
    return EvaluationReport(
        run_id="ignored",
        candidate_id=candidate.candidate_id,
        stage=stage,
        candidate_content_hash=candidate.content_hash,
        context_hash=context.context_hash,
        input_hash=sha256_json({"stage": stage, "candidate": candidate.candidate_id}),
        status="passed",
        metrics=metrics,
        warnings=(),
        artifact_path=None,
        artifact_sha256=None,
        error_code=None,
        error_message=None,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
    )


def _seed_review_ready(service: MiningWebService, key: str) -> tuple[str, str]:
    detail = _create_run(service, key)
    run_id = detail["run"]["run_id"]
    store = service.store
    run = store.get_run(run_id)
    context = context_from_run(run)
    definition = validate_expression("mean(close,5)", direction=1).to_candidate(
        mechanism="短期趋势",
        description="收盘价短期均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    store.add_candidate(run_id, definition)
    store.transition_run(
        run_id,
        "generating",
        event_key="test_generating",
        stage="generating",
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "validating",
        event_key="test_validating",
        stage="validating",
    )
    gate = TechnicalGate(
        run_id=run_id,
        candidate_id=definition.candidate_id,
        candidate_content_hash=definition.content_hash,
        context_hash=context.context_hash,
        rule_version="test-v1",
        passed=True,
        checks={"values": {"coverage": 0.8}},
    )
    store.record_technical_gate(gate)
    store.record_evaluation(
        _replace_run(
            _report(
                context,
                definition,
                "technical",
                {"values": {"coverage": 0.8}},
            ),
            run_id,
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "screening",
        event_key="test_screening",
        stage="screening",
    )
    store.record_evaluation(
        _replace_run(
            _report(
                context,
                definition,
                "quick",
                {
                    "rank_ic_mean": 0.03,
                    "rank_ic_ir": 0.4,
                    "valid_days": 100,
                },
            ),
            run_id,
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "evaluating",
        event_key="test_evaluating",
        stage="evaluating",
    )
    store.record_evaluation(
        _replace_run(
            _report(
                context,
                definition,
                "full",
                {
                    "events": {
                        "overview": {
                            "metrics": {
                                "ic_stats": {
                                    "ic": {"mean": 0.02, "ir": 0.3},
                                    "rank_ic": {"mean": 0.03, "ir": 0.4},
                                    "valid_days": 100,
                                }
                            }
                        }
                    }
                },
            ),
            run_id,
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "evaluated",
        event_key="test_evaluated",
        stage="evaluating",
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "holdout_running",
        event_key="test_holdout_running",
        stage="holdout",
    )
    store.record_evaluation(
        _replace_run(
            _report(
                context,
                definition,
                "holdout",
                {"valid_days": 100, "min_valid_days": 20, "complete": True},
            ),
            run_id,
        )
    )
    store.claim_holdout(
        run_id=run_id,
        candidate_ids=[definition.candidate_id],
        evidence_hash="f" * 64,
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "review_ready",
        event_key="test_review_ready",
        stage="holdout",
    )
    for target in ("screening", "evaluating", "holding_out", "review_ready"):
        store.transition_run(
            run_id,
            target,
            event_key=f"test_run_{target}",
            stage=target,
        )
    return run_id, definition.candidate_id


def _replace_run(report: EvaluationReport, run_id: str) -> EvaluationReport:
    """测试报告先构造再绑定实际运行 ID。"""

    payload = {
        field: getattr(report, field)
        for field in (
            "candidate_id",
            "stage",
            "candidate_content_hash",
            "context_hash",
            "input_hash",
            "status",
            "metrics",
            "warnings",
            "artifact_path",
            "artifact_sha256",
            "error_code",
            "error_message",
            "started_at",
            "finished_at",
            "attempt",
        )
    }
    payload["run_id"] = run_id
    return EvaluationReport(**payload)


def _app(service: MiningWebService):
    return create_app(mining_service=service)


def test_options_create_list_detail_events_cancel_and_resume(tmp_path: Path):
    service, orchestrator = _service(tmp_path)
    with TestClient(_app(service)) as client:
        options = client.get("/api/mining/options")
        assert options.status_code == 200
        assert options.json()["reviewer"] == "local-reviewer"
        assert options.json()["defaults"]["max_rounds"] == 1

        created = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期量价变化可能预测未来五日收益",
                "idempotency_key": "web-test-run-0001",
                "constraints": {"max_candidates": 2, "full_eval_top_k": 1},
            },
        )
        assert created.status_code == 200
        run_id = created.json()["run"]["run_id"]
        assert created.json()["run"]["status"] == "queued"

        listed = client.get("/api/mining/runs")
        assert listed.status_code == 200
        assert listed.json()[0]["run_id"] == run_id
        events = client.get(f"/api/mining/runs/{run_id}/events?after=0")
        assert events.status_code == 200
        assert events.json()["events"]
        assert events.json()["last_sequence"] >= 1

        cancelled = client.post(f"/api/mining/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        resumed_run = _create_run(service, "web-test-run-resume")
        resumed_id = resumed_run["run"]["run_id"]
        service.store.transition_run(
            resumed_id,
            "generating",
            event_key="test_resume_generating",
            stage="generating",
        )
        service.store.interrupt_active_runs()
        resumed = client.post(f"/api/mining/runs/{resumed_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["run"]["status"] == "generating"
    assert orchestrator.closed is True


def test_service_startup_marks_active_run_interrupted_without_resume(
    tmp_path: Path,
):
    service, _ = _service(tmp_path)
    detail = _create_run(service, "web-startup-interrupt")
    run_id = detail["run"]["run_id"]
    service.store.transition_run(
        run_id,
        "generating",
        event_key="web_startup_generating",
        stage="generating",
    )
    with TestClient(_app(service)):
        assert service.store.get_run(run_id)["status"] == "interrupted"
        assert service.store.get_run(run_id)["interrupted_from"] == "generating"


def test_review_stream_and_approval_are_bound_to_current_evidence(tmp_path: Path):
    service, _ = _service(tmp_path)
    run_id, candidate_id = _seed_review_ready(service, "web-review-ready")
    with TestClient(_app(service)) as client:
        detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}"
        )
        assert detail.status_code == 200
        review = detail.json()["review"]
        assert review["eligible"] is True
        assert len(review["evidence_hash"]) == 64

        stream = client.get(f"/api/mining/runs/{run_id}/stream?after=0")
        assert stream.status_code == 200
        assert "event: done" in stream.text
        assert "event: mining" in stream.text

        conflict = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "local-reviewer",
                "note": "质量与测试段均符合要求",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": "0" * 64,
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "MINING-108"

        missing_note = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "local-reviewer",
                "note": "",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
        assert missing_note.status_code == 400
        assert missing_note.json()["code"] == "MINING-101"

        approved = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "local-reviewer",
                "note": "质量与测试段均符合要求",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
        assert approved.status_code == 200
        assert approved.json()["approval"]["decision"] == "approved"
        assert approved.json()["candidate"]["status"] == "approved"

        repeated = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "local-reviewer",
                "note": "质量与测试段均符合要求",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
        assert repeated.status_code == 200
        assert repeated.json()["approval"]["approval_id"] == approved.json()["approval"]["approval_id"]


def test_snapshot_change_blocks_decision_and_marks_run_stale(tmp_path: Path):
    stale = {"value": False}

    def snapshot_check(_context, **_kwargs):
        if stale["value"]:
            raise SnapshotStaleError("测试数据快照已变化")

    service, _ = _service(tmp_path, snapshot_check=snapshot_check)
    run_id, candidate_id = _seed_review_ready(service, "web-review-stale")
    review = service.candidate_detail(run_id, candidate_id)["review"]
    stale["value"] = True
    with TestClient(_app(service)) as client:
        result = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "local-reviewer",
                "note": "快照变化前提交",
                "candidate_content_hash": review["candidate_content_hash"],
                "evidence_hash": review["evidence_hash"],
            },
        )
        assert result.status_code == 409
        assert result.json()["code"] == "MINING-120"
        assert service.store.get_run(run_id)["status"] == "stale"
        assert service.store.get_candidate(run_id, candidate_id)["status"] == "stale"


def test_rejected_candidate_cannot_be_approved_again(tmp_path: Path):
    service, _ = _service(tmp_path)
    run_id, candidate_id = _seed_review_ready(service, "web-review-rejected")
    review = service.candidate_detail(run_id, candidate_id)["review"]
    payload = {
        "decision": "rejected",
        "reviewer": "local-reviewer",
        "note": "研究解释不足",
        "candidate_content_hash": review["candidate_content_hash"],
        "evidence_hash": review["evidence_hash"],
    }
    with TestClient(_app(service)) as client:
        rejected = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json=payload,
        )
        assert rejected.status_code == 200
        assert rejected.json()["candidate"]["status"] == "rejected"
        payload["decision"] = "approved"
        conflict = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json=payload,
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "MINING-105"


def test_sse_envelope_is_json_and_has_resume_sequence(tmp_path: Path):
    service, _ = _service(tmp_path)
    run_id, _ = _seed_review_ready(service, "web-stream-envelope")
    page = service.events(run_id, after=0, limit=2)
    assert len(page["events"]) == 2
    assert page["has_more"] is True
    second = service.events(run_id, after=page["last_sequence"], limit=2)
    assert second["events"][0]["sequence"] > page["last_sequence"]
    with pytest.raises(Exception):
        service.events(run_id, after=-1)


def test_missing_worker_configuration_fails_without_mock_fallback(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("MINING_WORKER_URL", raising=False)
    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    service = MiningWebService(
        store=store,
        orchestrator=MiningOrchestrator(store=store),
        defaults=_constraints(),
        reviewer="local-reviewer",
        snapshot_check=lambda _context, **_kwargs: None,
    )
    with TestClient(_app(service)) as client:
        result = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期量价变化可能预测未来五日收益",
                "idempotency_key": "web-worker-missing",
                "constraints": {"max_candidates": 2, "full_eval_top_k": 1},
            },
        )
    assert result.status_code == 503
    assert result.json()["code"] == "MINING-119"


def test_publication_batch_api_starts_and_completes_approved_batch(
    tmp_path: Path,
):
    """S8 发布接口必须整批启动并把成功状态返回页面。"""

    service, _ = _service(tmp_path)
    run_id, candidate_id = _seed_review_ready(service, "web-publication-ready")
    review = service.candidate_detail(run_id, candidate_id)["review"]
    service.decide(
        run_id,
        candidate_id,
        {
            "decision": "approved",
            "reviewer": "local-reviewer",
            "note": "批准正式发布",
            "candidate_content_hash": review["candidate_content_hash"],
            "evidence_hash": review["evidence_hash"],
        },
    )
    candidate = service.store.get_candidate(run_id, candidate_id)
    service.publication_service._prepare = lambda **_kwargs: SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=(candidate["name"],),
    )
    service.publication_service._commit = lambda **_kwargs: None
    service.publication_service._rollback = lambda **_kwargs: None
    context_hash = service.store.get_run(run_id)["context"]["context_hash"]
    with TestClient(_app(service)) as client:
        started = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": [candidate_id],
                "expected_context_hash": context_hash,
            },
        )
        assert started.status_code == 202
        batch_id = started.json()["batch_id"]
        for _ in range(50):
            status = client.get(
                f"/api/mining/runs/{run_id}/publication-batches/{batch_id}"
            )
            if status.json()["status"] == "published":
                break
            time.sleep(0.02)
        assert status.json()["status"] == "published"
        detail = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}"
        )
        assert detail.json()["candidate"]["status"] == "published"
        assert detail.json()["publication"]["latest_batch"]["batch_id"] == batch_id
