"""S6 挖掘编排的预算、冻结、恢复和取消联合测试。"""

from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.candidate_batch import CandidateBatchFailure
from mining.contract_requests import FrozenContext, MiningConstraints, MiningRunRequest
from mining.contract_types import sha256_json, utc_now_iso
from mining.errors import MiningError, MiningNotFoundError, SnapshotStaleError
from mining.evaluation_contract import (
    CandidateEvaluationResult,
    FullEvaluationResult,
    HoldoutEvaluationResult,
)
from mining.orchestrator import MiningOrchestrator
from mining.orchestration_contracts import load_orchestration_constraints
from mining.snapshot import (
    assert_context_snapshot,
    data_snapshot_fingerprint,
)
from mining.storage import MiningStore
from mining.worker_client import MockWorkerClient
from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerationResponse,
)


def _draft(expression: str, description: str) -> WorkerCandidateDraft:
    """返回合法 Worker 候选草稿。"""

    return WorkerCandidateDraft(
        expression=expression,
        mechanism="价格或流动性趋势",
        description=description,
        direction=1,
        steps=("读取字段", "计算短期均值"),
    )


class CapturingWorker(MockWorkerClient):
    """保留每一轮请求的确定性 Worker。"""

    def __init__(self, response: WorkerGenerationResponse) -> None:
        super().__init__(response)
        self.requests = []

    def submit(self, request):
        if (
            not self.requests
            or self.requests[-1].request_id != request.request_id
        ):
            self.requests.append(request)
        return super().submit(request)


def _report(
    evaluation_input,
    stage: str,
    status: str,
    *,
    metrics: dict | None = None,
) -> EvaluationReport:
    """构造与当前候选绑定的测试报告。"""

    return EvaluationReport(
        run_id=evaluation_input.run_id,
        candidate_id=evaluation_input.candidate_id,
        stage=stage,
        candidate_content_hash=evaluation_input.candidate.content_hash,
        context_hash=evaluation_input.context.context_hash,
        input_hash=sha256_json(
            {
                "candidate": evaluation_input.candidate_id,
                "stage": stage,
            }
        ),
        status=status,
        metrics=metrics or {},
        warnings=(),
        artifact_path=None,
        artifact_sha256=None,
        error_code=None,
        error_message=None,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
    )


class FakeEvaluator:
    """返回固定研究期评估结果。"""

    def __init__(self, *, quick_status: str = "passed") -> None:
        self.quick_status = quick_status

    def __call__(self, evaluation_input, *, policy=None):
        seed = int(evaluation_input.candidate.content_hash[:8], 16)
        gate = TechnicalGate(
            run_id=evaluation_input.run_id,
            candidate_id=evaluation_input.candidate_id,
            candidate_content_hash=evaluation_input.candidate.content_hash,
            context_hash=evaluation_input.context.context_hash,
            rule_version="test-v1",
            passed=True,
            checks={"values": {"coverage": 0.8 + seed % 10 / 100}},
        )
        technical = _report(
            evaluation_input,
            "technical",
            "passed",
            metrics={"values": {"coverage": 0.8 + seed % 10 / 100}},
        )
        quick = _report(
            evaluation_input,
            "quick",
            self.quick_status,
            metrics={
                "rank_ic_mean": 0.01 + seed % 100 / 10000,
                "rank_ic_ir": 0.1 + seed % 10 / 100,
            },
        )
        return CandidateEvaluationResult(
            gate=gate,
            technical=technical,
            quick=quick,
            full=None,
            warnings=(),
            factor_glob="/tmp/factor",
            factor_sha256="f" * 64,
        )


class FakeBatchMaterializer:
    """跳过真实数据库，返回预设的候选级失败。"""

    def __init__(self, failures=(), *, fail_first: bool = False) -> None:
        self.failures = tuple(failures)
        self.fail_first = fail_first
        self._failed_once = False
        self.calls = []

    def __call__(self, items, context, *, start, end):
        items = tuple(items)
        self.calls.append(items)
        failures = list(self.failures)
        if self.fail_first and items and not self._failed_once:
            failures.append(
                CandidateBatchFailure(
                    candidate_id=items[0].candidate.candidate_id,
                    error_code="MINING-106",
                    message="测试批量物化失败",
                )
            )
            self._failed_once = True
        return SimpleNamespace(failures=tuple(failures))


class FakeFullEvaluator:
    """返回固定完整评估结果。"""

    def __call__(self, evaluation_input, *, policy=None):
        report = _report(
            evaluation_input,
            "full",
            "passed",
            metrics={"events": {"overview": {"metrics": {}}}},
        )
        return FullEvaluationResult(
            report=report,
            warnings=(),
            factor_glob="/tmp/factor",
            factor_sha256="f" * 64,
        )


class FakeHoldoutEvaluator:
    """返回固定测试段结果。"""

    def __init__(self, *, passed: bool = True) -> None:
        self.passed = passed

    def __call__(self, evaluation_input, *, policy=None):
        report = _report(
            evaluation_input,
            "holdout",
            "passed" if self.passed else "failed",
            metrics={"valid_days": 100, "complete": self.passed},
        )
        return HoldoutEvaluationResult(
            report=report,
            passed=self.passed,
            warnings=(),
            factor_glob="/tmp/holdout-factor",
            factor_sha256="e" * 64,
        )


def _constraints(**overrides) -> MiningConstraints:
    payload = {
        "max_rounds": 1,
        "max_candidates": 4,
        "full_eval_top_k": 2,
        "max_llm_calls": 2,
        "max_failure_rate": 0.75,
        "min_quality_checks_for_failure_stop": 4,
    }
    payload.update(overrides)
    payload["min_quality_checks_for_failure_stop"] = min(
        int(payload["min_quality_checks_for_failure_stop"]),
        int(payload["max_candidates"]),
    )
    return MiningConstraints(**payload)


def _seed_run(
    store: MiningStore,
    constraints: MiningConstraints,
) -> dict:
    """创建不依赖真实 Worker 的待执行运行。"""

    request = MiningRunRequest(
        "orchestrator-test-0001",
        "短期反转可能带来未来五日超额收益",
        constraints=constraints,
    )
    context = _context(constraints)
    return store.create_run(request, context, worker_mode="rdagent")


def _context(constraints: MiningConstraints) -> FrozenContext:
    """返回带最小 Worker 目录的测试上下文。"""

    return FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
        budget=constraints,
    )


def _orchestrator(
    store: MiningStore,
    worker,
    *,
    fake_evaluator=None,
    batch_materializer=None,
    holdout_passed: bool = True,
) -> MiningOrchestrator:
    """创建跳过真实快照、使用确定性评估的编排器。"""

    orchestrator = MiningOrchestrator(
        store=store,
        worker=worker,
        evaluator=fake_evaluator or FakeEvaluator(),
        batch_materializer=batch_materializer or FakeBatchMaterializer(),
        full_evaluator=FakeFullEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(passed=holdout_passed),
    )
    orchestrator._assert_snapshot = lambda _context: None
    return orchestrator


def _worker() -> CapturingWorker:
    """返回含四个合法候选的 Worker。"""

    return CapturingWorker(
        WorkerGenerationResponse(
            candidates=(
                _draft("mean(close,5)", "收盘价短期均值"),
                _draft("mean(volume_share,5)", "成交量短期均值"),
                _draft("mean(amount_cny,5)", "成交额短期均值"),
                _draft("mean(turnover_rate,5)", "换手率短期均值"),
            ),
            usage={"calls": 1, "mode": "mock"},
        )
    )


def test_full_orchestration_reaches_review_ready_and_claims_once(tmp_path: Path):
    """正常链路完成生成、筛选、完整评估和一次性测试段。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    worker = _worker()
    batch_materializer = FakeBatchMaterializer()
    orchestrator = _orchestrator(
        store,
        worker,
        batch_materializer=batch_materializer,
    )

    final = orchestrator.execute(run["run_id"])
    repeated = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "review_ready"
    assert repeated["status"] == "review_ready"
    statuses = [item["status"] for item in store.list_candidates(run["run_id"])]
    assert statuses.count("review_ready") == 2
    assert statuses.count("screened_out") == 2
    usage = store.get_holdout_usage()
    assert len(usage["candidate_ids"]) == 2
    progress = orchestrator.progress(run["run_id"])
    assert progress.llm_calls_used == 1
    assert progress.rounds_completed == 1
    assert progress.quality_checks_used == 4
    assert progress.full_evaluations_used == 2
    assert progress.holdout_claimed is True
    second_usage = store.get_holdout_usage()
    assert second_usage["batch_hash"] == usage["batch_hash"]
    assert len(worker.requests) == 1
    assert [len(items) for items in batch_materializer.calls] == [4, 2]


def test_batch_materialization_failure_isolated_per_candidate(tmp_path: Path):
    """一个候选批量物化失败后，其他候选仍完成筛选和测试段。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    batch_materializer = FakeBatchMaterializer(fail_first=True)
    orchestrator = _orchestrator(
        store,
        _worker(),
        batch_materializer=batch_materializer,
    )

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    candidates = store.list_candidates(run["run_id"])
    assert final["status"] == "review_ready"
    assert sum(item["status"] == "technical_failed" for item in candidates) == 1
    assert sum(item["status"] == "review_ready" for item in candidates) == 2
    assert sum(item["status"] == "screened_out" for item in candidates) == 1
    assert all(item["error_code"] == "MINING-106" for item in candidates if item["status"] == "technical_failed")
    assert [len(items) for items in batch_materializer.calls] == [4, 2]


def test_create_run_background_executes_with_frozen_context(tmp_path: Path):
    """公开创建入口会冻结真实目录和快照并后台执行。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    worker = CapturingWorker(
        WorkerGenerationResponse(
            candidates=(_draft("mean(close,5)", "收盘价短期均值"),),
            usage={"calls": 1},
        )
    )
    orchestrator = MiningOrchestrator(
        store=store,
        worker=worker,
        evaluator=FakeEvaluator(),
        full_evaluator=FakeFullEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
    )
    request = MiningRunRequest(
        "orchestrator-background-0001",
        "短期反转可能带来未来五日超额收益",
        constraints=_constraints(max_candidates=1, full_eval_top_k=1),
    )

    run = orchestrator.create_run(request)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            current = store.get_run(run["run_id"])
        except MiningError:
            time.sleep(0.2)
            continue
        if current["status"] in {"review_ready", "failed", "cancelled", "stale"}:
            break
        time.sleep(0.2)
    orchestrator.close()

    assert current["status"] == "review_ready"
    assert len(current["context"]["payload"]["field_catalog"]) == 84
    assert len(current["context"]["payload"]["operator_catalog"]) == 34


def test_orchestration_config_matches_contract_defaults():
    """默认配置和契约默认预算保持一致。"""

    assert load_orchestration_constraints() == MiningConstraints()


def test_feedback_contains_only_research_generation_summary(tmp_path: Path):
    """Worker 反馈不含测试段内容或原始公式。"""

    constraints = _constraints(
        max_rounds=2,
        max_llm_calls=2,
        max_candidates=8,
        full_eval_top_k=2,
    )
    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, constraints)
    worker = _worker()
    orchestrator = _orchestrator(store, worker)

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "review_ready"
    assert len(worker.requests) == 2
    feedback = worker.requests[1].to_dict()["feedback"]
    assert feedback
    text = str(feedback)
    assert "holdout" not in text
    assert "rank_ic" not in text
    assert "mean(close" not in text


def test_failure_rate_threshold_prevents_full_evaluation_and_holdout(
    tmp_path: Path,
):
    """四个候选均失败时在 75% 阈值处终止。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    orchestrator = _orchestrator(
        store,
        _worker(),
        fake_evaluator=FakeEvaluator(quick_status="failed"),
    )

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "failed"
    assert final["error_code"] == "MINING-106"
    assert all(
        item["status"] == "screened_out"
        for item in store.list_candidates(run["run_id"])
    )
    with pytest.raises(MiningNotFoundError):
        store.get_holdout_usage()


def test_all_holdout_failures_remain_visible_for_review(tmp_path: Path):
    """测试段失败是业务结论，不伪装成运行异常。"""

    constraints = _constraints(max_candidates=1, full_eval_top_k=1)
    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, constraints)
    worker = CapturingWorker(
        WorkerGenerationResponse(
            candidates=(_draft("mean(close,5)", "收盘价短期均值"),),
            usage={"calls": 1},
        )
    )
    orchestrator = _orchestrator(store, worker, holdout_passed=False)

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "review_ready"
    assert store.list_candidates(run["run_id"])[0]["status"] == "holdout_failed"
    assert len(store.get_holdout_usage()["candidate_ids"]) == 1


def test_invalid_worker_output_fails_without_holdout(tmp_path: Path):
    """Worker 候选全部非法时不打开测试段。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    worker = CapturingWorker(
        WorkerGenerationResponse(
            candidates=(
                _draft("SELECT * FROM md.daily_hfq", "非法 SQL"),
            ),
            usage={"calls": 1},
        )
    )
    orchestrator = _orchestrator(store, worker)

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "failed"
    assert final["error_code"] == "MINING-106"
    assert final["status"] != "holding_out"
    with pytest.raises(MiningNotFoundError):
        store.get_holdout_usage()


def test_cancel_is_atomic_for_run_and_candidate(tmp_path: Path):
    """取消运行同时取消未终态候选。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    worker = _worker()
    orchestrator = MiningOrchestrator(store=store, worker=worker)
    from mining.worker_client import generate_worker_candidates

    request = MiningRunRequest(
        "orchestrator-test-0001",
        "短期反转可能带来未来五日超额收益",
        constraints=_constraints(),
    )
    from mining.orchestration_support import (
        build_worker_request,
        context_from_run,
        request_from_run,
    )

    worker_request = build_worker_request(
        request=request_from_run(store.get_run(run["run_id"])),
        context=context_from_run(store.get_run(run["run_id"])),
        round_index=1,
        feedback=(),
        max_candidates=4,
    )
    definition = generate_worker_candidates(worker, worker_request).accepted[0]
    store.add_candidate(run["run_id"], definition)

    cancelled = orchestrator.cancel(run["run_id"])
    orchestrator.close()

    assert cancelled["status"] == "cancelled"
    assert store.list_candidates(run["run_id"])[0]["status"] == "cancelled"


def test_resume_reuses_completed_round_without_worker_call(tmp_path: Path):
    """恢复已生成运行时不重复调用 Worker。"""

    from mining.expression_validation import validate_expression
    from mining.orchestration_support import build_worker_request

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints(max_candidates=1, full_eval_top_k=1))
    store.transition_run(
        run["run_id"],
        "generating",
        event_key="run_generating",
        stage="generating",
    )
    request = MiningRunRequest(
        "orchestrator-test-0001",
        "短期反转可能带来未来五日超额收益",
        constraints=_constraints(max_candidates=1, full_eval_top_k=1),
    )
    worker_request = build_worker_request(
        request=request,
        context=_context(request.constraints),
        round_index=1,
        feedback=(),
        max_candidates=1,
    )
    definition = validate_expression("mean(close,5)", direction=1).to_candidate(
        mechanism="短期趋势",
        description="收盘价短期均值",
        steps=("计算收盘价短期均值",),
        discovery_round=1,
        discovery_index=1,
    )
    store.add_candidate(run["run_id"], definition)
    store.append_event(
        run["run_id"],
        event_key="round_1_requested",
        stage="generating",
        event_type="round_requested",
        payload={
            "round": 1,
            "max_candidates": 1,
            "feedback": [],
            "payload_hash": worker_request.payload_hash,
        },
    )
    store.append_event(
        run["run_id"],
        event_key="round_1_completed",
        stage="generating",
        event_type="round_completed",
        payload={
            "round": 1,
            "new_candidates": 1,
            "feedback_entry": {
                "round": 1,
                "stored": 1,
                "rejected": 0,
                "duplicates": 0,
                "error_codes": [],
            },
            "usage": {"calls": 1},
        },
    )
    store.interrupt_active_runs()
    worker = _worker()
    orchestrator = _orchestrator(store, worker)

    resumed = orchestrator.resume(run["run_id"], background=False)
    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert resumed["status"] == "generating"
    assert final["status"] == "review_ready"
    assert worker.requests == []


def test_snapshot_change_marks_run_and_candidate_stale(tmp_path: Path):
    """快照变化时运行与候选原子进入 stale。"""

    from mining.expression_validation import validate_expression

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    run = _seed_run(store, _constraints())
    definition = validate_expression("mean(close,5)", direction=1).to_candidate(
        mechanism="短期趋势",
        description="收盘价短期均值",
        steps=("计算收盘价短期均值",),
        discovery_round=1,
        discovery_index=1,
    )
    store.add_candidate(run["run_id"], definition)
    orchestrator = MiningOrchestrator(store=store, worker=_worker())

    def stale(_context):
        raise SnapshotStaleError("测试快照变化")

    orchestrator._assert_snapshot = stale
    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "stale"
    assert final["error_code"] == "MINING-120"
    assert store.list_candidates(run["run_id"])[0]["status"] == "stale"


def test_snapshot_fingerprint_and_context_check(tmp_path: Path):
    """数据文件或清单变化会被识别。"""

    database = tmp_path / "quant.duckdb"
    manifest = tmp_path / "manifest.json"
    database.write_bytes(b"database")
    manifest.write_text('{"data_version":"test"}', encoding="utf-8")
    fingerprint = data_snapshot_fingerprint(database, manifest)
    context = FrozenContext(data_fingerprint=fingerprint)

    assert_context_snapshot(context, db_file=database, manifest_file=manifest)
    manifest.write_text('{"data_version":"changed"}', encoding="utf-8")
    with pytest.raises(SnapshotStaleError) as error:
        assert_context_snapshot(context, db_file=database, manifest_file=manifest)
    assert error.value.code == "MINING-120"
