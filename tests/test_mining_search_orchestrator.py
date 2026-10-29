"""V2 逐轮搜索闭环测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_types import sha256_json, utc_now_iso
from mining.errors import MiningStateError, SnapshotStaleError, WorkerTimeoutError
from mining.evaluation_contract import (
    CandidateEvaluationResult,
    FullEvaluationResult,
    HoldoutEvaluationResult,
)
from mining.search_context import FrozenSearchContext
from mining.search_budget import SearchBudget
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store
from mining_worker.protocol import WorkerCandidateDraft, WorkerJobView
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchResponse,
)


def _response(turn_index: int) -> WorkerSearchResponse:
    if turn_index == 3:
        return WorkerSearchResponse(
            decision=WorkerDecisionProposal(
                action="stop",
                reason="开发期证据已经足够完成入围选择",
            ),
            next_hypothesis=None,
            candidates=(),
            usage={"calls": 1},
        )
    expression = "mean(close,5)" if turn_index == 1 else "mean(volume_share,5)"
    return WorkerSearchResponse(
        decision=WorkerDecisionProposal(
            action="refine",
            reason="继续缩短窗口以验证信号稳定性",
        ),
        next_hypothesis=WorkerSearchHypothesis(
            statement=f"第 {turn_index + 1} 轮短期反转假设继续验证",
            mechanism="短期价格修复",
            expected_direction="positive",
            falsification_criteria=("连续两轮没有改善",),
        ),
        candidates=(
            WorkerCandidateDraft(
                expression=expression,
                mechanism="短期价格修复",
                description=f"第 {turn_index} 轮候选",
                direction=1,
                steps=("读取字段", "计算短期均值"),
            ),
        ),
        usage={"calls": 1},
    )


class SequenceWorker:
    def __init__(self):
        self.requests = []

    def health(self):
        return {"status": "ready", "mode": "test"}

    def submit(self, request):
        self.requests.append(request)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=_response(request.turn_index).to_dict(),
        )

    def get_job(self, job_id):
        request = next(item for item in self.requests if item.job_id == job_id)
        return self.submit_without_record(request)

    def submit_without_record(self, request):
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=_response(request.turn_index).to_dict(),
        )

    def cancel(self, job_id):
        return WorkerJobView(
            job_id=job_id,
            request_id="cancelled-request",
            status="cancelled",
        )


def _report(evaluation_input, stage, status="passed", *, metrics=None):
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
                "start": evaluation_input.quick_window,
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


class FakeDevelopmentEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        if "amount_cny" in candidate.dependencies:
            input_value = SimpleNamespace(
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate,
                context=context,
                quick_window=(
                    context.windows.development_start,
                    context.windows.development_end,
                ),
            )
            return CandidateEvaluationResult(
                gate=TechnicalGate(
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    candidate_content_hash=candidate.content_hash,
                    context_hash="a" * 64,
                    rule_version="test-v1",
                    passed=False,
                    checks={"values": {"coverage": 0.1}},
                ),
                technical=_report(
                    input_value,
                    "technical",
                    "failed",
                    metrics={"values": {"coverage": 0.1}},
                ),
                quick=None,
                full=None,
                warnings=(),
                factor_glob="/tmp/factor",
                factor_sha256="b" * 64,
            )
        rank_ic = (
            0.05
            if "open" in candidate.dependencies
            else (
                0.04
                if "volume_share" in candidate.dependencies
                else 0.03
            )
        )
        input_value = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.development_start,
                context.windows.development_end,
            ),
        )
        gate = TechnicalGate(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate_content_hash=candidate.content_hash,
            context_hash="a" * 64,
            rule_version="test-v1",
            passed=True,
            checks={"values": {"coverage": 0.8}},
        )
        technical = _report(
            input_value,
            "technical",
            metrics={"values": {"coverage": 0.8}},
        )
        quick = _report(
            input_value,
            "quick",
            metrics={
                "rank_ic_mean": rank_ic,
                "rank_ic_ir": 0.4,
                "year_sign_consistency": 0.75,
                "valid_days": 100,
            },
        )
        return CandidateEvaluationResult(
            gate=gate,
            technical=technical,
            quick=quick,
            full=None,
            warnings=(),
            factor_glob="/tmp/factor",
            factor_sha256="b" * 64,
        )


class FakeSelectionEvaluator(FakeDevelopmentEvaluator):
    def __call__(self, *, run_id, candidate, context, stage_root):
        result = super().__call__(
            run_id=run_id,
            candidate=candidate,
            context=context,
            stage_root=stage_root,
        )
        return result


class FakeFinalEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        input_value = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.final_start,
                context.windows.final_end,
            ),
        )
        return FullEvaluationResult(
            report=_report(input_value, "full", metrics={"complete": True}),
            warnings=(),
            factor_glob="/tmp/final",
            factor_sha256="c" * 64,
        )


class FakeHoldoutEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        input_value = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.holdout_start,
                context.windows.holdout_end,
            ),
        )
        return HoldoutEvaluationResult(
            report=_report(
                input_value,
                "holdout",
                metrics={"complete": True, "valid_days": 100},
            ),
            passed=True,
            warnings=(),
            factor_glob="/tmp/holdout",
            factor_sha256="d" * 64,
        )


class FakeMaterializer:
    def __call__(self, items, context, *, start, end):
        return SimpleNamespace(failures=())


def _orchestrator(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    worker = SequenceWorker()
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=worker,
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    return orchestrator, store, worker


def test_search_closes_loop_and_reaches_review_ready(tmp_path: Path):
    orchestrator, store, worker = _orchestrator(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "volume_share"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-orchestrator-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    assert [request.turn_index for request in worker.requests] == [1, 2, 3]
    trace = store.get_search_trace(run["run_id"])
    assert [item["action"] for item in trace["decisions"]] == [
        "refine",
        "refine",
        "stop",
    ]
    assert len(trace["hypotheses"]) == 3
    assert len(trace["candidates"]) == 2
    assert all(
        item["status"] == "review_ready" for item in trace["candidates"]
    )
    assert store.trial_summary(run["run_id"])["counts"]["holdout_claimed"] == 1
    assert store.validate_search_trace(run["run_id"]) is True
    progress = orchestrator.progress(run["run_id"])
    assert progress.status == "review_ready"
    assert progress.turns_completed == 3
    assert progress.current_branch_id == "branch-0001"


def test_search_request_never_contains_selection_or_holdout_window(tmp_path: Path):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    # 构造不执行、只检查请求构造器所需的最小运行和上下文。
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    run = store.create_run(
        SearchRunRequest(
            "search-request-view-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )
    from mining.search_orchestrator_support import initial_hypothesis

    request = orchestrator._worker_request(
        run=run,
        context=context,
        current=initial_hypothesis(run),
        turn_index=1,
        previous_evidence=None,
        trace={"candidates": []},
    )
    encoded = str(request.to_dict())

    assert context.windows.selection_start not in encoded
    assert context.windows.final_start not in encoded
    assert context.windows.holdout_start not in encoded
    orchestrator.close()


class IsolationWorker(SequenceWorker):
    def submit(self, request):
        self.requests.append(request)
        if request.turn_index == 1:
            response = WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action="refine",
                    reason="保留通过候选并记录失败候选",
                ),
                next_hypothesis=WorkerSearchHypothesis(
                    statement="第二轮确认已有开发期证据是否稳定",
                    mechanism="稳定性确认",
                    expected_direction="positive",
                    falsification_criteria=("没有新增有效证据",),
                ),
                candidates=(
                    WorkerCandidateDraft(
                        expression="mean(close,5)",
                        mechanism="短期价格修复",
                        description="可评估候选",
                        direction=1,
                        steps=("读取收盘价",),
                    ),
                    WorkerCandidateDraft(
                        expression="mean(amount_cny,5)",
                        mechanism="流动性冲击",
                        description="门禁失败候选",
                        direction=1,
                        steps=("读取成交额",),
                    ),
                ),
                usage={"calls": 1},
            )
        else:
            response = _response(3)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=response.to_dict(),
        )

    def submit_without_record(self, request):
        self.requests = [item for item in self.requests if item is not request]
        return self.submit(request)


def test_single_candidate_failure_does_not_abort_turn(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=IsolationWorker(),
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "amount_cny"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-isolation-0001",
            "短期量价变化可能带来开发期超额收益",
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    trace = store.get_search_trace(run["run_id"])
    statuses = {
        item["candidate_id"]: item["status"] for item in trace["candidates"]
    }
    assert sorted(statuses.values()) == ["review_ready", "technical_failed"]
    counts = store.trial_summary(run["run_id"])["counts"]
    assert counts["gate_failed"] == 1
    assert counts["holdout_claimed"] == 1


class CrashOnSecondTurnWorker(SequenceWorker):
    def submit(self, request):
        if request.turn_index == 2:
            raise KeyboardInterrupt("模拟主服务进程退出")
        return super().submit(request)


def test_interrupted_run_resumes_from_completed_turn_without_replaying(
    tmp_path: Path,
):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "volume_share"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    run = store.create_run(
        SearchRunRequest(
            "search-recovery-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )
    first = MiningSearchOrchestrator(
        store=store,
        worker=CrashOnSecondTurnWorker(),
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    interrupted = store.transition_search_run(
        run["run_id"],
        "interrupted",
        event_key="run_interrupted_for_test",
        stage="searching",
    )
    assert interrupted["interrupted_from"] == "searching"

    worker = SequenceWorker()
    resumed = MiningSearchOrchestrator(
        store=store,
        worker=worker,
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    try:
        resumed.resume(run["run_id"], background=False)
        with pytest.raises(MiningStateError, match="中断"):
            resumed.resume(run["run_id"], background=False)
        result = resumed.execute(run["run_id"])
    finally:
        resumed.close()

    assert result["status"] == "review_ready"
    assert [request.turn_index for request in worker.requests] == [2, 3]
    assert resumed.progress(run["run_id"]).turns_completed == 3


def test_cancel_before_search_stops_without_worker_call(tmp_path: Path):
    orchestrator, store, worker = _orchestrator(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-cancel-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
        background=False,
    )
    store.transition_search_run(
        run["run_id"],
        "searching",
        event_key="run_searching",
        stage="searching",
    )
    orchestrator.cancel(run["run_id"])

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "cancelled"
    assert worker.requests == []
    orchestrator.close()


def test_stale_snapshot_marks_v2_run_stale(tmp_path: Path):
    def stale(*args, **kwargs):
        raise SnapshotStaleError("数据快照已变化")

    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=SequenceWorker(),
        snapshot_check=stale,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-stale-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "stale"
    orchestrator.close()


class TimeoutWorker(SequenceWorker):
    def submit(self, request):
        raise WorkerTimeoutError("Worker v2 超时")


def test_worker_timeout_has_stable_failure_state_and_code(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=TimeoutWorker(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-timeout-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "failed"
    failures = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "search_run_failed"
    ]
    assert failures[-1]["payload"]["error_code"] == "MINING-116"
    orchestrator.close()


def test_startup_interrupts_only_active_v2_runs(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    active = store.create_run(
        SearchRunRequest(
            "search-manual-interrupt-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )
    queued = store.create_run(
        SearchRunRequest(
            "search-manual-interrupt-0002",
            "估值修复可能带来开发期超额收益",
        ),
        context,
    )
    store.transition_search_run(
        active["run_id"],
        "searching",
        event_key="run_searching",
        stage="searching",
    )

    interrupted = store.interrupt_active_search_runs()

    assert [item["run_id"] for item in interrupted] == [active["run_id"]]
    assert store.get_run(active["run_id"])["status"] == "interrupted"
    assert store.get_run(queued["run_id"])["status"] == "queued"


def test_search_budget_exhaustion_promotes_existing_quality_candidate(
    tmp_path: Path,
):
    orchestrator, store, worker = _orchestrator(tmp_path)
    budget = SearchBudget(
        max_search_turns=1,
        max_hypothesis_branches=2,
        max_refines_per_branch=1,
        max_candidates_per_turn=1,
        max_candidates_per_run=4,
        max_full_evaluations=1,
        stop_after_no_improvement_turns=1,
        max_tool_calls_per_turn=0,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
        budget=budget,
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-budget-stop-0001",
            "短期反转可能带来开发期超额收益",
            budget=budget,
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    assert len(worker.requests) == 1
    decisions = store.get_search_trace(run["run_id"])["decisions"]
    assert decisions[0]["action"] == "promote"
    assert "预算耗尽" in decisions[0]["reason"]
    orchestrator.close()


class BranchWorker(SequenceWorker):
    def submit(self, request):
        self.requests.append(request)
        if request.turn_index == 4:
            response = WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action="stop",
                    reason="新分支已经产生可用证据",
                ),
                next_hypothesis=None,
                candidates=(),
                usage={"calls": 1},
            )
        else:
            expressions = {
                1: "mean(close,5)",
                2: "mean(low,5)",
                3: "mean(low,3)",
            }
            response = WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action="refine",
                    reason="继续修订同一假设",
                ),
                next_hypothesis=WorkerSearchHypothesis(
                    statement=f"第 {request.turn_index} 轮继续验证短期反转",
                    mechanism="短期价格修复",
                    expected_direction="positive",
                    falsification_criteria=("连续两轮没有改善",),
                ),
                candidates=(
                    WorkerCandidateDraft(
                        expression=expressions[request.turn_index],
                        mechanism="短期价格修复",
                        description="短期价格统计",
                        direction=1,
                        steps=("读取价格字段",),
                    ),
                ),
                usage={"calls": 1},
            )
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=response.to_dict(),
        )

    def submit_without_record(self, request):
        return self.submit(request)


def test_no_improvement_threshold_switches_branch(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=BranchWorker(),
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        snapshot_check=lambda *args, **kwargs: None,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "low"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        SearchRunRequest(
            "search-branch-stop-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
        background=False,
    )

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    decisions = store.get_search_trace(run["run_id"])["decisions"]
    assert [item["action"] for item in decisions] == [
        "refine",
        "refine",
        "branch",
        "stop",
    ]
    assert any(
        item["branch_id"] == "branch-0002"
        for item in store.get_search_trace(run["run_id"])["hypotheses"]
    )
    orchestrator.close()
