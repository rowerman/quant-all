"""V3 模块测试段在四类来源下的中断恢复测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from tests.mining_v3_recovery_support import (
    NoWorker,
    build_costeer_fixture,
    build_free_fixture,
    build_hypothesis_fixture,
    build_research_fixture,
    open_runtime,
    search_orchestrator,
)
from tests.test_mining_free_exploration import FreeBranchWorker
from tests.test_mining_research_run import (
    FakeEvaluator,
    FakeFinalEvaluator,
    FakeMaterializer,
    FakePortfolio,
    FakeResearchConverter,
    FakeResearchExtractor,
    FakeSearchWorker,
)
from tests.test_mining_search_orchestrator import (
    FakeDevelopmentEvaluator,
    FakeFinalEvaluator as SearchFakeFinalEvaluator,
    FakeHoldoutEvaluator as SearchFakeHoldoutEvaluator,
    FakeMaterializer as SearchFakeMaterializer,
    FakePortfolioSelector,
    FakeSelectionEvaluator,
    SequenceWorker,
)


SOURCE_KINDS = (
    "hypothesis",
    "free_exploration",
    "research_report",
    "costeer",
)
CHECKPOINTS = (
    "claim_written",
    "holdout_running",
    "evidence_saved",
    "candidates_review_ready",
)


class CrashHoldout(SearchFakeHoldoutEvaluator):
    """模拟模块测试段执行期间进程退出。"""

    def __call__(self, **kwargs):
        raise KeyboardInterrupt("模拟模块测试段执行中退出")


class RecordingHoldout(SearchFakeHoldoutEvaluator):
    """记录恢复后实际重新执行的候选。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *, candidate, **kwargs):
        self.calls.append(candidate.candidate_id)
        return super().__call__(candidate=candidate, **kwargs)


def _source_fixture(tmp_path: Path, kind: str):
    if kind == "hypothesis":
        runtime, contract, request, context = build_hypothesis_fixture(tmp_path)
        return runtime, contract, request, context, None, None
    if kind == "free_exploration":
        runtime, contract, request, context = build_free_fixture(tmp_path)
        return runtime, contract, request, context, None, None
    if kind == "research_report":
        (
            runtime,
            contract,
            request,
            context,
            research_input,
            root,
        ) = build_research_fixture(tmp_path)
        return runtime, contract, request, context, research_input, root
    runtime, contract, request, context, _candidate = build_costeer_fixture(
        tmp_path,
        "0301",
    )
    return runtime, contract, request, context, None, None


def _source_orchestrator(
    runtime,
    kind: str,
    *,
    material_root: Path | None,
    holdout_evaluator,
) -> MiningV3SearchOrchestrator:
    if kind == "research_report":
        return MiningV3SearchOrchestrator(
            store=runtime,
            worker=FakeSearchWorker(),
            research_material_root=material_root,
            research_extractor=FakeResearchExtractor(),
            research_converter=FakeResearchConverter(),
            materializer=FakeMaterializer(),
            development_evaluator=FakeEvaluator(),
            selection_evaluator=FakeEvaluator(),
            final_evaluator=FakeFinalEvaluator(),
            holdout_evaluator=holdout_evaluator,
            portfolio_selector=FakePortfolio(),
            snapshot_check=lambda *_args, **_kwargs: None,
        )
    worker = (
        FreeBranchWorker(("branch", "promote"))
        if kind == "free_exploration"
        else NoWorker()
        if kind == "costeer"
        else SequenceWorker()
    )
    return search_orchestrator(
        runtime,
        worker,
        holdout_evaluator=holdout_evaluator,
    )


def _install_crash(runtime, checkpoint: str) -> None:
    if checkpoint == "claim_written":
        original = runtime.transition_candidate

        def transition_candidate(*args, **kwargs):
            if kwargs.get("target") == "holdout_running" or (
                len(args) >= 3 and args[2] == "holdout_running"
            ):
                raise KeyboardInterrupt("模拟测试段领取后、执行前退出")
            return original(*args, **kwargs)

        runtime.transition_candidate = transition_candidate
        return
    if checkpoint == "evidence_saved":
        original = runtime.record_evaluation_evidence

        def record_evaluation_evidence(evidence):
            result = original(evidence)
            if evidence.stage == "holdout":
                raise KeyboardInterrupt("模拟测试段证据落库后退出")
            return result

        runtime.record_evaluation_evidence = record_evaluation_evidence
        return
    if checkpoint == "candidates_review_ready":
        original = runtime.transition_search_run

        def transition_search_run(run_id, target, **kwargs):
            if target == "review_ready":
                raise KeyboardInterrupt("模拟候选可审核后、运行收口前退出")
            return original(run_id, target, **kwargs)

        runtime.transition_search_run = transition_search_run


@pytest.mark.parametrize("kind", SOURCE_KINDS)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_v3_holdout_restart_recovers_exact_claimed_batch(
    tmp_path: Path,
    kind: str,
    checkpoint: str,
):
    (
        runtime,
        contract,
        request,
        context,
        research_input,
        material_root,
    ) = _source_fixture(tmp_path, kind)
    first = _source_orchestrator(
        runtime,
        kind,
        material_root=material_root,
        holdout_evaluator=(
            CrashHoldout()
            if checkpoint == "holdout_running"
            else SearchFakeHoldoutEvaluator()
        ),
    )
    _install_crash(runtime, checkpoint)
    create_kwargs = (
        {"research_input": research_input}
        if kind == "research_report"
        else {}
    )
    run = first.create_run(
        request,
        context,
        background=False,
        **create_kwargs,
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, reopened_contract = open_runtime(tmp_path)
    interrupted = runtime.interrupt_orphaned_search_runs()
    assert interrupted[0]["interrupted_from"] == "holding_out"
    events_before = runtime.list_events(run["run_id"])
    claim_events = [
        item for item in events_before
        if item["event_key"] == "holdout_claimed"
    ]
    assert len(claim_events) == 1
    claimed = list(claim_events[0]["payload"]["candidate_ids"])
    evidence_before = {
        row["evidence"]["candidate_id"]
        for row in runtime.list_evaluation_evidence(
            run["run_id"],
            stage="holdout",
        )
    }

    holdout = RecordingHoldout()
    second = _source_orchestrator(
        runtime,
        kind,
        material_root=material_root,
        holdout_evaluator=holdout,
    )
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    trace = runtime.get_search_trace(run["run_id"])
    statuses = {
        item["candidate_id"]: item["status"] for item in trace["candidates"]
    }
    events_after = runtime.list_events(run["run_id"])
    assert result["status"] == "review_ready", result
    assert set(claimed) <= set(statuses)
    assert {statuses[candidate_id] for candidate_id in claimed} == {
        "review_ready"
    }
    assert sorted(holdout.calls) == sorted(set(claimed) - evidence_before)
    assert sum(
        event["event_key"] == "holdout_claimed" for event in events_after
    ) == 1
    assert {
        review.candidate_id
        for review in reopened_contract.list_candidate_reviews(run["run_id"])
        if review.eligible
    } == set(claimed)


def test_v3_holdout_restart_reuses_failed_evidence(tmp_path: Path):
    runtime, _contract, request, context = build_hypothesis_fixture(tmp_path)

    class FailingHoldout(SearchFakeHoldoutEvaluator):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            return type(result)(
                report=replace(result.report, status="failed"),
                passed=False,
                warnings=("模块测试段未通过",),
                factor_glob=result.factor_glob,
                factor_sha256=result.factor_sha256,
            )

    first = search_orchestrator(
        runtime,
        SequenceWorker(),
        holdout_evaluator=FailingHoldout(),
    )
    _install_crash(runtime, "evidence_saved")
    run = first.create_run(request, context, background=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, _contract = open_runtime(tmp_path)
    runtime.interrupt_orphaned_search_runs()
    claim = next(
        event for event in runtime.list_events(run["run_id"])
        if event["event_key"] == "holdout_claimed"
    )
    claimed = list(claim["payload"]["candidate_ids"])
    evidence_before = {
        row["evidence"]["candidate_id"]
        for row in runtime.list_evaluation_evidence(
            run["run_id"],
            stage="holdout",
        )
    }
    holdout = RecordingHoldout()
    second = search_orchestrator(
        runtime,
        SequenceWorker(),
        holdout_evaluator=holdout,
    )
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    trace = runtime.get_search_trace(run["run_id"])
    statuses = {
        item["candidate_id"]: item["status"] for item in trace["candidates"]
    }
    assert result["status"] == "review_ready"
    assert evidence_before
    assert sorted(holdout.calls) == sorted(set(claimed) - evidence_before)
    assert {
        statuses[candidate_id]
        for candidate_id in evidence_before
    } == {"holdout_failed"}


def test_v3_holdout_recovery_rejects_candidate_state_without_evidence(
    tmp_path: Path,
):
    runtime, _contract, request, context = build_hypothesis_fixture(tmp_path)
    first = search_orchestrator(
        runtime,
        SequenceWorker(),
        holdout_evaluator=CrashHoldout(),
    )
    run = first.create_run(request, context, background=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, _contract = open_runtime(tmp_path)
    runtime.interrupt_orphaned_search_runs()
    claim = next(
        event for event in runtime.list_events(run["run_id"])
        if event["event_key"] == "holdout_claimed"
    )
    candidate_id = claim["payload"]["candidate_ids"][0]
    runtime.transition_candidate(
        run["run_id"],
        candidate_id,
        "review_ready",
    )

    second = search_orchestrator(runtime, SequenceWorker())
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    assert result["status"] == "failed"
    state = runtime.search_control_state(run["run_id"])
    assert "测试段状态与证据不一致" in state["stop_reason"]
