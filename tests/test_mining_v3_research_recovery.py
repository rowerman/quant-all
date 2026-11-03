"""V3 资料驱动运行的重启恢复测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mining.research_run_contracts import ResearchRunInput
from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from tests.mining_v3_recovery_support import (
    build_research_fixture,
    open_runtime,
    research_orchestrator,
)
from tests.test_mining_research_run import (
    FakeEvaluator,
    FakeFinalEvaluator,
    FakeHoldoutEvaluator,
    FakeMaterializer,
    FakePortfolio,
    FakeSearchWorker,
)


def test_research_restart_does_not_repeat_extraction(tmp_path: Path):
    (
        runtime,
        contract,
        request,
        context,
        research_input,
        root,
    ) = build_research_fixture(tmp_path)
    first = research_orchestrator(runtime, root, interrupting=True)
    run = first.create_run(
        request,
        context,
        background=False,
        research_input=research_input,
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, _contract = open_runtime(tmp_path)
    runtime.interrupt_orphaned_search_runs()
    recovered = runtime.recover_v3_run(run["run_id"])
    assert isinstance(recovered.research_input, ResearchRunInput)
    assert recovered.research_input == research_input

    second = MiningV3SearchOrchestrator(
        store=runtime,
        worker=FakeSearchWorker(),
        research_material_root=root,
        research_preparer=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("恢复不应重新执行资料抽取")
            )
        ),
        materializer=FakeMaterializer(),
        development_evaluator=FakeEvaluator(),
        selection_evaluator=FakeEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolio(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    assert result["status"] == "review_ready", result
    assert contract.load_input(run["run_id"]).input_mode == "research_report"
