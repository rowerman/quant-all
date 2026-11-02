"""自由探索从 Web 请求到审核态的联合验收。"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.free_exploration import direction_from_hypothesis
from mining.search_cards import HypothesisCard
from mining.search_context import load_search_config
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store
from tests.test_mining_free_exploration import (
    FreeBranchWorker,
    _free_request,
    _rich_context,
)
from tests.test_mining_search_orchestrator import (
    FakeDevelopmentEvaluator,
    FakeFinalEvaluator,
    FakeHoldoutEvaluator,
    FakeMaterializer,
    FakePortfolioSelector,
    FakeSelectionEvaluator,
)


def _orchestrator(
    tmp_path: Path,
    worker: FreeBranchWorker | None = None,
) -> tuple[MiningSearchOrchestrator, MiningV2Store, FreeBranchWorker]:
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    active_worker = worker or FreeBranchWorker(("branch", "promote"))
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=active_worker,
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolioSelector(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    return orchestrator, store, active_worker


def _frozen_windows() -> tuple[str, ...]:
    windows = load_search_config().windows
    return (
        windows.selection_start,
        windows.selection_end,
        windows.final_start,
        windows.final_end,
        windows.holdout_start,
        windows.holdout_end,
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _recursive_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _recursive_keys(item)


def _assert_agent_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    keys = set(_recursive_keys(value))
    assert not keys & {
        "selection_start",
        "selection_end",
        "final_start",
        "final_end",
        "holdout_start",
        "holdout_end",
        "test_segment",
        "module_test",
        "prompt",
        "api_key",
        "authorization",
    }
    for forbidden in _frozen_windows():
        assert forbidden not in encoded
    for forbidden in (
        "FACTOR_DEEPSEEK_API",
        "MINING_WORKER_URL",
        "/home/hanwen",
        "Bearer ",
    ):
        assert forbidden not in encoded


def _assert_public_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    keys = set(_recursive_keys(value))
    assert not keys & {
        "prompt",
        "api_key",
        "authorization",
        "request_json",
    }
    for forbidden in (
        "FACTOR_DEEPSEEK_API",
        "MINING_WORKER_URL",
        "/home/hanwen",
        "Bearer ",
    ):
        assert forbidden not in encoded


def test_free_exploration_web_request_reaches_review_ready_without_leakage(
    tmp_path: Path,
):
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    orchestrator, v2_store, worker = _orchestrator(tmp_path)
    search_service = MiningSearchWebService(
        store=v2_store,
        orchestrator=orchestrator,
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    service = MiningWebService(
        store=v1_store,
        snapshot_check=lambda *_args, **_kwargs: None,
        search_service=search_service,
    )
    app = create_app(mining_service=service)

    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()["search_v2"]
        created = client.post(
            "/api/mining/runs",
            json={
                "input_mode": FREE_EXPLORATION_INPUT_MODE,
                "exploration_seed": "20261102",
                "idempotency_key": "free-e2e-web-0001",
                "budget": options["defaults"],
            },
        )
        assert created.status_code == 200
        run_id = created.json()["run"]["run_id"]

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            detail = client.get(f"/api/mining/runs/{run_id}").json()
            if detail["run"]["status"] in {
                "review_ready",
                "failed",
                "cancelled",
                "stale",
            }:
                break
            time.sleep(0.02)

        assert detail["run"]["status"] == "review_ready"
        assert detail["run"]["input_mode"] == FREE_EXPLORATION_INPUT_MODE
        assert detail["run"]["effective_exploration_seed"] == 20261102
        assert detail["exploration"]["initial_direction"]["seed"] == 20261102
        assert [item["action"] for item in detail["turns"]] == [
            "branch",
            "promote",
        ]
        assert detail["candidates"]
        assert all(
            item["status"] == "review_ready"
            for item in detail["candidates"]
        )
        assert all(
            v2_store.search_approval_eligibility(
                run_id,
                item["candidate_id"],
            ).eligible
            for item in detail["candidates"]
        )
        assert [request.turn_index for request in worker.requests] == [1, 2]
        for request in worker.requests:
            _assert_agent_safe(request.to_dict())

        events = client.get(
            f"/api/mining/runs/{run_id}/events?limit=1000"
        ).json()["events"]
        _assert_agent_safe(events)
        _assert_public_safe(detail)
        context_payload = json.dumps(detail["run"]["context"], ensure_ascii=False)
        assert "windows" not in detail["run"]["context"]
        for forbidden in _frozen_windows():
            assert forbidden not in context_payload


def test_free_exploration_cancel_before_worker_is_terminal(tmp_path: Path):
    orchestrator, store, worker = _orchestrator(tmp_path)
    context = _rich_context()
    run = orchestrator.create_run(
        _free_request("free-e2e-cancel-0001", seed=41),
        context,
        background=False,
    )
    store.transition_search_run(
        run["run_id"],
        "searching",
        event_key="free_cancel_searching",
        stage="searching",
    )

    orchestrator.cancel(run["run_id"])
    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "cancelled"
    assert worker.requests == []
    orchestrator.close()


class CrashOnSecondFreeTurnWorker(FreeBranchWorker):
    """模拟第二个 Worker 步骤执行前主服务退出。"""

    def submit(self, request):
        if request.turn_index == 2:
            raise KeyboardInterrupt("模拟自由探索第二次 Worker 前退出")
        return super().submit(request)


def test_free_exploration_resume_reuses_seed_direction_and_completed_turn(
    tmp_path: Path,
):
    orchestrator, store, _worker = _orchestrator(
        tmp_path,
        CrashOnSecondFreeTurnWorker(("branch", "promote")),
    )
    context = _rich_context()
    request = _free_request("free-e2e-resume-0001", seed=53)
    run = orchestrator.create_run(request, context, background=False)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.execute(run["run_id"])
    orchestrator.close()

    interrupted = store.transition_search_run(
        run["run_id"],
        "interrupted",
        event_key="free_resume_interrupted",
        stage="searching",
    )
    assert interrupted["interrupted_from"] == "searching"
    before = store.get_search_trace(run["run_id"])
    first_direction = direction_from_hypothesis(
        HypothesisCard.from_dict(before["hypotheses"][0])
    )
    assert first_direction is not None
    assert first_direction.seed == 53

    resumed, _same_store, resumed_worker = _orchestrator(tmp_path)
    resumed.resume(run["run_id"], background=False)
    result = resumed.execute(run["run_id"])

    assert result["status"] == "review_ready", {
        key: result.get(key)
        for key in (
            "status",
            "error_code",
            "error_message",
            "stop_reason",
            "last_completed_stage",
        )
    }
    assert [item.turn_index for item in resumed_worker.requests] == [2]
    after = store.get_search_trace(run["run_id"])
    initial_direction = direction_from_hypothesis(
        HypothesisCard.from_dict(after["hypotheses"][0])
    )
    assert initial_direction == first_direction
    current = resumed.progress(run["run_id"])
    assert current.turns_completed == 2
    for item in resumed_worker.requests:
        _assert_agent_safe(item.to_dict())
    resumed.close()
