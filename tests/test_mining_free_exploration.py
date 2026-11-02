"""自由探索初始方向契约、生成和 V2 编排测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mining.catalog import load_catalog
from mining.errors import FreeExplorationError, MiningContractError
from mining.free_exploration import (
    direction_from_hypothesis,
    direction_to_hypothesis,
    generate_initial_direction,
)
from mining.free_exploration_contracts import (
    EXPLORATION_SEED_MAX,
    EXPRESSION_DEPTH_BY_COMPLEXITY,
    ResearchDirection,
)
from mining.free_exploration_pool import (
    build_direction_pool,
    plan_next_branch_direction,
)
from mining.free_exploration_profiles import MECHANISM_PROFILES
from mining.search_budget import SearchBudget
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_contracts import SearchRunRequest
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from tests.test_mining_search_orchestrator import _orchestrator
from mining_worker.protocol import WorkerCandidateDraft, WorkerJobView
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchResponse,
)


def _context() -> FrozenSearchContext:
    catalog = load_catalog()
    return FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
    )


def _free_request(
    key: str = "free-direction-0001",
    *,
    seed: int | None = 42,
) -> SearchRunRequest:
    return SearchRunRequest(
        key,
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
        exploration_seed=seed,
    )


def test_free_exploration_seed_is_optional_and_changes_request_hash():
    derived = _free_request("free-seed-derived", seed=None)
    explicit = _free_request("free-seed-derived", seed=42)
    maximum = _free_request("free-seed-maximum", seed=EXPLORATION_SEED_MAX)

    assert derived.exploration_seed is None
    assert derived.effective_exploration_seed == int(
        derived.request_hash[:16],
        16,
    )
    assert explicit.effective_exploration_seed == 42
    assert maximum.exploration_seed == EXPLORATION_SEED_MAX
    assert derived.request_hash != explicit.request_hash
    assert "exploration_seed" not in derived.material()
    assert explicit.material()["exploration_seed"] == 42
    assert SearchRunRequest.from_dict(
        {
            "idempotency_key": explicit.idempotency_key,
            **explicit.to_dict(),
        }
    ) == explicit


def test_free_exploration_accepts_p3_01_persisted_request_without_seed():
    request = _free_request("free-seed-legacy", seed=None)
    legacy = request.to_dict()
    legacy.pop("exploration_seed")

    loaded = SearchRunRequest.from_dict(
        {
            "idempotency_key": request.idempotency_key,
            **legacy,
        }
    )

    assert loaded == request
    assert loaded.request_hash == request.request_hash
    assert loaded.effective_exploration_seed == int(loaded.request_hash[:16], 16)


def test_seed_is_rejected_for_manual_mode_and_invalid_values():
    with pytest.raises(MiningContractError, match="不得提供 exploration_seed"):
        SearchRunRequest(
            "manual-seed-0001",
            "低估值股票可能获得未来超额收益",
            exploration_seed=1,
        )
    for invalid in (-1, EXPLORATION_SEED_MAX + 1, True, 1.5):
        with pytest.raises(MiningContractError, match="exploration_seed"):
            _free_request(seed=invalid)
    with pytest.raises(MiningContractError, match="只有自由探索"):
        SearchRunRequest(
            "manual-seed-0002",
            "低估值股票可能获得未来超额收益",
        ).effective_exploration_seed


def test_direction_generation_is_deterministic_and_uses_frozen_catalog():
    request = _free_request(seed=20261102)
    context = _context()

    first = generate_initial_direction(request, context)
    second = generate_initial_direction(request, context)

    assert first == second
    assert first.direction_hash == second.direction_hash
    assert first.direction_id == second.direction_id
    assert first.run_id == request.run_id
    assert first.seed == 20261102
    assert first.seed_source == "request"
    assert first.request_hash == request.request_hash
    assert first.context_hash == context.context_hash
    assert first.mechanism_category in {
        profile.category for profile in MECHANISM_PROFILES
    }
    assert set(first.data_dependencies) <= {
        str(item["name"]) for item in context.field_catalog
    }
    assert first.max_expression_depth == EXPRESSION_DEPTH_BY_COMPLEXITY[
        first.expression_complexity
    ]
    assert ResearchDirection.from_dict(first.to_dict()) == first


def test_generation_covers_profiles_dependencies_and_complexities():
    context = _context()
    names = {
        str(item["name"])
        for item in context.field_catalog
    }
    assert {
        profile.category for profile in MECHANISM_PROFILES
    } == {
        "price_trend",
        "short_reversal",
        "volatility",
        "liquidity",
        "valuation",
        "profitability",
        "growth",
        "cashflow_quality",
    }
    assert all(set(profile.fields) & names for profile in MECHANISM_PROFILES)

    directions = tuple(
        generate_initial_direction(
            _free_request(f"free-coverage-{seed:04d}", seed=seed),
            context,
        )
        for seed in range(32)
    )
    assert len({item.mechanism_category for item in directions}) > 1
    assert len({item.data_dependencies for item in directions}) > 1
    assert len({item.expression_complexity for item in directions}) == 3


def test_generation_uses_derived_seed_source_and_fails_without_fields():
    request = _free_request(seed=None)
    direction = generate_initial_direction(request, _context())
    assert direction.seed == request.effective_exploration_seed
    assert direction.seed_source == "request_hash"

    empty = FrozenSearchContext(
        data_fingerprint="b" * 64,
        field_catalog=({"name": "unknown_numeric_field"},),
        operator_catalog=({"name": "mean"},),
    )
    with pytest.raises(FreeExplorationError, match="研究机制") as error:
        generate_initial_direction(request, empty)
    assert error.value.code == "MINING-161"

    no_operators = replace(empty, operator_catalog=())
    with pytest.raises(FreeExplorationError, match="算子"):
        generate_initial_direction(request, no_operators)
    with pytest.raises(MiningContractError, match="只有自由探索"):
        generate_initial_direction(
            SearchRunRequest(
                "manual-direction-0001",
                "低估值股票可能获得未来超额收益",
            ),
            _context(),
        )


def test_direction_hypothesis_round_trip_and_tamper_detection():
    direction = generate_initial_direction(_free_request(), _context())
    hypothesis = direction_to_hypothesis(direction)

    assert direction_from_hypothesis(hypothesis) == direction
    assert hypothesis.hypothesis_id
    changed = replace(hypothesis, statement="被篡改但仍满足长度要求的研究假设")
    with pytest.raises(MiningContractError, match="不一致"):
        direction_from_hypothesis(changed)


def test_free_exploration_run_freezes_direction_before_worker(tmp_path):
    orchestrator, store, worker = _orchestrator(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close", "label": "后复权收盘价"},
            {"name": "volume_share", "label": "成交量"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    run = orchestrator.create_run(
        _free_request("free-orchestrator-0001", seed=17),
        context,
        background=False,
    )

    first = orchestrator._initial_search_hypothesis(run, context)
    second = orchestrator._initial_search_hypothesis(run, context)

    assert first.card_hash == second.card_hash
    direction = direction_from_hypothesis(first)
    assert direction is not None
    events = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "free_exploration_direction_planned"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["direction"] == direction.to_dict()
    assert events[0]["payload"]["direction_hash"] == direction.direction_hash

    request = orchestrator._worker_request(
        run=run,
        context=context,
        current=first,
        turn_index=1,
        previous_evidence=None,
        trace={"candidates": []},
    )
    assert request.hypothesis.statement == direction.statement
    assert request.max_expression_depth == direction.max_expression_depth
    encoded = str(events[0]) + str(request.to_dict())
    for forbidden in (
        context.windows.selection_start,
        context.windows.final_start,
        context.windows.holdout_start,
    ):
        assert forbidden not in encoded
    assert worker.requests == []
    orchestrator.close()


def test_manual_run_keeps_original_initial_hypothesis_and_depth(tmp_path):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close", "label": "后复权收盘价"},),
        operator_catalog=({"name": "mean"},),
    )
    request = SearchRunRequest(
        "manual-direction-0002",
        "短期反转可能带来开发期超额收益",
    )
    run = orchestrator.create_run(request, context, background=False)

    hypothesis = orchestrator._initial_search_hypothesis(run, context)
    worker_request = orchestrator._worker_request(
        run=run,
        context=context,
        current=hypothesis,
        turn_index=1,
        previous_evidence=None,
        trace={"candidates": []},
    )

    assert hypothesis.statement == request.hypothesis
    assert hypothesis.metadata == {"source": "user"}
    assert worker_request.max_expression_depth == 6
    assert not any(
        item["event_type"] == "free_exploration_direction_planned"
        for item in store.list_events(run["run_id"])
    )
    orchestrator.close()


def test_direction_semantic_hash_ignores_wording_seed_and_content_hash():
    direction = generate_initial_direction(
        _free_request("free-semantic-0001", seed=20261102),
        _context(),
    )
    rewritten = replace(
        direction,
        statement="另一种措辞描述同一个冻结研究机制和字段依赖",
        mechanism="同义机制说明",
        seed=direction.seed + 1,
    )

    assert rewritten.semantic_hash == direction.semantic_hash
    assert rewritten.direction_hash != direction.direction_hash
    assert direction.semantic_material()["data_dependencies"] == sorted(
        direction.data_dependencies
    )
    assert replace(
        direction,
        mechanism_category="different_mechanism",
    ).semantic_hash != direction.semantic_hash
    assert replace(
        direction,
        expected_direction=(
            "negative"
            if direction.expected_direction != "negative"
            else "positive"
        ),
    ).semantic_hash != direction.semantic_hash


def test_direction_pool_is_diverse_deterministic_and_skips_prior_directions():
    request = _free_request("free-pool-0001", seed=20261102)
    context = _context()
    pool = build_direction_pool(request, context)

    assert pool == build_direction_pool(request, context)
    first_categories = []
    for direction in pool:
        if direction.mechanism_category not in first_categories:
            first_categories.append(direction.mechanism_category)
    supported_categories = {
        direction.mechanism_category
        for direction in pool
    }
    assert set(first_categories[: len(supported_categories)]) == (
        supported_categories
    )

    prior = (
        direction_to_hypothesis(pool[0], turn_index=1),
        direction_to_hypothesis(pool[1], turn_index=2),
    )
    first = plan_next_branch_direction(request, context, prior)
    second = plan_next_branch_direction(request, context, prior)

    assert first == second
    assert first.direction == pool[2]
    assert [
        item.direction.semantic_hash
        for item in first.duplicate_attempts
    ] == [pool[0].semantic_hash, pool[1].semantic_hash]
    assert first.plan_hash == second.plan_hash
    assert first.pool_hash


def test_direction_plan_fails_when_frozen_pool_is_exhausted():
    request = _free_request("free-pool-exhausted", seed=7)
    context = _context()
    pool = build_direction_pool(request, context)
    prior = tuple(
        direction_to_hypothesis(direction, turn_index=index)
        for index, direction in enumerate(pool, start=1)
    )

    with pytest.raises(FreeExplorationError, match="没有未使用") as error:
        plan_next_branch_direction(request, context, prior)

    assert error.value.code == "MINING-161"


def test_duplicate_branch_direction_is_ledgered_once_without_materialization(
    tmp_path,
):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    context = _rich_context()
    request = _free_request("free-duplicate-ledger", seed=20261102)
    run = orchestrator.create_run(request, context, background=False)
    pool = build_direction_pool(request, context)
    current = direction_to_hypothesis(pool[0], turn_index=1)
    store.prepare_hypothesis(current)
    store.begin_turn(
        run["run_id"],
        current,
        ExperimentBatch(
            run_id=run["run_id"],
            hypothesis_id=current.hypothesis_id,
            turn_index=1,
            branch_id=current.branch_id,
            candidate_ids=(),
            parent_candidate_ids=(),
            development_start=context.windows.development_start,
            development_end=context.windows.development_end,
            input_hash="a" * 64,
        ),
    )

    first = orchestrator._prepare_free_branch_hypothesis(
        run=run,
        context=context,
        current=current,
        decision_turn=1,
        next_turn=2,
        next_branch="branch-0002",
    )
    second = orchestrator._prepare_free_branch_hypothesis(
        run=run,
        context=context,
        current=current,
        decision_turn=1,
        next_turn=2,
        next_branch="branch-0002",
    )

    direction = direction_from_hypothesis(first)
    assert first.hypothesis_id == second.hypothesis_id
    assert direction == pool[1]
    counts = store.trial_summary(run["run_id"])["counts"]
    assert counts["duplicate"] == 1
    assert len(store.get_search_trace(run["run_id"])["hypotheses"]) == 2
    orchestrator.close()


class FreeBranchWorker:
    """按动作序列返回自由探索搜索响应。"""

    def __init__(self, actions: tuple[str, ...]) -> None:
        self.actions = actions
        self.requests = []

    def health(self):
        return {"status": "ready", "mode": "test"}

    def _response(self, turn_index: int) -> WorkerSearchResponse:
        action = self.actions[turn_index - 1]
        if action in {"stop", "promote"}:
            return WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action=action,
                    reason="自由探索测试结束搜索",
                ),
                next_hypothesis=None,
                candidates=(),
                usage={"calls": 1},
            )
        expressions = {
            1: "mean(close,5)",
            2: "mean(amount_cny,5)",
            3: "mean(pe_ttm,5)",
        }
        return WorkerSearchResponse(
            decision=WorkerDecisionProposal(
                action=action,
                reason="自由探索测试进入下一分支",
            ),
            next_hypothesis=WorkerSearchHypothesis(
                statement=f"Worker 建议的分支方向 {turn_index}",
                mechanism="Worker 分支建议",
                expected_direction="positive",
                falsification_criteria=("没有新增有效证据",),
            ),
            candidates=(
                WorkerCandidateDraft(
                    expression=expressions[turn_index],
                    mechanism="自由探索测试候选",
                    description="用于验证分支规划的候选",
                    direction=1,
                    steps=("读取冻结字段",),
                ),
            ),
            usage={"calls": 1},
        )

    def submit(self, request):
        self.requests.append(request)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self._response(request.turn_index).to_dict(),
        )

    def get_job(self, job_id):
        request = next(item for item in self.requests if item.job_id == job_id)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self._response(request.turn_index).to_dict(),
        )

    def cancel(self, job_id):
        return WorkerJobView(
            job_id=job_id,
            request_id="cancelled-request",
            status="cancelled",
        )


def _rich_context(*, budget=None) -> FrozenSearchContext:
    values = {
        "data_fingerprint": "a" * 64,
        "field_catalog": (
            {"name": "close", "label": "收盘价"},
            {"name": "amount_cny", "label": "成交额"},
            {"name": "pe_ttm", "label": "滚动市盈率"},
            {"name": "fi_roe", "label": "净资产收益率"},
        ),
        "operator_catalog": ({"name": "mean"},),
    }
    if budget is not None:
        values["budget"] = budget
    return FrozenSearchContext(**values)


def test_free_branch_uses_main_planned_direction(tmp_path):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    worker = FreeBranchWorker(("branch", "promote"))
    orchestrator._worker_client_value = worker
    context = _rich_context()
    request = _free_request(
        "free-branch-main-0001",
        seed=20261102,
    )
    run = orchestrator.create_run(request, context, background=False)

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    trace = store.get_search_trace(run["run_id"])
    assert [item["action"] for item in trace["decisions"]] == [
        "branch",
        "promote",
    ]
    assert len(trace["hypotheses"]) == 2
    assert trace["hypotheses"][1]["branch_id"] == "branch-0002"
    assert all(
        not item["statement"].startswith("Worker 建议的分支方向")
        for item in trace["hypotheses"]
    )
    direction = direction_from_hypothesis(
        HypothesisCard.from_dict(trace["hypotheses"][1])
    )
    assert direction is not None
    assert worker.requests[1].hypothesis.statement == direction.statement
    assert worker.requests[1].max_expression_depth == direction.max_expression_depth
    planned = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "free_exploration_direction_planned"
        and item["payload"].get("target_turn") == 2
    ]
    assert len(planned) == 1
    assert planned[0]["payload"]["plan_hash"]
    orchestrator.close()


def test_free_refine_inherits_planned_direction_and_depth(tmp_path):
    orchestrator, store, worker = _orchestrator(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "volume_share"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    request = _free_request("free-refine-inherit", seed=19)
    run = orchestrator.create_run(request, context, background=False)

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    assert worker.requests[1].max_expression_depth == (
        worker.requests[0].max_expression_depth
    )
    trace = store.get_search_trace(run["run_id"])
    first = direction_from_hypothesis(
        HypothesisCard.from_dict(trace["hypotheses"][0])
    )
    second = direction_from_hypothesis(
        HypothesisCard.from_dict(trace["hypotheses"][1])
    )
    assert second == first
    assert trace["hypotheses"][1]["metadata"]["source"] == (
        "free_exploration_refine"
    )
    orchestrator.close()


def test_free_branch_honors_turn_budget_before_worker_suggestion(tmp_path):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    worker = FreeBranchWorker(("branch", "branch"))
    orchestrator._worker_client_value = worker
    budget = SearchBudget(max_search_turns=2)
    context = _rich_context(budget=budget)
    request = SearchRunRequest(
        "free-budget-stop-0001",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
        exploration_seed=23,
        budget=budget,
    )
    run = orchestrator.create_run(request, context, background=False)

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    assert [item.turn_index for item in worker.requests] == [1, 2]
    decisions = store.get_search_trace(run["run_id"])["decisions"]
    assert [item["action"] for item in decisions] == ["branch", "promote"]
    assert "轮次预算耗尽" in decisions[-1]["reason"]
    orchestrator.close()


def test_free_branch_recovers_after_decision_before_hypothesis(tmp_path):
    orchestrator, store, _worker = _orchestrator(tmp_path)
    orchestrator._worker_client_value = FreeBranchWorker(("branch", "promote"))
    context = _rich_context()
    request = _free_request("free-branch-recover", seed=31)
    run = orchestrator.create_run(request, context, background=False)
    original = orchestrator._prepare_free_branch_hypothesis

    def interrupt_before_plan(**kwargs):
        raise KeyboardInterrupt("模拟分支方向持久化前退出")

    orchestrator._prepare_free_branch_hypothesis = interrupt_before_plan
    try:
        with pytest.raises(KeyboardInterrupt):
            orchestrator.execute(run["run_id"])
    finally:
        orchestrator._prepare_free_branch_hypothesis = original

    trace = store.get_search_trace(run["run_id"])
    assert trace["decisions"][-1]["action"] == "branch"
    assert len(trace["hypotheses"]) == 1
    expected = plan_next_branch_direction(
        request,
        context,
        trace["hypotheses"],
    )
    store.transition_search_run(
        run["run_id"],
        "interrupted",
        event_key="run_interrupted_before_branch",
        stage="searching",
    )
    orchestrator.resume(run["run_id"], background=False)
    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    recovered = store.get_search_trace(run["run_id"])
    direction = direction_from_hypothesis(
        HypothesisCard.from_dict(recovered["hypotheses"][1])
    )
    assert direction == expected.direction
    planned = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "free_exploration_direction_planned"
        and item["payload"].get("target_turn") == 2
    ]
    assert len(planned) == 1
    assert planned[0]["payload"]["plan_hash"] == expected.plan_hash
    counts = store.trial_summary(run["run_id"])["counts"]
    assert counts["duplicate"] == len(expected.duplicate_attempts)
    orchestrator.close()
