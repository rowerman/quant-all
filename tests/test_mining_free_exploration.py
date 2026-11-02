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
from mining.free_exploration_profiles import MECHANISM_PROFILES
from mining.search_context import FrozenSearchContext
from mining.search_contracts import SearchRunRequest
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from tests.test_mining_search_orchestrator import _orchestrator


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
