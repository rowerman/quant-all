"""挖掘 V2 分层窗口、冻结上下文和访问角色测试。"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from common.paths import MINING_SEARCH_CONFIG_FILE
from mining.errors import MiningContractError, MiningVisibilityError
from mining.search_context import (
    AGENT_ROLE,
    CONTROLLER_ROLE,
    DEFAULT_SEARCH_WINDOWS,
    EVALUATOR_ROLE,
    HOLDOUT_ROLE,
    FrozenSearchContext,
    SearchConfig,
    SearchWindows,
    ensure_agent_visible,
    load_search_config,
)
from mining.search_contracts import (
    EvidencePacket,
    ExperimentBatch,
    HypothesisCard,
    TrialCounts,
)


def _context(**changes) -> FrozenSearchContext:
    payload = {
        "data_fingerprint": "a" * 64,
        "llm_profile": "default",
        "llm_model": "deepseek/deepseek-chat",
        "field_catalog": ({"name": "close"},),
        "operator_catalog": ({"name": "rank"},),
        "existing_factors": ("MOM_20",),
        "created_at": "2026-10-29T08:00:00+08:00",
    }
    payload.update(changes)
    return FrozenSearchContext(**payload)


def _hypothesis(**changes) -> HypothesisCard:
    payload = {
        "run_id": "run_search_0001",
        "turn_index": 1,
        "branch_id": "branch_value",
        "statement": "低估值股票可能获得更高的未来收益",
        "mechanism": "估值修复",
        "expected_direction": "positive",
        "falsification_criteria": ("年度同号率低于 0.6",),
    }
    payload.update(changes)
    return HypothesisCard(**payload)


def test_default_search_config_matches_frozen_windows_and_budget():
    config = load_search_config()
    assert config.windows == DEFAULT_SEARCH_WINDOWS
    assert config.windows.to_dict() == {
        "development_start": "2012-01-01",
        "development_end": "2019-12-31",
        "selection_start": "2020-01-01",
        "selection_end": "2022-12-31",
        "final_start": "2023-01-01",
        "final_end": "2025-10-15",
        "holdout_start": "2025-10-16",
        "holdout_end": "2026-10-15",
    }
    assert config.budget.max_search_turns == 6
    assert config.budget.max_candidates_per_run == 24
    assert config.control_policy.correlation_cluster_threshold == 0.90
    assert config.control_policy.stability_variant_limit == 2
    assert config.control_policy.stability_rank_ic_ratio_floor == 0.50
    assert SearchConfig.from_dict(config.to_dict()) == config
    assert MINING_SEARCH_CONFIG_FILE == Path("configs/mining/search.json").resolve()


def test_search_windows_reject_reversed_or_overlapping_ranges():
    with pytest.raises(MiningContractError, match="不得晚于"):
        SearchWindows(
            development_start="2020-01-01",
            development_end="2019-12-31",
        )
    with pytest.raises(MiningContractError, match="必须按顺序"):
        SearchWindows(selection_start="2019-01-01")
    with pytest.raises(MiningContractError, match="必须按顺序"):
        SearchWindows(holdout_start="2025-10-15")


def test_search_config_rejects_unknown_version_and_noncanonical_windows(tmp_path):
    payload = json.loads(MINING_SEARCH_CONFIG_FILE.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "search.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MiningContractError, match="未知"):
        load_search_config(path)

    payload.pop("unexpected")
    payload["schema_version"] = "mining_search_v3"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MiningContractError, match="schema_version"):
        load_search_config(path)

    payload["schema_version"] = "mining_search_v2"
    payload["windows"]["development_start"] = "2012-01-02"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MiningContractError, match="冻结定义"):
        load_search_config(path)


def test_role_ranges_isolate_selection_final_and_holdout_windows():
    windows = DEFAULT_SEARCH_WINDOWS
    windows.assert_access(AGENT_ROLE, "2012-01-01", "2019-12-31")
    windows.assert_access(CONTROLLER_ROLE, "2020-01-01", "2022-12-31")
    windows.assert_access(EVALUATOR_ROLE, "2023-01-01", "2025-10-15")
    windows.assert_access(HOLDOUT_ROLE, "2025-10-16", "2026-10-15")

    with pytest.raises(MiningVisibilityError, match="无权"):
        windows.assert_access(AGENT_ROLE, "2020-01-01", "2020-12-31")
    with pytest.raises(MiningVisibilityError, match="无权"):
        windows.assert_access(CONTROLLER_ROLE, "2019-12-01", "2020-01-31")
    with pytest.raises(MiningVisibilityError, match="无权"):
        windows.assert_access(EVALUATOR_ROLE, "2025-10-16", "2025-12-31")


def test_frozen_context_hash_covers_windows_budget_and_rules():
    context = _context()
    changed_time = replace(
        context,
        created_at="2026-10-30T00:00:00+00:00",
    )
    changed_budget = replace(
        context,
        budget=replace(context.budget, max_search_turns=5),
    )
    changed_catalog = replace(
        context,
        field_catalog=({"name": "amount"},),
    )
    assert context.context_hash == changed_time.context_hash
    assert len(
        {
            context.context_hash,
            changed_budget.context_hash,
            changed_catalog.context_hash,
        }
    ) == 3
    assert context.windows_hash
    assert context.context_id.startswith("searchctx_")


def test_search_context_round_trip_and_v1_context_rejection():
    context = _context()
    assert FrozenSearchContext.from_dict(context.to_dict()) == context
    with pytest.raises(MiningContractError, match="缺少"):
        FrozenSearchContext.from_dict(
            {
                "schema_version": "mining_v1",
                "data_fingerprint": "a" * 64,
            }
        )


def test_worker_view_exposes_only_development_window():
    context = _context()
    worker = context.worker_view()
    assert worker["development_start"] == "2012-01-01"
    assert worker["development_end"] == "2019-12-31"
    rendered = json.dumps(worker, ensure_ascii=False)
    for forbidden in ("selection", "final", "holdout", "2020-01-01", "2025-10-16"):
        assert forbidden not in rendered


def test_agent_payload_rejects_reserved_evidence_and_out_of_window_dates():
    ensure_agent_visible(
        {
            "development_start": "2012-01-01",
            "development_end": "2019-12-31",
            "metrics": {"rank_ic": 0.02},
        }
    )
    with pytest.raises(MiningVisibilityError, match="selection_metrics"):
        ensure_agent_visible({"selection_metrics": {"rank_ic": 0.03}})
    with pytest.raises(MiningVisibilityError, match="holdout"):
        ensure_agent_visible({"nested": [{"holdout_start": "2025-10-16"}]})
    with pytest.raises(MiningVisibilityError, match="超出开发期"):
        ensure_agent_visible(
            {
                "validation_start": "2019-01-01",
                "validation_end": "2020-01-31",
            }
        )

    context = _context()
    with pytest.raises(MiningVisibilityError, match="非开发期"):
        context.assert_worker_payload({"notes": "2020-01-01"})


def test_experiment_and_evidence_windows_are_limited_to_development_period():
    hypothesis = _hypothesis()
    common = {
        "run_id": hypothesis.run_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "turn_index": 1,
        "branch_id": hypothesis.branch_id,
        "candidate_ids": ("cand_value_0001",),
        "parent_candidate_ids": (),
        "input_hash": "a" * 64,
    }
    with pytest.raises(MiningContractError, match="必须位于"):
        ExperimentBatch(
            **common,
            development_start="2019-12-01",
            development_end="2020-01-31",
        )
    with pytest.raises(MiningContractError, match="必须位于"):
        EvidencePacket(
            run_id=hypothesis.run_id,
            hypothesis_id=hypothesis.hypothesis_id,
            batch_id="batch_value_0001",
            turn_index=1,
            development_start="2022-01-01",
            development_end="2022-12-31",
            metrics={"rank_ic": 0.03},
            failures=(),
            correlation_clusters=(),
            trial_counts=TrialCounts(proposed=1),
            budget_state={"turns_used": 1},
        )
