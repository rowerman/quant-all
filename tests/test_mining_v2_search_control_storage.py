"""V2 运行状态、评估证据和轨迹测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mining.errors import MiningStateError
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store


def _store(tmp_path: Path) -> MiningV2Store:
    return MiningV2Store(tmp_path / "factor_mining.duckdb", tmp_path / "data")


def _run(store: MiningV2Store) -> dict:
    return store.create_run(
        SearchRunRequest(
            "search-run-state-0001",
            "短期反转可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )


def test_v2_run_state_machine_and_cancel_request(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    run_id = run["run_id"]

    searching = store.transition_search_run(
        run_id,
        "searching",
        event_key="run_searching",
        stage="searching",
    )
    assert searching["status"] == "searching"
    assert searching["reused"] is False

    with pytest.raises(MiningStateError, match="不能"):
        store.transition_search_run(
            run_id,
            "holding_out",
            event_key="invalid_jump",
            stage="holding_out",
        )

    cancel = store.request_search_cancel(run_id)
    assert cancel["cancel_requested"] is True
    assert store.cancel_requested(run_id) is True


def test_interrupted_v2_run_only_resumes_original_stage(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    run_id = run["run_id"]
    store.transition_search_run(
        run_id,
        "searching",
        event_key="run_searching",
        stage="searching",
    )
    store.transition_search_run(
        run_id,
        "interrupted",
        event_key="run_interrupted",
        stage="searching",
    )

    with pytest.raises(MiningStateError, match="原阶段"):
        store.transition_search_run(
            run_id,
            "selecting",
            event_key="wrong_resume",
            stage="selecting",
        )
    resumed = store.transition_search_run(
        run_id,
        "searching",
        event_key="run_resumed",
        stage="searching",
    )
    assert resumed["status"] == "searching"
    assert resumed["interrupted_from"] is None


def test_evaluation_evidence_is_immutable_and_trace_checked(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    run_id = run["run_id"]
    card = HypothesisCard(
        run_id=run_id,
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("年度同号率低于 0.6",),
    )
    batch = ExperimentBatch(
        run_id=run_id,
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id="branch-0001",
        candidate_ids=("cand_0001",),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run_id, card, batch)
    store.record_candidate(
        run_id,
        candidate_id="cand_0001",
        candidate_hash="e" * 64,
        semantic_hash="f" * 64,
        name="MINE_TEST_0001",
        status="proposed",
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "mean(close, 5)"},
    )

    evidence = SearchEvaluationEvidence(
        run_id=run_id,
        candidate_id="cand_0001",
        stage="development",
        input_hash="d" * 64,
        status="passed",
        metrics={"rank_ic": 0.03, "coverage": 0.8},
        warnings=(),
    )
    first = store.record_evaluation_evidence(evidence)
    second = store.record_evaluation_evidence(evidence)

    assert first["reused"] is False
    assert second["reused"] is True
    assert len(
        store.list_evaluation_evidence(
            run_id,
            "cand_0001",
            stage="development",
        )
    ) == 1
    assert store.validate_search_trace(run_id) is True
