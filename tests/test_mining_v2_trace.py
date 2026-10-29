"""挖掘 V2 turn、候选谱系、证据决策和链式轨迹测试。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mining.contracts import canonical_json
from mining.errors import (
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.search_context import FrozenSearchContext
from mining.search_contracts import (
    EvidencePacket,
    ExperimentBatch,
    HypothesisCard,
    SearchDecision,
    TrialCounts,
)
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store


def _store(tmp_path: Path) -> MiningV2Store:
    return MiningV2Store(tmp_path / "factor_mining.duckdb", tmp_path / "data")


def _run(
    store: MiningV2Store,
    *,
    key: str = "search-run-0001",
    hypothesis: str = "低估值股票可能带来未来超额收益",
) -> dict:
    request = SearchRunRequest(key, hypothesis)
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    return store.create_run(request, context)


def _card(run_id: str, **changes) -> HypothesisCard:
    payload = {
        "run_id": run_id,
        "turn_index": 1,
        "branch_id": "branch_value",
        "statement": "低估值股票可能带来未来超额收益",
        "mechanism": "估值修复",
        "expected_direction": "positive",
        "falsification_criteria": ("年度同号率低于 0.6",),
        "parent_hypothesis_id": None,
        "metadata": {"source": "user"},
    }
    payload.update(changes)
    return HypothesisCard(**payload)


def _batch(card: HypothesisCard, **changes) -> ExperimentBatch:
    payload = {
        "run_id": card.run_id,
        "hypothesis_id": card.hypothesis_id,
        "turn_index": card.turn_index,
        "branch_id": card.branch_id,
        "candidate_ids": ("cand_value_0001",),
        "parent_candidate_ids": (),
        "development_start": "2012-01-01",
        "development_end": "2019-12-31",
        "input_hash": "b" * 64,
        "parameters": {"window": 20},
    }
    payload.update(changes)
    return ExperimentBatch(**payload)


def _candidate(store: MiningV2Store, card: HypothesisCard, batch: ExperimentBatch):
    return store.record_candidate(
        card.run_id,
        candidate_id=batch.candidate_ids[0],
        candidate_hash="c" * 64,
        semantic_hash="d" * 64,
        name="MINE_VALUE_0001",
        status="proposed",
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "rank(ep)"},
    )


def _evidence(card: HypothesisCard, batch: ExperimentBatch) -> EvidencePacket:
    return EvidencePacket(
        run_id=card.run_id,
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        turn_index=card.turn_index,
        development_start="2012-01-01",
        development_end="2019-12-31",
        metrics={"rank_ic": 0.03},
        failures=(),
        correlation_clusters=(),
        trial_counts=TrialCounts(proposed=1, evaluated=1),
        budget_state={"turns_used": 1},
    )


def _decision(
    card: HypothesisCard,
    evidence: EvidencePacket,
    *,
    action: str = "refine",
    reason: str = "覆盖不足，需要修正窗口",
) -> SearchDecision:
    return SearchDecision(
        run_id=card.run_id,
        hypothesis_id=card.hypothesis_id,
        evidence_id=evidence.evidence_id,
        turn_index=card.turn_index,
        action=action,
        reason=reason,
    )


def test_begin_turn_candidate_evidence_and_decision_are_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card = _card(run["run_id"])
    batch = _batch(card)

    first_turn = store.begin_turn(run["run_id"], card, batch, snapshot_hash="e" * 64)
    second_turn = store.begin_turn(run["run_id"], card, batch, snapshot_hash="e" * 64)
    assert first_turn["status"] == "requested"
    assert second_turn["reused"] is True
    assert len(store.list_events(run["run_id"])) == 2

    first_candidate = _candidate(store, card, batch)
    second_candidate = _candidate(store, card, batch)
    assert first_candidate["reused"] is False
    assert second_candidate["reused"] is True
    assert first_candidate["lineage"]["batch_id"] == batch.batch_id
    assert len(store.list_events(run["run_id"])) == 3

    evidence = _evidence(card, batch)
    first_evidence = store.record_evidence(evidence)
    second_evidence = store.record_evidence(evidence)
    assert first_evidence["reused"] is False
    assert second_evidence["reused"] is True
    assert store.get_turn(run["run_id"], 1)["status"] == "evidence_ready"
    assert len(store.list_events(run["run_id"])) == 4

    decision = _decision(card, evidence)
    first_decision = store.record_decision(decision)
    second_decision = store.record_decision(decision)
    assert first_decision["reused"] is False
    assert second_decision["reused"] is True
    assert store.get_turn(run["run_id"], 1)["status"] == "decided"
    assert store.transition_turn(run["run_id"], 1, "completed")["status"] == "completed"
    assert len(store.list_events(run["run_id"])) == 6


def test_same_turn_with_different_content_conflicts(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    first_card = _card(run["run_id"])
    store.begin_turn(run["run_id"], first_card, _batch(first_card))

    changed_card = _card(
        run["run_id"],
        statement="高估值股票可能带来未来超额收益",
    )
    with pytest.raises(MiningDuplicateError, match="同一 turn"):
        store.begin_turn(run["run_id"], changed_card, _batch(changed_card))


def test_parent_nodes_cannot_cross_runs(tmp_path: Path):
    store = _store(tmp_path)
    first_run = _run(store)
    first_card = _card(first_run["run_id"])
    first_batch = _batch(first_card)
    store.begin_turn(first_run["run_id"], first_card, first_batch)
    first_candidate = _candidate(store, first_card, first_batch)

    second_run = _run(
        store,
        key="search-run-0002",
        hypothesis="现金流改善可能带来未来超额收益",
    )
    cross_card = _card(
        second_run["run_id"],
        statement="现金流改善可能带来未来超额收益",
        parent_hypothesis_id=first_card.hypothesis_id,
    )
    with pytest.raises(MiningNotFoundError, match="假设不存在"):
        store.begin_turn(second_run["run_id"], cross_card, _batch(cross_card))

    local_card = _card(
        second_run["run_id"],
        statement="现金流改善可能带来未来超额收益",
    )
    local_batch = _batch(local_card, candidate_ids=("cand_cash_0001",))
    store.begin_turn(second_run["run_id"], local_card, local_batch)
    with pytest.raises(MiningNotFoundError, match="候选不存在"):
        store.record_candidate(
            second_run["run_id"],
            candidate_id="cand_cash_0001",
            candidate_hash="1" * 64,
            semantic_hash="2" * 64,
            name="MINE_CASH_0001",
            status="proposed",
            hypothesis_id=local_card.hypothesis_id,
            batch_id=local_batch.batch_id,
            parent_candidate_id=first_candidate["candidate_id"],
            payload={"canonical_expression": "rank(cfo)"},
        )


def test_turn_state_machine_rejects_invalid_transition(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card = _card(run["run_id"])
    store.begin_turn(run["run_id"], card, _batch(card))
    with pytest.raises(MiningStateError, match="不能"):
        store.transition_turn(run["run_id"], 1, "completed")
    assert store.transition_turn(run["run_id"], 1, "failed")["status"] == "failed"
    assert store.transition_turn(run["run_id"], 1, "failed")["reused"] is True


def test_trace_hash_changes_and_survives_store_reopen(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card = _card(run["run_id"])
    batch = _batch(card)
    before = store.get_run(run["run_id"])["trace_hash"]
    store.begin_turn(run["run_id"], card, batch)
    after_turn = store.get_run(run["run_id"])["trace_hash"]
    _candidate(store, card, batch)
    after_candidate = store.get_run(run["run_id"])["trace_hash"]
    assert len({before, after_turn, after_candidate}) == 3
    assert store.validate_search_trace(run["run_id"]) is True

    reopened = MiningV2Store(store.db_path, store.data_root)
    trace = reopened.get_search_trace(run["run_id"])
    assert len(trace["hypotheses"]) == 1
    assert len(trace["experiment_batches"]) == 1
    assert len(trace["candidates"]) == 1
    assert trace["trace_hash"] == after_candidate


def test_trace_detects_payload_tampering(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card = _card(run["run_id"])
    batch = _batch(card)
    store.begin_turn(run["run_id"], card, batch)
    evidence = _evidence(card, batch)
    store.record_evidence(evidence)

    payload = evidence.to_dict()
    payload["metrics"]["rank_ic"] = 0.99
    with duckdb.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE mining_v2.evidence_packet SET payload_json=? "
            "WHERE run_id=? AND evidence_id=?",
            [canonical_json(payload), run["run_id"], evidence.evidence_id],
        )
    with pytest.raises(MiningStorageError, match="内容哈希"):
        store.get_search_trace(run["run_id"])


def test_candidate_schema_conflicts_are_reported_without_partial_trace(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card = _card(run["run_id"])
    batch = _batch(
        card,
        candidate_ids=("cand_value_0001", "cand_value_0002"),
    )
    store.begin_turn(run["run_id"], card, batch)
    _candidate(store, card, batch)
    before = store.get_run(run["run_id"])["trace_hash"]
    with pytest.raises(MiningDuplicateError, match="语义"):
        store.record_candidate(
            run["run_id"],
            candidate_id="cand_value_0002",
            candidate_hash="f" * 64,
            semantic_hash="d" * 64,
            name="MINE_VALUE_0002",
            status="proposed",
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            payload={"canonical_expression": "rank(ep2)"},
        )
    assert store.get_run(run["run_id"])["trace_hash"] == before
