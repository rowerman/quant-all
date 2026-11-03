"""挖掘 V2 尝试台账、预算和候选状态事务测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mining.errors import (
    MiningDuplicateError,
    MiningStateError,
)
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext
from mining.search_contracts import ExperimentBatch, HypothesisCard, TrialRecord
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store


def _store(tmp_path: Path) -> MiningV2Store:
    return MiningV2Store(tmp_path / "factor_mining.duckdb", tmp_path / "data")


def _run(
    store: MiningV2Store,
    *,
    key: str = "search-run-0001",
    hypothesis: str = "低估值股票可能带来未来超额收益",
    budget: SearchBudget | None = None,
) -> dict:
    selected = budget or SearchBudget()
    request = SearchRunRequest(key, hypothesis, budget=selected)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=selected,
    )
    return store.create_run(request, context)


def _card(
    run_id: str,
    *,
    turn_index: int = 1,
    branch_id: str = "branch_value",
) -> HypothesisCard:
    return HypothesisCard(
        run_id=run_id,
        turn_index=turn_index,
        branch_id=branch_id,
        statement=f"第 {turn_index} 轮低估值股票可能带来未来超额收益",
        mechanism="估值修复",
        expected_direction="positive",
        falsification_criteria=("年度同号率低于 0.6",),
    )


def _batch(card: HypothesisCard, candidate_ids: tuple[str, ...]) -> ExperimentBatch:
    return ExperimentBatch(
        run_id=card.run_id,
        hypothesis_id=card.hypothesis_id,
        turn_index=card.turn_index,
        branch_id=card.branch_id,
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )


def _prepare_candidates(
    store: MiningV2Store,
    run: dict,
    count: int = 1,
    *,
    turn_index: int = 1,
    branch_id: str = "branch_value",
):
    card = _card(run["run_id"], turn_index=turn_index, branch_id=branch_id)
    candidate_ids = tuple(f"cand_value_{index:04d}" for index in range(1, count + 1))
    batch = _batch(card, candidate_ids)
    store.begin_turn(run["run_id"], card, batch)
    candidates = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        candidates.append(
            store.record_candidate(
                run["run_id"],
                candidate_id=candidate_id,
                candidate_hash=f"{index:x}" * 64,
                semantic_hash=f"{index + 4:x}" * 64,
                name=f"MINE_VALUE_{index:04d}",
                status="proposed",
                hypothesis_id=card.hypothesis_id,
                batch_id=batch.batch_id,
                payload={"canonical_expression": f"rank(ep_{index})"},
            )
        )
    return card, batch, candidates


def _trial(
    card: HypothesisCard,
    batch: ExperimentBatch,
    candidate_id: str,
    *,
    kind: str,
    key: str,
) -> TrialRecord:
    return TrialRecord(
        run_id=card.run_id,
        trial_key=key,
        kind=kind,
        turn_index=card.turn_index,
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        candidate_id=candidate_id,
        reason_code=None if kind != "invalid" else "contract",
        payload={"source": "test"},
    )


def test_trial_ledger_counts_every_attempt_and_is_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card, batch, candidates = _prepare_candidates(store, run)
    candidate_id = candidates[0]["candidate_id"]

    first = store.record_trial(
        _trial(
            card,
            batch,
            candidate_id,
            kind="proposed",
            key="trial-proposed-1",
        )
    )
    replay = store.record_trial(
        _trial(
            card,
            batch,
            candidate_id,
            kind="proposed",
            key="trial-proposed-1",
        )
    )
    assert first["reused"] is False
    assert replay["reused"] is True

    for kind in (
        "rejected",
        "duplicate",
        "invalid",
        "gate_failed",
        "evaluated",
        "full_evaluated",
        "holdout_claimed",
    ):
        store.record_trial(
            _trial(
                card,
                batch,
                candidate_id,
                kind=kind,
                key=f"trial-{kind}-1",
            )
        )
    counts = store.trial_counts(run["run_id"])
    assert counts.to_dict() == {
        "proposed": 1,
        "rejected": 1,
        "duplicate": 1,
        "invalid": 1,
        "gate_failed": 1,
        "evaluated": 1,
        "full_evaluated": 1,
        "holdout_claimed": 1,
        "invalidated": 0,
    }
    summary = store.trial_summary(run["run_id"])
    assert summary["total_events"] == 8
    assert summary["failed_attempts"] == 3
    assert len(store.list_trials(run["run_id"])) == 8
    assert store.get_search_trace(run["run_id"])["trial_count"] == 8

    changed = _trial(
        card,
        batch,
        candidate_id,
        kind="evaluated",
        key="trial-proposed-1",
    )
    with pytest.raises(MiningDuplicateError, match="trial_key"):
        store.record_trial(changed)


def test_trial_invalidation_is_append_only_and_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card, batch, candidates = _prepare_candidates(store, run)
    trial = store.record_trial(
        _trial(
            card,
            batch,
            candidates[0]["candidate_id"],
            kind="proposed",
            key="trial-proposed-1",
        )
    )
    first = store.invalidate_trial(
        run["run_id"],
        trial["trial_id"],
        reason_code="输入代码回滚",
    )
    second = store.invalidate_trial(
        run["run_id"],
        trial["trial_id"],
        reason_code="输入代码回滚",
    )
    assert first["reused"] is False
    assert second["reused"] is True
    counts = store.trial_counts(run["run_id"])
    assert counts.proposed == 1
    assert counts.invalidated == 1
    listed = store.list_trials(run["run_id"])
    assert listed[0]["invalidated"] is True
    with pytest.raises(MiningDuplicateError, match="作废原因"):
        store.invalidate_trial(
            run["run_id"],
            trial["trial_id"],
            reason_code="另一个原因",
        )


def test_candidate_transition_and_trial_commit_or_rollback_together(
    tmp_path: Path,
    monkeypatch,
):
    store = _store(tmp_path)
    run = _run(store)
    card, batch, candidates = _prepare_candidates(store, run)
    candidate_id = candidates[0]["candidate_id"]

    def fail_counter(*_args, **_kwargs):
        raise RuntimeError("injected counter failure")

    monkeypatch.setattr(store, "_increment_counter_conn", fail_counter)
    with pytest.raises(RuntimeError, match="injected"):
        store.transition_candidate(
            run["run_id"],
            candidate_id,
            "validating",
            trial=_trial(
                card,
                batch,
                candidate_id,
                kind="proposed",
                key="trial-proposed-1",
            ),
        )
    assert store.get_candidate(run["run_id"], candidate_id)["status"] == "proposed"
    assert store.trial_counts(run["run_id"]).proposed == 0

    monkeypatch.undo()
    updated = store.transition_candidate(
        run["run_id"],
        candidate_id,
        "validating",
        trial=_trial(
            card,
            batch,
            candidate_id,
            kind="proposed",
            key="trial-proposed-1",
        ),
    )
    assert updated["status"] == "validating"
    assert store.trial_counts(run["run_id"]).proposed == 1
    assert store.validate_search_trace(run["run_id"]) is True


def test_candidate_budget_is_atomic_and_cannot_be_expanded(tmp_path: Path):
    budget = SearchBudget(
        max_candidates_per_turn=2,
        max_candidates_per_run=2,
        max_full_evaluations=2,
    )
    store = _store(tmp_path)
    run = _run(store, budget=budget)
    card, batch, candidates = _prepare_candidates(store, run, count=3)
    for index in range(2):
        store.record_trial(
            _trial(
                card,
                batch,
                candidates[index]["candidate_id"],
                kind="proposed",
                key=f"trial-proposed-{index}",
            )
        )
    before = store.get_run(run["run_id"])["trace_hash"]
    with pytest.raises(MiningStateError, match="max_candidates_per_run"):
        store.record_trial(
            _trial(
                card,
                batch,
                candidates[2]["candidate_id"],
                kind="proposed",
                key="trial-proposed-overflow",
            )
        )
    assert store.get_run(run["run_id"])["trace_hash"] == before
    with pytest.raises(MiningStateError, match="max_candidates_per_run"):
        store.transition_candidate(
            run["run_id"],
            candidates[2]["candidate_id"],
            "validating",
            trial=_trial(
                card,
                batch,
                candidates[2]["candidate_id"],
                kind="proposed",
                key="trial-proposed-overflow",
            ),
        )
    assert store.get_candidate(
        run["run_id"],
        candidates[2]["candidate_id"],
    )["status"] == "proposed"


def test_full_evaluation_and_holdout_budgets_are_enforced(tmp_path: Path):
    budget = SearchBudget(max_full_evaluations=1)
    store = _store(tmp_path)
    run = _run(store, budget=budget)
    card, batch, candidates = _prepare_candidates(store, run)
    candidate_id = candidates[0]["candidate_id"]
    store.record_trial(
        _trial(
            card,
            batch,
            candidate_id,
            kind="full_evaluated",
            key="trial-full-1",
        )
    )
    with pytest.raises(MiningStateError, match="max_full_evaluations"):
        store.record_trial(
            _trial(
                card,
                batch,
                candidate_id,
                kind="full_evaluated",
                key="trial-full-2",
            )
        )
    store.record_trial(
        _trial(
            card,
            batch,
            candidate_id,
            kind="holdout_claimed",
            key="trial-holdout-1",
        )
    )
    with pytest.raises(MiningStateError, match="测试段"):
        store.record_trial(
            _trial(
                card,
                batch,
                candidate_id,
                kind="holdout_claimed",
                key="trial-holdout-2",
            )
        )


def test_per_turn_candidate_budget_is_enforced(tmp_path: Path):
    budget = SearchBudget(
        max_candidates_per_turn=2,
        max_candidates_per_run=4,
        max_full_evaluations=4,
    )
    store = _store(tmp_path)
    run = _run(store, budget=budget)
    card, batch, candidates = _prepare_candidates(store, run, count=3)
    for index in range(2):
        store.record_trial(
            _trial(
                card,
                batch,
                candidates[index]["candidate_id"],
                kind="proposed",
                key=f"trial-turn-limit-{index}",
            )
        )
    with pytest.raises(MiningStateError, match="max_candidates_per_turn"):
        store.record_trial(
            _trial(
                card,
                batch,
                candidates[2]["candidate_id"],
                kind="proposed",
                key="trial-turn-limit-overflow",
            )
        )


def test_turn_branch_and_refine_budgets_are_frozen(tmp_path: Path):
    one_turn = SearchBudget(max_search_turns=1)
    store = _store(tmp_path)
    run = _run(store, key="search-run-0001", budget=one_turn)
    _prepare_candidates(store, run)
    second_card = _card(run["run_id"], turn_index=2)
    with pytest.raises(MiningStateError, match="max_search_turns"):
        store.begin_turn(
            run["run_id"],
            second_card,
            _batch(second_card, ("cand_value_0002",)),
        )

    branch_budget = SearchBudget(max_hypothesis_branches=1)
    branch_store = MiningV2Store(
        tmp_path / "branch.duckdb",
        tmp_path / "branch-data",
    )
    branch_run = _run(
        branch_store,
        key="search-run-branch",
        budget=branch_budget,
    )
    _prepare_candidates(branch_store, branch_run)
    branch_card = _card(
        branch_run["run_id"],
        turn_index=2,
        branch_id="branch_other",
    )
    with pytest.raises(MiningStateError, match="max_hypothesis_branches"):
        branch_store.begin_turn(
            branch_run["run_id"],
            branch_card,
            _batch(branch_card, ("cand_other_0001",)),
        )

    refine_budget = SearchBudget(max_refines_per_branch=0)
    refine_store = MiningV2Store(
        tmp_path / "refine.duckdb",
        tmp_path / "refine-data",
    )
    refine_run = _run(
        refine_store,
        key="search-run-refine",
        budget=refine_budget,
    )
    _prepare_candidates(refine_store, refine_run)
    refine_card = _card(refine_run["run_id"], turn_index=2)
    with pytest.raises(MiningStateError, match="max_refines_per_branch"):
        refine_store.begin_turn(
            refine_run["run_id"],
            refine_card,
            _batch(refine_card, ("cand_value_0002",)),
        )


def test_ledger_rejects_missing_or_mismatched_references(tmp_path: Path):
    store = _store(tmp_path)
    run = _run(store)
    card, batch, candidates = _prepare_candidates(store, run)
    wrong = TrialRecord(
        run_id=run["run_id"],
        trial_key="trial-wrong-batch",
        kind="evaluated",
        turn_index=1,
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        candidate_id="cand_missing_0001",
        payload={},
    )
    with pytest.raises(Exception, match="候选不存在"):
        store.record_trial(wrong)
    assert store.trial_counts(run["run_id"]).evaluated == 0
