"""挖掘 V2 搜索契约、字段限制和稳定哈希测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mining.contract_types import sha256_json
from mining.errors import MiningContractError
from mining.search_contracts import (
    EvidencePacket,
    ExperimentBatch,
    FREE_EXPLORATION_INPUT_MODE,
    HYPOTHESIS_INPUT_MODE,
    HypothesisCard,
    SearchBudget,
    SearchDecision,
    SearchRunRequest,
    TrialCounts,
    TrialRecord,
)
from mining.search_evidence import SearchDecision as SearchDecisionFromModule


def _hypothesis(**changes) -> HypothesisCard:
    payload = {
        "run_id": "run_search_0001",
        "turn_index": 1,
        "branch_id": "branch_momentum",
        "statement": "短期收益延续可能带来下一期超额收益",
        "mechanism": "投资者反应不足导致价格逐步吸收新信息",
        "expected_direction": "positive",
        "falsification_criteria": ("年度同号率低于 0.6", "RankIC 为负"),
        "parent_hypothesis_id": None,
        "metadata": {"source": "user"},
        "created_at": "2026-10-29T08:00:00+08:00",
    }
    payload.update(changes)
    return HypothesisCard(**payload)


def _batch(**changes) -> ExperimentBatch:
    payload = {
        "run_id": "run_search_0001",
        "hypothesis_id": _hypothesis().hypothesis_id,
        "turn_index": 1,
        "branch_id": "branch_momentum",
        "candidate_ids": ("cand_momentum_0001",),
        "parent_candidate_ids": (),
        "development_start": "2012-01-01",
        "development_end": "2019-12-31",
        "input_hash": "a" * 64,
        "parameters": {"window": 20},
        "created_at": "2026-10-29T08:01:00+08:00",
    }
    payload.update(changes)
    return ExperimentBatch(**payload)


def _evidence(**changes) -> EvidencePacket:
    batch = _batch()
    payload = {
        "run_id": batch.run_id,
        "hypothesis_id": batch.hypothesis_id,
        "batch_id": batch.batch_id,
        "turn_index": 1,
        "development_start": "2012-01-01",
        "development_end": "2019-12-31",
        "metrics": {"rank_ic": 0.03, "icir": 0.4},
        "failures": ({"candidate_id": "cand_momentum_0001", "reason": "coverage"},),
        "correlation_clusters": ({"cluster": 1, "candidate_ids": ["cand_momentum_0001"]},),
        "trial_counts": TrialCounts(proposed=1, evaluated=1),
        "budget_state": {"turns_used": 1},
        "created_at": "2026-10-29T08:02:00+08:00",
    }
    payload.update(changes)
    return EvidencePacket(**payload)


def _decision(**changes) -> SearchDecision:
    evidence = _evidence()
    payload = {
        "run_id": evidence.run_id,
        "hypothesis_id": evidence.hypothesis_id,
        "evidence_id": evidence.evidence_id,
        "turn_index": 1,
        "action": "refine",
        "reason": "开发期覆盖不足，需要调整回看窗口",
        "target_hypothesis_id": None,
        "created_at": "2026-10-29T08:03:00+08:00",
    }
    payload.update(changes)
    return SearchDecision(**payload)


def _trial(**changes) -> TrialRecord:
    payload = {
        "run_id": "run_search_0001",
        "trial_key": "turn-1-proposed-1",
        "kind": "proposed",
        "turn_index": 1,
        "hypothesis_id": _hypothesis().hypothesis_id,
        "batch_id": _batch().batch_id,
        "candidate_id": "cand_momentum_0001",
        "reason_code": None,
        "payload": {"source": "worker"},
        "occurred_at": "2026-10-29T08:04:00+08:00",
    }
    payload.update(changes)
    return TrialRecord(**payload)


def test_search_budget_freezes_defaults_and_rejects_expansion():
    budget = SearchBudget()
    assert budget.to_dict() == {
        "max_search_turns": 6,
        "max_hypothesis_branches": 3,
        "max_refines_per_branch": 3,
        "max_candidates_per_turn": 4,
        "max_candidates_per_run": 24,
        "max_full_evaluations": 4,
        "stop_after_no_improvement_turns": 2,
        "max_tool_calls_per_turn": 4,
        "schema_version": "mining_search_v2",
    }
    assert SearchBudget.from_dict(budget.to_dict()) == budget
    assert budget.budget_hash == SearchBudget().budget_hash
    with pytest.raises(MiningContractError, match="max_search_turns"):
        SearchBudget(max_search_turns=7)
    with pytest.raises(MiningContractError, match="max_candidates_per_turn"):
        SearchBudget(max_candidates_per_turn=5)
    with pytest.raises(MiningContractError, match="不得超过"):
        SearchBudget(max_candidates_per_turn=4, max_candidates_per_run=3)


def test_search_run_request_generates_stable_hashes_for_both_input_modes():
    hypothesis = SearchRunRequest(
        "search-mode-0001",
        "短期反转可能带来开发期超额收益",
    )
    explicit = SearchRunRequest(
        "search-mode-0001",
        "短期反转可能带来开发期超额收益",
        input_mode=HYPOTHESIS_INPUT_MODE,
    )
    legacy_material = {
        "schema_version": "mining_v2",
        "hypothesis": "短期反转可能带来开发期超额收益",
        "budget": SearchBudget().to_dict(),
        "llm_profile": "default",
    }
    assert hypothesis.request_hash == explicit.request_hash
    assert hypothesis.request_hash == sha256_json(legacy_material)

    free = SearchRunRequest(
        "search-mode-0002",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
    )
    repeated = SearchRunRequest(
        "search-mode-0002",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
    )
    assert free.request_hash == repeated.request_hash
    assert free.material() == {
        **legacy_material,
        "hypothesis": None,
        "input_mode": FREE_EXPLORATION_INPUT_MODE,
    }
    assert free.request_hash != hypothesis.request_hash
    assert free.display_hypothesis == "自由探索"
    assert free.to_dict()["input_mode"] == FREE_EXPLORATION_INPUT_MODE


def test_search_run_request_accepts_legacy_dict_and_round_trips_mode():
    request = SearchRunRequest(
        "search-mode-0003",
        "短期反转可能带来开发期超额收益",
    )
    legacy = {
        "idempotency_key": request.idempotency_key,
        **request.to_dict(),
    }
    legacy.pop("input_mode")
    assert SearchRunRequest.from_dict(legacy) == request

    free = SearchRunRequest(
        "search-mode-0004",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
    )
    payload = {
        "idempotency_key": free.idempotency_key,
        **free.to_dict(),
    }
    assert SearchRunRequest.from_dict(payload) == free
    with pytest.raises(MiningContractError, match="未知"):
        SearchRunRequest.from_dict({**payload, "unexpected": True})


def test_search_run_request_rejects_invalid_modes_and_empty_inputs():
    with pytest.raises(MiningContractError, match="input_mode"):
        SearchRunRequest("search-mode-0005", None, input_mode="manual")
    with pytest.raises(MiningContractError, match="hypothesis"):
        SearchRunRequest("search-mode-0006", None)
    with pytest.raises(MiningContractError, match="hypothesis"):
        SearchRunRequest("search-mode-0007", "")
    with pytest.raises(MiningContractError, match="不得提供"):
        SearchRunRequest(
            "search-mode-0008",
            "自由探索不应同时提供假设",
            input_mode=FREE_EXPLORATION_INPUT_MODE,
        )


def test_contract_hashes_ignore_only_time_fields():
    hypothesis = _hypothesis()
    changed_hypothesis = replace(
        hypothesis,
        created_at="2026-10-30T00:00:00+00:00",
    )
    assert hypothesis.card_hash == changed_hypothesis.card_hash
    assert hypothesis.hypothesis_id == changed_hypothesis.hypothesis_id

    batch = _batch()
    changed_batch = replace(batch, created_at="2026-10-30T00:00:00+00:00")
    assert batch.batch_hash == changed_batch.batch_hash

    evidence = _evidence()
    changed_evidence = replace(
        evidence,
        created_at="2026-10-30T00:00:00+00:00",
    )
    assert evidence.evidence_hash == changed_evidence.evidence_hash

    decision = _decision()
    changed_decision = replace(
        decision,
        created_at="2026-10-30T00:00:00+00:00",
    )
    assert decision.decision_hash == changed_decision.decision_hash

    trial = _trial()
    changed_trial = replace(trial, occurred_at="2026-10-30T00:00:00+00:00")
    assert trial.trial_hash == changed_trial.trial_hash
    assert trial.trial_id == changed_trial.trial_id
    assert SearchDecisionFromModule is SearchDecision


def test_contract_hashes_change_with_semantic_content():
    base = _hypothesis()
    changed = _hypothesis(statement="短期收益反转可能带来下一期超额收益")
    assert base.card_hash != changed.card_hash
    assert base.hypothesis_id != changed.hypothesis_id

    assert _batch().batch_hash != _batch(parameters={"window": 30}).batch_hash
    assert _evidence().evidence_hash != _evidence(metrics={"rank_ic": 0.04}).evidence_hash
    assert _decision().decision_hash != _decision(action="branch").decision_hash
    assert _trial().trial_hash != _trial(kind="evaluated").trial_hash


@pytest.mark.parametrize(
    ("contract", "payload", "name"),
    [
        (
            HypothesisCard,
            {**_hypothesis().to_dict()},
            "HypothesisCard",
        ),
        (
            ExperimentBatch,
            {**_batch().to_dict()},
            "ExperimentBatch",
        ),
        (
            EvidencePacket,
            {**_evidence().to_dict()},
            "EvidencePacket",
        ),
        (
            SearchDecision,
            {**_decision().to_dict()},
            "SearchDecision",
        ),
        (
            TrialRecord,
            {**_trial().to_dict()},
            "TrialRecord",
        ),
    ],
)
def test_contract_round_trip_missing_and_unknown_fields(contract, payload, name):
    assert contract.from_dict(payload).to_dict() == payload

    missing = dict(payload)
    missing.pop("schema_version")
    with pytest.raises(MiningContractError, match="缺少"):
        contract.from_dict(missing)

    unknown = {**payload, "unexpected": True}
    with pytest.raises(MiningContractError, match="未知"):
        contract.from_dict(unknown)


def test_contract_limits_and_enums_are_strict():
    with pytest.raises(MiningContractError, match="statement"):
        _hypothesis(statement="太短")
    with pytest.raises(MiningContractError, match="falsification"):
        _hypothesis(falsification_criteria=tuple(f"条件 {i}" for i in range(13)))
    with pytest.raises(MiningContractError, match="expected_direction"):
        _hypothesis(expected_direction="up")
    with pytest.raises(MiningContractError, match="candidate_ids"):
        _batch(candidate_ids=tuple(f"cand_{i}" for i in range(5)))
    with pytest.raises(MiningContractError, match="development"):
        _batch(development_start="2020-01-01", development_end="2019-12-31")
    with pytest.raises(MiningContractError, match="action"):
        _decision(action="continue")
    with pytest.raises(MiningContractError, match="kind"):
        _trial(kind="unknown")
    with pytest.raises(MiningContractError, match="payload"):
        _trial(payload={"blob": "x" * 17000})


def test_trial_counts_reject_negative_counts_and_normalize_sequences():
    assert TrialCounts.from_dict(TrialCounts(proposed=2).to_dict()).proposed == 2
    with pytest.raises(MiningContractError, match="proposed"):
        TrialCounts(proposed=-1)
    with pytest.raises(MiningContractError, match="缺少"):
        TrialCounts.from_dict({"proposed": 1})

    hypothesis = _hypothesis(
        falsification_criteria=("条件一", "条件一", "条件二"),
    )
    assert hypothesis.falsification_criteria == ("条件一", "条件二")
