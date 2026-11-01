"""V2 候选审批资格、最终决定和证据失效测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import duckdb
import pytest

from mining.contract_evidence import ApprovalDecision
from mining.contract_types import canonical_json
from mining.errors import (
    ApprovalConflictError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_review import SEARCH_APPROVAL_REQUIRED_STAGES
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store


def _review_ready_run(
    tmp_path: Path,
    *,
    omit_stage: str | None = None,
    failed_stage: str | None = None,
) -> tuple[MiningV2Store, str, tuple[str, ...]]:
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "search-approval-0001",
            "短期趋势可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )
    card = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期趋势可能带来开发期超额收益",
        mechanism="价格趋势延续",
        expected_direction="positive",
        falsification_criteria=("完整评估未通过",),
    )
    definitions = (
        validate_expression("mean(close, 5)", direction=1).to_candidate(
            mechanism="短期趋势",
            description="五日收盘均值",
            steps=("读取收盘价", "计算五日均值"),
            discovery_round=1,
            discovery_index=1,
        ),
        validate_expression("mean(close, 10)", direction=1).to_candidate(
            mechanism="中期趋势",
            description="十日收盘均值",
            steps=("读取收盘价", "计算十日均值"),
            discovery_round=1,
            discovery_index=2,
        ),
    )
    candidate_ids = tuple(item.candidate_id for item in definitions)
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id=card.branch_id,
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], card, batch)
    for index, candidate in enumerate(definitions, start=1):
        store.record_candidate(
            run["run_id"],
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.content_hash,
            semantic_hash=candidate.semantic_hash,
            name=candidate.name,
            status="proposed",
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            payload=candidate_payload(candidate),
        )
        for target in (
            "validating",
            "screening",
            "evaluating",
            "evaluated",
            "holdout_running",
            "review_ready",
        ):
            store.transition_candidate(
                run["run_id"],
                candidate.candidate_id,
                target,
            )
        for stage_index, stage in enumerate(SEARCH_APPROVAL_REQUIRED_STAGES):
            if stage == omit_stage:
                continue
            suffix = f"{index:x}{stage_index + 1:x}" * 32
            status = "failed" if stage == failed_stage else "passed"
            store.record_evaluation_evidence(
                SearchEvaluationEvidence(
                    run_id=run["run_id"],
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    input_hash=suffix,
                    status=status,
                    metrics={
                        "rank_ic_mean": 0.03,
                        "rank_ic_ir": 0.4,
                        "year_sign_consistency": 0.75,
                        "valid_days": 100,
                        "values": {"coverage": 0.8, "rows": 10000},
                    },
                )
            )
    for target in (
        "searching",
        "selecting",
        "evaluating",
        "holding_out",
        "review_ready",
    ):
        store.transition_search_run(
            run["run_id"],
            target,
            event_key=f"run_{target}",
            stage=target,
        )
    store.validate_search_trace(run["run_id"])
    return store, run["run_id"], candidate_ids


def _decision(
    store: MiningV2Store,
    run_id: str,
    candidate_id: str,
    *,
    decision: str = "approved",
    reviewer: str = "reviewer-1",
    note: str = "证据完整，允许进入发布准备",
) -> ApprovalDecision:
    eligibility = store.search_approval_eligibility(run_id, candidate_id)
    return ApprovalDecision(
        run_id=run_id,
        candidate_id=candidate_id,
        decision=decision,
        reviewer=reviewer,
        note=note,
        candidate_content_hash=eligibility.candidate_content_hash,
        semantic_hash=eligibility.semantic_hash,
        context_hash=eligibility.context_hash,
        evidence_hash=eligibility.evidence_hash,
    )


def test_approval_eligibility_is_stable_and_requires_all_stages(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)

    first = store.search_approval_eligibility(run_id, candidate_ids[0])
    second = store.search_approval_eligibility(run_id, candidate_ids[0])

    assert first == second
    assert first.eligible is True
    assert first.blocking_reasons == ()
    assert first.approval_id is None
    assert {item.stage for item in first.evidence} == set(
        SEARCH_APPROVAL_REQUIRED_STAGES
    )
    assert len(first.evidence_hash) == 64
    assert store.search_approval_evidence(run_id, candidate_ids[0]) == (
        first.evidence_hash
    )


@pytest.mark.parametrize(
    "omit_stage,failed_stage,reason",
    [
        ("holdout", None, "MISSING_EVIDENCE:holdout"),
        (None, "full", "EVIDENCE_NOT_PASSED:full"),
    ],
)
def test_missing_or_failed_required_evidence_blocks_approval(
    tmp_path: Path,
    omit_stage: str | None,
    failed_stage: str | None,
    reason: str,
):
    store, run_id, candidate_ids = _review_ready_run(
        tmp_path,
        omit_stage=omit_stage,
        failed_stage=failed_stage,
    )

    eligibility = store.search_approval_eligibility(
        run_id,
        candidate_ids[0],
    )

    assert eligibility.eligible is False
    assert reason in eligibility.blocking_reasons
    with pytest.raises(ApprovalConflictError, match=reason):
        store.record_search_approval(
            _decision(store, run_id, candidate_ids[0])
        )


def test_approval_is_idempotent_persistent_and_candidate_scoped(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    first_id, second_id = candidate_ids
    second_before = store.search_approval_eligibility(run_id, second_id)
    decision = _decision(store, run_id, first_id)

    saved = store.record_search_approval(decision)
    repeated = store.record_search_approval(decision)

    assert saved["approval_id"] == decision.approval_id
    assert saved["decision"] == "approved"
    assert repeated["reused"] is True
    assert store.get_run(run_id)["status"] == "review_ready"
    assert store.get_candidate(run_id, first_id)["status"] == "approved"
    assert store.search_approval_is_current(run_id, first_id) is True
    assert store.search_approval_evidence(run_id, second_id) == (
        second_before.evidence_hash
    )
    assert store.search_approval_eligibility(run_id, second_id).eligible is True
    store.validate_search_trace(run_id)

    reopened = MiningV2Store(store.db_path, store.data_root)
    assert reopened.get_search_approval(run_id, first_id)["approval_id"] == (
        decision.approval_id
    )
    assert reopened.search_approval_is_current(run_id, first_id) is True

    with pytest.raises(ApprovalConflictError, match="不同"):
        reopened.record_search_approval(
            _decision(
                reopened,
                run_id,
                first_id,
                decision="rejected",
            )
        )
    with pytest.raises(ApprovalConflictError, match="不同"):
        reopened.record_search_approval(
            _decision(
                reopened,
                run_id,
                first_id,
                note="更改审批备注",
            )
        )


def test_new_evidence_invalidates_existing_approval(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    candidate_id = candidate_ids[0]
    decision = _decision(store, run_id, candidate_id)
    store.record_search_approval(decision)

    store.record_evaluation_evidence(
        SearchEvaluationEvidence(
            run_id=run_id,
            candidate_id=candidate_id,
            stage="technical",
            input_hash="9" * 64,
            status="passed",
            metrics={"values": {"coverage": 0.9, "rows": 12000}},
        )
    )

    assert store.search_approval_is_current(run_id, candidate_id) is False
    with pytest.raises(ApprovalConflictError, match="证据已变化"):
        store.record_search_approval(decision)


@pytest.mark.parametrize("stage", SEARCH_APPROVAL_REQUIRED_STAGES)
def test_tampered_stage_evidence_breaks_approval_integrity(
    tmp_path: Path,
    stage: str,
):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    candidate_id = candidate_ids[0]
    _decision(store, run_id, candidate_id)
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT evaluation_id, payload_json "
            "FROM mining_v2.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? AND stage=?",
            [run_id, candidate_id, stage],
        ).fetchone()
        payload = json.loads(row[1])
        payload["metrics"]["rank_ic_mean"] = 0.99
        conn.execute(
            "UPDATE mining_v2.evaluation_evidence SET payload_json=? "
            "WHERE run_id=? AND evaluation_id=?",
            [canonical_json(payload), run_id, row[0]],
        )

    with pytest.raises(MiningStorageError, match="哈希"):
        store.search_approval_evidence(run_id, candidate_id)


def test_tampered_candidate_or_context_is_rejected(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json FROM mining_v2.candidate_trace "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_ids[0]],
        ).fetchone()
        payload = json.loads(row[0])
        payload["description"] = "被篡改的候选说明"
        conn.execute(
            "UPDATE mining_v2.candidate_trace SET payload_json=? "
            "WHERE run_id=? AND candidate_id=?",
            [canonical_json(payload), run_id, candidate_ids[0]],
        )
    with pytest.raises(MiningStorageError, match="哈希"):
        store.search_approval_evidence(run_id, candidate_ids[0])

    second_store, second_run_id, second_ids = _review_ready_run(
        tmp_path / "context"
    )
    with duckdb.connect(str(second_store.db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json FROM mining_v2.context "
            "JOIN mining_v2.run USING(context_id) WHERE run_id=?",
            [second_run_id],
        ).fetchone()
        payload = json.loads(row[0])
        payload["data_fingerprint"] = "f" * 64
        conn.execute(
            "UPDATE mining_v2.context SET payload_json=? "
            "WHERE context_id=(SELECT context_id FROM mining_v2.run "
            "WHERE run_id=?)",
            [canonical_json(payload), second_run_id],
        )
    with pytest.raises(MiningStorageError, match="上下文"):
        second_store.search_approval_evidence(second_run_id, second_ids[0])


def test_tampered_approval_breaks_search_trace(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    candidate_id = candidate_ids[0]
    store.record_search_approval(_decision(store, run_id, candidate_id))
    with duckdb.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE mining_v2.search_approval SET note='被篡改的审批备注' "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        )

    with pytest.raises(MiningStorageError, match="哈希"):
        store.validate_search_trace(run_id)


def test_rejection_is_persisted_and_remains_current(tmp_path: Path):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    candidate_id = candidate_ids[0]
    saved = store.record_search_approval(
        _decision(store, run_id, candidate_id, decision="rejected")
    )

    assert saved["decision"] == "rejected"
    assert store.get_candidate(run_id, candidate_id)["status"] == "rejected"
    assert store.search_approval_is_current(run_id, candidate_id) is True
    eligibility = store.search_approval_eligibility(run_id, candidate_id)
    assert eligibility.eligible is False
    assert "CANDIDATE_NOT_REVIEW_READY" in eligibility.blocking_reasons


def test_concurrent_approve_and_reject_have_single_final_decision(
    tmp_path: Path,
):
    store, run_id, candidate_ids = _review_ready_run(tmp_path)
    candidate_id = candidate_ids[0]
    approve = _decision(store, run_id, candidate_id)
    reject = _decision(store, run_id, candidate_id, decision="rejected")

    def decide(decision: ApprovalDecision) -> str:
        try:
            store.record_search_approval(decision)
            return decision.decision
        except ApprovalConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, (approve, reject)))

    assert sorted(results) in (
        ["approved", "conflict"],
        ["conflict", "rejected"],
    )
    approval = store.get_search_approval(run_id, candidate_id)
    assert approval["decision"] == store.get_candidate(
        run_id,
        candidate_id,
    )["status"]
    assert store.search_approval_is_current(run_id, candidate_id) is True
    store.validate_search_trace(run_id)


def test_existing_v2_database_gets_approval_table_on_open(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    with duckdb.connect(str(store.db_path)) as conn:
        conn.execute("DROP TABLE mining_v2.search_approval")

    reopened = MiningV2Store(store.db_path, store.data_root)
    with duckdb.connect(str(reopened.db_path), read_only=True) as conn:
        count = conn.execute(
            "SELECT count(*) FROM duckdb_tables() "
            "WHERE schema_name='mining_v2' AND table_name='search_approval'"
        ).fetchone()[0]
    assert count == 1
    with pytest.raises(MiningNotFoundError):
        reopened.get_search_approval("searchrun_missing", "cand_missing")
