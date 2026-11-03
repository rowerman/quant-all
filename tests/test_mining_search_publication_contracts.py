"""V2 到 V1 发布桥接契约测试。"""

from __future__ import annotations

from pathlib import Path
import re

import duckdb
import pytest

from mining.contract_requests import FrozenContext, MiningRunRequest
from mining.contract_types import canonical_json
from mining.errors import MiningContractError, MiningStorageError
from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_publication_contracts import (
    SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION,
)
from mining.search_publication_loader import load_search_publication_inputs
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


def _review_ready_run(
    tmp_path: Path,
) -> tuple[MiningV2Store, str, str]:
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "publication-bridge-0001",
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
    candidate = validate_expression(
        "mean(close, 5)",
        direction=1,
    ).to_candidate(
        mechanism="短期趋势",
        description="五日收盘均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id=card.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], card, batch)
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
    for index, (stage, suffix) in enumerate(
        (
            ("technical", "1"),
            ("development", "2"),
            ("selection", "3"),
            ("full", "4"),
            ("holdout", "5"),
        ),
        start=1,
    ):
        store.record_evaluation_evidence(
            SearchEvaluationEvidence(
                run_id=run["run_id"],
                candidate_id=candidate.candidate_id,
                stage=stage,
                input_hash=suffix * 64,
                status="passed",
                metrics={
                    "rank_ic_mean": 0.03,
                    "rank_ic_ir": 0.4,
                    "year_sign_consistency": 0.75,
                    "valid_days": 100 + index,
                    "values": {"coverage": 0.8, "rows": 10000},
                },
                created_at=f"2026-11-01T00:00:{index:02d}Z",
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
    return store, run["run_id"], candidate.candidate_id


def test_bridge_maps_review_ready_candidate_without_writing(tmp_path: Path):
    store, run_id, candidate_id = _review_ready_run(tmp_path)
    before_events = store.list_events(run_id)
    before_trace = store.get_run(run_id)["trace_hash"]

    first = load_search_publication_inputs(store, run_id, [candidate_id])
    second = load_search_publication_inputs(store, run_id, (candidate_id,))

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.schema_version == SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION
    assert len(first.candidates) == 1
    candidate = first.candidates[0]
    assert candidate.candidate_id == candidate_id
    assert candidate.status == "review_ready"
    assert candidate.content_hash == candidate.definition.content_hash
    assert candidate.semantic_hash == candidate.definition.semantic_hash
    assert candidate.publication_candidate()["definition"][
        "canonical_expression"
    ] == "mean(close, 5)"
    assert {item.stage for item in candidate.evidence} == {
        "technical",
        "development",
        "selection",
        "full",
        "holdout",
    }
    assert first.builder_context.full_eval_start == "2023-01-01"
    assert first.builder_context.full_eval_end == "2025-10-15"
    assert first.builder_context.holdout_start == "2025-10-16"
    assert first.builder_context.context_hash != first.source_context_hash
    assert len(candidate.evidence_set_hash) == 64
    assert store.list_events(run_id) == before_events
    assert store.get_run(run_id)["trace_hash"] == before_trace


def test_bridge_rejects_non_review_ready_run(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "publication-bridge-queued",
            "短期趋势可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )

    with pytest.raises(MiningContractError, match="运行状态 queued"):
        load_search_publication_inputs(store, run["run_id"], ["cand_missing"])


def test_bridge_rejects_candidate_payload_tampering(tmp_path: Path):
    store, run_id, candidate_id = _review_ready_run(tmp_path)
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json FROM mining_v2.candidate_trace "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        ).fetchone()
        payload = __import__("json").loads(row[0])
        payload["description"] = "被篡改的候选说明"
        conn.execute(
            "UPDATE mining_v2.candidate_trace SET payload_json=? "
            "WHERE run_id=? AND candidate_id=?",
            [canonical_json(payload), run_id, candidate_id],
        )

    with pytest.raises(MiningStorageError, match="内容哈希"):
        load_search_publication_inputs(store, run_id, [candidate_id])


def test_bridge_rejects_evaluation_evidence_tampering(tmp_path: Path):
    store, run_id, candidate_id = _review_ready_run(tmp_path)
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT evaluation_id, payload_json FROM mining_v2.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? AND stage='holdout'",
            [run_id, candidate_id],
        ).fetchone()
        payload = __import__("json").loads(row[1])
        payload["metrics"]["rank_ic_mean"] = 0.99
        conn.execute(
            "UPDATE mining_v2.evaluation_evidence SET payload_json=? "
            "WHERE run_id=? AND evaluation_id=?",
            [canonical_json(payload), run_id, row[0]],
        )

    with pytest.raises(MiningStorageError, match="内容哈希"):
        load_search_publication_inputs(store, run_id, [candidate_id])


def test_bridge_keeps_existing_v1_candidate_readable(tmp_path: Path):
    v2_store, search_run_id, search_candidate_id = _review_ready_run(tmp_path)
    v1_store = MiningStore(v2_store.db_path, v2_store.data_root)
    v1_run = v1_store.create_run(
        MiningRunRequest(
            "publication-bridge-v1-run",
            "短期低估值可能带来超额收益",
        ),
        FrozenContext(data_fingerprint="f" * 64),
    )
    v1_candidate = validate_expression(
        "mean(close, 10)",
        direction=1,
    ).to_candidate(
        mechanism="中期趋势",
        description="十日均值",
        steps=("读取收盘价", "计算十日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    v1_store.add_candidate(v1_run["run_id"], v1_candidate)
    before = v1_store.get_candidate(v1_run["run_id"], v1_candidate.candidate_id)

    bridged = load_search_publication_inputs(
        v2_store,
        search_run_id,
        [search_candidate_id],
    )

    after = v1_store.get_candidate(v1_run["run_id"], v1_candidate.candidate_id)
    assert after == before
    assert after["definition"]["canonical_expression"] == "mean(close, 10)"
    assert bridged.candidates[0].candidate_id == search_candidate_id


def test_bridge_has_no_public_mutation_api():
    from mining import search_publication_contracts
    from mining import search_publication_loader

    public_names = set(search_publication_contracts.__all__)
    public_names.update(search_publication_loader.__all__)
    assert not any(
        re.search(r"(record|write|publish|approve|transition)", name)
        for name in public_names
    )
