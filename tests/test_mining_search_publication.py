"""V2 已批准候选进入共享正式发布链路的测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from factors.registry import FactorSpec
from mining.contract_evidence import ApprovalDecision
from mining.errors import MiningStorageError, PublicationConflictError
from mining.expression_validation import validate_expression
from mining.publication_contracts import new_publication_batch
from mining.publication_builder import _entry
from mining.publication_service import MiningPublicationService
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_review import SEARCH_APPROVAL_REQUIRED_STAGES
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


def _approved_run(tmp_path: Path) -> tuple[MiningV2Store, str, tuple[str, ...]]:
    """创建带两个已批准候选的 V2 审核态运行。"""

    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "search-publication-0001",
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
    for candidate_index, candidate in enumerate(definitions, start=1):
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
            suffix = f"{candidate_index:x}{stage_index + 1:x}" * 32
            store.record_evaluation_evidence(
                SearchEvaluationEvidence(
                    run_id=run["run_id"],
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    input_hash=suffix,
                    status="passed",
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
    for candidate_id in candidate_ids:
        eligibility = store.search_approval_eligibility(
            run["run_id"],
            candidate_id,
        )
        store.record_search_approval(
            ApprovalDecision(
                run_id=run["run_id"],
                candidate_id=candidate_id,
                decision="approved",
                reviewer="reviewer-1",
                note="证据完整，允许发布",
                candidate_content_hash=eligibility.candidate_content_hash,
                semantic_hash=eligibility.semantic_hash,
                context_hash=eligibility.context_hash,
                evidence_hash=eligibility.evidence_hash,
            )
        )
    return store, run["run_id"], candidate_ids


def _batch(
    store: MiningV2Store,
    run_id: str,
    candidate_ids: tuple[str, ...],
    tmp_path: Path,
):
    approvals = [
        store.get_search_approval(run_id, item)["approval_id"]
        for item in candidate_ids
    ]
    return new_publication_batch(
        run_id=run_id,
        context_hash=store.get_run(run_id)["context_hash"],
        candidate_ids=candidate_ids,
        approval_ids=tuple(approvals),
        staging_path=str(tmp_path / "staging"),
        attempt=1,
        source_kind="v2_search",
    )


def _staged(candidates, **_kwargs):
    return SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=tuple(item["name"] for item in candidates),
    )


def test_v2_publication_service_publishes_entire_approved_batch(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return _staged(kwargs["candidates"])

    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=prepare,
        commit=lambda **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )

    result = service.start_batch(
        run_id,
        candidate_ids,
        expected_context_hash=store.get_run(run_id)["context_hash"],
        background=False,
    )

    assert result["status"] == "published"
    assert result["source_kind"] == "v2_search"
    assert {item["status"] for item in result["publications"]} == {"published"}
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"published"}
    assert captured["context"].holdout_end == "2026-10-15"
    assert {
        item["source_kind"] for item in captured["candidates"]
    } == {"v2_search"}
    assert all(item["source_context_hash"] for item in captured["candidates"])
    assert all(item["scorecard_hash"] for item in captured["candidates"])


def test_v2_manifest_entry_keeps_source_provenance():
    spec = FactorSpec(
        name="MINE_SOURCE_TEST",
        family="mined",
        formula="mean(close, 5)",
        description="来源审计测试",
        steps=("读取收盘价", "计算五日均值"),
        layers=("mean(close, 5)",),
        lookback=4,
    )
    candidate = {
        "candidate_id": "cand_source",
        "content_hash": "a" * 64,
        "semantic_hash": "b" * 64,
        "source_kind": "v2_search",
        "source_context_hash": "c" * 64,
        "source_trace_hash": "d" * 64,
        "scorecard_hash": "e" * 64,
        "approval_evidence_hash": "f" * 64,
    }

    entry = _entry(
        spec,
        candidate,
        run_id="searchrun_source",
        approval_id="approval_source",
    )

    assert entry["source_kind"] == "v2_search"
    assert entry["source_context_hash"] == "c" * 64
    assert entry["source_trace_hash"] == "d" * 64
    assert entry["scorecard_hash"] == "e" * 64
    assert entry["approval_evidence_hash"] == "f" * 64


def test_v2_publication_requires_all_publishable_candidates(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PublicationConflictError, match="全部已批准候选"):
        service.start_batch(
            run_id,
            candidate_ids[:1],
            expected_context_hash=store.get_run(run_id)["context_hash"],
            background=False,
        )


def test_v2_publication_failure_is_retryable(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("模拟 V2 构建失败")
        ),
        rollback=lambda **_kwargs: None,
    )

    failed = service.start_batch(
        run_id,
        candidate_ids,
        expected_context_hash=store.get_run(run_id)["context_hash"],
        background=False,
    )
    assert failed["status"] == "failed"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"build_failed"}

    service._prepare = _staged
    service._commit = lambda **_kwargs: None
    retried = service.start_batch(
        run_id,
        candidate_ids,
        expected_context_hash=store.get_run(run_id)["context_hash"],
        background=False,
    )
    assert retried["status"] == "published"
    assert retried["attempt"] == 2


def test_v2_publication_start_is_atomic(tmp_path: Path, monkeypatch):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    batch = _batch(store, run_id, candidate_ids, tmp_path)

    def fail_after_candidate_updates(*_args, **_kwargs):
        raise RuntimeError("模拟批次记录写入失败")

    monkeypatch.setattr(
        store,
        "_insert_publication_batch_conn",
        fail_after_candidate_updates,
    )
    with pytest.raises(RuntimeError, match="批次记录写入失败"):
        store.start_publication_batch(batch)

    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"approved"}
    assert store.list_publication_batches(run_id) == []
    with duckdb.connect(str(store.db_path), read_only=True) as conn:
        count = conn.execute(
            "SELECT count(*) FROM mining.publication WHERE batch_id=?",
            [batch.batch_id],
        ).fetchone()[0]
    assert count == 0


def test_v2_publication_detects_evidence_tampering(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    candidate_id = candidate_ids[0]
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT evaluation_id, payload_json "
            "FROM mining_v2.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? AND stage='holdout'",
            [run_id, candidate_id],
        ).fetchone()
        payload = json.loads(row[1])
        payload["metrics"]["rank_ic_mean"] = 0.99
        conn.execute(
            "UPDATE mining_v2.evaluation_evidence SET payload_json=? "
            "WHERE run_id=? AND evaluation_id=?",
            [
                json.dumps(payload, ensure_ascii=False),
                run_id,
                row[0],
            ],
        )

    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(MiningStorageError, match="内容哈希"):
        service.start_batch(
            run_id,
            candidate_ids,
            expected_context_hash=store.get_run(run_id)["context_hash"],
            background=False,
        )


def test_v2_recovery_only_processes_v2_incomplete_batches(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    batch = _batch(store, run_id, candidate_ids, tmp_path)
    store.start_publication_batch(batch)
    with duckdb.connect(str(store.db_path)) as conn:
        conn.execute(
            "INSERT INTO mining.publication_batch "
            "(batch_id, run_id, context_hash, candidate_ids_json, "
            "approval_ids_json, status, staging_path, manifest_hash, "
            "artifact_sha256, error_code, error_message, attempt, "
            "started_at, finished_at, source_kind) "
            "VALUES (?, ?, ?, ?, ?, 'staging', ?, NULL, NULL, NULL, NULL, "
            "1, ?, NULL, 'v1')",
            [
                "pub_batch_v1_isolated",
                "run_v1_isolated",
                "c" * 64,
                "[]",
                "[]",
                str(tmp_path / "v1-staging"),
                "2026-11-01T00:00:00Z",
            ],
        )

    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    service.initialize()

    recovered = store.get_publication_batch(batch.batch_id)
    assert recovered["status"] == "failed"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"build_failed"}
    with duckdb.connect(str(store.db_path), read_only=True) as conn:
        v1_status = conn.execute(
            "SELECT status FROM mining.publication_batch WHERE batch_id=?",
            ["pub_batch_v1_isolated"],
        ).fetchone()[0]
    assert v1_status == "staging"


def test_v2_recovery_finishes_committed_batch(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    batch = _batch(store, run_id, candidate_ids, tmp_path)
    store.start_publication_batch(batch)
    staging = Path(batch.staging_path)
    staging.mkdir(parents=True)
    version = tmp_path / "factors" / ".mined-versions" / "recovered"
    version.mkdir(parents=True)
    names = [
        store.get_candidate(run_id, item)["name"]
        for item in candidate_ids
    ]
    (staging / "commit.json").write_text(
        json.dumps(
            {
                "state": "committed",
                "new_link": ".mined-versions/recovered",
            }
        ),
        encoding="utf-8",
    )
    (version / "definitions.json").write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "candidate_id": candidate_id,
                        "name": name,
                    }
                    for candidate_id, name in zip(
                        candidate_ids,
                        names,
                        strict=True,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "data",
        factor_dir=tmp_path / "factors",
        snapshot_check=lambda *_args, **_kwargs: None,
    )

    service.initialize()

    assert store.get_publication_batch(batch.batch_id)["status"] == "published"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"published"}


def test_v1_store_does_not_read_v2_batches(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    batch = _batch(store, run_id, candidate_ids, tmp_path)
    store.start_publication_batch(batch)
    v1_store = MiningStore(store.db_path, tmp_path / "v1-data")

    assert v1_store.incomplete_publication_batches() == []
    assert store.incomplete_publication_batches()[0]["batch_id"] == batch.batch_id
    with pytest.raises(PublicationConflictError, match="V1 来源"):
        v1_store.get_publication_batch(batch.batch_id)


def test_concurrent_v2_publication_starts_only_one_batch(tmp_path: Path):
    store, run_id, candidate_ids = _approved_run(tmp_path)
    first = _batch(store, run_id, candidate_ids, tmp_path / "first")
    approval_ids = tuple(
        store.get_search_approval(run_id, item)["approval_id"]
        for item in candidate_ids
    )
    second = new_publication_batch(
        run_id=run_id,
        context_hash=store.get_run(run_id)["context_hash"],
        candidate_ids=candidate_ids,
        approval_ids=approval_ids,
        staging_path=str(tmp_path / "second"),
        attempt=2,
        source_kind="v2_search",
    )

    def start(batch):
        try:
            return store.start_publication_batch(batch)["status"]
        except PublicationConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(start, (first, second)))

    assert sorted(results) == ["conflict", "staging"]
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"building"}
