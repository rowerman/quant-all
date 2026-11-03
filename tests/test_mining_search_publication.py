"""V2 已批准候选进入共享正式发布链路的测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from factors.registry import FactorSpec
from mining.errors import MiningStorageError, PublicationConflictError
from mining.publication_contracts import new_publication_batch
from mining.publication_builder import _entry
from mining.publication_service import MiningPublicationService
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store
from tests.mining_v2_fixtures import approved_search_run


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


def test_v2_stale_transition_is_atomic_and_idempotent(
    tmp_path: Path,
    monkeypatch,
):
    """快照失效收口要么整体成功，要么不改变任何状态。"""

    store, run_id, candidate_ids = approved_search_run(tmp_path)
    original = store._append_event_conn
    fail_once = {"value": True}

    def fail_run_event(*args, **kwargs):
        event_type = kwargs.get("event_type", args[3] if len(args) > 3 else None)
        if event_type == "search_run_state_changed" and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("模拟运行失效事件写入失败")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_append_event_conn", fail_run_event)
    with pytest.raises(RuntimeError, match="运行失效事件写入失败"):
        store.mark_search_run_stale(
            run_id,
            event_key="web_decision_stale",
            error_code="MINING-120",
            error_message="审批前快照已变化",
        )

    assert store.get_run(run_id)["status"] == "review_ready"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"approved"}

    first = store.mark_search_run_stale(
        run_id,
        event_key="web_decision_stale",
        error_code="MINING-120",
        error_message="审批前快照已变化",
    )
    second = store.mark_search_run_stale(
        run_id,
        event_key="web_decision_stale",
        error_code="MINING-120",
        error_message="审批前快照已变化",
    )

    assert first["status"] == "stale"
    assert first["reused"] is False
    assert second["reused"] is True
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"stale"}
    assert store.validate_search_trace(run_id) is True
    assert sum(
        item["event_key"] == "web_decision_stale"
        for item in store.list_events(run_id)
    ) == 1


def test_v2_publication_service_publishes_entire_approved_batch(tmp_path: Path):
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
    store, run_id, candidate_ids = approved_search_run(tmp_path)
    batch = _batch(store, run_id, candidate_ids, tmp_path)
    store.start_publication_batch(batch)
    v1_store = MiningStore(store.db_path, tmp_path / "v1-data")

    assert v1_store.incomplete_publication_batches() == []
    assert store.incomplete_publication_batches()[0]["batch_id"] == batch.batch_id
    with pytest.raises(PublicationConflictError, match="V1 来源"):
        v1_store.get_publication_batch(batch.batch_id)


def test_concurrent_v2_publication_starts_only_one_batch(tmp_path: Path):
    store, run_id, candidate_ids = approved_search_run(tmp_path)
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
