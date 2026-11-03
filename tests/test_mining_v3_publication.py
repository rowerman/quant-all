"""V3 已批准候选进入共享正式发布链路的测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import duckdb
import pytest

from mining.contract_evidence import ApprovalDecision
from mining.errors import ApprovalConflictError, PublicationConflictError
from mining.publication_builder import _entry
from mining.publication_contracts import new_publication_batch
from mining.publication_service import MiningPublicationService
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.v3_contracts import CandidateEnvelope, EvidenceReference
from tests.mining_v3_fixtures import (
    approved_v3_run,
    make_v3_candidate_review_ready,
    v3_candidate_envelope,
)


def _staged(candidates, **_kwargs):
    return SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=tuple(item["name"] for item in candidates),
    )


def _batch(
    store: MiningV3Store,
    run_id: str,
    candidate_ids: tuple[str, ...],
    tmp_path: Path,
):
    approvals = [
        store.get_approval(run_id, item)["approval_id"]
        for item in candidate_ids
    ]
    return new_publication_batch(
        run_id=run_id,
        context_hash=store.load_input(run_id).context_hash,
        candidate_ids=candidate_ids,
        approval_ids=tuple(approvals),
        staging_path=str(tmp_path / "staging"),
        attempt=1,
        source_kind="v3_research",
    )


def test_v3_publication_service_publishes_entire_approved_batch(
    tmp_path: Path,
):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return _staged(kwargs["candidates"])

    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=prepare,
        commit=lambda **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )

    result = service.start_batch(
        input_envelope.run_id,
        (envelope.candidate.candidate_id,),
        expected_context_hash=input_envelope.context_hash,
        background=False,
    )

    assert result["status"] == "published"
    assert result["source_kind"] == "v3_research"
    assert {item["status"] for item in result["publications"]} == {"published"}
    assert {
        item["candidate_envelope_hash"] for item in result["publications"]
    } == {envelope.envelope_hash}
    assert store.get_candidate(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )["status"] == "published"
    assert captured["context"].holdout_end == "2026-10-15"
    assert captured["candidates"][0]["source_kind"] == "v3_research"
    assert store.publication_source_kind == "v3_research"
    assert (
        captured["candidates"][0]["candidate_envelope_hash"]
        == envelope.envelope_hash
    )
    assert [
        item["candidate_id"] for item in store.list_candidates(input_envelope.run_id)
    ] == [envelope.candidate.candidate_id]


def test_v3_publication_requires_every_approved_candidate(tmp_path: Path):
    store, input_envelope, first = approved_v3_run(tmp_path)
    second = v3_candidate_envelope(
        input_envelope,
        name="V3_SECOND_APPROVED",
    )
    store.record_candidate_envelope(input_envelope.run_id, second)
    make_v3_candidate_review_ready(
        store,
        input_envelope.run_id,
        second,
    )
    review = store.candidate_review(
        input_envelope.run_id,
        second.candidate.candidate_id,
        second.source.source_id,
    )
    store.record_approval(
        ApprovalDecision(
            run_id=review.run_id,
            candidate_id=review.candidate_id,
            decision="approved",
            reviewer="reviewer-b",
            note="第二个候选批准",
            candidate_content_hash=review.candidate_content_hash,
            semantic_hash=review.semantic_hash,
            context_hash=review.context_hash,
            evidence_hash=review.approval_evidence_hash,
            created_at="2026-11-02T00:01:00+00:00",
        ),
        second.source.source_id,
    )
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PublicationConflictError, match="全部已批准候选"):
        service.start_batch(
            input_envelope.run_id,
            (first.candidate.candidate_id,),
            expected_context_hash=input_envelope.context_hash,
            background=False,
        )


def test_v3_publication_failure_is_retryable(tmp_path: Path):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("模拟 V3 构建失败")
        ),
        rollback=lambda **_kwargs: None,
    )
    candidate_ids = (envelope.candidate.candidate_id,)

    failed = service.start_batch(
        input_envelope.run_id,
        candidate_ids,
        expected_context_hash=input_envelope.context_hash,
        background=False,
    )
    assert failed["status"] == "failed"
    assert store.get_candidate(
        input_envelope.run_id,
        candidate_ids[0],
    )["status"] == "build_failed"

    service._prepare = _staged
    service._commit = lambda **_kwargs: None
    retried = service.start_batch(
        input_envelope.run_id,
        candidate_ids,
        expected_context_hash=input_envelope.context_hash,
        background=False,
    )
    assert retried["status"] == "published"
    assert retried["attempt"] == 2


def test_v3_publication_start_is_atomic(tmp_path: Path, monkeypatch):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    batch = _batch(
        store,
        input_envelope.run_id,
        (envelope.candidate.candidate_id,),
        tmp_path,
    )

    monkeypatch.setattr(
        store,
        "_insert_publication_batch_conn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("模拟共享批次写入失败")
        ),
    )
    with pytest.raises(RuntimeError, match="共享批次写入失败"):
        store.start_publication_batch(batch)

    assert store.get_candidate(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )["status"] == "approved"
    assert store.list_publication_batches(input_envelope.run_id) == []
    with duckdb.connect(str(store.db_path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM mining.publication WHERE batch_id=?",
            [batch.batch_id],
        ).fetchone()[0] == 0


def test_v3_publication_detects_evidence_change(tmp_path: Path):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    extra = EvidenceReference(
        candidate_id=envelope.candidate.candidate_id,
        stage="holdout",
        evaluation_id="eval_holdout_recheck",
        input_hash="f" * 64,
        evidence_hash="e" * 64,
        status="passed",
    )
    store.record_candidate_envelope(
        input_envelope.run_id,
        CandidateEnvelope(
            candidate=envelope.candidate,
            source=envelope.source,
            evidence=(*envelope.evidence, extra),
        ),
    )
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ApprovalConflictError, match="审批证据"):
        service.start_batch(
            input_envelope.run_id,
            (envelope.candidate.candidate_id,),
            expected_context_hash=input_envelope.context_hash,
            background=False,
        )


def test_v3_publication_allows_only_one_active_batch(tmp_path: Path):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    prepare_started = Event()
    release_prepare = Event()

    def prepare(**kwargs):
        prepare_started.set()
        assert release_prepare.wait(timeout=5)
        return _staged(kwargs["candidates"])

    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=prepare,
        commit=lambda **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )
    candidate_ids = (envelope.candidate.candidate_id,)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            service.start_batch,
            input_envelope.run_id,
            candidate_ids,
            expected_context_hash=input_envelope.context_hash,
            background=True,
        )
        assert prepare_started.wait(timeout=5)
        assert first.result(timeout=5)["status"] == "staging"
        with pytest.raises(PublicationConflictError, match="正在执行"):
            store.start_publication_batch(
                _batch(
                    store,
                    input_envelope.run_id,
                    candidate_ids,
                    tmp_path,
                )
            )
        release_prepare.set()
        service.close()

    batches = store.list_publication_batches(input_envelope.run_id)
    assert [item["status"] for item in batches] == ["published"]


def test_v3_service_restart_rolls_back_uncommitted_batch(tmp_path: Path):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    batch = _batch(
        store,
        input_envelope.run_id,
        (envelope.candidate.candidate_id,),
        tmp_path,
    )
    store.start_publication_batch(batch)
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )

    service.initialize()

    assert service.get_batch(
        input_envelope.run_id,
        batch.batch_id,
    )["status"] == "failed"
    assert store.get_candidate(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
    )["status"] == "build_failed"


@pytest.mark.parametrize("run_status", ("failed", "interrupted", "stale"))
def test_v3_publication_rejects_non_review_ready_run(
    tmp_path: Path,
    run_status: str,
):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    candidate_ids = (envelope.candidate.candidate_id,)

    with duckdb.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE mining_v3.runtime_run_control SET status=? "
            "WHERE run_id=?",
            [run_status, input_envelope.run_id],
        )

    with pytest.raises(PublicationConflictError, match=run_status):
        store.load_publication_inputs(
            input_envelope.run_id,
            candidate_ids,
            expected_context_hash=input_envelope.context_hash,
        )


def test_v3_publication_rechecks_run_before_commit(tmp_path: Path):
    store, input_envelope, envelope = approved_v3_run(tmp_path)
    runtime = MiningV3RuntimeStore(
        store.db_path,
        store.data_root,
        contract_store=store,
    )
    candidate_ids = (envelope.candidate.candidate_id,)
    batch = _batch(
        store,
        input_envelope.run_id,
        candidate_ids,
        tmp_path,
    )
    store.start_publication_batch(batch)
    store.mark_publication_validating(batch.batch_id)
    runtime.transition_search_run(
        input_envelope.run_id,
        "stale",
        event_key="run_stale_before_commit",
        stage="review_ready",
    )

    with pytest.raises(PublicationConflictError, match="stale"):
        store.complete_publication_batch(
            batch.batch_id,
            manifest_hash="a" * 64,
            artifact_sha256="b" * 64,
            formal_factor_names=(envelope.candidate.name,),
        )

    assert store.get_candidate(
        input_envelope.run_id,
        candidate_ids[0],
    )["status"] == "building"
    assert store.get_publication_batch(batch.batch_id)["status"] == "validating"


def test_v3_manifest_entry_keeps_source_and_envelope_identity():
    from factors.registry import FactorSpec

    spec = FactorSpec(
        name="V3_SOURCE_TEST",
        family="mined",
        formula="mean(close, 5)",
        description="V3 来源审计",
        steps=("读取收盘价", "计算五日均值"),
        layers=("mean(close, 5)",),
        lookback=4,
    )
    candidate = {
        "candidate_id": "cand_v3_source",
        "content_hash": "a" * 64,
        "semantic_hash": "b" * 64,
        "source_kind": "v3_research",
        "source_context_hash": "c" * 64,
        "candidate_envelope_hash": "d" * 64,
        "source_reference_hash": "e" * 64,
        "approval_evidence_hash": "f" * 64,
    }

    entry = _entry(
        spec,
        candidate,
        run_id="researchrun_source",
        approval_id="approval_source",
    )

    assert entry["source_kind"] == "v3_research"
    assert entry["candidate_envelope_hash"] == "d" * 64
    assert entry["source_reference_hash"] == "e" * 64
    assert entry["approval_evidence_hash"] == "f" * 64
