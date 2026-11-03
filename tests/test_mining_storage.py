"""挖掘候选库的事务、幂等、并发和恢复测试。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import multiprocessing
from pathlib import Path

import duckdb
import pytest

from mining.contracts import (
    ApprovalDecision,
    CandidateDefinition,
    EvaluationReport,
    FrozenContext,
    MiningRunRequest,
    PublicationAttempt,
    TechnicalGate,
    sha256_json,
    utc_now_iso,
)
from mining.errors import (
    ApprovalConflictError,
    HoldoutError,
    MiningBusyError,
    MiningContractError,
    MiningDuplicateError,
    MiningError,
    MiningNotFoundError,
    MiningRecoveryError,
    MiningStateError,
)
from mining.storage import MiningStore


def _request(key: str = "client-run-0001") -> MiningRunRequest:
    """返回稳定测试请求。"""

    return MiningRunRequest(key, "短期价格反转假设")


def _context() -> FrozenContext:
    """返回与请求预算一致的冻结上下文。"""

    return FrozenContext(data_fingerprint="a" * 64)


def _candidate(
    index: int = 1,
    *,
    expression: str | None = None,
    description: str = "五日价格变化取负",
) -> CandidateDefinition:
    """返回可重复构造的候选。"""

    return CandidateDefinition(
        name=f"MINE_REVERSAL_{index:08d}",
        mechanism="short reversal",
        description=description,
        canonical_expression=expression or "neg(sub(close,lag(close,5)))",
        direction=1,
        lookback=5,
        dependencies=("close",),
        output_type="number",
        steps=("计算五日变化", "取负"),
        discovery_round=1,
        discovery_index=index,
    )


@pytest.fixture
def store(tmp_path: Path) -> MiningStore:
    """创建只存在临时目录中的候选库。"""

    return MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "mining")


def _run(store: MiningStore) -> dict:
    """创建并返回测试运行。"""

    return store.create_run(_request(), _context())


def _to_review_ready(
    store: MiningStore,
    run_id: str,
    candidate: CandidateDefinition,
    *,
    claim_holdout: bool = True,
) -> str:
    """把候选推进到待审核，并返回与技术门禁和测试报告一致的证据哈希。"""

    candidate_id = candidate.candidate_id
    run = store.get_run(run_id)
    context_hash = run["context"]["context_hash"]
    for target, stage in (
        ("validating", "technical"),
        ("screening", "screening"),
        ("evaluating", "evaluation"),
        ("evaluated", "evaluation"),
        ("holdout_running", "holdout"),
    ):
        store.transition_candidate(
            run_id,
            candidate_id,
            target,
            event_key=f"to_{target}",
            stage=stage,
        )
    gate = store.record_technical_gate(
        TechnicalGate(
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_content_hash=candidate.content_hash,
            context_hash=context_hash,
            rule_version="technical-v1",
            passed=True,
            checks={"finite": True},
        )
    )
    report = store.record_evaluation(
        EvaluationReport(
            run_id=run_id,
            candidate_id=candidate_id,
            stage="holdout",
            candidate_content_hash=candidate.content_hash,
            context_hash=context_hash,
            input_hash="9" * 64,
            status="passed",
            metrics={"rank_ic_mean": 0.02},
            warnings=(),
            artifact_path=None,
            artifact_sha256=None,
            error_code=None,
            error_message=None,
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
    )
    usage = None
    if claim_holdout:
        usage = store.claim_holdout(
            run_id=run_id,
            candidate_ids=[candidate_id],
            evidence_hash="8" * 64,
        )
    store.transition_candidate(
        run_id,
        candidate_id,
        "review_ready",
        event_key="to_review_ready",
        stage="review",
    )
    material = {
        "gate_hash": gate["check_hash"],
        "holdout_report_ids": [report["report_id"]],
    }
    if usage is not None:
        material["holdout_batch_hash"] = usage["batch_hash"]
        material["holdout_evidence_hash"] = usage["evidence_hash"]
    return sha256_json(material)


def test_create_run_is_idempotent_and_conflicts_on_changed_request(store: MiningStore):
    """相同幂等请求复用运行，不同请求不能覆盖。"""

    first = store.create_run(_request(), _context())
    second = store.create_run(_request(), _context())
    assert first["run_id"] == second["run_id"]
    assert second["reused"] is True
    assert len(store.list_events(first["run_id"])) == 1

    changed = MiningRunRequest("client-run-0001", "另一条研究假设，要求产生候选")
    with pytest.raises(MiningDuplicateError, match="幂等键"):
        store.create_run(changed, _context())
    with pytest.raises(MiningDuplicateError, match="幂等键"):
        store.create_run(_request(), _context(), worker_mode="mock")


def test_create_run_rolls_back_metadata_when_event_write_fails(
    store: MiningStore,
    monkeypatch,
):
    """运行、上下文和首个事件在同一事务，事件失败时不留下半完成记录。"""

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("injected event failure")

    monkeypatch.setattr(store, "_append_event_conn", fail_event)
    with pytest.raises(RuntimeError, match="injected"):
        store.create_run(_request(), _context())

    with duckdb.connect(str(store.db_path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM mining.run").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM mining.run_context").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM mining.event").fetchone()[0] == 0


def test_state_transition_rolls_back_when_event_write_fails(
    store: MiningStore,
    monkeypatch,
):
    """状态更新和事件处于同一事务，事件失败不能留下新状态。"""

    run = _run(store)

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("injected transition failure")

    monkeypatch.setattr(store, "_append_event_conn", fail_event)
    with pytest.raises(RuntimeError, match="injected"):
        store.transition_run(
            run["run_id"],
            "generating",
            event_key="broken_generating",
            stage="generating",
        )
    assert store.get_run(run["run_id"])["status"] == "queued"
    assert len(store.list_events(run["run_id"])) == 1


def test_candidate_insert_is_idempotent_and_rejects_semantic_duplicate(store: MiningStore):
    """完整内容重复幂等，语义相同但展示不同则冲突。"""

    run = _run(store)
    candidate = _candidate()
    first = store.add_candidate(run["run_id"], candidate)
    second = store.add_candidate(run["run_id"], candidate)
    assert second["reused"] is True

    described = _candidate(description="另一种展示文字")
    with pytest.raises(MiningDuplicateError, match="语义重复"):
        store.add_candidate(run["run_id"], described)
    assert first["content_hash"] == candidate.content_hash


def test_candidate_state_transition_and_event_are_atomic(store: MiningStore):
    """状态迁移与事件序号一起提交，重复事件调用幂等。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    updated = store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "validating",
        event_key="candidate_validating",
        stage="validation",
    )
    assert updated["status"] == "validating"
    repeated = store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "validating",
        event_key="candidate_validating",
        stage="validation",
    )
    assert repeated["status"] == "validating"
    assert [item["sequence"] for item in store.list_events(run["run_id"])] == [1, 2, 3]


def test_candidate_state_machine_rejects_gate_bypass(store: MiningStore):
    """存储层不能通过直接调用绕过状态机。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    with pytest.raises(MiningStateError):
        store.transition_candidate(
            run["run_id"],
            candidate.candidate_id,
            "review_ready",
            event_key="illegal_review",
            stage="review",
        )


def test_evaluation_and_gate_bind_current_candidate(store: MiningStore):
    """评估和门禁只能绑定当前候选内容与上下文。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "validating",
        event_key="to_validating",
        stage="validation",
    )
    store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "screening",
        event_key="to_screening",
        stage="screening",
    )
    context_hash = run["context"]["context_hash"]
    report = EvaluationReport(
        run_id=run["run_id"],
        candidate_id=candidate.candidate_id,
        stage="quick",
        candidate_content_hash=candidate.content_hash,
        context_hash=context_hash,
        input_hash="b" * 64,
        status="passed",
        metrics={"rank_ic_mean": 0.03},
        warnings=(),
        artifact_path="runs/test/quick.parquet",
        artifact_sha256="c" * 64,
        error_code=None,
        error_message=None,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
    )
    saved = store.record_evaluation(report)
    assert saved["metrics"]["rank_ic_mean"] == 0.03
    assert store.record_evaluation(report)["report_id"] == report.report_id

    gate = TechnicalGate(
        run_id=run["run_id"],
        candidate_id=candidate.candidate_id,
        candidate_content_hash=candidate.content_hash,
        context_hash=context_hash,
        rule_version="technical-v1",
        passed=True,
        checks={"finite": True, "lookback": True},
    )
    assert store.record_technical_gate(gate)["passed"] is True

    wrong = EvaluationReport(
        **{
            **report.__dict__,
            "candidate_id": candidate.candidate_id,
            "candidate_content_hash": "d" * 64,
            "input_hash": "e" * 64,
        }
    )
    with pytest.raises(ApprovalConflictError, match="内容"):
        store.record_evaluation(wrong)


def test_approval_is_final_and_repeated_request_is_idempotent(store: MiningStore):
    """首次审批写入后，相同请求返回原记录，相反决定冲突。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    evidence_hash = _to_review_ready(store, run["run_id"], candidate)
    approved = ApprovalDecision(
        run_id=run["run_id"],
        candidate_id=candidate.candidate_id,
        decision="approved",
        reviewer="local",
        note="质量门禁和测试段结论已核对",
        candidate_content_hash=candidate.content_hash,
        semantic_hash=candidate.semantic_hash,
        context_hash=run["context"]["context_hash"],
        evidence_hash=evidence_hash,
    )
    saved = store.record_approval(approved)
    assert saved["decision"] == "approved"
    assert store.record_approval(approved)["approval_id"] == approved.approval_id

    rejected = ApprovalDecision(
        **{
            **approved.__dict__,
            "decision": "rejected",
            "note": "后来改为拒绝",
        }
    )
    with pytest.raises(MiningDuplicateError):
        store.record_approval(rejected)


def test_approval_requires_claimed_holdout_segment(store: MiningStore):
    """即使门禁和报告通过，未领取测试段也不能审批。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    evidence_hash = _to_review_ready(
        store,
        run["run_id"],
        candidate,
        claim_holdout=False,
    )
    with pytest.raises(ApprovalConflictError, match="尚未领取"):
        store.record_approval(
            ApprovalDecision(
                run_id=run["run_id"],
                candidate_id=candidate.candidate_id,
                decision="approved",
                reviewer="local",
                note="试图绕过测试段领取",
                candidate_content_hash=candidate.content_hash,
                semantic_hash=candidate.semantic_hash,
                context_hash=run["context"]["context_hash"],
                evidence_hash=evidence_hash,
            )
        )


def test_publication_attempt_requires_approval_and_is_idempotent(store: MiningStore):
    """发布记录必须绑定批准，同一尝试重复写入不重复追加。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    evidence_hash = _to_review_ready(store, run["run_id"], candidate)
    approved = store.record_approval(
        ApprovalDecision(
            run_id=run["run_id"],
            candidate_id=candidate.candidate_id,
            decision="approved",
            reviewer="local",
            note="批准进入暂存构建",
            candidate_content_hash=candidate.content_hash,
            semantic_hash=candidate.semantic_hash,
            context_hash=run["context"]["context_hash"],
            evidence_hash=evidence_hash,
        )
    )
    attempt = PublicationAttempt(
        run_id=run["run_id"],
        candidate_id=candidate.candidate_id,
        approval_id=approved["approval_id"],
        status="staging",
        staging_path=str(store.candidate_dir(run["run_id"], candidate.candidate_id) / "staging"),
        manifest_hash=None,
        formal_factor_name=None,
        artifact_sha256=None,
        error_code=None,
        error_message=None,
        attempt=1,
        started_at=utc_now_iso(),
    )
    first = store.record_publication(attempt)
    second = store.record_publication(attempt)
    assert first["publication_id"] == second["publication_id"]
    assert first["status"] == "staging"
    validating = store.transition_publication(
        first["publication_id"],
        "validating",
        event_key="publication_validating",
    )
    assert validating["status"] == "validating"
    published = store.transition_publication(
        first["publication_id"],
        "published",
        event_key="publication_published",
        manifest_hash="a" * 64,
        formal_factor_name="MINE_PUBLISHED_0001",
        artifact_sha256="b" * 64,
    )
    assert published["status"] == "published"


def _race_approval(
    db_path: str,
    data_root: str,
    run_id: str,
    candidate_id: str,
    content_hash: str,
    semantic_hash: str,
    context_hash: str,
    evidence_hash: str,
    decision: str,
) -> str:
    """子进程尝试同一候选的相反审批。"""

    try:
        store = MiningStore(db_path, data_root)
        store.record_approval(
            ApprovalDecision(
                run_id=run_id,
                candidate_id=candidate_id,
                decision=decision,
                reviewer="local",
                note=f"并发审批：{decision}",
                candidate_content_hash=content_hash,
                semantic_hash=semantic_hash,
                context_hash=context_hash,
                evidence_hash=evidence_hash,
            )
        )
        return decision
    except MiningError:
        return "conflict"


def _race_approval_or_cancel(
    db_path: str,
    data_root: str,
    run_id: str,
    candidate_id: str,
    content_hash: str,
    semantic_hash: str,
    context_hash: str,
    evidence_hash: str,
    action: str,
) -> str:
    """子进程竞争候选审批和取消。"""

    try:
        store = MiningStore(db_path, data_root)
        if action == "cancel":
            store.transition_candidate(
                run_id,
                candidate_id,
                "cancelled",
                event_key="concurrent_cancel",
                stage="review",
            )
            return "cancelled"
        store.record_approval(
            ApprovalDecision(
                run_id=run_id,
                candidate_id=candidate_id,
                decision="approved",
                reviewer="local",
                note="并发审批通过",
                candidate_content_hash=content_hash,
                semantic_hash=semantic_hash,
                context_hash=context_hash,
                evidence_hash=evidence_hash,
            )
        )
        return "approved"
    except MiningError:
        return "conflict"


def test_concurrent_approval_has_single_final_decision(store: MiningStore):
    """批准与拒绝并发时只写入一个最终结论。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    evidence_hash = _to_review_ready(store, run["run_id"], candidate)
    common = (
        str(store.db_path),
        str(store.data_root),
        run["run_id"],
        candidate.candidate_id,
        candidate.content_hash,
        candidate.semantic_hash,
        run["context"]["context_hash"],
        evidence_hash,
    )
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(_race_approval, *common, decision)
            for decision in ("approved", "rejected")
        ]
        results = sorted(future.result() for future in futures)
    assert results[0] in {"approved", "conflict"}
    assert results[1] in {"rejected", "conflict"}
    assert "conflict" in results
    status = store.get_candidate(run["run_id"], candidate.candidate_id)["status"]
    assert status in {"approved", "rejected"}
    approval = store.get_approval(run["run_id"], candidate.candidate_id)
    assert approval["decision"] == status


def test_concurrent_approval_and_cancel_has_single_winner(store: MiningStore):
    """审批与取消共用状态锁，不能同时成功。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    evidence_hash = _to_review_ready(store, run["run_id"], candidate)
    common = (
        str(store.db_path),
        str(store.data_root),
        run["run_id"],
        candidate.candidate_id,
        candidate.content_hash,
        candidate.semantic_hash,
        run["context"]["context_hash"],
        evidence_hash,
    )
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(_race_approval_or_cancel, *common, action)
            for action in ("approve", "cancel")
        ]
        results = sorted(future.result() for future in futures)
    assert "conflict" in results
    status = store.get_candidate(run["run_id"], candidate.candidate_id)["status"]
    assert status in {"approved", "cancelled"}


def test_holdout_segment_can_only_be_claimed_once(store: MiningStore):
    """模块测试段同一批次重试幂等，其他批次明确冲突。"""

    run = _run(store)
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    for target, stage in (
        ("validating", "technical"),
        ("screening", "screening"),
        ("evaluating", "evaluation"),
        ("evaluated", "evaluation"),
    ):
        store.transition_candidate(
            run["run_id"],
            candidate.candidate_id,
            target,
            event_key=f"to_{target}",
            stage=stage,
        )
    first = store.claim_holdout(
        run_id=run["run_id"],
        candidate_ids=[candidate.candidate_id],
        evidence_hash="1" * 64,
    )
    repeated = store.claim_holdout(
        run_id=run["run_id"],
        candidate_ids=[candidate.candidate_id],
        evidence_hash="1" * 64,
    )
    assert repeated["batch_hash"] == first["batch_hash"]
    with pytest.raises(HoldoutError, match="已被"):
        store.claim_holdout(
            run_id=run["run_id"],
            candidate_ids=[candidate.candidate_id],
            evidence_hash="2" * 64,
        )


def test_interrupted_run_releases_slot_and_resumes_exact_stage(store: MiningStore):
    """重启恢复不自动执行，只标记中断并允许显式回到原阶段。"""

    run = _run(store)
    store.transition_run(
        run["run_id"],
        "generating",
        event_key="start_generation",
        stage="generating",
    )
    candidate = _candidate()
    store.add_candidate(run["run_id"], candidate)
    store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "validating",
        event_key="validate_candidate",
        stage="validation",
    )

    assert [item["run_id"] for item in [store.active_run()] if item] == [run["run_id"]]
    assert store.interrupt_active_runs() == [run["run_id"]]
    assert store.active_run() is None
    interrupted = store.get_run(run["run_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["interrupted_from"] == "generating"
    assert store.get_candidate(run["run_id"], candidate.candidate_id)["status"] == "interrupted"

    resumed = store.resume_run(run["run_id"])
    assert resumed["status"] == "generating"
    assert resumed["interruption_count"] == 1
    assert store.get_candidate(run["run_id"], candidate.candidate_id)["status"] == "validating"


def _race_run_transition(db_path: str, data_root: str, run_id: str) -> str:
    """子进程尝试同一运行状态迁移。"""

    try:
        store = MiningStore(db_path, data_root)
        store.transition_run(
            run_id,
            "generating",
            event_key=f"start_{__import__('os').getpid()}",
            stage="generating",
        )
        return "ok"
    except MiningStateError:
        return "conflict"


def test_concurrent_run_transition_has_single_winner(store: MiningStore):
    """两个进程竞争同一 CAS 时只有一个迁移成功。"""

    run = _run(store)
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                _race_run_transition,
                str(store.db_path),
                str(store.data_root),
                run["run_id"],
            )
            for _ in range(2)
        ]
        results = sorted(future.result() for future in futures)
    assert results == ["conflict", "ok"]
    assert store.get_run(run["run_id"])["status"] == "generating"


def test_only_one_run_can_occupy_execution_slot(store: MiningStore):
    """不同幂等请求可以排队，但同一时刻只有一个活动运行。"""

    first = _run(store)
    second = store.create_run(
        MiningRunRequest("client-run-0002", "第二条独立研究假设"),
        _context(),
    )
    store.transition_run(
        first["run_id"],
        "generating",
        event_key="first_generating",
        stage="generating",
    )
    with pytest.raises(MiningBusyError, match="正在运行"):
        store.transition_run(
            second["run_id"],
            "generating",
            event_key="second_generating",
            stage="generating",
        )
    assert store.get_run(second["run_id"])["status"] == "queued"


def test_manual_transition_cannot_forge_interrupted_state(store: MiningStore):
    """中断状态必须由重启恢复流程写入。"""

    run = _run(store)
    store.transition_run(
        run["run_id"],
        "generating",
        event_key="start_generation",
        stage="generating",
    )
    with pytest.raises(MiningRecoveryError, match="启动恢复"):
        store.transition_run(
            run["run_id"],
            "interrupted",
            event_key="fake_interrupt",
            stage="generating",
        )


def test_cleanup_only_removes_old_transient_paths_for_interrupted_runs(
    store: MiningStore,
):
    """清理函数不碰活动运行，也不碰新鲜临时文件。"""

    run = _run(store)
    store.transition_run(
        run["run_id"],
        "generating",
        event_key="start_generation",
        stage="generating",
    )
    store.interrupt_active_runs()
    candidate_dir = store.candidate_dir(run["run_id"], _candidate().candidate_id)
    old_tmp = candidate_dir / "quick.parquet.tmp"
    old_tmp.write_text("partial", encoding="utf-8")
    factor_staging = candidate_dir / ".factor.staging-old"
    factor_staging.mkdir()
    (factor_staging / "part.parquet").write_bytes(b"partial")
    batch_staging = candidate_dir.parent / ".batch-materialization.staging-old"
    batch_staging.mkdir()
    (batch_staging / "combined.parquet").write_bytes(b"partial")
    old_time = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    import os

    os.utime(old_tmp, (old_time, old_time))
    os.utime(factor_staging, (old_time, old_time))
    os.utime(batch_staging, (old_time, old_time))
    removed = store.cleanup_transient_temp(
        now=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    assert set(removed) == {
        old_tmp.resolve(),
        factor_staging.resolve(),
        batch_staging.resolve(),
    }
    assert not old_tmp.exists()
    assert not factor_staging.exists()
    assert not batch_staging.exists()


def test_unknown_schema_version_is_rejected(tmp_path: Path):
    """已有候选库使用未知 schema 时拒绝启动。"""

    path = tmp_path / "factor_mining.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE SCHEMA mining")
        conn.execute("CREATE TABLE mining.schema_meta (key VARCHAR, value VARCHAR)")
        conn.execute("INSERT INTO mining.schema_meta VALUES ('schema_version', 'future_v9')")
    with pytest.raises(Exception, match="schema"):
        MiningStore(path, tmp_path / "data")


def test_event_payload_rejects_credentials_before_insert(store: MiningStore):
    """敏感键不能通过公开事件入口落入候选库。"""

    run = _run(store)
    with pytest.raises(MiningContractError, match="api_key"):
        store.append_event(
            run["run_id"],
            event_key="bad_secret",
            stage="generating",
            event_type="note",
            payload={"api_key": "should-not-persist"},
        )
    assert len(store.list_events(run["run_id"])) == 1


def test_run_context_metadata_and_events_are_json_decoded(store: MiningStore):
    """查询结果保持可读结构而不是数据库内部 JSON 字符串。"""

    run = _run(store)
    loaded = store.get_run(run["run_id"])
    assert loaded["request"]["hypothesis"] == "短期价格反转假设"
    assert isinstance(loaded["context"]["payload"], dict)
    assert store.list_runs()[0]["run_id"] == run["run_id"]
    with pytest.raises(MiningNotFoundError):
        store.get_run("run_missing")
