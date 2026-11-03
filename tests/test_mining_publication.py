"""S8 发布批次、全量构建、原子提交和已发布因子读取测试。"""

from __future__ import annotations

from pathlib import Path
import shutil
import os
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

import factors
import quant
from calculate import resolve_factor
from common.paths import DB_FILE, FACTOR_DB_FILE
from mining.contracts import (
    ApprovalDecision,
    EvaluationReport,
    FrozenContext,
    MiningRunRequest,
    TechnicalGate,
    utc_now_iso,
)
from mining.expression_validation import validate_expression
from mining.publication_builder import (
    StagedPublication,
    commit_publication,
    prepare_publication,
)
from mining.publication_contracts import new_publication_batch
from mining.publication_service import MiningPublicationService
from mining.storage import MiningStore
from scripts.build_factors import catalog
from trading.backtest.snapshot import _factor_family
from alpha.final_holdout_data import _score_b4


def _approved_candidate(store: MiningStore) -> tuple[str, dict]:
    """通过正常门禁、测试段和审批路径产生一个已批准候选。"""

    request = MiningRunRequest("publication-test-0001", "短期趋势可能有效")
    context = FrozenContext(data_fingerprint="a" * 64)
    run = store.create_run(request, context)
    candidate = validate_expression("mean(close,5)", direction=1).to_candidate(
        mechanism="短期趋势",
        description="五日收盘均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    store.add_candidate(run["run_id"], candidate)
    context_hash = run["context"]["context_hash"]
    for target, stage in (
        ("validating", "technical"),
        ("screening", "screening"),
        ("evaluating", "evaluation"),
        ("evaluated", "evaluation"),
        ("holdout_running", "holdout"),
    ):
        store.transition_candidate(
            run["run_id"],
            candidate.candidate_id,
            target,
            event_key=f"publication_{target}",
            stage=stage,
        )
    store.record_technical_gate(
        TechnicalGate(
            run_id=run["run_id"],
            candidate_id=candidate.candidate_id,
            candidate_content_hash=candidate.content_hash,
            context_hash=context_hash,
            rule_version="publication-test",
            passed=True,
            checks={"values": True},
        )
    )
    store.record_evaluation(
        EvaluationReport(
            run_id=run["run_id"],
            candidate_id=candidate.candidate_id,
            stage="holdout",
            candidate_content_hash=candidate.content_hash,
            context_hash=context_hash,
            input_hash="1" * 64,
            status="passed",
            metrics={"valid_days": 30},
            warnings=(),
            artifact_path=None,
            artifact_sha256=None,
            error_code=None,
            error_message=None,
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
    )
    store.claim_holdout(
        run_id=run["run_id"],
        candidate_ids=[candidate.candidate_id],
        evidence_hash="2" * 64,
    )
    store.transition_candidate(
        run["run_id"],
        candidate.candidate_id,
        "review_ready",
        event_key="publication_review_ready",
        stage="review",
    )
    evidence = store.approval_evidence(
        run["run_id"],
        candidate.candidate_id,
        candidate_content_hash=candidate.content_hash,
        context_hash=context_hash,
    )
    store.record_approval(
        ApprovalDecision(
            run_id=run["run_id"],
            candidate_id=candidate.candidate_id,
            decision="approved",
            reviewer="local",
            note="发布测试批准",
            candidate_content_hash=candidate.content_hash,
            semantic_hash=candidate.semantic_hash,
            context_hash=context_hash,
            evidence_hash=evidence,
        )
    )
    return run["run_id"], store.get_candidate(
        run["run_id"], candidate.candidate_id
    )


def test_publication_batch_transitions_all_approved_candidates(tmp_path: Path):
    """整批启动和失败状态必须原子落到批次与候选。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "mining")
    run_id, candidate = _approved_candidate(store)
    approval = store.get_approval(run_id, candidate["candidate_id"])
    batch = new_publication_batch(
        run_id=run_id,
        context_hash=store.get_run(run_id)["context"]["context_hash"],
        candidate_ids=(candidate["candidate_id"],),
        approval_ids=(approval["approval_id"],),
        staging_path=str(tmp_path / "staging"),
        attempt=1,
    )
    started = store.start_publication_batch(batch)
    assert started["status"] == "staging"
    assert store.get_candidate(run_id, candidate["candidate_id"])["status"] == "building"
    failed = store.fail_publication_batch(
        batch.batch_id,
        error_code="MINING-122",
        error_message="测试失败",
    )
    assert failed["status"] == "failed"
    assert (
        store.get_candidate(run_id, candidate["candidate_id"])["status"]
        == "build_failed"
    )


def test_publication_service_failure_is_reported_and_retryable(tmp_path: Path):
    """后台构建异常要返回稳定错误码，并把候选留为可重试状态。"""

    store = MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "mining")
    run_id, candidate = _approved_candidate(store)
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "mining",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("模拟构建失败")
        ),
    )
    batch = service.start_batch(
        run_id,
        [candidate["candidate_id"]],
        expected_context_hash=store.get_run(run_id)["context"]["context_hash"],
        background=False,
    )
    assert batch["status"] == "failed"
    assert batch["error_code"] == "MINING-109"
    assert store.get_candidate(run_id, candidate["candidate_id"])["status"] == "build_failed"
    service._prepare = lambda **_kwargs: type(
        "Staged",
        (),
        {
            "manifest_hash": "a" * 64,
            "artifact_sha256": "b" * 64,
            "factor_names": (candidate["name"],),
        },
    )()
    service._commit = lambda **_kwargs: None
    service._rollback = lambda **_kwargs: None
    retried = service.start_batch(
        run_id,
        [candidate["candidate_id"]],
        expected_context_hash=store.get_run(run_id)["context"]["context_hash"],
        background=False,
    )
    assert retried["status"] == "published"
    assert retried["attempt"] == 2
    assert store.get_candidate(run_id, candidate["candidate_id"])["status"] == "published"


def test_committed_recovery_maps_factor_names_by_candidate_id(tmp_path: Path):
    """启动恢复必须按候选 ID 对应正式名，不能依赖清单排序。"""

    import json

    staging = tmp_path / "staging"
    version = tmp_path / "factors" / ".mined-versions" / "batch"
    version.mkdir(parents=True)
    staging.mkdir()
    (staging / "commit.json").write_text(
        json.dumps({"new_link": ".mined-versions/batch"}),
        encoding="utf-8",
    )
    (version / "definitions.json").write_text(
        json.dumps(
            {
                "factors": [
                    {"candidate_id": "cand_a", "name": "MINE_A"},
                    {"candidate_id": "cand_b", "name": "MINE_B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    names = MiningPublicationService._committed_factor_names(
        {
            "staging_path": str(staging),
            "candidate_ids": ["cand_b", "cand_a"],
        },
        tmp_path / "factors",
    )
    assert names == ["MINE_B", "MINE_A"]


def test_existing_publication_table_is_migrated_with_batch_id(tmp_path: Path):
    """旧候选库升级后保留发布记录，并新增批次关联列。"""

    db_path = tmp_path / "factor_mining.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE SCHEMA mining")
    conn.execute(
        "CREATE TABLE mining.publication ("
        "publication_id VARCHAR PRIMARY KEY, run_id VARCHAR, candidate_id VARCHAR, "
        "approval_id VARCHAR, status VARCHAR, staging_path VARCHAR, "
        "manifest_hash VARCHAR, formal_factor_name VARCHAR, artifact_sha256 VARCHAR, "
        "error_code VARCHAR, error_message VARCHAR, attempt INTEGER, "
        "started_at VARCHAR, finished_at VARCHAR)"
    )
    conn.close()
    MiningStore(db_path, tmp_path / "data")
    with duckdb.connect(str(db_path), read_only=True) as check:
        columns = [
            row[0]
            for row in check.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='mining' AND table_name='publication'"
            ).fetchall()
        ]
    assert "batch_id" in columns
    assert "source_kind" in columns
    with duckdb.connect(str(db_path), read_only=True) as check:
        batch_columns = [
            row[0]
            for row in check.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='mining' AND table_name='publication_batch'"
            ).fetchall()
        ]
    assert "source_kind" in batch_columns


def test_commit_failure_restores_previous_link_and_catalog(
    tmp_path: Path,
    monkeypatch,
):
    """目录库替换失败时，旧正式链接和因子库必须保持不变。"""

    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    old_version = factor_dir / ".mined-versions" / "old"
    old_version.mkdir(parents=True)
    (old_version / "definitions.json").write_text(
        '{"schema_version":"mining_published_factors_v1","factors":[]}',
        encoding="utf-8",
    )
    (factor_dir / "mined").symlink_to(old_version.relative_to(factor_dir))
    catalog_file = tmp_path / "factors.duckdb"
    with catalog.connect(catalog_file) as conn:
        catalog.ensure_catalog(conn)
    before_catalog = catalog_file.read_bytes()
    staging = tmp_path / "staging"
    family_root = staging / "mined"
    family_root.mkdir(parents=True)
    (family_root / "definitions.json").write_text(
        '{"schema_version":"mining_published_factors_v1","factors":[]}',
        encoding="utf-8",
    )
    staged_catalog = staging / "factors.staged.duckdb"
    with catalog.connect(staged_catalog) as conn:
        catalog.ensure_catalog(conn)
    staged = StagedPublication(
        staging_root=staging,
        family_root=family_root,
        definitions_path=family_root / "definitions.json",
        catalog_path=staged_catalog,
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=(),
        specs=(),
        coverage=(),
    )
    real_replace = os.replace

    def fail_catalog_replace(source, target):
        if Path(source) == staged_catalog and Path(target) == catalog_file:
            raise OSError("模拟目录库替换失败")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_catalog_replace)
    with pytest.raises(Exception, match="提交失败"):
        commit_publication(
            staged=staged,
            batch_id="pub_batch_failure",
            factor_dir=factor_dir,
            catalog_file=catalog_file,
        )
    assert os.readlink(factor_dir / "mined") == old_version.relative_to(
        factor_dir
    ).as_posix()
    assert catalog_file.read_bytes() == before_catalog


def test_published_mined_factor_is_atomic_and_readable(tmp_path: Path):
    """真实行情临时目录完成全量发布，并可被清单、quant 和评估解析。"""

    if not DB_FILE.exists() or not FACTOR_DB_FILE.exists():
        pytest.skip("需要真实行情库和因子目录库")
    if os.environ.get("MINING_PUBLICATION_CHILD") != "1":
        env = dict(os.environ)
        env["MINING_PUBLICATION_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                    "-m",
                    "",
                f"{Path(__file__).resolve()}::"
                "test_published_mined_factor_is_atomic_and_readable",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    catalog_file = tmp_path / "factors.duckdb"
    shutil.copy2(FACTOR_DB_FILE, catalog_file)
    candidates = tuple(
        validate_expression(expression, direction=1).to_candidate(
            mechanism="短期趋势",
            description=f"{expression} 正式发布测试",
            steps=("读取收盘价", "按登记窗口计算均值"),
            discovery_round=1,
            discovery_index=index,
        )
        for index, expression in enumerate(
            ("mean(close,5)", "mean(close,10)"),
            start=1,
        )
    )
    candidate_records = tuple(
        {
            "candidate_id": candidate.candidate_id,
            "semantic_hash": candidate.semantic_hash,
            "content_hash": candidate.content_hash,
            "name": candidate.name,
            "definition": {
                "canonical_expression": candidate.canonical_expression,
                "direction": candidate.direction,
                "description": candidate.description,
                "steps": list(candidate.steps),
            },
        }
        for candidate in candidates
    )
    context = FrozenContext(data_fingerprint="a" * 64)
    staging = tmp_path / "publication"
    staged = prepare_publication(
        batch_id="pub_batch_test",
        run_id="run_test_publication",
        candidates=candidate_records,
        approval_ids=("approval_test_1", "approval_test_2"),
        context=context,
        staging_root=staging,
        db_file=DB_FILE,
        source_catalog=catalog_file,
        factor_dir=factor_dir,
    )
    commit_publication(
        staged=staged,
        batch_id="pub_batch_test",
        factor_dir=factor_dir,
        catalog_file=catalog_file,
    )
    assert (factor_dir / "mined").is_symlink()
    assert (factor_dir / "mined" / "definitions.json").is_file()
    for candidate in candidates:
        assert factors.resolve(candidate.name, factor_dir).family == "mined"
        assert _factor_family(candidate.name, factor_dir)[0] == "mined"
    listed = set(quant.load_factor_list(db_path=catalog_file)["name"])
    assert {candidate.name for candidate in candidates} <= listed
    panel = quant.load_factor_panel(
        start="2024-01-02",
        end="2024-01-05",
        names=[candidate.name for candidate in candidates],
        factor_dir=factor_dir,
    )
    assert not panel.empty
    assert resolve_factor(candidates[0].name, factor_dir).formal is True
    with duckdb.connect(str(catalog_file), read_only=True) as conn:
        count = conn.execute("SELECT count(*) FROM v.mined").fetchone()[0]
    assert count > 1_000_000
    model_data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "stock_code": ["SH600000", "SZ000001"],
            "eligible": [True, True],
            candidates[0].name: [1.0, 2.0],
            candidates[1].name: [3.0, 1.0],
        }
    )
    b4 = _score_b4(
        model_data,
        pd.DatetimeIndex(["2024-01-02"]),
        (candidates[0].name, candidates[1].name),
        factor_dir,
    )
    assert set(b4["stock_code"]) == {"SH600000", "SZ000001"}

    appended = validate_expression("std(close,10)", direction=1).to_candidate(
        mechanism="短期波动",
        description="十日收盘波动",
        steps=("读取收盘价", "计算十日标准差"),
        discovery_round=1,
        discovery_index=3,
    )
    append_record = {
        "candidate_id": appended.candidate_id,
        "semantic_hash": appended.semantic_hash,
        "content_hash": appended.content_hash,
        "name": appended.name,
        "definition": {
            "canonical_expression": appended.canonical_expression,
            "direction": appended.direction,
            "description": appended.description,
            "steps": list(appended.steps),
        },
    }
    append_staging = tmp_path / "publication-append"
    append_staged = prepare_publication(
        batch_id="pub_batch_append",
        run_id="run_test_publication_append",
        candidates=(append_record,),
        approval_ids=("approval_test_3",),
        context=context,
        staging_root=append_staging,
        db_file=DB_FILE,
        source_catalog=catalog_file,
        factor_dir=factor_dir,
    )
    commit_publication(
        staged=append_staged,
        batch_id="pub_batch_append",
        factor_dir=factor_dir,
        catalog_file=catalog_file,
    )
    published = factors.published.load_manifest(factor_dir)
    names = {item["name"] for item in published["factors"]}
    assert {candidate.name for candidate in candidates} | {appended.name} <= names
