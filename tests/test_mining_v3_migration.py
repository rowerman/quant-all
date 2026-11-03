"""V1/V2 到 V3 的显式迁移、回滚和 legacy 读取测试。"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from mining.contract_candidates import CandidateDefinition
from mining.contracts import FrozenContext, MiningRunRequest
from mining.errors import (
    MiningMigrationError,
    MiningStorageError,
    V3MigrationConflictError,
    V3MigrationRollbackError,
)
from mining.migrate_to_v3 import main as migration_cli_main
from mining.migration_v1_v2 import migrate_v1_to_v2
from mining.migration_v1_v2_v3 import (
    migrate_v1_v2_to_v3,
    rollback_v3_migration,
    v3_migration_status,
)
from mining.search_context import FrozenSearchContext
from mining.search_contracts import FREE_EXPLORATION_INPUT_MODE
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store
from mining.storage_v3 import MiningV3Store
from tests.mining_v3_fixtures import v3_context, v3_input


def _v1_store(tmp_path: Path) -> MiningStore:
    return MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "mining")


def _populate_v1(
    store: MiningStore,
    *,
    key: str = "legacy-run-0001",
) -> tuple[dict, CandidateDefinition]:
    run = store.create_run(
        MiningRunRequest(key, "短期价格反转假设"),
        FrozenContext(data_fingerprint="a" * 64),
    )
    candidate = CandidateDefinition(
        name="LEGACY_V3_REVERSAL",
        mechanism="短期反转",
        description="五日价格变化取负",
        canonical_expression="neg(sub(close,lag(close,5)))",
        direction=1,
        lookback=5,
        dependencies=("close",),
        output_type="number",
        steps=("计算五日变化", "取负"),
        discovery_round=1,
        discovery_index=1,
    )
    store.add_candidate(run["run_id"], candidate)
    return run, candidate


def _insert_v2_candidate(path: Path, run_id: str) -> None:
    candidate = CandidateDefinition(
        name="LEGACY_V2_VALUE",
        mechanism="低估值",
        description="低估值候选",
        canonical_expression="neg(mean(close,5))",
        direction=1,
        lookback=5,
        dependencies=("close",),
        output_type="number",
        steps=("计算五日均值", "取负"),
        discovery_round=1,
        discovery_index=1,
    )
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO mining_v2.candidate_trace VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                candidate.candidate_id,
                candidate.content_hash,
                candidate.semantic_hash,
                candidate.name,
                "review_ready",
                json.dumps(
                    {
                        "name": candidate.name,
                        "mechanism": candidate.mechanism,
                        "description": candidate.description,
                        "canonical_expression": candidate.canonical_expression,
                        "direction": candidate.direction,
                        "lookback": candidate.lookback,
                        "dependencies": list(candidate.dependencies),
                        "output_type": candidate.output_type,
                        "steps": list(candidate.steps),
                        "discovery_round": 1,
                        "discovery_index": 1,
                        "parameters": {},
                    },
                    ensure_ascii=False,
                ),
                "2026-11-02T00:00:00+00:00",
                "2026-11-02T00:00:00+00:00",
            ],
        )


def test_v1_migration_is_explicit_idempotent_and_read_only(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    run, candidate = _populate_v1(_v1_store(tmp_path))

    with pytest.raises(MiningMigrationError, match="显式 V3 迁移"):
        MiningV3Store(path, tmp_path / "data")

    first = migrate_v1_v2_to_v3(path)
    second = migrate_v1_v2_to_v3(path)

    assert first["migrated"] is True
    assert first["reused"] is False
    assert first["sources"]["mining"]["counts"] == {
        "context": 1,
        "run": 1,
        "candidate": 1,
    }
    assert second["migrated"] is False
    assert second["reused"] is True
    assert (
        second["sources"]["mining"]["source_snapshot_hash"]
        == first["sources"]["mining"]["source_snapshot_hash"]
    )

    store = MiningV3Store(path, tmp_path / "data")
    summary = store.migration_summary()
    legacy = store.get_legacy_run(run["run_id"])
    legacy_candidate = store.get_legacy_candidate(
        run["run_id"],
        candidate.candidate_id,
    )

    assert summary["sources"]["mining"]["counts"]["run"] == 1
    assert legacy["read_only"] is True
    assert legacy["resumable"] is False
    assert legacy["request"]["hypothesis"] == "短期价格反转假设"
    assert legacy["context"]["payload"]["data_fingerprint"] == "a" * 64
    assert legacy["candidates"][0]["candidate_id"] == candidate.candidate_id
    assert legacy_candidate["definition"]["name"] == candidate.name
    assert store.list_legacy_runs(source_schema="mining")[0]["run_id"] == run["run_id"]

    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM mining.run").fetchone()[0] == 1
        assert conn.execute(
            "SELECT status FROM mining.run WHERE run_id=?",
            [run["run_id"]],
        ).fetchone()[0] == "queued"


def test_v2_only_migration_copies_run_and_candidate(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v2 = MiningV2Store(path, tmp_path / "data")
    run = v2.create_run(
        SearchRunRequest(
            "v2-only-run-0001",
            None,
            input_mode=FREE_EXPLORATION_INPUT_MODE,
        ),
        FrozenSearchContext(data_fingerprint="b" * 64),
    )
    _insert_v2_candidate(path, run["run_id"])
    with duckdb.connect(str(path)) as conn:
        conn.execute("DROP SCHEMA mining CASCADE")

    result = migrate_v1_v2_to_v3(path)
    store = MiningV3Store(path, tmp_path / "data")
    legacy = store.get_legacy_run(run["run_id"])

    assert set(result["sources"]) == {"mining_v2"}
    assert legacy["input_mode"] == FREE_EXPLORATION_INPUT_MODE
    assert legacy["hypothesis"] == "自由探索"
    assert legacy["candidates"][0]["name"] == "LEGACY_V2_VALUE"
    assert legacy["read_only"] is True


def test_both_sources_are_snapshotted_independently(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v1_run, _ = _populate_v1(_v1_store(tmp_path))
    migrate_v1_to_v2(path)
    v2_run = MiningV2Store(path, tmp_path / "data").create_run(
        SearchRunRequest("v2-both-run-0001", "现金流改善可能带来未来收益"),
        FrozenSearchContext(data_fingerprint="c" * 64),
    )

    result = migrate_v1_v2_to_v3(path)
    store = MiningV3Store(path, tmp_path / "data")

    assert set(result["sources"]) == {"mining", "mining_v2"}
    assert store.get_legacy_run(v1_run["run_id"])["source_schema"] == "mining"
    assert (
        store.get_legacy_run(v2_run["run_id"])["source_schema"]
        == "mining_v2"
    )


def test_v2_research_run_remains_research_report_in_legacy_view(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v2 = MiningV2Store(path, tmp_path / "data")
    run = v2.create_run(
        SearchRunRequest("v2-research-run-0001", "资料驱动研究：1 份研报资料"),
        FrozenSearchContext(data_fingerprint="e" * 64),
    )
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO mining_v2.research_run VALUES "
            "(?, ?, ?, ?, ?, NULL, ?, ?)",
            [
                run["run_id"],
                "a" * 64,
                "b" * 64,
                "review_ready",
                "{}",
                "2026-11-02T00:00:00+00:00",
                "2026-11-02T00:00:00+00:00",
            ],
        )

    migrate_v1_v2_to_v3(path)
    legacy = MiningV3Store(path, tmp_path / "data").get_legacy_run(
        run["run_id"]
    )

    assert legacy["input_mode"] == "research_report"
    assert legacy["read_only"] is True


def test_migration_detects_source_drift_after_completion(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v1 = _v1_store(tmp_path)
    _populate_v1(v1)
    migrate_v1_v2_to_v3(path)
    v1.create_run(
        MiningRunRequest("legacy-run-0002", "第二条旧版研究假设"),
        FrozenContext(data_fingerprint="a" * 64),
    )

    with pytest.raises(V3MigrationConflictError, match="发生变化"):
        migrate_v1_v2_to_v3(path)


@pytest.mark.parametrize("schema", ("mining", "mining_v2"))
def test_migration_rejects_active_source_run(tmp_path: Path, schema: str):
    path = tmp_path / "factor_mining.duckdb"
    if schema == "mining":
        store = _v1_store(tmp_path)
        run, _ = _populate_v1(store)
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "UPDATE mining.run SET status='generating' WHERE run_id=?",
                [run["run_id"]],
            )
    else:
        store = MiningV2Store(path, tmp_path / "data")
        run = store.create_run(
            SearchRunRequest("active-v2-run-0001", "活动 V2 测试假设"),
            FrozenSearchContext(data_fingerprint="d" * 64),
        )
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "UPDATE mining_v2.run_control SET status='searching' "
                "WHERE run_id=?",
                [run["run_id"]],
            )

    with pytest.raises(V3MigrationConflictError, match="活动 V"):
        migrate_v1_v2_to_v3(path)


def test_migration_interruption_rolls_back_schema_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "factor_mining.duckdb"
    _populate_v1(_v1_store(tmp_path))

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("injected V3 migration interruption")

    monkeypatch.setattr(
        "mining.migration_v1_v2_v3._copy_snapshot_conn",
        fail_copy,
    )
    with pytest.raises(RuntimeError, match="injected"):
        migrate_v1_v2_to_v3(path)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM duckdb_schemas() WHERE schema_name='mining_v3'"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM mining.run").fetchone()[0] == 1

    monkeypatch.undo()
    retried = migrate_v1_v2_to_v3(path)
    assert retried["migrated"] is True


def test_rollback_is_allowed_before_v3_business_writes(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    run, _ = _populate_v1(_v1_store(tmp_path))
    migrate_v1_v2_to_v3(path)

    status = v3_migration_status(path)
    first = rollback_v3_migration(path)
    second = rollback_v3_migration(path)
    retried = migrate_v1_v2_to_v3(path)

    assert status["rollback_eligible"] is True
    assert first == {
        "rolled_back": True,
        "reused": False,
        "sources": ["mining"],
    }
    assert second["rolled_back"] is False
    assert second["reused"] is True
    assert retried["migrated"] is True
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM mining.run WHERE run_id=?",
            [run["run_id"]],
        ).fetchone()[0] == 1


def test_rollback_is_forbidden_after_v3_business_write(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    _populate_v1(_v1_store(tmp_path))
    migrate_v1_v2_to_v3(path)
    store = MiningV3Store(path, tmp_path / "data")
    context = v3_context()
    store.register_input(v3_input("hypothesis", context))

    status = v3_migration_status(path)

    assert status["rollback_eligible"] is False
    assert "V3_BUSINESS_WRITES" in status["rollback_reason"]
    with pytest.raises(V3MigrationRollbackError, match="业务写入"):
        rollback_v3_migration(path)
    assert status["sources"]["mining"]["counts"]["run"] == 1


def test_legacy_snapshot_tamper_is_detected(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    run, _ = _populate_v1(_v1_store(tmp_path))
    migrate_v1_v2_to_v3(path)
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "UPDATE mining_v3.legacy_run SET hypothesis='tampered' "
            "WHERE source_schema='mining' AND run_id=?",
            [run["run_id"]],
        )

    store = MiningV3Store(path, tmp_path / "data")
    with pytest.raises(MiningStorageError, match="篡改"):
        store.get_legacy_run(run["run_id"])


def test_fresh_v3_is_not_migration_managed(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    MiningV3Store(path, tmp_path / "data")

    status = v3_migration_status(path)

    assert status["migrated"] is False
    assert status["rollback_eligible"] is False
    assert status["rollback_reason"] == "NO_MIGRATION_RECORD"
    with pytest.raises(V3MigrationRollbackError, match="不是由显式迁移"):
        rollback_v3_migration(path)


def test_cli_prints_safe_migration_summary(tmp_path: Path, capsys):
    path = tmp_path / "factor_mining.duckdb"
    _populate_v1(_v1_store(tmp_path))

    assert migration_cli_main(["--db", str(path), "migrate"]) == 0
    output = capsys.readouterr().out
    assert '"migrated": true' in output
    assert "短期价格反转假设" not in output
    assert str(tmp_path) not in output
