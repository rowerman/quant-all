"""挖掘 V2 schema、运行存储和 V1 显式迁移测试。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mining.contracts import CandidateDefinition, FrozenContext, MiningRunRequest
from mining.errors import MiningDuplicateError, MiningMigrationError
from mining.migrate_v1_to_v2 import main as migration_main
from mining.migration_v1_v2 import migrate_v1_to_v2
from mining.search_context import FrozenSearchContext
from mining.search_contracts import FREE_EXPLORATION_INPUT_MODE
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


def _v1_context() -> FrozenContext:
    return FrozenContext(data_fingerprint="a" * 64)


def _v1_store(tmp_path: Path) -> MiningStore:
    return MiningStore(tmp_path / "factor_mining.duckdb", tmp_path / "mining")


def _populate_v1(store: MiningStore) -> tuple[dict, CandidateDefinition]:
    run = store.create_run(
        MiningRunRequest("legacy-run-0001", "短期价格反转假设"),
        _v1_context(),
    )
    candidate = CandidateDefinition(
        name="LEGACY_REVERSAL_0001",
        mechanism="short reversal",
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


def _v2_request(key: str = "search-run-0001") -> SearchRunRequest:
    return SearchRunRequest(key, "低估值股票可能带来未来超额收益")


def _v2_context() -> FrozenSearchContext:
    return FrozenSearchContext(data_fingerprint="b" * 64)


def test_v2_store_initializes_fresh_database_and_run_is_idempotent(tmp_path: Path):
    store = MiningV2Store(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    first = store.create_run(_v2_request(), _v2_context())
    second = store.create_run(_v2_request(), _v2_context())
    assert first["run_id"].startswith("searchrun_")
    assert second["reused"] is True
    assert second["status"] == "queued"
    assert store.get_run(first["run_id"])["context"]["budget"]["max_search_turns"] == 6
    assert len(store.list_events(first["run_id"])) == 1
    assert store.list_runs()[0]["run_id"] == first["run_id"]

    changed = SearchRunRequest(
        "search-run-0001",
        "现金流改善可能带来未来超额收益",
    )
    with pytest.raises(MiningDuplicateError, match="幂等键"):
        store.create_run(changed, _v2_context())


def test_v2_store_persists_free_exploration_request_without_schema_change(
    tmp_path: Path,
):
    store = MiningV2Store(tmp_path / "factor_mining.duckdb", tmp_path / "data")
    request = SearchRunRequest(
        "search-free-0001",
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
    )
    first = store.create_run(request, _v2_context())
    second = store.create_run(request, _v2_context())
    loaded = store.get_run(first["run_id"])

    assert second["reused"] is True
    assert loaded["hypothesis"] == "自由探索"
    assert loaded["request"]["input_mode"] == FREE_EXPLORATION_INPUT_MODE
    assert loaded["request"]["hypothesis"] is None
    assert store.list_runs()[0]["hypothesis"] == "自由探索"

    changed = SearchRunRequest(
        "search-free-0001",
        "相同幂等键切换到手工假设",
    )
    with pytest.raises(MiningDuplicateError, match="幂等键"):
        store.create_run(changed, _v2_context())


def test_v2_store_refuses_v1_database_without_creating_v2_schema(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v1 = MiningStore(path, tmp_path / "mining")
    _populate_v1(v1)
    with pytest.raises(MiningMigrationError, match="显式"):
        MiningV2Store(path, tmp_path / "data")
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM duckdb_schemas() WHERE schema_name='mining_v2'"
        ).fetchone()[0] == 0


def test_v1_migration_copies_read_only_snapshots_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    v1 = _v1_store(tmp_path)
    run, candidate = _populate_v1(v1)

    first = migrate_v1_to_v2(path)
    assert first["migrated"] is True
    assert first["reused"] is False
    assert first["counts"] == {"context": 1, "run": 1, "candidate": 1}

    second = migrate_v1_to_v2(path)
    assert second["migrated"] is False
    assert second["reused"] is True
    assert second["source_snapshot_hash"] == first["source_snapshot_hash"]

    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM mining.run").fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM mining_v2.legacy_source"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM mining_v2.legacy_run"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM mining_v2.legacy_candidate"
        ).fetchone()[0] == 1

    v2 = MiningV2Store(path, tmp_path / "data")
    legacy = v2.get_legacy_run(run["run_id"])
    assert legacy["read_only"] is True
    assert legacy["resumable"] is False
    assert legacy["request"]["hypothesis"] == "短期价格反转假设"
    assert legacy["candidates"][0]["candidate_id"] == candidate.candidate_id
    assert v2.get_legacy_candidate(
        run["run_id"],
        candidate.candidate_id,
    )["name"] == candidate.name
    assert v2.migration_summary()["source_schema"] == "mining_v1"


def test_migration_rolls_back_schema_when_copy_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / "factor_mining.duckdb"
    _populate_v1(_v1_store(tmp_path))

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("injected migration interruption")

    monkeypatch.setattr("mining.migration_v1_v2._copy_legacy_rows", fail_copy)
    with pytest.raises(RuntimeError, match="injected"):
        migrate_v1_to_v2(path)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM duckdb_schemas() WHERE schema_name='mining_v2'"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM mining.run").fetchone()[0] == 1


def test_migration_rejects_active_v1_run(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = _v1_store(tmp_path)
    run, _ = _populate_v1(store)
    store.transition_run(
        run["run_id"],
        "generating",
        event_key="legacy_generating",
        stage="generating",
    )
    with pytest.raises(MiningMigrationError, match="活动 V1 运行"):
        migrate_v1_to_v2(path)


def test_migration_detects_source_changes_after_completion(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = _v1_store(tmp_path)
    _populate_v1(store)
    migrate_v1_to_v2(path)
    store.create_run(
        MiningRunRequest("legacy-run-0002", "第二条旧版研究假设"),
        _v1_context(),
    )
    with pytest.raises(MiningMigrationError, match="发生变化"):
        migrate_v1_to_v2(path)


def test_unknown_v2_schema_is_rejected(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = MiningV2Store(path, tmp_path / "data")
    store.close() if hasattr(store, "close") else None
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "UPDATE mining_v2.schema_meta SET value='mining_v3' "
            "WHERE key='schema_version'"
        )
    with pytest.raises(MiningMigrationError, match="schema"):
        MiningV2Store(path, tmp_path / "data")


def test_migration_cli_prints_safe_summary(tmp_path: Path, capsys):
    path = tmp_path / "factor_mining.duckdb"
    _populate_v1(_v1_store(tmp_path))
    assert migration_main(["--db", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"migrated": true' in output
    assert "短期价格反转假设" not in output
