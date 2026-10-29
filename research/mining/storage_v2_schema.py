"""挖掘 V2 DuckDB schema 定义和版本检查。"""

from __future__ import annotations

import duckdb

from mining.errors import MiningMigrationError, MiningStorageError
from mining.search_contract_types import SEARCH_SCHEMA_VERSION

V2_DDL = (
    "CREATE SCHEMA IF NOT EXISTS mining_v2",
    """
    CREATE TABLE IF NOT EXISTS mining_v2.schema_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.context (
        context_id VARCHAR PRIMARY KEY,
        context_hash VARCHAR NOT NULL UNIQUE,
        data_version VARCHAR NOT NULL,
        data_fingerprint VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        universe VARCHAR NOT NULL,
        price_channel VARCHAR NOT NULL,
        development_start DATE NOT NULL,
        development_end DATE NOT NULL,
        selection_start DATE NOT NULL,
        selection_end DATE NOT NULL,
        final_start DATE NOT NULL,
        final_end DATE NOT NULL,
        holdout_start DATE NOT NULL,
        holdout_end DATE NOT NULL,
        field_catalog_hash VARCHAR NOT NULL,
        operator_catalog_hash VARCHAR NOT NULL,
        existing_factors_hash VARCHAR NOT NULL,
        budget_hash VARCHAR NOT NULL,
        windows_hash VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.run (
        run_id VARCHAR PRIMARY KEY,
        context_id VARCHAR NOT NULL REFERENCES mining_v2.context(context_id),
        idempotency_key VARCHAR NOT NULL UNIQUE,
        request_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        hypothesis TEXT NOT NULL,
        request_json TEXT NOT NULL,
        trace_hash VARCHAR,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.hypothesis (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        hypothesis_id VARCHAR NOT NULL,
        parent_hypothesis_id VARCHAR,
        turn_index INTEGER NOT NULL,
        branch_id VARCHAR NOT NULL,
        card_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, hypothesis_id),
        UNIQUE (run_id, card_hash),
        UNIQUE (run_id, turn_index, branch_id, card_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.experiment_batch (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        batch_id VARCHAR NOT NULL,
        hypothesis_id VARCHAR NOT NULL,
        turn_index INTEGER NOT NULL,
        branch_id VARCHAR NOT NULL,
        batch_hash VARCHAR NOT NULL,
        candidate_ids_json TEXT NOT NULL,
        parent_candidate_ids_json TEXT NOT NULL,
        input_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, batch_id),
        UNIQUE (run_id, batch_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.candidate_trace (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        candidate_id VARCHAR NOT NULL,
        candidate_hash VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        name VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id),
        UNIQUE (run_id, candidate_hash),
        UNIQUE (run_id, semantic_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.candidate_lineage (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        parent_candidate_id VARCHAR,
        hypothesis_id VARCHAR NOT NULL,
        batch_id VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id),
        UNIQUE (run_id, parent_candidate_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.turn (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        turn_index INTEGER NOT NULL,
        hypothesis_id VARCHAR,
        experiment_batch_id VARCHAR,
        status VARCHAR NOT NULL,
        request_hash VARCHAR NOT NULL,
        evidence_id VARCHAR,
        decision_id VARCHAR,
        snapshot_hash VARCHAR,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, turn_index),
        UNIQUE (run_id, request_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.evidence_packet (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        turn_index INTEGER NOT NULL,
        evidence_id VARCHAR NOT NULL,
        hypothesis_id VARCHAR NOT NULL,
        batch_id VARCHAR NOT NULL,
        evidence_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, evidence_id),
        UNIQUE (run_id, evidence_hash),
        UNIQUE (run_id, turn_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.search_decision (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        turn_index INTEGER NOT NULL,
        decision_id VARCHAR NOT NULL,
        hypothesis_id VARCHAR NOT NULL,
        evidence_id VARCHAR NOT NULL,
        decision_hash VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, decision_id),
        UNIQUE (run_id, decision_hash),
        UNIQUE (run_id, turn_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.trial_record (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        trial_id VARCHAR NOT NULL,
        trial_key VARCHAR NOT NULL,
        kind VARCHAR NOT NULL,
        turn_index INTEGER NOT NULL,
        hypothesis_id VARCHAR,
        batch_id VARCHAR,
        candidate_id VARCHAR,
        trial_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        occurred_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, trial_id),
        UNIQUE (run_id, trial_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.trial_invalidation (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        invalidation_id VARCHAR NOT NULL,
        trial_id VARCHAR NOT NULL,
        reason_code VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, invalidation_id),
        UNIQUE (run_id, trial_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.trial_counter (
        run_id VARCHAR PRIMARY KEY REFERENCES mining_v2.run(run_id),
        proposed BIGINT NOT NULL,
        rejected BIGINT NOT NULL,
        duplicate BIGINT NOT NULL,
        invalid BIGINT NOT NULL,
        gate_failed BIGINT NOT NULL,
        evaluated BIGINT NOT NULL,
        full_evaluated BIGINT NOT NULL,
        holdout_claimed BIGINT NOT NULL,
        invalidated BIGINT NOT NULL,
        updated_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.event (
        run_id VARCHAR NOT NULL REFERENCES mining_v2.run(run_id),
        sequence BIGINT NOT NULL,
        event_key VARCHAR NOT NULL,
        event_type VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, sequence),
        UNIQUE (run_id, event_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.legacy_source (
        source_schema VARCHAR PRIMARY KEY,
        source_snapshot_hash VARCHAR NOT NULL,
        context_count BIGINT NOT NULL,
        run_count BIGINT NOT NULL,
        candidate_count BIGINT NOT NULL,
        migrated_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.legacy_context (
        context_id VARCHAR PRIMARY KEY,
        context_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.legacy_run (
        run_id VARCHAR PRIMARY KEY,
        context_id VARCHAR NOT NULL,
        idempotency_key VARCHAR NOT NULL,
        request_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        hypothesis TEXT NOT NULL,
        request_json TEXT NOT NULL,
        worker_mode VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        resumable BOOLEAN NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v2.legacy_candidate (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        content_hash VARCHAR NOT NULL,
        name VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        definition_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_v2_run_status
    ON mining_v2.run(status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_v2_hypothesis_turn
    ON mining_v2.hypothesis(run_id, turn_index)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_v2_trial_kind
    ON mining_v2.trial_record(run_id, kind)
    """,
)


def schema_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    """判断 DuckDB schema 是否存在。"""

    row = conn.execute(
        "SELECT count(*) FROM duckdb_schemas() WHERE schema_name=?",
        [name],
    ).fetchone()
    return bool(row and row[0])


def schema_version(conn: duckdb.DuckDBPyConnection, schema: str) -> str | None:
    """读取指定 schema 的版本，缺少元表时返回空值。"""

    if not schema_exists(conn, schema):
        return None
    try:
        row = conn.execute(
            f"SELECT value FROM {schema}.schema_meta WHERE key='schema_version'"
        ).fetchone()
    except duckdb.Error as error:
        raise MiningStorageError(f"{schema} schema 元数据不可用：{error}") from error
    return None if row is None else str(row[0])


def initialize_v2_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """首次初始化 V2 schema。"""

    for statement in V2_DDL:
        conn.execute(statement)
    row = conn.execute(
        "SELECT value FROM mining_v2.schema_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO mining_v2.schema_meta VALUES ('schema_version', ?)",
            [SEARCH_SCHEMA_VERSION],
        )
    elif row[0] != SEARCH_SCHEMA_VERSION:
        raise MiningMigrationError(
            f"V2 候选库 schema 为 {row[0]}，当前代码只支持 {SEARCH_SCHEMA_VERSION}"
        )


def ensure_v2_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """确认已有 V2 schema 可写且版本匹配。"""

    version = schema_version(conn, "mining_v2")
    if version is None:
        raise MiningMigrationError("候选库尚未初始化 mining_v2")
    if version != SEARCH_SCHEMA_VERSION:
        raise MiningMigrationError(
            f"V2 候选库 schema 为 {version}，当前代码只支持 {SEARCH_SCHEMA_VERSION}"
        )
    for statement in V2_DDL:
        conn.execute(statement)
