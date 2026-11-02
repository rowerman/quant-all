"""挖掘 V3 DuckDB schema 定义和版本检查。"""

from __future__ import annotations

import duckdb

from mining.errors import MiningMigrationError, MiningStorageError
from mining.storage_v2_schema import schema_exists
from mining.v3_contracts import MINING_V3_SCHEMA_VERSION

V3_DDL = (
    "CREATE SCHEMA IF NOT EXISTS mining_v3",
    """
    CREATE TABLE IF NOT EXISTS mining_v3.schema_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.research_input (
        run_id VARCHAR PRIMARY KEY,
        idempotency_key VARCHAR NOT NULL UNIQUE,
        input_mode VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        request_hash VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        input_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.source_reference (
        run_id VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        source_kind VARCHAR NOT NULL,
        source_hash VARCHAR NOT NULL,
        artifact_manifest_hash VARCHAR,
        promotion_hash VARCHAR,
        reference_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.candidate_source (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        content_hash VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        candidate_source_hash VARCHAR NOT NULL,
        candidate_payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id, source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.evaluation_evidence (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        evaluation_id VARCHAR NOT NULL,
        stage VARCHAR NOT NULL,
        input_hash VARCHAR NOT NULL,
        evidence_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id, source_id, evaluation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.run_context (
        run_id VARCHAR PRIMARY KEY,
        context_hash VARCHAR NOT NULL,
        payload_json TEXT NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.candidate_approval (
        approval_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        decision VARCHAR NOT NULL,
        reviewer VARCHAR NOT NULL,
        note VARCHAR NOT NULL,
        candidate_content_hash VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        source_reference_hash VARCHAR NOT NULL,
        candidate_envelope_hash VARCHAR NOT NULL,
        evidence_hash VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        UNIQUE (run_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.candidate_state (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        error_code VARCHAR,
        error_message VARCHAR,
        updated_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining_v3.event (
        run_id VARCHAR NOT NULL,
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
    CREATE INDEX IF NOT EXISTS idx_mining_v3_candidate_source_candidate
    ON mining_v3.candidate_source(run_id, candidate_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_v3_evaluation_candidate
    ON mining_v3.evaluation_evidence(run_id, candidate_id, stage)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_v3_candidate_state_status
    ON mining_v3.candidate_state(run_id, status)
    """,
)


def v3_schema_version(conn: duckdb.DuckDBPyConnection) -> str | None:
    """读取 V3 schema 版本。"""

    if not schema_exists(conn, MINING_V3_SCHEMA_VERSION):
        return None
    try:
        row = conn.execute(
            "SELECT value FROM mining_v3.schema_meta "
            "WHERE key='schema_version'"
        ).fetchone()
    except duckdb.Error as error:
        raise MiningStorageError(
            f"mining_v3 schema 元数据不可用：{error}"
        ) from error
    return None if row is None else str(row[0])


def initialize_v3_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """首次初始化 V3 schema。"""

    for statement in V3_DDL:
        conn.execute(statement)
    row = conn.execute(
        "SELECT value FROM mining_v3.schema_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO mining_v3.schema_meta VALUES ('schema_version', ?)",
            [MINING_V3_SCHEMA_VERSION],
        )
    elif row[0] != MINING_V3_SCHEMA_VERSION:
        raise MiningMigrationError(
            f"V3 schema 为 {row[0]}，当前代码只支持 "
            f"{MINING_V3_SCHEMA_VERSION}"
        )


def ensure_v3_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """确认已有 V3 schema 可写且版本匹配。"""

    version = v3_schema_version(conn)
    if version is None:
        raise MiningMigrationError("数据库尚未初始化 mining_v3")
    if version != MINING_V3_SCHEMA_VERSION:
        raise MiningMigrationError(
            f"V3 schema 为 {version}，当前代码只支持 "
            f"{MINING_V3_SCHEMA_VERSION}"
        )
    for statement in V3_DDL:
        conn.execute(statement)


__all__ = [
    "V3_DDL",
    "ensure_v3_schema",
    "initialize_v3_schema",
    "v3_schema_version",
]
