"""候选 DuckDB 的 schema 定义与版本检查。"""

from __future__ import annotations

import duckdb

from mining.contracts import SCHEMA_VERSION
from mining.errors import MiningStorageError

DDL = (
    "CREATE SCHEMA IF NOT EXISTS mining",
    """
    CREATE TABLE IF NOT EXISTS mining.schema_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.run_context (
        context_id VARCHAR PRIMARY KEY,
        context_hash VARCHAR NOT NULL,
        data_version VARCHAR NOT NULL,
        data_fingerprint VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        universe VARCHAR NOT NULL,
        price_channel VARCHAR NOT NULL,
        research_start DATE NOT NULL,
        research_end DATE NOT NULL,
        full_eval_start DATE NOT NULL,
        full_eval_end DATE NOT NULL,
        holdout_start DATE NOT NULL,
        holdout_end DATE NOT NULL,
        llm_profile VARCHAR NOT NULL,
        llm_model VARCHAR NOT NULL,
        field_catalog_hash VARCHAR NOT NULL,
        operator_catalog_hash VARCHAR NOT NULL,
        existing_factors_hash VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.run (
        run_id VARCHAR PRIMARY KEY,
        context_id VARCHAR NOT NULL,
        idempotency_key VARCHAR NOT NULL UNIQUE,
        request_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        hypothesis VARCHAR NOT NULL,
        request_json VARCHAR NOT NULL,
        worker_mode VARCHAR NOT NULL,
        interruption_count INTEGER NOT NULL DEFAULT 0,
        interrupted_from VARCHAR,
        last_completed_stage VARCHAR,
        error_code VARCHAR,
        error_message VARCHAR,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        started_at VARCHAR,
        finished_at VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.candidate (
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        content_hash VARCHAR NOT NULL,
        name VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        discovery_round INTEGER NOT NULL,
        discovery_index INTEGER NOT NULL,
        definition_json VARCHAR NOT NULL,
        interrupted_from VARCHAR,
        failure_stage VARCHAR,
        error_code VARCHAR,
        error_message VARCHAR,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL,
        finished_at VARCHAR,
        PRIMARY KEY (run_id, candidate_id),
        UNIQUE (run_id, semantic_hash),
        UNIQUE (run_id, content_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.evaluation (
        report_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        stage VARCHAR NOT NULL,
        candidate_content_hash VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        input_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        metrics_json VARCHAR NOT NULL,
        warnings_json VARCHAR NOT NULL,
        artifact_path VARCHAR,
        artifact_sha256 VARCHAR,
        error_code VARCHAR,
        error_message VARCHAR,
        attempt INTEGER NOT NULL,
        started_at VARCHAR NOT NULL,
        finished_at VARCHAR,
        UNIQUE (
            run_id, candidate_id, stage, candidate_content_hash,
            context_hash, input_hash, attempt
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.technical_gate (
        gate_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        candidate_content_hash VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        rule_version VARCHAR NOT NULL,
        passed BOOLEAN NOT NULL,
        checks_json VARCHAR NOT NULL,
        check_hash VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.approval (
        approval_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        decision VARCHAR NOT NULL,
        reviewer VARCHAR NOT NULL,
        note VARCHAR NOT NULL,
        candidate_content_hash VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        evidence_hash VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        UNIQUE (run_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.publication (
        publication_id VARCHAR PRIMARY KEY,
        batch_id VARCHAR,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        approval_id VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        staging_path VARCHAR NOT NULL,
        manifest_hash VARCHAR,
        formal_factor_name VARCHAR,
        artifact_sha256 VARCHAR,
        error_code VARCHAR,
        error_message VARCHAR,
        attempt INTEGER NOT NULL,
        started_at VARCHAR NOT NULL,
        finished_at VARCHAR,
        source_kind VARCHAR NOT NULL DEFAULT 'v1',
        candidate_envelope_hash VARCHAR,
        source_reference_hash VARCHAR,
        UNIQUE (run_id, candidate_id, approval_id, attempt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.publication_batch (
        batch_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        context_hash VARCHAR NOT NULL,
        candidate_ids_json VARCHAR NOT NULL,
        approval_ids_json VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        staging_path VARCHAR NOT NULL,
        manifest_hash VARCHAR,
        artifact_sha256 VARCHAR,
        error_code VARCHAR,
        error_message VARCHAR,
        attempt INTEGER NOT NULL,
        started_at VARCHAR NOT NULL,
        finished_at VARCHAR,
        source_kind VARCHAR NOT NULL DEFAULT 'v1',
        UNIQUE (run_id, context_hash, attempt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.event (
        run_id VARCHAR NOT NULL,
        sequence BIGINT NOT NULL,
        event_key VARCHAR NOT NULL,
        stage VARCHAR NOT NULL,
        event_type VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        PRIMARY KEY (run_id, sequence),
        UNIQUE (run_id, event_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mining.holdout_usage (
        segment_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        batch_hash VARCHAR NOT NULL,
        candidate_ids_json VARCHAR NOT NULL,
        evidence_hash VARCHAR NOT NULL,
        consumed_at VARCHAR NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_run_status
    ON mining.run(status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_candidate_run_status
    ON mining.candidate(run_id, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_evaluation_candidate_stage
    ON mining.evaluation(run_id, candidate_id, stage)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mining_publication_batch_status
    ON mining.publication_batch(status)
    """,
)

SCHEMA_MIGRATIONS = (
    "ALTER TABLE mining.publication ADD COLUMN IF NOT EXISTS batch_id VARCHAR",
    "ALTER TABLE mining.publication ADD COLUMN IF NOT EXISTS "
    "source_kind VARCHAR DEFAULT 'v1'",
    "ALTER TABLE mining.publication_batch ADD COLUMN IF NOT EXISTS "
    "source_kind VARCHAR DEFAULT 'v1'",
    "ALTER TABLE mining.publication ADD COLUMN IF NOT EXISTS "
    "candidate_envelope_hash VARCHAR",
    "ALTER TABLE mining.publication ADD COLUMN IF NOT EXISTS "
    "source_reference_hash VARCHAR",
    "UPDATE mining.publication SET source_kind='v1' WHERE source_kind IS NULL",
    "UPDATE mining.publication_batch SET source_kind='v1' WHERE source_kind IS NULL",
)


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """创建当前 schema，并拒绝未知 schema 版本。"""

    for statement in DDL:
        conn.execute(statement)
    for statement in SCHEMA_MIGRATIONS:
        conn.execute(statement)
    row = conn.execute(
        "SELECT value FROM mining.schema_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO mining.schema_meta VALUES ('schema_version', ?)",
            [SCHEMA_VERSION],
        )
    elif row[0] != SCHEMA_VERSION:
        raise MiningStorageError(
            f"候选库 schema 为 {row[0]}，当前代码只支持 {SCHEMA_VERSION}"
        )
