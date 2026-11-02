"""V1/V2 迁移来源的稳定快照、复制和漂移检测。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.errors import (
    MiningContractError,
    V3MigrationConflictError,
)
from mining.search_run import SearchRunRequest
from mining.search_states import SEARCH_RUN_ACTIVE_STATUSES
from mining.states import RUN_ACTIVE_STATUSES
from mining.storage_v2_schema import schema_exists, schema_version
from mining.v3_migration_contracts import (
    LEGACY_CANDIDATE_FIELDS,
    LEGACY_CONTEXT_FIELDS,
    LEGACY_RUN_FIELDS,
    LEGACY_SOURCE_VERSIONS,
    legacy_snapshot_hash,
)


def _fetch_rows(
    conn: duckdb.DuckDBPyConnection,
    fields: tuple[str, ...],
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """执行查询并按约定字段构造行字典。"""

    rows = conn.execute(sql, params or []).fetchall()
    return [
        dict(zip(fields, row, strict=True))
        for row in rows
    ]


def _read_v1_snapshot(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """读取并校验 V1 上下文、运行和候选摘要。"""

    if not schema_exists(conn, "mining"):
        raise V3MigrationConflictError("候选库缺少 mining_v1 schema")
    version = schema_version(conn, "mining")
    if version != LEGACY_SOURCE_VERSIONS["mining"]:
        raise V3MigrationConflictError(
            f"仅支持从 mining_v1 迁移，当前 V1 版本为 {version!r}"
        )
    placeholders = ",".join("?" for _ in RUN_ACTIVE_STATUSES)
    active = conn.execute(
        f"SELECT run_id FROM mining.run WHERE status IN ({placeholders}) "
        "ORDER BY created_at LIMIT 1",
        list(RUN_ACTIVE_STATUSES),
    ).fetchone()
    if active is not None:
        raise V3MigrationConflictError(
            f"存在活动 V1 运行，停止服务并等待运行结束后再迁移：{active[0]}"
        )
    contexts = _fetch_rows(
        conn,
        LEGACY_CONTEXT_FIELDS,
        "SELECT context_id, context_hash, payload_json, created_at "
        "FROM mining.run_context ORDER BY context_id",
    )
    runs = _fetch_rows(
        conn,
        LEGACY_RUN_FIELDS,
        "SELECT run_id, context_id, idempotency_key, request_hash, "
        "'hypothesis' AS input_mode, status, hypothesis, request_json, "
        "created_at, updated_at FROM mining.run ORDER BY run_id",
    )
    candidates = _fetch_rows(
        conn,
        LEGACY_CANDIDATE_FIELDS,
        "SELECT run_id, candidate_id, semantic_hash, content_hash, name, status, "
        "definition_json, created_at, updated_at "
        "FROM mining.candidate ORDER BY run_id, candidate_id",
    )
    return _snapshot("mining", contexts, runs, candidates)


def _read_v2_snapshot(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """读取并校验 V2 上下文、运行和候选摘要。"""

    if not schema_exists(conn, "mining_v2"):
        raise V3MigrationConflictError("候选库缺少 mining_v2 schema")
    version = schema_version(conn, "mining_v2")
    if version != LEGACY_SOURCE_VERSIONS["mining_v2"]:
        raise V3MigrationConflictError(
            f"仅支持从 mining_v2 迁移，当前 V2 版本为 {version!r}"
        )
    placeholders = ",".join("?" for _ in SEARCH_RUN_ACTIVE_STATUSES)
    active = conn.execute(
        f"SELECT r.run_id FROM mining_v2.run r "
        "LEFT JOIN mining_v2.run_control ctl ON ctl.run_id=r.run_id "
        f"WHERE coalesce(ctl.status, r.status) IN ({placeholders}) "
        "ORDER BY r.created_at LIMIT 1",
        list(SEARCH_RUN_ACTIVE_STATUSES),
    ).fetchone()
    if active is not None:
        raise V3MigrationConflictError(
            f"存在活动 V2 运行，停止服务并等待运行结束后再迁移：{active[0]}"
        )
    contexts = _fetch_rows(
        conn,
        LEGACY_CONTEXT_FIELDS,
        "SELECT context_id, context_hash, payload_json, created_at "
        "FROM mining_v2.context ORDER BY context_id",
    )
    raw_runs = _fetch_rows(
        conn,
        (
            "run_id",
            "context_id",
            "idempotency_key",
            "request_hash",
            "status",
            "hypothesis",
            "request_json",
            "created_at",
            "updated_at",
            "is_research",
        ),
        "SELECT r.run_id, r.context_id, r.idempotency_key, r.request_hash, "
        "coalesce(ctl.status, r.status) AS status, r.hypothesis, r.request_json, "
        "r.created_at, coalesce(ctl.updated_at, r.updated_at) AS updated_at, "
        "CASE WHEN rr.run_id IS NULL THEN 0 ELSE 1 END AS is_research "
        "FROM mining_v2.run r "
        "LEFT JOIN mining_v2.run_control ctl ON ctl.run_id=r.run_id "
        "LEFT JOIN mining_v2.research_run rr ON rr.run_id=r.run_id "
        "ORDER BY r.run_id",
    )
    runs = []
    for row in raw_runs:
        try:
            request = SearchRunRequest.from_dict(
                {
                    "idempotency_key": row["idempotency_key"],
                    **json.loads(row["request_json"]),
                }
            )
        except (
            MiningContractError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise V3MigrationConflictError(
                f"V2 运行请求无法解析：{row['run_id']}"
            ) from error
        runs.append(
            {
                "run_id": row["run_id"],
                "context_id": row["context_id"],
                "idempotency_key": row["idempotency_key"],
                "request_hash": row["request_hash"],
                "input_mode": (
                    "research_report"
                    if row["is_research"]
                    else request.input_mode
                ),
                "status": row["status"],
                "hypothesis": row["hypothesis"],
                "request_json": row["request_json"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    candidates = _fetch_rows(
        conn,
        LEGACY_CANDIDATE_FIELDS,
        "SELECT run_id, candidate_id, semantic_hash, candidate_hash, name, status, "
        "payload_json, created_at, updated_at "
        "FROM mining_v2.candidate_trace ORDER BY run_id, candidate_id",
    )
    return _snapshot("mining_v2", contexts, runs, candidates)


def _snapshot(
    source_schema: str,
    contexts: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """组合一个来源的稳定快照。"""

    return {
        "source_schema": source_schema,
        "source_schema_version": LEGACY_SOURCE_VERSIONS[source_schema],
        "snapshot_hash": legacy_snapshot_hash(
            source_schema,
            contexts,
            runs,
            candidates,
        ),
        "counts": {
            "context": len(contexts),
            "run": len(runs),
            "candidate": len(candidates),
        },
        "contexts": contexts,
        "runs": runs,
        "candidates": candidates,
    }


def read_snapshots(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, dict[str, Any]]:
    """读取数据库中存在的全部受支持旧 schema。"""

    snapshots = {}
    if schema_exists(conn, "mining"):
        snapshots["mining"] = _read_v1_snapshot(conn)
    if schema_exists(conn, "mining_v2"):
        snapshots["mining_v2"] = _read_v2_snapshot(conn)
    if not snapshots:
        raise V3MigrationConflictError("候选库缺少可迁移的 V1/V2 schema")
    return snapshots


def copy_snapshot_conn(
    conn: duckdb.DuckDBPyConnection,
    snapshot: dict[str, Any],
    migrated_at: str,
) -> None:
    """在已有事务中复制一个来源快照。"""

    source_schema = snapshot["source_schema"]
    context_values = [
        (
            source_schema,
            row["context_id"],
            row["context_hash"],
            row["payload_json"],
            row["created_at"],
        )
        for row in snapshot["contexts"]
    ]
    if context_values:
        conn.executemany(
            "INSERT INTO mining_v3.legacy_context "
            "(source_schema, context_id, context_hash, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            context_values,
        )
    run_values = [
        (
            source_schema,
            row["run_id"],
            row["context_id"],
            row["idempotency_key"],
            row["request_hash"],
            row["input_mode"],
            row["status"],
            row["hypothesis"],
            row["request_json"],
            row["created_at"],
            row["updated_at"],
        )
        for row in snapshot["runs"]
    ]
    if run_values:
        conn.executemany(
            "INSERT INTO mining_v3.legacy_run "
            "(source_schema, run_id, context_id, idempotency_key, request_hash, "
            "input_mode, status, hypothesis, request_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            run_values,
        )
    candidate_values = [
        (
            source_schema,
            row["run_id"],
            row["candidate_id"],
            row["semantic_hash"],
            row["content_hash"],
            row["name"],
            row["status"],
            row["definition_json"],
            row["created_at"],
            row["updated_at"],
        )
        for row in snapshot["candidates"]
    ]
    if candidate_values:
        conn.executemany(
            "INSERT INTO mining_v3.legacy_candidate "
            "(source_schema, run_id, candidate_id, semantic_hash, content_hash, "
            "name, status, definition_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            candidate_values,
        )
    conn.execute(
        "INSERT INTO mining_v3.migration_source VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            source_schema,
            snapshot["source_schema_version"],
            snapshot["snapshot_hash"],
            snapshot["counts"]["context"],
            snapshot["counts"]["run"],
            snapshot["counts"]["candidate"],
            migrated_at,
        ],
    )


def compare_migration(
    current: dict[str, dict[str, Any]],
    stored: dict[str, dict[str, Any]],
) -> None:
    """确认迁移记录的来源集合和快照仍与当前旧库一致。"""

    if set(current) != set(stored):
        raise V3MigrationConflictError(
            "迁移后旧 schema 集合发生变化，拒绝自动合并新来源"
        )
    for source, snapshot in current.items():
        if stored[source]["source_snapshot_hash"] != snapshot["snapshot_hash"]:
            raise V3MigrationConflictError(
                f"{source} 在迁移后发生变化，拒绝自动合并新数据"
            )


def source_drift(
    current: dict[str, dict[str, Any]],
    stored: dict[str, dict[str, Any]],
) -> list[str]:
    """列出迁移后来源集合或快照内容的漂移。"""

    drift = []
    if set(current) != set(stored):
        drift.append("SCHEMA_SET_CHANGED")
    for source in sorted(set(current) & set(stored)):
        if stored[source]["source_snapshot_hash"] != current[source]["snapshot_hash"]:
            drift.append(f"{source}:SNAPSHOT_CHANGED")
    return drift


__all__ = [
    "compare_migration",
    "copy_snapshot_conn",
    "read_snapshots",
    "source_drift",
]
