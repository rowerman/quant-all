"""显式、事务化的挖掘 V1 到 V2 schema 迁移。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from common.paths import MINING_DB_FILE
from mining.contract_types import sha256_json, utc_now_iso
from mining.errors import MiningMigrationError
from mining.logging_conf import format_event, get_mining_logger
from mining.states import RUN_ACTIVE_STATUSES
from mining.storage_lock import database_state_lock
from mining.storage_v2_schema import (
    initialize_v2_schema,
    schema_exists,
    schema_version,
)

_LOGGER = get_mining_logger("mining.migration")


def _read_source_snapshot(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """读取 V1 元数据计数和稳定哈希。"""

    version = schema_version(conn, "mining")
    if version != "mining_v1":
        raise MiningMigrationError(
            f"仅支持从 mining_v1 迁移，当前源版本为 {version!r}"
        )
    rows = {
        "context": conn.execute(
            "SELECT context_id, context_hash, payload_json, created_at "
            "FROM mining.run_context ORDER BY context_id"
        ).fetchall(),
        "run": conn.execute(
            "SELECT run_id, context_id, idempotency_key, request_hash, status, "
            "hypothesis, request_json, worker_mode, created_at, updated_at "
            "FROM mining.run ORDER BY run_id"
        ).fetchall(),
        "candidate": conn.execute(
            "SELECT run_id, candidate_id, semantic_hash, content_hash, name, status, "
            "definition_json, created_at, updated_at "
            "FROM mining.candidate ORDER BY run_id, candidate_id"
        ).fetchall(),
    }
    counts = {name: len(values) for name, values in rows.items()}
    snapshot_hash = sha256_json(
        {
            "schema_version": version,
            "context": rows["context"],
            "run": rows["run"],
            "candidate": rows["candidate"],
        }
    )
    return {
        "snapshot_hash": snapshot_hash,
        "counts": counts,
        "rows": rows,
    }


def _copy_legacy_rows(
    conn: duckdb.DuckDBPyConnection,
    snapshot: dict[str, Any],
    migrated_at: str,
) -> None:
    """把已冻结的 V1 行复制为 V2 只读快照。"""

    rows = snapshot["rows"]
    conn.executemany(
        "INSERT INTO mining_v2.legacy_context VALUES (?, ?, ?)",
        [
            (context_id, context_hash, payload_json)
            for context_id, context_hash, payload_json, _created_at in rows["context"]
        ],
    )
    conn.executemany(
        "INSERT INTO mining_v2.legacy_run VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (*row, False)
            for row in rows["run"]
        ],
    )
    conn.executemany(
        "INSERT INTO mining_v2.legacy_candidate VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows["candidate"],
    )
    conn.execute(
        "INSERT INTO mining_v2.legacy_source VALUES (?, ?, ?, ?, ?, ?)",
        [
            "mining_v1",
            snapshot["snapshot_hash"],
            snapshot["counts"]["context"],
            snapshot["counts"]["run"],
            snapshot["counts"]["candidate"],
            migrated_at,
        ],
    )


def migrate_v1_to_v2(
    db_path: Path | str = MINING_DB_FILE,
) -> dict[str, Any]:
    """显式把 V1 元数据快照迁移到同库的 mining_v2 schema。"""

    target = Path(db_path)
    if not target.is_file():
        raise MiningMigrationError(f"候选库不存在：{target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with database_state_lock(target):
            with duckdb.connect(str(target)) as conn:
                if not schema_exists(conn, "mining"):
                    raise MiningMigrationError("候选库缺少 mining_v1 schema")
                if schema_exists(conn, "mining_v2"):
                    version = schema_version(conn, "mining_v2")
                    if version != "mining_v2":
                        raise MiningMigrationError(
                            f"未知 mining_v2 schema 版本：{version!r}"
                        )
                    source = conn.execute(
                        "SELECT * FROM mining_v2.legacy_source "
                        "WHERE source_schema='mining_v1'"
                    ).fetchone()
                    if source is None:
                        raise MiningMigrationError(
                            "mining_v2 已存在但没有 V1 迁移记录"
                        )
                    current = _read_source_snapshot(conn)
                    if source[1] != current["snapshot_hash"]:
                        raise MiningMigrationError(
                            "V1 库在迁移后发生变化，拒绝自动合并新数据"
                        )
                    return {
                        "migrated": False,
                        "reused": True,
                        "source_snapshot_hash": source[1],
                        "counts": current["counts"],
                        "migrated_at": source[5],
                    }

                placeholders = ",".join("?" for _ in RUN_ACTIVE_STATUSES)
                active = conn.execute(
                    f"SELECT run_id FROM mining.run WHERE status IN ({placeholders}) "
                    "ORDER BY created_at LIMIT 1",
                    list(RUN_ACTIVE_STATUSES),
                ).fetchone()
                if active is not None:
                    raise MiningMigrationError(
                        f"存在活动 V1 运行，停止服务并等待运行结束后再迁移：{active[0]}"
                    )

                migrated_at = utc_now_iso()
                conn.begin()
                try:
                    snapshot = _read_source_snapshot(conn)
                    initialize_v2_schema(conn)
                    _copy_legacy_rows(conn, snapshot, migrated_at)
                except BaseException:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
                result = {
                    "migrated": True,
                    "reused": False,
                    "source_snapshot_hash": snapshot["snapshot_hash"],
                    "counts": snapshot["counts"],
                    "migrated_at": migrated_at,
                }
    except MiningMigrationError:
        raise
    except (OSError, duckdb.Error) as error:
        _LOGGER.error(
            format_event(
                {
                    "operation": "migrate_v1_to_v2",
                    "error_code": "MINING-126",
                }
            )
        )
        raise MiningMigrationError(f"V1 到 V2 迁移失败：{error}") from error
    _LOGGER.info(
        format_event(
            {
                "operation": "migrate_v1_to_v2",
                "source_snapshot_hash": result["source_snapshot_hash"],
                "run_count": result["counts"]["run"],
                "candidate_count": result["counts"]["candidate"],
            }
        )
    )
    return result
