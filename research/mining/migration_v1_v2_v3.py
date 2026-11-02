"""显式、事务化的挖掘 V1/V2 到 V3 只读迁移。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from common.paths import MINING_DB_FILE
from mining.contract_types import utc_now_iso
from mining.errors import (
    MiningMigrationError,
    MiningStorageError,
    V3MigrationConflictError,
    V3MigrationRollbackError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.storage_lock import database_state_lock
from mining.storage_v2_schema import schema_exists
from mining.storage_v3_schema import (
    ensure_v3_schema,
    initialize_v3_schema,
    v3_schema_version,
)
from mining.v3_contracts import MINING_V3_SCHEMA_VERSION
from mining.v3_migration_sources import (
    compare_migration,
    copy_snapshot_conn as _copy_snapshot_conn,
    read_snapshots as _read_snapshots,
    source_drift as _source_drift,
)
from mining.v3_migration_state import (
    migration_rows_conn as _migration_rows_conn,
    table_exists_conn as _table_exists_conn,
    v3_business_counts_conn as _v3_business_counts_conn,
    v3_publication_count_conn as _v3_publication_count_conn,
)

_LOGGER = get_mining_logger("mining.migration")


def _result(
    *,
    migrated: bool,
    reused: bool,
    snapshots: dict[str, dict[str, Any]],
    migrated_at: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": MINING_V3_SCHEMA_VERSION,
        "migrated": migrated,
        "reused": reused,
        "migrated_at": migrated_at,
        "sources": {
            source: {
                "source_schema_version": snapshot["source_schema_version"],
                "source_snapshot_hash": snapshot["snapshot_hash"],
                "counts": snapshot["counts"],
            }
            for source, snapshot in sorted(snapshots.items())
        },
    }


def migrate_v1_v2_to_v3(
    db_path: Path | str = MINING_DB_FILE,
) -> dict[str, Any]:
    """显式迁移已有 V1/V2 摘要到 mining_v3。"""

    target = Path(db_path)
    if not target.is_file():
        raise MiningMigrationError(f"候选库不存在：{target}")
    try:
        with database_state_lock(target):
            with duckdb.connect(str(target)) as conn:
                if schema_exists(conn, "mining_v3"):
                    ensure_v3_schema(conn)
                    current = _read_snapshots(conn)
                    stored = _migration_rows_conn(conn)
                    if stored:
                        compare_migration(current, stored)
                        result = _result(
                            migrated=False,
                            reused=True,
                            snapshots=current,
                            migrated_at=max(
                                str(row["migrated_at"])
                                for row in stored.values()
                            ),
                        )
                        _log_migration(result)
                        return result
                    if any(_v3_business_counts_conn(conn).values()):
                        raise V3MigrationConflictError(
                            "mining_v3 已包含业务事实但没有迁移记录"
                        )
                else:
                    current = _read_snapshots(conn)

                migrated_at = utc_now_iso()
                conn.begin()
                try:
                    initialize_v3_schema(conn)
                    for snapshot in current.values():
                        _copy_snapshot_conn(conn, snapshot, migrated_at)
                except BaseException:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
                result = _result(
                    migrated=True,
                    reused=False,
                    snapshots=current,
                    migrated_at=migrated_at,
                )
    except (MiningMigrationError, MiningStorageError):
        raise
    except (OSError, duckdb.Error, json.JSONDecodeError) as error:
        _LOGGER.error(
            format_event(
                {
                    "operation": "migrate_v1_v2_to_v3",
                    "error_code": "MINING-126",
                }
            )
        )
        raise MiningMigrationError(f"V1/V2 到 V3 迁移失败：{error}") from error
    _log_migration(result)
    return result


def _log_migration(result: dict[str, Any]) -> None:
    counts = {
        source: info["counts"]["run"]
        for source, info in result["sources"].items()
    }
    _LOGGER.info(
        format_event(
            {
                "operation": "migrate_v1_v2_to_v3",
                "migrated": result["migrated"],
                "reused": result["reused"],
                "run_counts": json.dumps(counts, sort_keys=True),
            }
        )
    )


def v3_migration_status(
    db_path: Path | str = MINING_DB_FILE,
) -> dict[str, Any]:
    """返回迁移状态、源漂移和回滚资格，不修改数据库。"""

    target = Path(db_path)
    if not target.is_file():
        raise MiningMigrationError(f"候选库不存在：{target}")
    try:
        with duckdb.connect(str(target), read_only=True) as conn:
            if not schema_exists(conn, "mining_v3"):
                return {
                    "schema_version": None,
                    "migrated": False,
                    "sources": {},
                    "source_drift": [],
                    "rollback_eligible": False,
                    "rollback_reason": "mining_v3 不存在",
                }
            version = v3_schema_version(conn)
            if version != MINING_V3_SCHEMA_VERSION:
                raise V3MigrationConflictError(
                    f"V3 schema 为 {version!r}，当前代码只支持 "
                    f"{MINING_V3_SCHEMA_VERSION}"
                )
            stored = (
                _migration_rows_conn(conn)
                if _table_exists_conn(conn, "mining_v3", "migration_source")
                else {}
            )
            current = (
                _read_snapshots(conn)
                if schema_exists(conn, "mining")
                or schema_exists(conn, "mining_v2")
                else {}
            )
            drift = _source_drift(current, stored)
            counts = _v3_business_counts_conn(conn)
            publication_count = _v3_publication_count_conn(conn)
            blocked = {
                "V3_BUSINESS_WRITES": sum(counts.values()),
                "V3_PUBLICATION": publication_count,
            }
            reasons = [
                f"{name}:{value}"
                for name, value in blocked.items()
                if value
            ]
            if not stored:
                reasons.append("NO_MIGRATION_RECORD")
    except (MiningMigrationError, MiningStorageError):
        raise
    except (OSError, duckdb.Error) as error:
        raise MiningStorageError(f"读取 V3 迁移状态失败：{error}") from error
    return {
        "schema_version": version,
        "migrated": bool(stored),
        "sources": {
            source: {
                "source_schema_version": row["source_schema_version"],
                "source_snapshot_hash": row["source_snapshot_hash"],
                "counts": {
                    "context": int(row["context_count"]),
                    "run": int(row["run_count"]),
                    "candidate": int(row["candidate_count"]),
                },
                "migrated_at": row["migrated_at"],
            }
            for source, row in sorted(stored.items())
        },
        "source_drift": drift,
        "rollback_eligible": not reasons,
        "rollback_reason": None if not reasons else ", ".join(reasons),
    }


def rollback_v3_migration(
    db_path: Path | str = MINING_DB_FILE,
) -> dict[str, Any]:
    """仅在 V3 无业务写入时删除迁移创建的 mining_v3 schema。"""

    target = Path(db_path)
    if not target.is_file():
        raise MiningMigrationError(f"候选库不存在：{target}")
    try:
        with database_state_lock(target):
            with duckdb.connect(str(target)) as conn:
                if not schema_exists(conn, "mining_v3"):
                    result = {
                        "rolled_back": False,
                        "reused": True,
                        "sources": [],
                    }
                    _log_rollback(result)
                    return result
                version = v3_schema_version(conn)
                if version != MINING_V3_SCHEMA_VERSION:
                    raise V3MigrationRollbackError(
                        f"V3 schema 为 {version!r}，不能确认回滚资格"
                    )
                if not _table_exists_conn(
                    conn,
                    "mining_v3",
                    "migration_source",
                ):
                    raise V3MigrationRollbackError(
                        "mining_v3 不是由显式迁移创建，禁止回滚"
                    )
                stored = _migration_rows_conn(conn)
                if not stored:
                    raise V3MigrationRollbackError(
                        "mining_v3 不是由显式迁移创建，禁止回滚"
                    )
                counts = _v3_business_counts_conn(conn)
                if any(counts.values()):
                    raise V3MigrationRollbackError(
                        "mining_v3 已包含业务写入，禁止回滚迁移"
                    )
                if _v3_publication_count_conn(conn):
                    raise V3MigrationRollbackError(
                        "已存在 V3 正式发布批次，禁止回滚迁移"
                    )
                sources = sorted(stored)
                conn.begin()
                try:
                    conn.execute("DROP SCHEMA mining_v3 CASCADE")
                except BaseException:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
                result = {
                    "rolled_back": True,
                    "reused": False,
                    "sources": sources,
                }
    except (MiningMigrationError, MiningStorageError):
        raise
    except (OSError, duckdb.Error) as error:
        raise MiningStorageError(f"回滚 V3 迁移失败：{error}") from error
    _log_rollback(result)
    return result


def _log_rollback(result: dict[str, Any]) -> None:
    _LOGGER.info(
        format_event(
            {
                "operation": "rollback_v3_migration",
                "rolled_back": result["rolled_back"],
                "reused": result["reused"],
                "source_count": len(result["sources"]),
            }
        )
    )


__all__ = [
    "migrate_v1_v2_to_v3",
    "rollback_v3_migration",
    "v3_migration_status",
]
