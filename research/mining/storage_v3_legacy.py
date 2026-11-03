"""迁移后 V1/V2 运行的只读兼容查询。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.errors import MiningNotFoundError, MiningStorageError
from mining.v3_migration_contracts import (
    LEGACY_CANDIDATE_FIELDS,
    LEGACY_CONTEXT_FIELDS,
    LEGACY_RUN_FIELDS,
    LEGACY_SOURCE_SCHEMAS,
    LEGACY_SOURCE_VERSIONS,
    legacy_snapshot_hash,
)


class V3LegacyStoreMixin:
    """读取迁移时固化的 V1/V2 只读摘要。"""

    def migration_summary(self) -> dict[str, Any]:
        """返回 V1/V2 来源快照和迁移计数。"""

        with self._connection() as conn:
            rows = self._migration_source_rows_conn(conn)
            self._validate_legacy_sources_conn(conn)
        sources = {
            str(row["source_schema"]): {
                "source_schema_version": row["source_schema_version"],
                "source_snapshot_hash": row["source_snapshot_hash"],
                "counts": {
                    "context": int(row["context_count"]),
                    "run": int(row["run_count"]),
                    "candidate": int(row["candidate_count"]),
                },
                "migrated_at": row["migrated_at"],
            }
            for row in rows
        }
        return {
            "schema_version": "mining_v3",
            "sources": sources,
        }

    def list_legacy_runs(
        self,
        limit: int = 50,
        *,
        source_schema: str | None = None,
    ) -> list[dict[str, Any]]:
        """按创建时间倒序列出迁移后的 V1/V2 运行摘要。"""

        self._validate_source_filter(source_schema)
        limit = max(1, min(int(limit), 200))
        with self._connection() as conn:
            self._validate_legacy_sources_conn(conn)
            if source_schema is None:
                rows = conn.execute(
                    "SELECT r.*, c.context_hash "
                    "FROM mining_v3.legacy_run r "
                    "JOIN mining_v3.legacy_context c "
                    "ON c.source_schema=r.source_schema "
                    "AND c.context_id=r.context_id "
                    "ORDER BY r.created_at DESC, r.source_schema, r.run_id LIMIT ?",
                    [limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT r.*, c.context_hash "
                    "FROM mining_v3.legacy_run r "
                    "JOIN mining_v3.legacy_context c "
                    "ON c.source_schema=r.source_schema "
                    "AND c.context_id=r.context_id "
                    "WHERE r.source_schema=? "
                    "ORDER BY r.created_at DESC, r.run_id LIMIT ?",
                    [source_schema, limit],
                ).fetchall()
            columns = [item[0] for item in conn.description]
        return [
            self._legacy_run_view(dict(zip(columns, row, strict=True)))
            for row in rows
        ]

    def get_legacy_run(
        self,
        run_id: str,
        *,
        source_schema: str | None = None,
    ) -> dict[str, Any]:
        """读取一个旧运行、上下文和候选摘要，并明确标记只读。"""

        self._validate_source_filter(source_schema)
        with self._connection() as conn:
            self._validate_legacy_sources_conn(conn)
            row = self._legacy_run_row_conn(conn, run_id, source_schema)
            context = self._row(
                conn,
                "SELECT * FROM mining_v3.legacy_context "
                "WHERE source_schema=? AND context_id=?",
                [row["source_schema"], row["context_id"]],
            )
            candidates = conn.execute(
                "SELECT * FROM mining_v3.legacy_candidate "
                "WHERE source_schema=? AND run_id=? "
                "ORDER BY created_at, candidate_id",
                [row["source_schema"], run_id],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        if context is None:
            raise MiningStorageError(f"legacy 运行缺少上下文：{run_id}")
        result = self._legacy_run_view(row)
        result["context"] = {
            "context_id": context["context_id"],
            "context_hash": context["context_hash"],
            "payload": json.loads(context["payload_json"]),
            "created_at": context["created_at"],
        }
        result["candidates"] = [
            self._legacy_candidate_view(dict(zip(columns, item, strict=True)))
            for item in candidates
        ]
        return result

    def get_legacy_candidate(
        self,
        run_id: str,
        candidate_id: str,
        *,
        source_schema: str | None = None,
    ) -> dict[str, Any]:
        """读取一个旧候选摘要，不允许恢复或继续运行。"""

        self._validate_source_filter(source_schema)
        with self._connection() as conn:
            self._validate_legacy_sources_conn(conn)
            if source_schema is None:
                row = self._row(
                    conn,
                    "SELECT * FROM mining_v3.legacy_candidate "
                    "WHERE run_id=? AND candidate_id=?",
                    [run_id, candidate_id],
                )
            else:
                row = self._row(
                    conn,
                    "SELECT * FROM mining_v3.legacy_candidate "
                    "WHERE source_schema=? AND run_id=? AND candidate_id=?",
                    [source_schema, run_id, candidate_id],
                )
        if row is None:
            raise MiningNotFoundError(f"legacy 候选不存在：{candidate_id}")
        return self._legacy_candidate_view(row)

    def _migration_source_rows_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM mining_v3.migration_source ORDER BY source_schema"
        ).fetchall()
        columns = [item[0] for item in conn.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def _validate_legacy_sources_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """重算每个 legacy 来源，拒绝迁移后被修改的快照。"""

        for source in self._migration_source_rows_conn(conn):
            source_schema = str(source["source_schema"])
            if source_schema not in LEGACY_SOURCE_SCHEMAS:
                raise MiningStorageError(f"未知 legacy 来源：{source_schema}")
            contexts, runs, candidates, counts = self._legacy_material_conn(
                conn,
                source_schema,
            )
            expected_counts = {
                "context": int(source["context_count"]),
                "run": int(source["run_count"]),
                "candidate": int(source["candidate_count"]),
            }
            actual_counts = {
                "context": len(contexts),
                "run": len(runs),
                "candidate": len(candidates),
            }
            snapshot_hash = legacy_snapshot_hash(
                source_schema,
                contexts,
                runs,
                candidates,
            )
            if (
                source["source_schema_version"]
                != LEGACY_SOURCE_VERSIONS[source_schema]
                or source["source_snapshot_hash"] != snapshot_hash
                or expected_counts != actual_counts
                or counts != actual_counts
            ):
                raise MiningStorageError(
                    f"{source_schema} legacy 快照已变化或被篡改"
                )

    @staticmethod
    def _legacy_material_conn(
        conn: duckdb.DuckDBPyConnection,
        source_schema: str,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
    ]:
        def read(
            table: str,
            fields: tuple[str, ...],
            order_by: tuple[str, ...],
        ) -> list[dict[str, Any]]:
            rows = conn.execute(
                f"SELECT {', '.join(fields)} FROM mining_v3.{table} "
                f"WHERE source_schema=? ORDER BY {', '.join(order_by)}",
                [source_schema],
            ).fetchall()
            return [
                dict(zip(fields, row, strict=True))
                for row in rows
            ]

        contexts = read(
            "legacy_context",
            LEGACY_CONTEXT_FIELDS,
            ("context_id",),
        )
        runs = read(
            "legacy_run",
            LEGACY_RUN_FIELDS,
            ("run_id",),
        )
        candidates = read(
            "legacy_candidate",
            LEGACY_CANDIDATE_FIELDS,
            ("run_id", "candidate_id"),
        )
        counts = {
            "context": len(contexts),
            "run": len(runs),
            "candidate": len(candidates),
        }
        return contexts, runs, candidates, counts

    def _legacy_run_row_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        source_schema: str | None,
    ) -> dict[str, Any]:
        if source_schema is None:
            rows = conn.execute(
                "SELECT * FROM mining_v3.legacy_run WHERE run_id=?",
                [run_id],
            ).fetchall()
            if len(rows) > 1:
                raise MiningStorageError(
                    f"legacy 运行 ID 跨来源重复：{run_id}"
                )
            row = rows[0] if rows else None
            columns = [item[0] for item in conn.description]
        else:
            row = self._row(
                conn,
                "SELECT * FROM mining_v3.legacy_run "
                "WHERE source_schema=? AND run_id=?",
                [source_schema, run_id],
            )
            columns = []
        if row is None:
            raise MiningNotFoundError(f"legacy 运行不存在：{run_id}")
        if isinstance(row, dict):
            return row
        return dict(zip(columns, row, strict=True))

    @staticmethod
    def _validate_source_filter(source_schema: str | None) -> None:
        if source_schema is not None and source_schema not in LEGACY_SOURCE_SCHEMAS:
            raise MiningNotFoundError(f"未知 legacy 来源：{source_schema}")

    @staticmethod
    def _legacy_run_view(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        result["read_only"] = True
        result["resumable"] = False
        return result

    @staticmethod
    def _legacy_candidate_view(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["definition"] = json.loads(result.pop("definition_json"))
        result["read_only"] = True
        return result


__all__ = ["V3LegacyStoreMixin"]
