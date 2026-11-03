"""迁移后的 V1 只读兼容查询。"""

from __future__ import annotations

import json
from typing import Any

from mining.errors import MiningNotFoundError


class V2LegacyStoreMixin:
    """读取迁移时固化的 V1 元数据快照。"""

    def migration_summary(self) -> dict[str, Any] | None:
        """返回 V1 迁移证据。"""

        with self._connection() as conn:
            return self._row(
                conn,
                "SELECT * FROM mining_v2.legacy_source "
                "WHERE source_schema='mining_v1'",
            )

    def list_legacy_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出迁移后的 V1 运行快照。"""

        limit = max(1, min(int(limit), 200))
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mining_v2.legacy_run ORDER BY created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        return [self._legacy_run_view(dict(zip(columns, row, strict=True))) for row in rows]

    def get_legacy_run(self, run_id: str) -> dict[str, Any]:
        """读取单条 V1 运行快照，并明确拒绝恢复。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.legacy_run WHERE run_id=?",
                [run_id],
            )
            context = self._row(
                conn,
                "SELECT * FROM mining_v2.legacy_context "
                "WHERE context_id=(SELECT context_id FROM mining_v2.legacy_run "
                "WHERE run_id=?)",
                [run_id],
            )
            candidates = conn.execute(
                "SELECT * FROM mining_v2.legacy_candidate "
                "WHERE run_id=? ORDER BY created_at",
                [run_id],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        if row is None or context is None:
            raise MiningNotFoundError(f"V1 快照运行不存在：{run_id}")
        result = self._legacy_run_view(row)
        result["context"] = {
            "context_id": context["context_id"],
            "context_hash": context["context_hash"],
            "payload": json.loads(context["payload_json"]),
        }
        result["candidates"] = [
            self._legacy_candidate_view(dict(zip(columns, values, strict=True)))
            for values in candidates
        ]
        return result

    def get_legacy_candidate(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """读取单条 V1 候选快照。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.legacy_candidate "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        if row is None:
            raise MiningNotFoundError(f"V1 快照候选不存在：{candidate_id}")
        return self._legacy_candidate_view(row)

    @staticmethod
    def _legacy_run_view(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        result["resumable"] = False
        result["read_only"] = True
        return result

    @staticmethod
    def _legacy_candidate_view(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["definition"] = json.loads(result.pop("definition_json"))
        result["read_only"] = True
        return result

