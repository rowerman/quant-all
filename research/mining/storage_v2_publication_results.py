"""V2 发布校验、成功和失败结果收口。"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mining.contract_types import utc_now_iso
from mining.errors import MiningStateError, PublicationConflictError
from mining.logging_conf import format_event
from mining.storage_lock import database_state_lock


class V2PublicationResultMixin:
    """把 V2 发布结果同步到正式发布表和候选轨迹。"""

    def mark_publication_validating(self, batch_id: str) -> dict[str, Any]:
        """把 V2 发布批次推进到 validating。"""

        return self._transition_v2_publication_batch(
            batch_id,
            "validating",
            publications_from="staging",
        )

    def validate_publication_manifest(
        self,
        batch_id: str,
        *,
        manifest_hash: str,
        artifact_sha256: str,
    ) -> dict[str, Any]:
        """保存 V2 发布批次已通过的清单和产物哈希。"""

        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    updated = conn.execute(
                        "UPDATE mining.publication_batch SET "
                        "manifest_hash=?, artifact_sha256=? "
                        "WHERE batch_id=? AND source_kind='v2_search' "
                        "AND status='validating' RETURNING batch_id",
                        [manifest_hash, artifact_sha256, batch_id],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError("V2 发布批次不在 validating 状态")
                    conn.execute(
                        "UPDATE mining.publication SET manifest_hash=?, "
                        "artifact_sha256=? WHERE batch_id=? "
                        "AND source_kind='v2_search'",
                        [manifest_hash, artifact_sha256, batch_id],
                    )
        return self.get_publication_batch(batch_id)

    def complete_publication_batch(
        self,
        batch_id: str,
        *,
        manifest_hash: str,
        artifact_sha256: str,
        formal_factor_names: Sequence[str],
    ) -> dict[str, Any]:
        """原子完成 V2 批次并把候选推进到 published。"""

        timestamp = utc_now_iso()
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["source_kind"] != "v2_search":
                        raise PublicationConflictError("发布批次不是 V2 来源")
                    if batch["status"] != "validating":
                        raise MiningStateError("只有 validating 批次可以完成发布")
                    candidate_ids = json.loads(batch["candidate_ids_json"])
                    if len(candidate_ids) != len(formal_factor_names):
                        raise MiningStateError(
                            "正式因子名称数量与候选数量不一致"
                        )
                    for candidate_id, factor_name in zip(
                        candidate_ids,
                        formal_factor_names,
                        strict=True,
                    ):
                        self._transition_v2_publication_candidate_conn(
                            conn,
                            batch,
                            candidate_id,
                            "building",
                            "published",
                            timestamp,
                            {
                                "batch_id": batch_id,
                                "formal_factor_name": factor_name,
                                "artifact_sha256": artifact_sha256,
                            },
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status='published', "
                        "formal_factor_name=?, manifest_hash=?, "
                        "artifact_sha256=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind='v2_search'",
                        [
                            ",".join(formal_factor_names),
                            manifest_hash,
                            artifact_sha256,
                            timestamp,
                            batch_id,
                        ],
                    )
                    conn.execute(
                        "UPDATE mining.publication_batch SET status='published', "
                        "manifest_hash=?, artifact_sha256=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind='v2_search' "
                        "AND status='validating'",
                        [manifest_hash, artifact_sha256, timestamp, batch_id],
                    )
            result = self.get_publication_batch(batch_id)
        self.logger.info(
            format_event(
                {
                    "operation": "complete_v2_publication_batch",
                    "batch_id": batch_id,
                    "candidate_count": len(formal_factor_names),
                    "artifact_sha256": artifact_sha256,
                }
            )
        )
        return result

    def fail_publication_batch(
        self,
        batch_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        """把失败 V2 批次和 building 候选收口到 build_failed。"""

        timestamp = utc_now_iso()
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["source_kind"] != "v2_search":
                        raise PublicationConflictError("发布批次不是 V2 来源")
                    if batch["status"] in {"published", "rolled_back"}:
                        return self.get_publication_batch(batch_id)
                    candidate_ids = json.loads(batch["candidate_ids_json"])
                    for candidate_id in candidate_ids:
                        candidate = self._row(
                            conn,
                            "SELECT status FROM mining_v2.candidate_trace "
                            "WHERE run_id=? AND candidate_id=?",
                            [batch["run_id"], candidate_id],
                        )
                        if candidate is not None and candidate["status"] == "building":
                            self._transition_v2_publication_candidate_conn(
                                conn,
                                batch,
                                candidate_id,
                                "building",
                                "build_failed",
                                timestamp,
                                {"batch_id": batch_id},
                                error_code=error_code,
                                error_message=error_message,
                            )
                    conn.execute(
                        "UPDATE mining.publication SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind='v2_search'",
                        [error_code, error_message, timestamp, batch_id],
                    )
                    conn.execute(
                        "UPDATE mining.publication_batch SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind='v2_search'",
                        [error_code, error_message, timestamp, batch_id],
                    )
            result = self.get_publication_batch(batch_id)
        self.logger.error(
            format_event(
                {
                    "operation": "fail_v2_publication_batch",
                    "batch_id": batch_id,
                    "error_code": error_code,
                }
            )
        )
        return result

    def _transition_v2_publication_batch(
        self,
        batch_id: str,
        status: str,
        *,
        publications_from: str,
    ) -> dict[str, Any]:
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    updated = conn.execute(
                        "UPDATE mining.publication_batch SET status=? "
                        "WHERE batch_id=? AND source_kind='v2_search' "
                        "AND status=? RETURNING batch_id",
                        [status, batch_id, publications_from],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError(
                            f"发布批次不能从 {publications_from} 变为 {status}"
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status=? "
                        "WHERE batch_id=? AND source_kind='v2_search' "
                        "AND status=?",
                        [status, batch_id, publications_from],
                    )
        return self.get_publication_batch(batch_id)
