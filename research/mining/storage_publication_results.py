"""发布批次校验、成功和失败结果的原子状态收口。"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mining.contracts import utc_now_iso
from mining.errors import MiningStateError
from mining.logging_conf import format_event
from mining.states import CandidateStateMachine


class PublicationResultStoreMixin:
    """把发布结果同步到批次、逐候选记录和候选状态。"""

    def mark_publication_validating(self, batch_id: str) -> dict[str, Any]:
        """把整批发布记录推进到 validating。"""

        return self._transition_publication_batch(
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
        """保存已通过校验的清单和产物哈希。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    updated = conn.execute(
                        "UPDATE mining.publication_batch SET "
                        "manifest_hash=?, artifact_sha256=? "
                        "WHERE batch_id=? AND status='validating' "
                        "RETURNING batch_id",
                        [manifest_hash, artifact_sha256, batch_id],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError("发布批次不在 validating 状态")
                    conn.execute(
                        "UPDATE mining.publication SET manifest_hash=?, "
                        "artifact_sha256=? WHERE batch_id=?",
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
        """在一笔事务内把批次和全部候选标记为 published。"""

        timestamp = utc_now_iso()
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["status"] != "validating":
                        raise MiningStateError("只有 validating 批次可以完成发布")
                    candidate_ids = json.loads(batch["candidate_ids_json"])
                    if len(candidate_ids) != len(formal_factor_names):
                        raise MiningStateError("正式因子名称数量与候选数量不一致")
                    for candidate_id, factor_name in zip(
                        candidate_ids,
                        formal_factor_names,
                        strict=True,
                    ):
                        CandidateStateMachine.ensure("building", "published")
                        self._transition_candidate_conn(
                            conn,
                            batch["run_id"],
                            candidate_id,
                            "building",
                            "published",
                            f"publication_published_{batch_id}_{candidate_id}",
                            "publication",
                            {
                                "batch_id": batch_id,
                                "formal_factor_name": factor_name,
                                "artifact_sha256": artifact_sha256,
                            },
                            None,
                            None,
                            None,
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status='published', "
                        "formal_factor_name=?, manifest_hash=?, artifact_sha256=?, "
                        "finished_at=? WHERE batch_id=?",
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
                        "WHERE batch_id=? AND status='validating'",
                        [manifest_hash, artifact_sha256, timestamp, batch_id],
                    )
            result = self.get_publication_batch(batch_id)
        self.logger.info(
            format_event(
                {
                    "operation": "complete_publication_batch",
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
        """把失败批次及其 building 候选整体收口到 build_failed。"""

        timestamp = utc_now_iso()
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["status"] in {"published", "rolled_back"}:
                        return self.get_publication_batch(batch_id)
                    candidate_ids = json.loads(batch["candidate_ids_json"])
                    for candidate_id in candidate_ids:
                        current = self._row(
                            conn,
                            "SELECT status FROM mining.candidate "
                            "WHERE run_id=? AND candidate_id=?",
                            [batch["run_id"], candidate_id],
                        )
                        if current is not None and current["status"] == "building":
                            CandidateStateMachine.ensure("building", "build_failed")
                            self._transition_candidate_conn(
                                conn,
                                batch["run_id"],
                                candidate_id,
                                "building",
                                "build_failed",
                                f"publication_failed_{batch_id}_{candidate_id}",
                                "publication",
                                {"batch_id": batch_id},
                                "publication",
                                error_code,
                                error_message,
                            )
                    conn.execute(
                        "UPDATE mining.publication SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=?",
                        [error_code, error_message, timestamp, batch_id],
                    )
                    conn.execute(
                        "UPDATE mining.publication_batch SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=?",
                        [error_code, error_message, timestamp, batch_id],
                    )
            result = self.get_publication_batch(batch_id)
        self.logger.error(
            format_event(
                {
                    "operation": "fail_publication_batch",
                    "batch_id": batch_id,
                    "error_code": error_code,
                }
            )
        )
        return result

    def _transition_publication_batch(
        self,
        batch_id: str,
        status: str,
        *,
        publications_from: str,
    ) -> dict[str, Any]:
        """同步更新批次与逐候选发布记录的状态。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    updated = conn.execute(
                        "UPDATE mining.publication_batch SET status=? "
                        "WHERE batch_id=? AND status=? RETURNING batch_id",
                        [status, batch_id, publications_from],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError(
                            f"发布批次不能从 {publications_from} 变为 {status}"
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status=? "
                        "WHERE batch_id=? AND status=?",
                        [status, batch_id, publications_from],
                    )
        return self.get_publication_batch(batch_id)
