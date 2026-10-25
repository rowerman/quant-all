"""批量正式发布资格、状态和逐候选关联的持久化。"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mining.contracts import canonical_json
from mining.errors import (
    MiningNotFoundError,
    MiningStateError,
    PublicationConflictError,
)
from mining.logging_conf import format_event
from mining.publication_contracts import (
    PUBLICATION_ACTIVE_STATUSES,
    PublicationBatch,
)
from mining.states import CandidateStateMachine


class PublicationStoreMixin:
    """管理发布批次与候选状态之间的原子关系。"""

    def start_publication_batch(
        self,
        batch: PublicationBatch,
    ) -> dict[str, Any]:
        """资格全部通过后，把整批候选原子改为 building。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    self._ensure_no_active_publication_conn(conn)
                    run = self._row(
                        conn,
                        "SELECT c.context_hash FROM mining.run r "
                        "JOIN mining.run_context c ON c.context_id=r.context_id "
                        "WHERE r.run_id=?",
                        [batch.run_id],
                    )
                    if run is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{batch.run_id}")
                    if run["context_hash"] != batch.context_hash:
                        raise PublicationConflictError("发布上下文已变化")
                    publishable = {
                        row[0]
                        for row in conn.execute(
                            "SELECT candidate_id FROM mining.candidate "
                            "WHERE run_id=? AND status IN ('approved', 'build_failed')",
                            [batch.run_id],
                        ).fetchall()
                    }
                    if publishable != set(batch.candidate_ids):
                        raise PublicationConflictError(
                            "发布批次必须包含该运行内全部已批准候选"
                        )
                    for candidate_id, approval_id in zip(
                        batch.candidate_ids,
                        batch.approval_ids,
                        strict=True,
                    ):
                        self._start_publication_candidate_conn(
                            conn,
                            batch,
                            candidate_id,
                            approval_id,
                        )
                    conn.execute(
                        "INSERT INTO mining.publication_batch VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            batch.batch_id,
                            batch.run_id,
                            batch.context_hash,
                            canonical_json(list(batch.candidate_ids)),
                            canonical_json(list(batch.approval_ids)),
                            batch.status,
                            batch.staging_path,
                            batch.manifest_hash,
                            batch.artifact_sha256,
                            batch.error_code,
                            batch.error_message,
                            batch.attempt,
                            batch.started_at,
                            batch.finished_at,
                        ],
                    )
            result = self.get_publication_batch(batch.batch_id)
        self.logger.info(
            format_event(
                {
                    "operation": "start_publication_batch",
                    "batch_id": batch.batch_id,
                    "run_id": batch.run_id,
                    "candidate_count": len(batch.candidate_ids),
                }
            )
        )
        return result

    def _ensure_no_active_publication_conn(self, conn) -> None:
        """同一候选库只允许一个批次处于构建或校验。"""

        placeholders = ",".join("?" for _ in PUBLICATION_ACTIVE_STATUSES)
        active = conn.execute(
            f"SELECT batch_id FROM mining.publication_batch "
            f"WHERE status IN ({placeholders}) LIMIT 1",
            list(PUBLICATION_ACTIVE_STATUSES),
        ).fetchone()
        if active is not None:
            raise PublicationConflictError(
                f"已有发布批次正在执行：{active[0]}"
            )

    def _start_publication_candidate_conn(
        self,
        conn,
        batch: PublicationBatch,
        candidate_id: str,
        approval_id: str,
    ) -> None:
        """写入逐候选发布记录并执行 approved -> building。"""

        candidate = self._row(
            conn,
            "SELECT status FROM mining.candidate "
            "WHERE run_id=? AND candidate_id=?",
            [batch.run_id, candidate_id],
        )
        if candidate is None or candidate["status"] not in {
            "approved",
            "build_failed",
        }:
            raise PublicationConflictError(f"候选不可发布：{candidate_id}")
        approval = self._row(
            conn,
            "SELECT approval_id, decision FROM mining.approval "
            "WHERE run_id=? AND candidate_id=?",
            [batch.run_id, candidate_id],
        )
        if (
            approval is None
            or approval["approval_id"] != approval_id
            or approval["decision"] != "approved"
        ):
            raise PublicationConflictError(f"候选批准记录无效：{candidate_id}")
        publication_id = (
            f"pub_{batch.batch_id}_{candidate_id.replace(':', '_')}"
        )[:160]
        conn.execute(
            "INSERT INTO mining.publication "
            "(publication_id, batch_id, run_id, candidate_id, approval_id, "
            "status, staging_path, manifest_hash, formal_factor_name, "
            "artifact_sha256, error_code, error_message, attempt, "
            "started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, 'staging', ?, NULL, NULL, NULL, NULL, "
            "NULL, ?, ?, NULL)",
            [
                publication_id,
                batch.batch_id,
                batch.run_id,
                candidate_id,
                approval_id,
                batch.staging_path,
                batch.attempt,
                batch.started_at,
            ],
        )
        CandidateStateMachine.ensure(candidate["status"], "building")
        self._transition_candidate_conn(
            conn,
            batch.run_id,
            candidate_id,
            candidate["status"],
            "building",
            f"publication_building_{batch.batch_id}_{candidate_id}",
            "publication",
            {"batch_id": batch.batch_id},
            None,
            None,
            None,
        )

    def get_publication_batch(self, batch_id: str) -> dict[str, Any]:
        """读取发布批次及逐候选发布记录。"""

        with self._connection() as conn:
            batch = self._publication_batch_row(conn, batch_id)
            publications = conn.execute(
                "SELECT * FROM mining.publication WHERE batch_id=? "
                "ORDER BY candidate_id",
                [batch_id],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        batch["candidate_ids"] = json.loads(batch.pop("candidate_ids_json"))
        batch["approval_ids"] = json.loads(batch.pop("approval_ids_json"))
        batch["publications"] = [
            dict(zip(columns, row, strict=True)) for row in publications
        ]
        return batch

    def list_publication_batches(
        self,
        run_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按时间倒序列出运行的发布批次。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT batch_id FROM mining.publication_batch "
                "WHERE run_id=? ORDER BY started_at DESC LIMIT ?",
                [run_id, max(1, min(int(limit), 100))],
            ).fetchall()
        return [self.get_publication_batch(row[0]) for row in rows]

    def incomplete_publication_batches(self) -> list[dict[str, Any]]:
        """返回启动恢复需要收口的批次。"""

        placeholders = ",".join("?" for _ in PUBLICATION_ACTIVE_STATUSES)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id FROM mining.publication_batch "
                f"WHERE status IN ({placeholders}) ORDER BY started_at",
                list(PUBLICATION_ACTIVE_STATUSES),
            ).fetchall()
        return [self.get_publication_batch(row[0]) for row in rows]

    def next_publication_attempt(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
    ) -> int:
        """返回当前批次组合的下一个尝试编号。"""

        encoded = canonical_json(sorted(candidate_ids))
        with self._connection() as conn:
            row = conn.execute(
                "SELECT max(attempt) FROM mining.publication_batch "
                "WHERE run_id=? AND candidate_ids_json=?",
                [run_id, encoded],
            ).fetchone()
        return int(row[0] or 0) + 1

    def _publication_batch_row(self, conn, batch_id: str) -> dict[str, Any]:
        """读取单行发布批次。"""

        row = self._row(
            conn,
            "SELECT * FROM mining.publication_batch WHERE batch_id=?",
            [batch_id],
        )
        if row is None:
            raise MiningNotFoundError(f"发布批次不存在：{batch_id}")
        return row
