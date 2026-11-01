"""V2 发布批次启动和候选状态写入。"""

from __future__ import annotations

from typing import Any

import duckdb

from mining.contract_types import canonical_json, sha256_json
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
    PublicationConflictError,
)
from mining.logging_conf import format_event
from mining.publication_contracts import (
    PUBLICATION_ACTIVE_STATUSES,
    PublicationBatch,
)
from mining.states import CandidateStateMachine
from mining.storage_lock import database_state_lock
from mining.storage_v2_publication_read import PUBLISHABLE_STATUSES


class V2PublicationWriteMixin:
    """启动 V2 批次并原子迁移候选发布状态。"""

    def start_publication_batch(
        self,
        batch: PublicationBatch,
    ) -> dict[str, Any]:
        """原子写入发布批次并把 V2 候选推进到 building。"""

        if batch.source_kind != "v2_search":
            raise PublicationConflictError("V2 存储只能启动搜索来源批次")
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._ensure_no_active_publication_conn(conn)
                        run = self._row(
                            conn,
                            "SELECT c.context_hash, coalesce(ctl.status, r.status) "
                            "AS run_status "
                            "FROM mining_v2.run r "
                            "JOIN mining_v2.context c "
                            "ON c.context_id=r.context_id "
                            "LEFT JOIN mining_v2.run_control ctl "
                            "ON ctl.run_id=r.run_id "
                            "WHERE r.run_id=?",
                            [batch.run_id],
                        )
                        if run is None:
                            raise MiningNotFoundError(
                                f"V2 搜索运行不存在：{batch.run_id}"
                            )
                        if run["context_hash"] != batch.context_hash:
                            raise PublicationConflictError("发布上下文已变化")
                        if run["run_status"] != "review_ready":
                            raise PublicationConflictError(
                                "V2 运行不在 review_ready 状态"
                            )
                        publishable = {
                            item["candidate_id"]
                            for item in self._candidate_rows_conn(
                                conn,
                                batch.run_id,
                            )
                            if item["status"] in PUBLISHABLE_STATUSES
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
                            self._start_v2_publication_candidate_conn(
                                conn,
                                batch,
                                candidate_id,
                                approval_id,
                            )
                        self._insert_publication_batch_conn(conn, batch)
                result = self.get_publication_batch(batch.batch_id)
        except (
            ApprovalConflictError,
            MiningContractError,
            MiningNotFoundError,
            MiningStateError,
            PublicationConflictError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(
                f"启动 V2 发布批次失败：{error}"
            ) from error
        self.logger.info(
            format_event(
                {
                    "operation": "start_v2_publication_batch",
                    "batch_id": batch.batch_id,
                    "run_id": batch.run_id,
                    "candidate_count": len(batch.candidate_ids),
                }
            )
        )
        return result

    def _ensure_no_active_publication_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """V1 和 V2 共用正式因子输出，因此全库只允许一个活动批次。"""

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

    def _start_v2_publication_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch: PublicationBatch,
        candidate_id: str,
        approval_id: str,
    ) -> None:
        candidate = self._require_candidate_conn(
            conn,
            batch.run_id,
            candidate_id,
        )
        if candidate["status"] not in PUBLISHABLE_STATUSES:
            raise PublicationConflictError(f"候选不可发布：{candidate_id}")
        approval = self._row(
            conn,
            "SELECT approval_id, decision FROM mining_v2.search_approval "
            "WHERE run_id=? AND candidate_id=?",
            [batch.run_id, candidate_id],
        )
        if (
            approval is None
            or approval["approval_id"] != approval_id
            or approval["decision"] != "approved"
        ):
            raise PublicationConflictError(f"候选批准记录无效：{candidate_id}")
        snapshot = self._search_approval_snapshot_conn(
            conn,
            batch.run_id,
            candidate_id,
        )
        stored = self._row(
            conn,
            "SELECT evidence_hash FROM mining_v2.search_approval "
            "WHERE run_id=? AND candidate_id=?",
            [batch.run_id, candidate_id],
        )
        if (
            stored is None
            or stored["evidence_hash"] != snapshot["evidence"].evidence_hash
        ):
            raise ApprovalConflictError("审批证据已变化，不能发布")

        publication_id = (
            f"pub_{batch.batch_id}_{candidate_id.replace(':', '_')}"
        )[:160]
        conn.execute(
            "INSERT INTO mining.publication "
            "(publication_id, batch_id, run_id, candidate_id, approval_id, "
            "status, staging_path, manifest_hash, formal_factor_name, "
            "artifact_sha256, error_code, error_message, attempt, "
            "started_at, finished_at, source_kind) "
            "VALUES (?, ?, ?, ?, ?, 'staging', ?, NULL, NULL, NULL, NULL, "
            "NULL, ?, ?, NULL, ?)",
            [
                publication_id,
                batch.batch_id,
                batch.run_id,
                candidate_id,
                approval_id,
                batch.staging_path,
                batch.attempt,
                batch.started_at,
                batch.source_kind,
            ],
        )
        self._transition_v2_publication_candidate_conn(
            conn,
            batch,
            candidate_id,
            candidate["status"],
            "building",
            batch.started_at,
            {"batch_id": batch.batch_id},
        )

    def _transition_v2_publication_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch: dict[str, Any] | PublicationBatch,
        candidate_id: str,
        previous: str,
        target: str,
        timestamp: str,
        payload: dict[str, Any],
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        CandidateStateMachine.ensure(previous, target)
        run_id = (
            batch.run_id if isinstance(batch, PublicationBatch) else batch["run_id"]
        )
        batch_id = (
            batch.batch_id if isinstance(batch, PublicationBatch) else batch["batch_id"]
        )
        updated = conn.execute(
            "UPDATE mining_v2.candidate_trace SET status=?, updated_at=? "
            "WHERE run_id=? AND candidate_id=? AND status=? "
            "RETURNING candidate_id",
            [target, timestamp, run_id, candidate_id, previous],
        ).fetchall()
        if not updated:
            raise MiningStateError("V2 候选状态已被其他请求修改")
        self._append_trace_conn(
            conn,
            run_id,
            "candidate_state",
            f"candidate:{candidate_id}:{target}:{batch_id}",
            sha256_json({"status": target, "batch_id": batch_id}),
            timestamp,
        )
        self._append_event_conn(
            conn,
            run_id,
            f"candidate_{candidate_id}_{target}_{batch_id}",
            "candidate_state_changed",
            {
                "candidate_id": candidate_id,
                "from": previous,
                "to": target,
                "batch_id": batch_id,
                "error_code": error_code,
                **payload,
            },
            created_at=timestamp,
        )

    @staticmethod
    def _insert_publication_batch_conn(
        conn: duckdb.DuckDBPyConnection,
        batch: PublicationBatch,
    ) -> None:
        conn.execute(
            "INSERT INTO mining.publication_batch "
            "(batch_id, run_id, context_hash, candidate_ids_json, "
            "approval_ids_json, status, staging_path, manifest_hash, "
            "artifact_sha256, error_code, error_message, attempt, "
            "started_at, finished_at, source_kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                batch.source_kind,
            ],
        )
