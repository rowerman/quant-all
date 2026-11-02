"""V3 已批准候选接入共享正式发布链路的存储适配。"""

from __future__ import annotations

import json
from typing import Any, Sequence

import duckdb

from mining.contract_types import canonical_json, identifier, utc_now_iso
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
    PublicationConflictError,
)
from mining.publication_contracts import (
    PUBLICATION_ACTIVE_STATUSES,
    PublicationBatch,
    PublicationInputs,
)
from mining.search_orchestrator_support import candidate_payload
from mining.search_publication_contracts import publication_context
from mining.storage_lock import database_state_lock
from mining.v3_contracts import V3_PUBLICATION_SOURCE_KIND

_PUBLISHABLE_STATUSES = ("approved", "build_failed")


class V3PublicationStoreMixin:
    """把 V3 审批事实映射为共享发布服务输入和状态迁移。"""

    @property
    def publication_source_kind(self) -> str:
        """返回 V3 共享发布来源。"""

        return V3_PUBLICATION_SOURCE_KIND

    def load_publication_inputs(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
        *,
        expected_context_hash: str,
    ) -> PublicationInputs:
        """校验 V3 整批发布资格并构造共享构建输入。"""

        run_id = identifier(run_id, "run_id")
        if isinstance(candidate_ids, str) or not isinstance(
            candidate_ids,
            Sequence,
        ):
            raise MiningContractError("candidate_ids 必须是标识序列")
        selected = tuple(
            sorted(identifier(item, "candidate_id") for item in candidate_ids)
        )
        if not selected:
            raise PublicationConflictError("发布批次至少需要一个候选")
        if len(set(selected)) != len(selected):
            raise MiningContractError("candidate_ids 不能重复")

        with self._connection() as conn:
            input_envelope = self._input_envelope_conn(conn, run_id)
            context = self._context_conn(conn, run_id)
            publishable = self._publishable_candidate_ids_conn(conn, run_id)
        if expected_context_hash != input_envelope.context_hash:
            raise PublicationConflictError("页面上下文已变化，请刷新")
        if set(publishable) != set(selected):
            raise PublicationConflictError(
                "发布批次必须包含该运行内全部已批准候选"
            )

        candidates = []
        approval_ids = []
        for candidate_id in selected:
            candidate = self.get_candidate(run_id, candidate_id)
            approval = self.get_approval(run_id, candidate_id)
            if approval["decision"] != "approved":
                raise PublicationConflictError("候选没有批准记录")
            if not self.approval_is_current(run_id, candidate_id):
                raise ApprovalConflictError("审批证据已变化，不能发布")
            source_id = str(approval["source_id"])
            envelope = self._candidate_envelope(run_id, candidate_id, source_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "content_hash": envelope.candidate.content_hash,
                    "semantic_hash": envelope.candidate.semantic_hash,
                    "name": envelope.candidate.name,
                    "status": candidate["status"],
                    "definition": candidate_payload(envelope.candidate),
                    "source_kind": V3_PUBLICATION_SOURCE_KIND,
                    "source_context_hash": input_envelope.context_hash,
                    "candidate_envelope_hash": envelope.envelope_hash,
                    "source_reference_hash": envelope.source.reference_hash,
                    "approval_evidence_hash": approval["evidence_hash"],
                }
            )
            approval_ids.append(approval["approval_id"])
        return PublicationInputs(
            run_id=run_id,
            source_kind=V3_PUBLICATION_SOURCE_KIND,
            source_context_hash=input_envelope.context_hash,
            builder_context=publication_context(context),
            candidates=tuple(candidates),
            approval_ids=tuple(approval_ids),
        )

    def assert_publication_batch_current(
        self,
        batch: PublicationBatch,
    ) -> None:
        """确认构建期间 V3 候选、来源和审批证据没有变化。"""

        if batch.source_kind != V3_PUBLICATION_SOURCE_KIND:
            raise PublicationConflictError("发布批次不是 V3 研究来源")
        for candidate_id, approval_id in zip(
            batch.candidate_ids,
            batch.approval_ids,
            strict=True,
        ):
            candidate = self.get_candidate(batch.run_id, candidate_id)
            if candidate["status"] != "building":
                raise PublicationConflictError(
                    f"候选 {candidate_id} 已离开构建状态"
                )
            approval = self.get_approval(batch.run_id, candidate_id)
            if (
                approval["approval_id"] != approval_id
                or approval["decision"] != "approved"
            ):
                raise ApprovalConflictError("候选批准记录已变化")
            if not self.approval_is_current(batch.run_id, candidate_id):
                raise ApprovalConflictError("审批证据已变化，不能提交发布")

    def get_candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """读取 V3 候选状态及最终批准来源。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        with self._connection() as conn:
            state = self._candidate_state_conn(conn, run_id, candidate_id)
            envelope = self._candidate_envelope_conn(
                conn,
                run_id,
                candidate_id,
                str(state["source_id"]),
            )
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "candidate_hash": envelope.candidate.content_hash,
            "semantic_hash": envelope.candidate.semantic_hash,
            "name": envelope.candidate.name,
            "status": str(state["status"]),
            "error_code": state["error_code"],
            "error_message": state["error_message"],
            "source_id": str(state["source_id"]),
            "candidate_envelope_hash": envelope.envelope_hash,
            "source_reference_hash": envelope.source.reference_hash,
        }

    def list_candidates(self, run_id: str) -> list[dict[str, Any]]:
        """列出已完成最终审批的 V3 候选。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT candidate_id FROM mining_v3.candidate_state "
                "WHERE run_id=? ORDER BY candidate_id",
                [run_id],
            ).fetchall()
        return [self.get_candidate(run_id, str(row[0])) for row in rows]

    def start_publication_batch(
        self,
        batch: PublicationBatch,
    ) -> dict[str, Any]:
        """原子写入共享批次并把 V3 已批准候选推进到构建。"""

        if batch.source_kind != V3_PUBLICATION_SOURCE_KIND:
            raise PublicationConflictError("V3 存储只能启动 V3 研究来源批次")
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._ensure_no_active_publication_conn(conn)
                        input_envelope = self._input_envelope_conn(
                            conn,
                            batch.run_id,
                        )
                        if input_envelope.context_hash != batch.context_hash:
                            raise PublicationConflictError("发布上下文已变化")
                        self._context_conn(conn, batch.run_id)
                        publishable = self._publishable_candidate_ids_conn(
                            conn,
                            batch.run_id,
                        )
                        if set(publishable) != set(batch.candidate_ids):
                            raise PublicationConflictError(
                                "发布批次必须包含该运行内全部已批准候选"
                            )
                        for candidate_id, approval_id in zip(
                            batch.candidate_ids,
                            batch.approval_ids,
                            strict=True,
                        ):
                            self._start_v3_publication_candidate_conn(
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
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise MiningStorageError(
                f"启动 V3 发布批次失败：{error}"
            ) from error
        return result

    def get_publication_batch(self, batch_id: str) -> dict[str, Any]:
        """读取共享发布批次及逐候选记录。"""

        with self._connection() as conn:
            batch = self._publication_batch_row(conn, batch_id)
            if batch["source_kind"] != V3_PUBLICATION_SOURCE_KIND:
                raise PublicationConflictError("发布批次不是 V3 来源")
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
        """列出 V3 运行共享发布批次。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT batch_id FROM mining.publication_batch "
                "WHERE run_id=? AND source_kind=? "
                "ORDER BY started_at DESC LIMIT ?",
                [
                    identifier(run_id, "run_id"),
                    V3_PUBLICATION_SOURCE_KIND,
                    max(1, min(int(limit), 100)),
                ],
            ).fetchall()
        return [self.get_publication_batch(str(row[0])) for row in rows]

    def incomplete_publication_batches(self) -> list[dict[str, Any]]:
        """返回 V3 来源中需要恢复的活动批次。"""

        placeholders = ",".join("?" for _ in PUBLICATION_ACTIVE_STATUSES)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id FROM mining.publication_batch "
                f"WHERE source_kind=? AND status IN ({placeholders}) "
                f"ORDER BY started_at",
                [V3_PUBLICATION_SOURCE_KIND, *PUBLICATION_ACTIVE_STATUSES],
            ).fetchall()
        return [self.get_publication_batch(str(row[0])) for row in rows]

    def next_publication_attempt(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
    ) -> int:
        """返回 V3 候选组合的下一个发布尝试编号。"""

        encoded = canonical_json(sorted(candidate_ids))
        with self._connection() as conn:
            row = conn.execute(
                "SELECT max(attempt) FROM mining.publication_batch "
                "WHERE run_id=? AND source_kind=? AND candidate_ids_json=?",
                [
                    identifier(run_id, "run_id"),
                    V3_PUBLICATION_SOURCE_KIND,
                    encoded,
                ],
            ).fetchone()
        return int(row[0] or 0) + 1

    def mark_publication_validating(self, batch_id: str) -> dict[str, Any]:
        """把 V3 共享发布批次推进到校验状态。"""

        return self._transition_v3_publication_batch(
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
        """保存 V3 批次已通过的清单和产物哈希。"""

        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    updated = conn.execute(
                        "UPDATE mining.publication_batch SET "
                        "manifest_hash=?, artifact_sha256=? "
                        "WHERE batch_id=? AND source_kind=? "
                        "AND status='validating' RETURNING batch_id",
                        [
                            manifest_hash,
                            artifact_sha256,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError(
                            "V3 发布批次不在 validating 状态"
                        )
                    conn.execute(
                        "UPDATE mining.publication SET manifest_hash=?, "
                        "artifact_sha256=? WHERE batch_id=? AND source_kind=?",
                        [
                            manifest_hash,
                            artifact_sha256,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
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
        """原子完成 V3 共享批次并把候选推进到 published。"""

        timestamp = utc_now_iso()
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["source_kind"] != V3_PUBLICATION_SOURCE_KIND:
                        raise PublicationConflictError("发布批次不是 V3 来源")
                    if batch["status"] != "validating":
                        raise MiningStateError(
                            "只有 validating 批次可以完成发布"
                        )
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
                        state = self._candidate_state_conn(
                            conn,
                            batch["run_id"],
                            candidate_id,
                        )
                        self._transition_v3_candidate_conn(
                            conn,
                            run_id=batch["run_id"],
                            candidate_id=candidate_id,
                            previous="building",
                            target="published",
                            timestamp=timestamp,
                            batch_id=batch_id,
                            source_id=str(state["source_id"]),
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status='published', "
                        "formal_factor_name=?, manifest_hash=?, "
                        "artifact_sha256=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind=?",
                        [
                            ",".join(formal_factor_names),
                            manifest_hash,
                            artifact_sha256,
                            timestamp,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
                    )
                    conn.execute(
                        "UPDATE mining.publication_batch SET status='published', "
                        "manifest_hash=?, artifact_sha256=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind=? "
                        "AND status='validating'",
                        [
                            manifest_hash,
                            artifact_sha256,
                            timestamp,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
                    )
            result = self.get_publication_batch(batch_id)
        return result

    def fail_publication_batch(
        self,
        batch_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        """把失败 V3 批次和构建中候选收口到 build_failed。"""

        timestamp = utc_now_iso()
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    batch = self._publication_batch_row(conn, batch_id)
                    if batch["source_kind"] != V3_PUBLICATION_SOURCE_KIND:
                        raise PublicationConflictError("发布批次不是 V3 来源")
                    if batch["status"] in {"published", "rolled_back"}:
                        return self.get_publication_batch(batch_id)
                    candidate_ids = json.loads(batch["candidate_ids_json"])
                    for candidate_id in candidate_ids:
                        state = self._candidate_state_conn(
                            conn,
                            batch["run_id"],
                            candidate_id,
                            missing_ok=True,
                        )
                        if state is not None and state["status"] == "building":
                            self._transition_v3_candidate_conn(
                                conn,
                                run_id=batch["run_id"],
                                candidate_id=candidate_id,
                                previous="building",
                                target="build_failed",
                                timestamp=timestamp,
                                batch_id=batch_id,
                                source_id=str(state["source_id"]),
                                error_code=error_code,
                                error_message=error_message,
                            )
                    conn.execute(
                        "UPDATE mining.publication SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind=?",
                        [
                            error_code,
                            error_message,
                            timestamp,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
                    )
                    conn.execute(
                        "UPDATE mining.publication_batch SET status='failed', "
                        "error_code=?, error_message=?, finished_at=? "
                        "WHERE batch_id=? AND source_kind=?",
                        [
                            error_code,
                            error_message,
                            timestamp,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                        ],
                    )
            result = self.get_publication_batch(batch_id)
        return result

    def _candidate_envelope(
        self,
        run_id: str,
        candidate_id: str,
        source_id: str,
    ):
        with self._connection() as conn:
            return self._candidate_envelope_conn(
                conn,
                run_id,
                candidate_id,
                source_id,
            )

    def _publishable_candidate_ids_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> tuple[str, ...]:
        placeholders = ",".join("?" for _ in _PUBLISHABLE_STATUSES)
        return tuple(
            str(row[0])
            for row in conn.execute(
                f"SELECT candidate_id FROM mining_v3.candidate_state "
                f"WHERE run_id=? AND status IN ({placeholders}) "
                f"ORDER BY candidate_id",
                [run_id, *_PUBLISHABLE_STATUSES],
            ).fetchall()
        )

    def _start_v3_publication_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch: PublicationBatch,
        candidate_id: str,
        approval_id: str,
    ) -> None:
        state = self._candidate_state_conn(conn, batch.run_id, candidate_id)
        if state["status"] not in _PUBLISHABLE_STATUSES:
            raise PublicationConflictError(f"候选不可发布：{candidate_id}")
        source_id = str(state["source_id"])
        approval = self._approval_row_conn(conn, batch.run_id, candidate_id)
        if (
            approval["approval_id"] != approval_id
            or approval["decision"] != "approved"
            or approval["source_id"] != source_id
        ):
            raise PublicationConflictError(f"候选批准记录无效：{candidate_id}")
        envelope = self._candidate_envelope_conn(
            conn,
            batch.run_id,
            candidate_id,
            source_id,
        )
        review = self._candidate_review_conn(
            conn,
            batch.run_id,
            candidate_id,
            source_id,
        )
        if (
            review.approval_evidence_hash is None
            or review.approval_evidence_hash != approval["evidence_hash"]
            or envelope.envelope_hash != approval["candidate_envelope_hash"]
            or envelope.source.reference_hash
            != approval["source_reference_hash"]
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
            "started_at, finished_at, source_kind, candidate_envelope_hash, "
            "source_reference_hash) "
            "VALUES (?, ?, ?, ?, ?, 'staging', ?, NULL, NULL, NULL, NULL, "
            "NULL, ?, ?, NULL, ?, ?, ?)",
            [
                publication_id,
                batch.batch_id,
                batch.run_id,
                candidate_id,
                approval_id,
                batch.staging_path,
                batch.attempt,
                batch.started_at,
                V3_PUBLICATION_SOURCE_KIND,
                envelope.envelope_hash,
                envelope.source.reference_hash,
            ],
        )
        self._transition_v3_candidate_conn(
            conn,
            run_id=batch.run_id,
            candidate_id=candidate_id,
            previous=str(state["status"]),
            target="building",
            timestamp=batch.started_at,
            batch_id=batch.batch_id,
            source_id=source_id,
        )

    @staticmethod
    def _ensure_no_active_publication_conn(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
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

    def _transition_v3_publication_batch(
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
                        "WHERE batch_id=? AND source_kind=? "
                        "AND status=? RETURNING batch_id",
                        [
                            status,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                            publications_from,
                        ],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError(
                            f"发布批次不能从 {publications_from} 变为 {status}"
                        )
                    conn.execute(
                        "UPDATE mining.publication SET status=? "
                        "WHERE batch_id=? AND source_kind=? AND status=?",
                        [
                            status,
                            batch_id,
                            V3_PUBLICATION_SOURCE_KIND,
                            publications_from,
                        ],
                    )
        return self.get_publication_batch(batch_id)

    @staticmethod
    def _publication_batch_row(
        conn: duckdb.DuckDBPyConnection,
        batch_id: str,
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM mining.publication_batch WHERE batch_id=?",
            [batch_id],
        ).fetchone()
        if row is None:
            raise MiningNotFoundError(f"发布批次不存在：{batch_id}")
        columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=True))


__all__ = ["V3PublicationStoreMixin"]
