"""人工审批、发布记录和一次性测试段存储。"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mining.contracts import (
    ApprovalDecision,
    HOLDOUT_SEGMENT_ID,
    PublicationAttempt,
    canonical_json,
    sha256_json,
    utc_now_iso,
)
from mining.contract_types import identifier, sha256_text, text
from mining.errors import (
    ApprovalConflictError,
    HoldoutError,
    MiningNotFoundError,
    MiningStateError,
)
from mining.logging_conf import format_event
from mining.states import CandidateStateMachine, PublicationStateMachine


class ReviewStoreMixin:
    """审批、发布尝试与测试段领取的持久化方法。"""

    def record_approval(self, decision: ApprovalDecision) -> dict[str, Any]:
        """保存最终人工审批，并同步候选状态。"""

        target = "approved" if decision.decision == "approved" else "rejected"
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    candidate = self._row(
                        conn,
                        "SELECT status, content_hash, semantic_hash FROM mining.candidate "
                        "WHERE run_id=? AND candidate_id=?",
                        [decision.run_id, decision.candidate_id],
                    )
                    if candidate is None:
                        raise MiningNotFoundError(f"候选不存在：{decision.candidate_id}")
                    context = self._context_for_run_conn(conn, decision.run_id)
                    if context is None or context["context_hash"] != decision.context_hash:
                        raise ApprovalConflictError("审批上下文已变化")
                    if (
                        candidate["content_hash"] != decision.candidate_content_hash
                        or candidate["semantic_hash"] != decision.semantic_hash
                    ):
                        raise ApprovalConflictError("候选内容已变化")
                    expected_evidence = self._approval_evidence_conn(
                        conn,
                        run_id=decision.run_id,
                        candidate_id=decision.candidate_id,
                        candidate_content_hash=decision.candidate_content_hash,
                        context_hash=decision.context_hash,
                    )
                    if decision.evidence_hash != expected_evidence:
                        raise ApprovalConflictError("审批证据哈希不一致")
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining.approval WHERE run_id=? AND candidate_id=?",
                        [decision.run_id, decision.candidate_id],
                    )
                    if existing is not None:
                        self._assert_same_row(
                            existing,
                            {
                                "approval_id": decision.approval_id,
                                "decision": decision.decision,
                                "reviewer": decision.reviewer,
                                "note": decision.note,
                                "candidate_content_hash": decision.candidate_content_hash,
                                "semantic_hash": decision.semantic_hash,
                                "context_hash": decision.context_hash,
                                "evidence_hash": decision.evidence_hash,
                            },
                            (
                                "approval_id",
                                "decision",
                                "reviewer",
                                "note",
                                "candidate_content_hash",
                                "semantic_hash",
                                "context_hash",
                                "evidence_hash",
                            ),
                        )
                    else:
                        self._insert_approval_conn(conn, decision, candidate, target)
            result = self.get_approval(decision.run_id, decision.candidate_id)
        self.logger.info(
            format_event(
                {
                    "operation": "record_approval",
                    "run_id": decision.run_id,
                    "candidate_id": decision.candidate_id,
                    "decision": decision.decision,
                    "approval_id": decision.approval_id,
                }
            )
        )
        return result

    def _approval_evidence_conn(
        self,
        conn,
        *,
        run_id: str,
        candidate_id: str,
        candidate_content_hash: str,
        context_hash: str,
    ) -> str:
        """确认门禁和测试段结论存在，并返回其合成证据哈希。"""

        gate = self._row(
            conn,
            "SELECT gate_id, passed, check_hash FROM mining.technical_gate "
            "WHERE run_id=? AND candidate_id=? AND candidate_content_hash=? "
            "AND context_hash=? ORDER BY created_at DESC LIMIT 1",
            [
                run_id,
                candidate_id,
                candidate_content_hash,
                context_hash,
            ],
        )
        if gate is None or not gate["passed"]:
            raise ApprovalConflictError("候选没有通过技术门禁")
        holdouts = conn.execute(
            "SELECT report_id, status FROM mining.evaluation "
            "WHERE run_id=? AND candidate_id=? AND stage='holdout' "
            "AND candidate_content_hash=? AND context_hash=? "
            "ORDER BY started_at, attempt",
            [
                run_id,
                candidate_id,
                candidate_content_hash,
                context_hash,
            ],
        ).fetchall()
        if not holdouts or any(status != "passed" for _, status in holdouts):
            raise ApprovalConflictError("候选尚无通过的模块测试段报告")
        usage = self._row(
            conn,
            "SELECT run_id, candidate_ids_json, batch_hash, evidence_hash "
            "FROM mining.holdout_usage WHERE run_id=?",
            [run_id],
        )
        if usage is None:
            raise ApprovalConflictError("候选尚未领取模块测试段")
        candidate_ids = json.loads(usage["candidate_ids_json"])
        if candidate_id not in candidate_ids:
            raise ApprovalConflictError("候选不在本次测试段领取批次中")
        return sha256_json(
            {
                "gate_hash": gate["check_hash"],
                "holdout_report_ids": sorted(
                    report_id for report_id, _ in holdouts
                ),
                "holdout_batch_hash": usage["batch_hash"],
                "holdout_evidence_hash": usage["evidence_hash"],
            }
        )

    def approval_evidence(
        self,
        run_id: str,
        candidate_id: str,
        *,
        candidate_content_hash: str,
        context_hash: str,
    ) -> str:
        """返回当前可审批证据哈希，供页面预览和提交共同使用。"""

        run_id = self._validate_id(run_id, "run_id")
        candidate_id = self._validate_id(candidate_id, "candidate_id")
        candidate_content_hash = sha256_text(
            candidate_content_hash,
            "candidate_content_hash",
        )
        context_hash = sha256_text(context_hash, "context_hash")
        with self._connection() as conn:
            return self._approval_evidence_conn(
                conn,
                run_id=run_id,
                candidate_id=candidate_id,
                candidate_content_hash=candidate_content_hash,
                context_hash=context_hash,
            )

    def _insert_approval_conn(
        self,
        conn,
        decision: ApprovalDecision,
        candidate: dict[str, Any],
        target: str,
    ) -> None:
        """插入审批并执行带 CAS 的候选状态更新。"""

        if candidate["status"] != "review_ready":
            raise ApprovalConflictError(
                f"候选状态 {candidate['status']} 不允许审批"
            )
        timestamp = utc_now_iso()
        conn.execute(
            "INSERT INTO mining.approval VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                decision.approval_id,
                decision.run_id,
                decision.candidate_id,
                decision.decision,
                decision.reviewer,
                decision.note,
                decision.candidate_content_hash,
                decision.semantic_hash,
                decision.context_hash,
                decision.evidence_hash,
                timestamp,
            ],
        )
        CandidateStateMachine.ensure("review_ready", target)
        updated = conn.execute(
            "UPDATE mining.candidate SET status=?, updated_at=?, "
            "finished_at=? WHERE run_id=? AND candidate_id=? "
            "AND status='review_ready' RETURNING candidate_id",
            [
                target,
                timestamp,
                timestamp if target == "rejected" else None,
                decision.run_id,
                decision.candidate_id,
            ],
        ).fetchall()
        if not updated:
            raise MiningStateError("候选状态已被其他请求修改")
        self._append_event_conn(
            conn,
            decision.run_id,
            f"approval_{decision.approval_id}",
            "review",
            "approval_recorded",
            {
                "candidate_id": decision.candidate_id,
                "approval_id": decision.approval_id,
                "decision": decision.decision,
                "evidence_hash": decision.evidence_hash,
            },
            created_at=timestamp,
        )

    def get_approval(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """读取候选最终审批。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.approval WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        if row is None:
            raise MiningNotFoundError(f"候选尚无审批：{candidate_id}")
        return row

    def record_publication(
        self,
        attempt: PublicationAttempt,
    ) -> dict[str, Any]:
        """保存发布尝试；不执行正式因子写入。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    approval = self._row(
                        conn,
                        "SELECT approval_id, decision FROM mining.approval "
                        "WHERE run_id=? AND candidate_id=?",
                        [attempt.run_id, attempt.candidate_id],
                    )
                    if approval is None or approval["decision"] != "approved":
                        raise ApprovalConflictError("候选没有可用的批准记录")
                    if approval["approval_id"] != attempt.approval_id:
                        raise ApprovalConflictError("发布尝试绑定错误审批")
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining.publication WHERE publication_id=?",
                        [attempt.publication_id],
                    )
                    if existing is not None:
                        self._assert_same_row(
                            existing,
                            self._publication_row(attempt),
                            self._publication_columns(),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO mining.publication "
                            "(publication_id, run_id, candidate_id, approval_id, "
                            "status, staging_path, manifest_hash, formal_factor_name, "
                            "artifact_sha256, error_code, error_message, attempt, "
                            "started_at, finished_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            self._publication_values(attempt),
                        )
                        self._append_event_conn(
                            conn,
                            attempt.run_id,
                            f"publication_{attempt.publication_id}",
                            "publication",
                            "publication_recorded",
                            {
                                "candidate_id": attempt.candidate_id,
                                "publication_id": attempt.publication_id,
                                "status": attempt.status,
                            },
                        )
            result = self.get_publication(attempt.publication_id)
        self.logger.info(
            format_event(
                {
                    "operation": "record_publication",
                    "run_id": attempt.run_id,
                    "candidate_id": attempt.candidate_id,
                    "publication_id": attempt.publication_id,
                    "status": attempt.status,
                }
            )
        )
        return result

    @staticmethod
    def _publication_columns() -> tuple[str, ...]:
        """返回发布表字段顺序。"""

        return (
            "publication_id",
            "run_id",
            "candidate_id",
            "approval_id",
            "status",
            "staging_path",
            "manifest_hash",
            "formal_factor_name",
            "artifact_sha256",
            "error_code",
            "error_message",
            "attempt",
            "started_at",
            "finished_at",
        )

    @classmethod
    def _publication_values(cls, attempt: PublicationAttempt) -> list[Any]:
        """把发布尝试转为字段值。"""

        return [
            attempt.publication_id,
            attempt.run_id,
            attempt.candidate_id,
            attempt.approval_id,
            attempt.status,
            attempt.staging_path,
            attempt.manifest_hash,
            attempt.formal_factor_name,
            attempt.artifact_sha256,
            attempt.error_code,
            attempt.error_message,
            attempt.attempt,
            attempt.started_at,
            attempt.finished_at,
        ]

    @classmethod
    def _publication_row(cls, attempt: PublicationAttempt) -> dict[str, Any]:
        """返回发布尝试的字段映射。"""

        return dict(
            zip(
                cls._publication_columns(),
                cls._publication_values(attempt),
                strict=True,
            )
        )

    def get_publication(self, publication_id: str) -> dict[str, Any]:
        """读取发布尝试。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.publication WHERE publication_id=?",
                [publication_id],
            )
        if row is None:
            raise MiningNotFoundError(f"发布尝试不存在：{publication_id}")
        return row

    def transition_publication(
        self,
        publication_id: str,
        status: str,
        *,
        event_key: str,
        manifest_hash: str | None = None,
        formal_factor_name: str | None = None,
        artifact_sha256: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """迁移发布尝试状态；正式构建由 S8 调用。"""

        publication_id = self._validate_id(publication_id, "publication_id")
        event_key = self._validate_id(event_key, "event_key")
        manifest_hash = (
            sha256_text(manifest_hash, "manifest_hash")
            if manifest_hash is not None
            else None
        )
        artifact_sha256 = (
            sha256_text(artifact_sha256, "artifact_sha256")
            if artifact_sha256 is not None
            else None
        )
        formal_factor_name = (
            identifier(formal_factor_name, "formal_factor_name")
            if formal_factor_name is not None
            else None
        )
        error_code = (
            text(error_code, "error_code", maximum=200)
            if error_code is not None
            else None
        )
        error_message = (
            text(error_message, "error_message", maximum=2000)
            if error_message is not None
            else None
        )
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._row(
                        conn,
                        "SELECT run_id, status FROM mining.publication "
                        "WHERE publication_id=?",
                        [publication_id],
                    )
                    if current is None:
                        raise MiningNotFoundError(
                            f"发布尝试不存在：{publication_id}"
                        )
                    previous = current["status"]
                    if previous == status:
                        event = self._row(
                            conn,
                            "SELECT sequence FROM mining.event "
                            "WHERE run_id=? AND event_key=?",
                            [current["run_id"], event_key],
                        )
                        if event is None:
                            raise MiningStateError(f"发布尝试已经处于 {status}")
                    else:
                        PublicationStateMachine.ensure(previous, status)
                        timestamp = utc_now_iso()
                        finished = timestamp if status in {
                            "published",
                            "failed",
                            "rolled_back",
                        } else None
                        updated = conn.execute(
                            "UPDATE mining.publication SET status=?, "
                            "manifest_hash=coalesce(?, manifest_hash), "
                            "formal_factor_name=coalesce(?, formal_factor_name), "
                            "artifact_sha256=coalesce(?, artifact_sha256), "
                            "error_code=coalesce(?, error_code), "
                            "error_message=coalesce(?, error_message), "
                            "finished_at=coalesce(?, finished_at) "
                            "WHERE publication_id=? AND status=? "
                            "RETURNING publication_id",
                            [
                                status,
                                manifest_hash,
                                formal_factor_name,
                                artifact_sha256,
                                error_code,
                                error_message,
                                finished,
                                publication_id,
                                previous,
                            ],
                        ).fetchall()
                        if not updated:
                            raise MiningStateError("发布状态已被其他请求修改")
                        self._append_event_conn(
                            conn,
                            current["run_id"],
                            event_key,
                            "publication",
                            "publication_status_changed",
                            {
                                "publication_id": publication_id,
                                "from": previous,
                                "to": status,
                                "error_code": error_code,
                            },
                            created_at=timestamp,
                        )
            result = self.get_publication(publication_id)
        self.logger.info(
            format_event(
                {
                    "operation": "transition_publication",
                    "publication_id": publication_id,
                    "status": status,
                    "error_code": error_code,
                }
            )
        )
        return result

    def claim_holdout(
        self,
        *,
        run_id: str,
        candidate_ids: Sequence[str],
        evidence_hash: str,
        segment_id: str = HOLDOUT_SEGMENT_ID,
    ) -> dict[str, Any]:
        """以唯一批次领取模块测试段，同证据重复调用保持幂等。"""

        run_id = self._validate_id(run_id, "run_id")
        segment_id = self._validate_id(segment_id, "segment_id")
        evidence_hash = sha256_text(evidence_hash, "evidence_hash")
        candidates = sorted(
            {self._validate_id(value, "candidate_id") for value in candidate_ids}
        )
        if not candidates:
            raise HoldoutError("测试段批次至少需要一个候选")
        batch_hash = sha256_json(
            {
                "segment_id": segment_id,
                "run_id": run_id,
                "candidate_ids": candidates,
            }
        )
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining.holdout_usage WHERE segment_id=?",
                        [segment_id],
                    )
                    if existing is not None:
                        if (
                            existing["batch_hash"] == batch_hash
                            and existing["evidence_hash"] == evidence_hash
                        ):
                            existing["candidate_ids"] = json.loads(
                                existing.pop("candidate_ids_json")
                            )
                            return existing
                        raise HoldoutError("模块测试段已被其他批次或证据领取")
                    for candidate_id in candidates:
                        candidate = self._row(
                            conn,
                            "SELECT status FROM mining.candidate "
                            "WHERE run_id=? AND candidate_id=?",
                            [run_id, candidate_id],
                        )
                        if candidate is None:
                            raise MiningNotFoundError(f"候选不存在：{candidate_id}")
                        if candidate["status"] not in {
                            "holdout_running",
                            "evaluated",
                        }:
                            raise HoldoutError(
                                f"候选 {candidate_id} 尚未冻结到可领取测试段的阶段"
                            )
                    timestamp = utc_now_iso()
                    conn.execute(
                        "INSERT INTO mining.holdout_usage VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            segment_id,
                            run_id,
                            batch_hash,
                            canonical_json(candidates),
                            evidence_hash,
                            timestamp,
                        ],
                    )
                    self._append_event_conn(
                        conn,
                        run_id,
                        "holdout_claimed",
                        "holdout",
                        "holdout_claimed",
                        {
                            "segment_id": segment_id,
                            "batch_hash": batch_hash,
                            "candidate_ids": candidates,
                        },
                        created_at=timestamp,
                    )
            return self.get_holdout_usage(segment_id)

    def get_holdout_usage(
        self,
        segment_id: str = HOLDOUT_SEGMENT_ID,
    ) -> dict[str, Any]:
        """读取模块测试段领取记录。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.holdout_usage WHERE segment_id=?",
                [segment_id],
            )
        if row is None:
            raise MiningNotFoundError(f"测试段尚未领取：{segment_id}")
        row["candidate_ids"] = json.loads(row.pop("candidate_ids_json"))
        return row
