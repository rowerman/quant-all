"""V3 冻结上下文、审核资格和单一来源最终审批存储。"""

from __future__ import annotations

import hmac
import json
from typing import Any, Mapping

import duckdb

from mining.contract_evidence import ApprovalDecision
from mining.contract_types import canonical_json, identifier, utc_now_iso
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.search_context import FrozenSearchContext
from mining.storage_lock import database_state_lock
from mining.states import CandidateStateMachine
from mining.v3_contracts import V3_REQUIRED_EVIDENCE_STAGES, CandidateEnvelope
from mining.v3_review import V3CandidateReview


class V3ReviewStoreMixin:
    """为 V3 存储提供上下文、审批和候选状态事实。"""

    def register_context(
        self,
        run_id: str,
        context: FrozenSearchContext,
    ) -> dict[str, Any]:
        """幂等登记与运行输入绑定的冻结搜索上下文。"""

        run_id = identifier(run_id, "run_id")
        if not isinstance(context, FrozenSearchContext):
            raise MiningContractError("context 必须是 FrozenSearchContext")
        encoded = canonical_json(context.to_dict())
        reused = False
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        input_envelope = self._input_envelope_conn(conn, run_id)
                        if input_envelope.context_hash != context.context_hash:
                            raise MiningContractError(
                                "V3 输入与冻结上下文哈希不一致"
                            )
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v3.run_context WHERE run_id=?",
                            [run_id],
                        )
                        if existing is None:
                            conn.execute(
                                "INSERT INTO mining_v3.run_context VALUES "
                                "(?, ?, ?, ?)",
                                [
                                    run_id,
                                    context.context_hash,
                                    encoded,
                                    utc_now_iso(),
                                ],
                            )
                            self._append_v3_event_conn(
                                conn,
                                run_id,
                                "v3_context_registered",
                                "v3_context_registered",
                                {"context_hash": context.context_hash},
                            )
                        elif (
                            existing["context_hash"] != context.context_hash
                            or existing["payload_json"] != encoded
                        ):
                            raise MiningDuplicateError(
                                "V3 运行已绑定不同冻结上下文"
                            )
                        else:
                            reused = True
        except (
            ApprovalConflictError,
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
            MiningStateError,
            MiningStorageError,
        ):
            raise
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise MiningStorageError(f"登记 V3 冻结上下文失败：{error}") from error
        return {
            "run_id": run_id,
            "context_hash": context.context_hash,
            "reused": reused,
        }

    def load_context(self, run_id: str) -> FrozenSearchContext:
        """读取并校验 V3 冻结搜索上下文。"""

        with self._connection() as conn:
            return self._context_conn(conn, identifier(run_id, "run_id"))

    def initialize_candidate_state(
        self,
        run_id: str,
        candidate_id: str,
        source_id: str,
        *,
        status: str = "proposed",
    ) -> dict[str, Any]:
        """为进入 V3 运行链路的候选建立可迁移状态。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        source_id = identifier(source_id, "source_id")
        if status != "proposed":
            raise MiningContractError("V3 候选初始状态必须是 proposed")
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    self._candidate_envelope_conn(
                        conn,
                        run_id,
                        candidate_id,
                        source_id,
                    )
                    existing = self._candidate_state_conn(
                        conn,
                        run_id,
                        candidate_id,
                        missing_ok=True,
                    )
                    if existing is None:
                        conn.execute(
                            "INSERT INTO mining_v3.candidate_state VALUES "
                            "(?, ?, ?, ?, NULL, NULL, ?)",
                            [
                                run_id,
                                candidate_id,
                                source_id,
                                status,
                                utc_now_iso(),
                            ],
                        )
                        reused = False
                    elif existing["source_id"] != source_id:
                        raise MiningDuplicateError(
                            "V3 候选状态已绑定其他来源"
                        )
                    else:
                        reused = True
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "source_id": source_id,
            "status": status,
            "reused": reused,
        }

    def candidate_source_id(self, run_id: str, candidate_id: str) -> str:
        """返回运行候选的唯一来源；多来源候选由最终审批选择来源。"""

        with self._connection() as conn:
            return self._single_candidate_source_conn(
                conn,
                identifier(run_id, "run_id"),
                identifier(candidate_id, "candidate_id"),
            )

    def sync_candidate_state(self, run_id: str, candidate_id: str, target: str) -> dict[str, Any]:
        """把共享运行状态同步到 V3 候选状态表。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        timestamp = utc_now_iso()
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    state = self._candidate_state_conn(conn, run_id, candidate_id)
                    previous = str(state["status"])
                    if previous == target:
                        reused = True
                    else:
                        CandidateStateMachine.ensure(previous, target)
                        updated = conn.execute(
                            "UPDATE mining_v3.candidate_state SET status=?, "
                            "updated_at=? WHERE run_id=? AND candidate_id=? "
                            "AND status=? RETURNING candidate_id",
                            [target, timestamp, run_id, candidate_id, previous],
                        ).fetchall()
                        if not updated:
                            raise MiningStateError(
                                "V3 候选状态已被其他请求修改"
                            )
                        self._append_v3_event_conn(
                            conn,
                            run_id,
                            f"v3_runtime_candidate_{candidate_id}_{target}",
                            "v3_candidate_state_changed",
                            {
                                "candidate_id": candidate_id,
                                "from": previous,
                                "to": target,
                            },
                            created_at=timestamp,
                        )
                        reused = False
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "status": target,
            "reused": reused,
        }

    def candidate_review(
        self,
        run_id: str,
        candidate_id: str,
        source_id: str,
    ) -> V3CandidateReview:
        """返回一个候选来源的当前审批资格。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        source_id = identifier(source_id, "source_id")
        with self._connection() as conn:
            review = self._candidate_review_conn(
                conn,
                run_id,
                candidate_id,
                source_id,
            )
        return review

    def list_candidate_reviews(self, run_id: str) -> tuple[V3CandidateReview, ...]:
        """按候选和来源稳定顺序列出全部审核摘要。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            self._input_envelope_conn(conn, run_id)
            rows = conn.execute(
                "SELECT candidate_id, source_id "
                "FROM mining_v3.candidate_source WHERE run_id=? "
                "ORDER BY candidate_id, source_id",
                [run_id],
            ).fetchall()
            return tuple(
                self._candidate_review_conn(
                    conn,
                    run_id,
                    str(candidate_id),
                    str(source_id),
                )
                for candidate_id, source_id in rows
            )

    def record_approval(
        self,
        decision: ApprovalDecision,
        source_id: str,
    ) -> dict[str, Any]:
        """原子保存一个候选来源的最终人工决定。"""

        if not isinstance(decision, ApprovalDecision):
            raise MiningContractError("decision 必须是 ApprovalDecision")
        source_id = identifier(source_id, "source_id")
        reused = False
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        envelope = self._candidate_envelope_conn(
                            conn,
                            decision.run_id,
                            decision.candidate_id,
                            source_id,
                        )
                        review = self._candidate_review_conn(
                            conn,
                            decision.run_id,
                            decision.candidate_id,
                            source_id,
                        )
                        existing = self._approval_row_conn(
                            conn,
                            decision.run_id,
                            decision.candidate_id,
                            missing_ok=True,
                        )
                        if existing is not None:
                            self._check_current_approval(
                                existing,
                                decision,
                                envelope,
                                review,
                                source_id,
                            )
                            reused = True
                        else:
                            if not review.eligible:
                                raise ApprovalConflictError(
                                    "候选当前不能审批："
                                    + ", ".join(review.blocking_reasons)
                                )
                            self._validate_v3_approval_decision(
                                decision,
                                review,
                            )
                            timestamp = decision.created_at
                            conn.execute(
                                "INSERT INTO mining_v3.candidate_approval VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    decision.approval_id,
                                    decision.run_id,
                                    decision.candidate_id,
                                    source_id,
                                    decision.decision,
                                    decision.reviewer,
                                    decision.note,
                                    decision.candidate_content_hash,
                                    decision.semantic_hash,
                                    decision.context_hash,
                                    envelope.source.reference_hash,
                                    envelope.envelope_hash,
                                    decision.evidence_hash,
                                    timestamp,
                                ],
                            )
                            target = (
                                "approved"
                                if decision.decision == "approved"
                                else "rejected"
                            )
                            state = self._candidate_state_conn(
                                conn,
                                decision.run_id,
                                decision.candidate_id,
                                missing_ok=True,
                            )
                            if state is None:
                                conn.execute(
                                    "INSERT INTO mining_v3.candidate_state VALUES "
                                    "(?, ?, ?, ?, NULL, NULL, ?)",
                                    [
                                        decision.run_id,
                                        decision.candidate_id,
                                        source_id,
                                        target,
                                        timestamp,
                                    ],
                                )
                            else:
                                if state["source_id"] != source_id:
                                    raise ApprovalConflictError(
                                        "候选状态绑定了不同来源"
                                    )
                                CandidateStateMachine.ensure(
                                    str(state["status"]),
                                    target,
                                )
                                conn.execute(
                                    "UPDATE mining_v3.candidate_state SET "
                                    "status=?, error_code=NULL, error_message=NULL, "
                                    "updated_at=? WHERE run_id=? AND candidate_id=? "
                                    "AND source_id=?",
                                    [
                                        target,
                                        timestamp,
                                        decision.run_id,
                                        decision.candidate_id,
                                        source_id,
                                    ],
                                )
                            self._append_v3_event_conn(
                                conn,
                                decision.run_id,
                                f"v3_approval_{decision.approval_id}",
                                "v3_candidate_approved",
                                {
                                    "candidate_id": decision.candidate_id,
                                    "source_id": source_id,
                                    "approval_id": decision.approval_id,
                                    "decision": decision.decision,
                                    "evidence_hash": decision.evidence_hash,
                                },
                                created_at=timestamp,
                            )
            result = self.get_approval(
                decision.run_id,
                decision.candidate_id,
            )
        except (
            ApprovalConflictError,
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
            MiningStateError,
            MiningStorageError,
        ):
            raise
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise MiningStorageError(f"写入 V3 候选审批失败：{error}") from error
        result["reused"] = reused
        return result

    def get_approval(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """读取一个 V3 候选最终审批。"""

        with self._connection() as conn:
            row = self._approval_row_conn(
                conn,
                identifier(run_id, "run_id"),
                identifier(candidate_id, "candidate_id"),
            )
        return row

    def approval_is_current(self, run_id: str, candidate_id: str) -> bool:
        """判断最终审批是否仍绑定当前候选包络和证据。"""

        approval = self.get_approval(run_id, candidate_id)
        with self._connection() as conn:
            envelope = self._candidate_envelope_conn(
                conn,
                run_id,
                candidate_id,
                approval["source_id"],
            )
            review = self._candidate_review_conn(
                conn,
                run_id,
                candidate_id,
                approval["source_id"],
            )
        return (
            review.approval_evidence_hash is not None
            and hmac.compare_digest(
                approval["evidence_hash"],
                review.approval_evidence_hash,
            )
            and hmac.compare_digest(
                approval["candidate_envelope_hash"],
                envelope.envelope_hash,
            )
            and hmac.compare_digest(
                approval["source_reference_hash"],
                envelope.source.reference_hash,
            )
        )

    def list_events(
        self,
        run_id: str,
        after: int = 0,
    ) -> list[dict[str, Any]]:
        """读取 V3 运行审计事件。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT sequence, event_key, event_type, payload_json, created_at "
                "FROM mining_v3.event WHERE run_id=? AND sequence>? "
                "ORDER BY sequence",
                [identifier(run_id, "run_id"), max(0, int(after))],
            ).fetchall()
        return [
            {
                "sequence": int(sequence),
                "event_key": event_key,
                "event_type": event_type,
                "payload": json.loads(payload_json),
                "created_at": created_at,
            }
            for sequence, event_key, event_type, payload_json, created_at in rows
        ]

    def _candidate_review_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
        source_id: str,
    ) -> V3CandidateReview:
        input_envelope = self._input_envelope_conn(conn, run_id)
        envelope = self._candidate_envelope_conn(
            conn,
            run_id,
            candidate_id,
            source_id,
        )
        context_row = self._row(
            conn,
            "SELECT context_hash, payload_json "
            "FROM mining_v3.run_context WHERE run_id=?",
            [run_id],
        )
        context_valid = context_row is not None
        if context_row is not None:
            context = FrozenSearchContext.from_dict(
                json.loads(context_row["payload_json"])
            )
            if (
                context.context_hash != context_row["context_hash"]
                or context.context_hash != input_envelope.context_hash
            ):
                raise MiningStorageError("V3 冻结上下文列与正文不一致")

        blockers: list[str] = []
        if not context_valid:
            blockers.append("CONTEXT_NOT_REGISTERED")
        by_stage: dict[str, list[str]] = {}
        for item in envelope.evidence:
            by_stage.setdefault(item.stage, []).append(item.status)
        for stage in V3_REQUIRED_EVIDENCE_STAGES:
            statuses = by_stage.get(stage, [])
            if not statuses:
                blockers.append(f"MISSING_EVIDENCE:{stage}")
            elif any(status != "passed" for status in statuses):
                blockers.append(f"EVIDENCE_NOT_PASSED:{stage}")

        approval = self._approval_row_conn(
            conn,
            run_id,
            candidate_id,
            missing_ok=True,
        )
        if approval is not None:
            blockers.append("FINAL_DECISION_EXISTS")
        state = self._candidate_state_conn(
            conn,
            run_id,
            candidate_id,
            missing_ok=True,
        )
        candidate_status = (
            str(state["status"]) if state is not None else "review_ready"
        )
        approval_hash = None
        if not any(
            reason.startswith(("MISSING_EVIDENCE:", "EVIDENCE_NOT_PASSED:"))
            for reason in blockers
        ) and context_valid:
            approval_hash = envelope.approval_evidence_hash(
                input_envelope.context_hash
            )
        return V3CandidateReview(
            run_id=run_id,
            candidate_id=candidate_id,
            source_id=source_id,
            source_reference_hash=envelope.source.reference_hash,
            candidate_content_hash=envelope.candidate.content_hash,
            semantic_hash=envelope.candidate.semantic_hash,
            context_hash=input_envelope.context_hash,
            candidate_envelope_hash=envelope.envelope_hash,
            candidate_status=candidate_status,
            eligible=not blockers,
            blocking_reasons=tuple(blockers),
            approval_evidence_hash=approval_hash,
            approval_id=(
                str(approval["approval_id"])
                if approval is not None
                else None
            ),
            decision=(
                str(approval["decision"])
                if approval is not None
                else None
            ),
            evidence=envelope.evidence,
        )

    def _single_candidate_source_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
    ) -> str:
        rows = conn.execute(
            "SELECT source_id FROM mining_v3.candidate_source "
            "WHERE run_id=? AND candidate_id=? ORDER BY source_id",
            [run_id, candidate_id],
        ).fetchall()
        if not rows:
            raise MiningNotFoundError(
                f"V3 候选来源不存在：{run_id}/{candidate_id}"
            )
        if len(rows) != 1:
            raise MiningContractError(
                "运行候选必须只绑定一个来源，最终审批后才能选择来源"
            )
        return str(rows[0][0])

    @staticmethod
    def _validate_v3_approval_decision(
        decision: ApprovalDecision,
        review: V3CandidateReview,
    ) -> None:
        expected = {
            "candidate_content_hash": review.candidate_content_hash,
            "semantic_hash": review.semantic_hash,
            "context_hash": review.context_hash,
            "evidence_hash": review.approval_evidence_hash,
        }
        if any(getattr(decision, name) != value for name, value in expected.items()):
            raise ApprovalConflictError("审批依据与当前 V3 候选证据不一致")

    @staticmethod
    def _check_current_approval(
        approval: Mapping[str, Any],
        decision: ApprovalDecision,
        envelope: CandidateEnvelope,
        review: V3CandidateReview,
        source_id: str,
    ) -> None:
        requested = {
            "approval_id": decision.approval_id,
            "source_id": source_id,
            "decision": decision.decision,
            "reviewer": decision.reviewer,
            "note": decision.note,
            "candidate_content_hash": decision.candidate_content_hash,
            "semantic_hash": decision.semantic_hash,
            "context_hash": decision.context_hash,
            "evidence_hash": decision.evidence_hash,
            "candidate_envelope_hash": envelope.envelope_hash,
            "source_reference_hash": envelope.source.reference_hash,
            "created_at": decision.created_at,
        }
        stored = {name: approval[name] for name in requested}
        if stored != requested:
            raise ApprovalConflictError("候选已有不同的最终 V3 审批")
        if review.approval_evidence_hash is None or not hmac.compare_digest(
            str(approval["evidence_hash"]),
            review.approval_evidence_hash,
        ):
            raise ApprovalConflictError("候选审批证据已变化")

    def _approval_row_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.candidate_approval "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        )
        if row is None:
            if missing_ok:
                return None
            raise MiningNotFoundError(
                f"V3 候选尚无审批：{run_id}/{candidate_id}"
            )
        return row

    def _candidate_state_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.candidate_state "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        )
        if row is None and not missing_ok:
            raise MiningNotFoundError(
                f"V3 候选状态不存在：{run_id}/{candidate_id}"
            )
        return row

    def _context_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> FrozenSearchContext:
        input_envelope = self._input_envelope_conn(conn, run_id)
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.run_context WHERE run_id=?",
            [run_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V3 冻结上下文不存在：{run_id}")
        context = FrozenSearchContext.from_dict(json.loads(row["payload_json"]))
        if (
            row["context_hash"] != context.context_hash
            or context.context_hash != input_envelope.context_hash
        ):
            raise MiningStorageError("V3 冻结上下文列与正文不一致")
        return context

    def _transition_v3_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        run_id: str,
        candidate_id: str,
        previous: str,
        target: str,
        timestamp: str,
        batch_id: str,
        source_id: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        CandidateStateMachine.ensure(previous, target)
        updated = conn.execute(
            "UPDATE mining_v3.candidate_state SET status=?, error_code=?, "
            "error_message=?, updated_at=? "
            "WHERE run_id=? AND candidate_id=? AND source_id=? AND status=? "
            "RETURNING candidate_id",
            [
                target,
                error_code,
                error_message,
                timestamp,
                run_id,
                candidate_id,
                source_id,
                previous,
            ],
        ).fetchall()
        if not updated:
            raise MiningStateError("V3 候选状态已被其他请求修改")
        self._append_v3_event_conn(
            conn,
            run_id,
            f"v3_candidate_{candidate_id}_{target}_{batch_id}",
            "v3_candidate_state_changed",
            {
                "candidate_id": candidate_id,
                "source_id": source_id,
                "from": previous,
                "to": target,
                "batch_id": batch_id,
                "error_code": error_code,
            },
            created_at=timestamp,
        )

    def _append_v3_event_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> int:
        encoded = canonical_json(dict(payload))
        existing = self._row(
            conn,
            "SELECT sequence, event_type, payload_json "
            "FROM mining_v3.event WHERE run_id=? AND event_key=?",
            [run_id, event_key],
        )
        if existing is not None:
            if (
                existing["event_type"] != event_type
                or existing["payload_json"] != encoded
            ):
                raise MiningDuplicateError(f"V3 事件键 {event_key} 内容冲突")
            return int(existing["sequence"])
        sequence = int(
            conn.execute(
                "SELECT coalesce(max(sequence), 0) + 1 "
                "FROM mining_v3.event WHERE run_id=?",
                [run_id],
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO mining_v3.event VALUES (?, ?, ?, ?, ?, ?)",
            [
                run_id,
                sequence,
                event_key,
                event_type,
                encoded,
                created_at or utc_now_iso(),
            ],
        )
        return sequence


__all__ = ["V3ReviewStoreMixin"]
