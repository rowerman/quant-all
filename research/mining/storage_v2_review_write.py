"""V2 审核候选最终审批的原子写入。"""

from __future__ import annotations

import hmac
from typing import Any

import duckdb

from mining.contract_evidence import ApprovalDecision
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.logging_conf import format_event
from mining.search_review import (
    SearchApprovalEvidence,
    search_approval_record_hash,
)
from mining.storage_lock import database_state_lock


class V2ReviewWriteMixin:
    """V2 候选最终审批、状态迁移和审计写入。"""

    def record_search_approval(
        self,
        decision: ApprovalDecision,
    ) -> dict[str, Any]:
        """原子保存 V2 最终审批并迁移候选状态。"""

        if not isinstance(decision, ApprovalDecision):
            raise MiningContractError("decision 必须是 ApprovalDecision")
        try:
            with database_state_lock(self.db_path):
                self.validate_search_trace(decision.run_id)
                with self._connection() as conn:
                    with self._transaction(conn):
                        snapshot = self._search_approval_snapshot_conn(
                            conn,
                            decision.run_id,
                            decision.candidate_id,
                        )
                        evidence = snapshot["evidence"]
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.search_approval "
                            "WHERE run_id=? AND candidate_id=?",
                            [decision.run_id, decision.candidate_id],
                        )
                        if existing is not None:
                            reused = self._check_existing_search_approval(
                                existing,
                                decision,
                                evidence.evidence_hash,
                            )
                        else:
                            reused = False
                            blockers = self._search_approval_blockers(
                                snapshot["run_status"],
                                snapshot["candidate_status"],
                                evidence,
                            )
                            if blockers:
                                raise ApprovalConflictError(
                                    "候选当前不能审批："
                                    + ", ".join(blockers)
                                )
                            self._validate_search_approval_decision(
                                decision,
                                evidence,
                            )
                            self._insert_search_approval_conn(
                                conn,
                                decision,
                                evidence.evidence_hash,
                            )
                    result = self.get_search_approval(
                        decision.run_id,
                        decision.candidate_id,
                    )
        except (
            ApprovalConflictError,
            MiningContractError,
            MiningNotFoundError,
            MiningStateError,
            MiningStorageError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"写入 V2 候选审批失败：{error}") from error
        result["reused"] = reused
        self.logger.info(
            format_event(
                {
                    "operation": "record_search_approval",
                    "run_id": decision.run_id,
                    "candidate_id": decision.candidate_id,
                    "approval_id": result["approval_id"],
                    "decision": result["decision"],
                    "reused": reused,
                }
            )
        )
        return result

    @staticmethod
    def _validate_search_approval_decision(
        decision: ApprovalDecision,
        evidence: SearchApprovalEvidence,
    ) -> None:
        """确认审批请求绑定当前候选上下文和证据。"""

        expected = {
            "candidate_content_hash": evidence.candidate_content_hash,
            "semantic_hash": evidence.semantic_hash,
            "context_hash": evidence.context_hash,
            "evidence_hash": evidence.evidence_hash,
        }
        actual = {
            name: getattr(decision, name)
            for name in expected
        }
        if actual != expected:
            raise ApprovalConflictError("审批依据与当前候选证据不一致")

    def _check_existing_search_approval(
        self,
        existing: dict[str, Any],
        decision: ApprovalDecision,
        current_evidence_hash: str,
    ) -> bool:
        """校验重复请求；证据变化后不能继续视为幂等成功。"""

        requested = {
            "approval_id": decision.approval_id,
            "decision": decision.decision,
            "reviewer": decision.reviewer,
            "note": decision.note,
            "candidate_content_hash": decision.candidate_content_hash,
            "semantic_hash": decision.semantic_hash,
            "context_hash": decision.context_hash,
            "evidence_hash": decision.evidence_hash,
        }
        stored = {name: existing[name] for name in requested}
        if stored != requested:
            raise ApprovalConflictError("候选已有不同的最终审批")
        if not hmac.compare_digest(
            str(existing["evidence_hash"]),
            current_evidence_hash,
        ):
            raise ApprovalConflictError("候选审批证据已变化")
        return True

    def _insert_search_approval_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        decision: ApprovalDecision,
        evidence_hash: str,
    ) -> None:
        """插入审批，以 CAS 迁移候选并追加防篡改轨迹和事件。"""

        target = "approved" if decision.decision == "approved" else "rejected"
        conn.execute(
            "INSERT INTO mining_v2.search_approval VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                evidence_hash,
                decision.created_at,
            ],
        )
        updated = conn.execute(
            "UPDATE mining_v2.candidate_trace "
            "SET status=?, updated_at=? "
            "WHERE run_id=? AND candidate_id=? AND status='review_ready' "
            "RETURNING candidate_id",
            [
                target,
                decision.created_at,
                decision.run_id,
                decision.candidate_id,
            ],
        ).fetchall()
        if not updated:
            raise MiningStateError("V2 候选状态已被其他请求修改")
        self._append_trace_conn(
            conn,
            decision.run_id,
            "search_approval",
            decision.approval_id,
            search_approval_record_hash(decision),
            decision.created_at,
        )
        self._append_event_conn(
            conn,
            decision.run_id,
            f"search_approval_{decision.approval_id}",
            "search_approval_recorded",
            {
                "candidate_id": decision.candidate_id,
                "approval_id": decision.approval_id,
                "decision": decision.decision,
                "evidence_hash": evidence_hash,
            },
            created_at=decision.created_at,
        )
