"""V2 审核候选的审批证据、资格和只读视图。"""

from __future__ import annotations

import hmac
import json
from typing import Any

import duckdb

from mining.contract_types import identifier
from mining.errors import MiningNotFoundError, MiningStorageError
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_from_v2_record
from mining.search_publication_contracts import SearchEvidenceReference
from mining.search_review import (
    SEARCH_APPROVAL_REQUIRED_STAGES,
    SearchApprovalEvidence,
    SearchApprovalEligibility,
)


class V2ReviewReadMixin:
    """V2 候选审批证据、资格和最终审批读取。"""

    def search_approval_evidence(
        self,
        run_id: str,
        candidate_id: str,
    ) -> str:
        """返回当前候选级审批证据哈希。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        self.validate_search_trace(run_id)
        with self._connection() as conn:
            snapshot = self._search_approval_snapshot_conn(
                conn,
                run_id,
                candidate_id,
            )
        return snapshot["evidence"].evidence_hash

    def search_approval_eligibility(
        self,
        run_id: str,
        candidate_id: str,
    ) -> SearchApprovalEligibility:
        """返回当前审批资格及可能导致审批失效的证据身份。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        self.validate_search_trace(run_id)
        with self._connection() as conn:
            snapshot = self._search_approval_snapshot_conn(
                conn,
                run_id,
                candidate_id,
            )
            approval = self._row(
                conn,
                "SELECT approval_id, decision FROM mining_v2.search_approval "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        evidence = snapshot["evidence"]
        blockers = self._search_approval_blockers(
            snapshot["run_status"],
            snapshot["candidate_status"],
            evidence,
        )
        return SearchApprovalEligibility(
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_status=snapshot["candidate_status"],
            eligible=not blockers,
            blocking_reasons=blockers,
            candidate_content_hash=evidence.candidate_content_hash,
            semantic_hash=evidence.semantic_hash,
            context_hash=evidence.context_hash,
            evidence_hash=evidence.evidence_hash,
            evidence=evidence.evidence,
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
        )

    def get_search_approval(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """读取 V2 候选最终审批。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.search_approval "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        if row is None:
            raise MiningNotFoundError(
                f"V2 候选尚无审批：{run_id}/{candidate_id}"
            )
        return row

    def search_approval_is_current(
        self,
        run_id: str,
        candidate_id: str,
    ) -> bool:
        """判断已登记审批是否仍绑定当前候选研究证据。"""

        approval = self.get_search_approval(run_id, candidate_id)
        current = self.search_approval_evidence(run_id, candidate_id)
        return hmac.compare_digest(str(approval["evidence_hash"]), current)

    def _search_approval_snapshot_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """在单个连接内重建候选、上下文和全部候选级证据。"""

        run_row = self._row(
            conn,
            "SELECT r.*, c.payload_json AS context_payload_json, "
            "c.context_hash AS stored_context_hash, "
            "ctl.status AS control_status "
            "FROM mining_v2.run r "
            "JOIN mining_v2.context c ON c.context_id=r.context_id "
            "LEFT JOIN mining_v2.run_control ctl ON ctl.run_id=r.run_id "
            "WHERE r.run_id=?",
            [run_id],
        )
        if run_row is None:
            raise MiningNotFoundError(f"V2 搜索运行不存在：{run_id}")
        context = FrozenSearchContext.from_dict(
            json.loads(run_row.pop("context_payload_json"))
        )
        if run_row.pop("stored_context_hash") != context.context_hash:
            raise MiningStorageError("V2 冻结上下文哈希与正文不一致")

        candidate_row = self._row(
            conn,
            "SELECT c.*, l.parent_candidate_id, l.hypothesis_id, l.batch_id "
            "FROM mining_v2.candidate_trace c "
            "JOIN mining_v2.candidate_lineage l "
            "ON l.run_id=c.run_id AND l.candidate_id=c.candidate_id "
            "WHERE c.run_id=? AND c.candidate_id=?",
            [run_id, candidate_id],
        )
        if candidate_row is None:
            raise MiningNotFoundError(f"V2 候选不存在：{candidate_id}")
        record = self._public_row(candidate_row)
        record["payload"] = json.loads(record.pop("payload_json"))
        record["lineage"] = {
            "run_id": record["run_id"],
            "candidate_id": record["candidate_id"],
            "parent_candidate_id": record.pop("parent_candidate_id"),
            "hypothesis_id": record.pop("hypothesis_id"),
            "batch_id": record.pop("batch_id"),
        }
        definition = candidate_from_v2_record(record)
        if (
            record["candidate_hash"] != definition.content_hash
            or record["semantic_hash"] != definition.semantic_hash
            or record["candidate_id"] != definition.candidate_id
            or record["name"] != definition.name
        ):
            raise MiningStorageError("V2 候选内容哈希或身份与冻结定义不一致")

        evidence_rows = conn.execute(
            "SELECT * FROM mining_v2.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? "
            "ORDER BY stage, input_hash, evaluation_id",
            [run_id, candidate_id],
        ).fetchall()
        columns = [item[0] for item in conn.description]
        references = []
        for values in evidence_rows:
            row = self._public_row(
                dict(zip(columns, values, strict=True))
            )
            payload = json.loads(row.pop("payload_json"))
            stored = SearchEvaluationEvidence.from_dict(payload)
            for name in (
                "run_id",
                "candidate_id",
                "stage",
                "input_hash",
                "status",
            ):
                if row[name] != getattr(stored, name):
                    raise MiningStorageError(
                        f"V2 评估证据 {name} 与正文不一致"
                    )
            if (
                row["evaluation_id"] != stored.evaluation_id
                or row["evidence_hash"] != stored.evidence_hash
            ):
                raise MiningStorageError("V2 评估证据身份与正文不一致")
            references.append(
                SearchEvidenceReference(
                    stage=stored.stage,
                    evaluation_id=stored.evaluation_id,
                    input_hash=stored.input_hash,
                    evidence_hash=stored.evidence_hash,
                    status=stored.status,
                )
            )
        evidence = SearchApprovalEvidence(
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_content_hash=definition.content_hash,
            semantic_hash=definition.semantic_hash,
            context_hash=context.context_hash,
            evidence=tuple(references),
        )
        return {
            "run_status": str(
                run_row.pop("control_status") or run_row["status"]
            ),
            "candidate_status": str(record["status"]),
            "evidence": evidence,
        }

    @staticmethod
    def _search_approval_blockers(
        run_status: str,
        candidate_status: str,
        evidence: SearchApprovalEvidence,
    ) -> tuple[str, ...]:
        """按稳定原因码列出不能审批的条件。"""

        reasons = []
        if run_status != "review_ready":
            reasons.append("RUN_NOT_REVIEW_READY")
        if candidate_status != "review_ready":
            reasons.append("CANDIDATE_NOT_REVIEW_READY")
        by_stage: dict[str, list[str]] = {}
        for item in evidence.evidence:
            by_stage.setdefault(item.stage, []).append(item.status)
        for stage in SEARCH_APPROVAL_REQUIRED_STAGES:
            statuses = by_stage.get(stage, [])
            if not statuses:
                reasons.append(f"MISSING_EVIDENCE:{stage}")
            elif any(status != "passed" for status in statuses):
                reasons.append(f"EVIDENCE_NOT_PASSED:{stage}")
        return tuple(reasons)
