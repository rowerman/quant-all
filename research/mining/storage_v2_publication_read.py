"""V2 发布输入校验与批次只读查询。"""

from __future__ import annotations

import json
from typing import Any, Sequence

import duckdb

from mining.contract_types import canonical_json
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    MiningStorageError,
    PublicationConflictError,
)
from mining.publication_contracts import (
    PUBLICATION_ACTIVE_STATUSES,
    PublicationBatch,
    PublicationInputs,
)
from mining.search_context import FrozenSearchContext
from mining.search_orchestrator_support import candidate_from_v2_record
from mining.search_publication_contracts import publication_context
from mining.search_scorecard import build_candidate_scorecard

PUBLISHABLE_STATUSES = ("approved", "build_failed")


class V2PublicationReadMixin:
    """读取 V2 已批准候选并核对整批发布资格。"""

    @property
    def publication_source_kind(self) -> str:
        """返回 V2 搜索发布来源。"""

        return "v2_search"

    def load_publication_inputs(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
        *,
        expected_context_hash: str,
    ) -> PublicationInputs:
        """校验 V2 整批发布资格并返回共享发布器输入。"""

        if isinstance(candidate_ids, str) or not isinstance(
            candidate_ids,
            Sequence,
        ):
            raise MiningContractError("candidate_ids 必须是标识序列")
        selected = tuple(sorted(dict.fromkeys(str(item) for item in candidate_ids)))
        if not selected:
            raise PublicationConflictError("发布批次至少需要一个候选")
        run = self.get_run(run_id)
        if run["status"] != "review_ready":
            raise PublicationConflictError(
                f"V2 运行状态 {run['status']} 不能进入发布"
            )
        if expected_context_hash != run["context_hash"]:
            raise PublicationConflictError("页面上下文已变化，请刷新")
        self.validate_search_trace(run_id)
        with self._connection() as conn:
            rows = self._candidate_rows_conn(conn, run_id)
        by_id = {row["candidate_id"]: row for row in rows}
        publishable = {
            row["candidate_id"]
            for row in rows
            if row["status"] in PUBLISHABLE_STATUSES
        }
        if publishable != set(selected):
            raise PublicationConflictError(
                "发布批次必须包含该运行内全部已批准候选"
            )

        candidates = []
        approval_ids = []
        for candidate_id in selected:
            record = by_id.get(candidate_id)
            if record is None:
                raise MiningNotFoundError(f"V2 候选不存在：{candidate_id}")
            definition = candidate_from_v2_record(record)
            if (
                record["candidate_hash"] != definition.content_hash
                or record["semantic_hash"] != definition.semantic_hash
                or record["name"] != definition.name
            ):
                raise MiningStorageError("V2 候选身份与冻结定义不一致")
            approval = self.get_search_approval(run_id, candidate_id)
            if approval["decision"] != "approved":
                raise PublicationConflictError("候选没有批准记录")
            current_evidence = self.search_approval_evidence(
                run_id,
                candidate_id,
            )
            if current_evidence != approval["evidence_hash"]:
                raise ApprovalConflictError("审批证据已变化，不能发布")
            scorecard = build_candidate_scorecard(
                self,
                run_id,
                candidate_id,
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "content_hash": definition.content_hash,
                    "semantic_hash": definition.semantic_hash,
                    "name": definition.name,
                    "status": record["status"],
                    "definition": dict(record["payload"]),
                    "source_kind": "v2_search",
                    "source_context_hash": run["context_hash"],
                    "source_trace_hash": run["trace_hash"],
                    "scorecard_hash": scorecard.scorecard_hash,
                    "approval_evidence_hash": current_evidence,
                }
            )
            approval_ids.append(approval["approval_id"])
        context = FrozenSearchContext.from_dict(run["context"])
        return PublicationInputs(
            run_id=run_id,
            source_kind="v2_search",
            source_context_hash=run["context_hash"],
            builder_context=publication_context(context),
            candidates=tuple(candidates),
            approval_ids=tuple(approval_ids),
        )

    def assert_publication_batch_current(
        self,
        batch: PublicationBatch,
    ) -> None:
        """确认构建期间 V2 候选和审批证据没有变化。"""

        if batch.source_kind != "v2_search":
            raise PublicationConflictError("发布批次不是 V2 搜索来源")
        self.validate_search_trace(batch.run_id)
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
            approval = self.get_search_approval(
                batch.run_id,
                candidate_id,
            )
            if (
                approval["approval_id"] != approval_id
                or approval["decision"] != "approved"
            ):
                raise ApprovalConflictError("候选批准记录已变化")
            if not self.search_approval_is_current(
                batch.run_id,
                candidate_id,
            ):
                raise ApprovalConflictError("审批证据已变化，不能提交发布")

    def get_publication_batch(self, batch_id: str) -> dict[str, Any]:
        """读取发布批次及逐候选发布记录。"""

        with self._connection() as conn:
            batch = self._publication_batch_row(conn, batch_id)
            if batch["source_kind"] != "v2_search":
                raise PublicationConflictError("发布批次不是 V2 来源")
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
        """列出 V2 运行的发布批次。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT batch_id FROM mining.publication_batch "
                "WHERE run_id=? AND source_kind=? "
                "ORDER BY started_at DESC LIMIT ?",
                [run_id, "v2_search", max(1, min(int(limit), 100))],
            ).fetchall()
        return [self.get_publication_batch(row[0]) for row in rows]

    def incomplete_publication_batches(self) -> list[dict[str, Any]]:
        """返回 V2 来源中需要启动恢复的批次。"""

        placeholders = ",".join("?" for _ in PUBLICATION_ACTIVE_STATUSES)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id FROM mining.publication_batch "
                f"WHERE source_kind=? AND status IN ({placeholders}) "
                f"ORDER BY started_at",
                ["v2_search", *PUBLICATION_ACTIVE_STATUSES],
            ).fetchall()
        return [self.get_publication_batch(row[0]) for row in rows]

    def next_publication_attempt(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
    ) -> int:
        """返回 V2 当前候选组合的下一个发布尝试编号。"""

        encoded = canonical_json(sorted(candidate_ids))
        with self._connection() as conn:
            row = conn.execute(
                "SELECT max(attempt) FROM mining.publication_batch "
                "WHERE run_id=? AND source_kind=? AND candidate_ids_json=?",
                [run_id, "v2_search", encoded],
            ).fetchone()
        return int(row[0] or 0) + 1

    def _publication_batch_row(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining.publication_batch WHERE batch_id=?",
            [batch_id],
        )
        if row is None:
            raise MiningNotFoundError(f"发布批次不存在：{batch_id}")
        return row
