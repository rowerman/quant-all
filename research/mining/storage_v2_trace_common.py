"""挖掘 V2 轨迹共用的记录读取、插入和链式哈希工具。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contract_types import canonical_json, sha256_json
from mining.errors import (
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
)
from mining.search_budget import SearchBudget
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.storage_v2_trace_states import TURN_TRANSITIONS, ZERO_HASH


class V2TraceCommonMixin:
    """轨迹 mixin 共用的持久化原语。"""

    def _insert_hypothesis_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        hypothesis: HypothesisCard,
    ) -> bool:
        if hypothesis.parent_hypothesis_id is not None:
            self._require_hypothesis_conn(
                conn,
                hypothesis.run_id,
                hypothesis.parent_hypothesis_id,
            )
        existing = self._row(
            conn,
            "SELECT card_hash FROM mining_v2.hypothesis "
            "WHERE run_id=? AND hypothesis_id=?",
            [hypothesis.run_id, hypothesis.hypothesis_id],
        )
        if existing is not None:
            if existing["card_hash"] != hypothesis.card_hash:
                raise MiningDuplicateError("假设 ID 已绑定不同内容")
            return False
        try:
            conn.execute(
                "INSERT INTO mining_v2.hypothesis VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    hypothesis.run_id,
                    hypothesis.hypothesis_id,
                    hypothesis.parent_hypothesis_id,
                    hypothesis.turn_index,
                    hypothesis.branch_id,
                    hypothesis.card_hash,
                    canonical_json(hypothesis.to_dict()),
                    hypothesis.created_at,
                ],
            )
        except duckdb.ConstraintException as error:
            raise MiningDuplicateError("运行内已有相同假设内容") from error
        return True

    def _insert_batch_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch: ExperimentBatch,
    ) -> None:
        self._require_hypothesis_conn(conn, batch.run_id, batch.hypothesis_id)
        existing = self._row(
            conn,
            "SELECT batch_hash FROM mining_v2.experiment_batch "
            "WHERE run_id=? AND batch_id=?",
            [batch.run_id, batch.batch_id],
        )
        if existing is not None:
            if existing["batch_hash"] != batch.batch_hash:
                raise MiningDuplicateError("实验批次 ID 已绑定不同内容")
            return
        try:
            conn.execute(
                "INSERT INTO mining_v2.experiment_batch VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    batch.run_id,
                    batch.batch_id,
                    batch.hypothesis_id,
                    batch.turn_index,
                    batch.branch_id,
                    batch.batch_hash,
                    canonical_json(list(batch.candidate_ids)),
                    canonical_json(list(batch.parent_candidate_ids)),
                    batch.input_hash,
                    canonical_json(batch.to_dict()),
                    batch.created_at,
                ],
            )
        except duckdb.ConstraintException as error:
            raise MiningDuplicateError("运行内已有相同实验批次") from error

    def _require_run_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.run WHERE run_id=?",
            [run_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V2 搜索运行不存在：{run_id}")
        return row

    def _require_hypothesis_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        hypothesis_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.hypothesis "
            "WHERE run_id=? AND hypothesis_id=?",
            [run_id, hypothesis_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V2 假设不存在：{hypothesis_id}")
        return row

    def _require_batch_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        batch_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.experiment_batch "
            "WHERE run_id=? AND batch_id=?",
            [run_id, batch_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V2 实验批次不存在：{batch_id}")
        return row

    def _require_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.candidate_trace "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V2 候选不存在：{candidate_id}")
        return row

    def _require_turn_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        turn_index: int,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.turn WHERE run_id=? AND turn_index=?",
            [run_id, int(turn_index)],
        )
        if row is None:
            raise MiningNotFoundError(f"V2 turn 不存在：{run_id}/{turn_index}")
        return row

    def _update_turn_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        turn_index: int,
        target: str,
        *,
        evidence_id: str | None = None,
        decision_id: str | None = None,
        updated_at: str,
    ) -> None:
        current = self._require_turn_conn(conn, run_id, turn_index)
        if target not in TURN_TRANSITIONS[current["status"]]:
            raise MiningStateError(
                f"turn 不能从 {current['status']} 变为 {target}"
            )
        conn.execute(
            "UPDATE mining_v2.turn SET status=?, evidence_id=coalesce(?, evidence_id), "
            "decision_id=coalesce(?, decision_id), updated_at=? "
            "WHERE run_id=? AND turn_index=? AND status=?",
            [
                target,
                evidence_id,
                decision_id,
                updated_at,
                run_id,
                turn_index,
                current["status"],
            ],
        )

    def _append_trace_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        entry_type: str,
        entry_id: str,
        entry_hash: str,
        created_at: str,
    ) -> str:
        run = self._require_run_conn(conn, run_id)
        previous = run["trace_hash"] or ZERO_HASH
        trace_hash = sha256_json(
            {
                "previous_hash": previous,
                "entry_type": entry_type,
                "entry_id": entry_id,
                "entry_hash": entry_hash,
            }
        )
        sequence = int(
            conn.execute(
                "SELECT coalesce(max(sequence), 0) + 1 "
                "FROM mining_v2.trace_entry WHERE run_id=?",
                [run_id],
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO mining_v2.trace_entry VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                sequence,
                entry_type,
                entry_id,
                entry_hash,
                previous,
                trace_hash,
                created_at,
            ],
        )
        conn.execute(
            "UPDATE mining_v2.run SET trace_hash=?, updated_at=? WHERE run_id=?",
            [trace_hash, created_at, run_id],
        )
        return trace_hash

    def _budget_for_run_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> SearchBudget:
        row = conn.execute(
            "SELECT c.payload_json FROM mining_v2.context c "
            "JOIN mining_v2.run r ON r.context_id=c.context_id "
            "WHERE r.run_id=?",
            [run_id],
        ).fetchone()
        if row is None:
            raise MiningNotFoundError(f"V2 搜索运行不存在：{run_id}")
        payload = json.loads(row[0])
        return SearchBudget.from_dict(payload["budget"])

    def _ensure_turn_budget_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        hypothesis: HypothesisCard,
    ) -> None:
        budget = self._budget_for_run_conn(conn, hypothesis.run_id)
        if hypothesis.turn_index > budget.max_search_turns:
            raise MiningStateError("搜索预算已用尽：max_search_turns")
        count = int(
            conn.execute(
                "SELECT count(*) FROM mining_v2.hypothesis WHERE run_id=?",
                [hypothesis.run_id],
            ).fetchone()[0]
        )
        count -= int(
            conn.execute(
                "SELECT count(*) FROM mining_v2.hypothesis "
                "WHERE run_id=? AND hypothesis_id=?",
                [hypothesis.run_id, hypothesis.hypothesis_id],
            ).fetchone()[0]
        )
        branch_count = int(
            conn.execute(
                "SELECT count(DISTINCT branch_id) FROM mining_v2.hypothesis "
                "WHERE run_id=? AND hypothesis_id<>?",
                [hypothesis.run_id, hypothesis.hypothesis_id],
            ).fetchone()[0]
        )
        branch_cards = int(
            conn.execute(
                "SELECT count(*) FROM mining_v2.hypothesis "
                "WHERE run_id=? AND branch_id=? AND hypothesis_id<>?",
                [
                    hypothesis.run_id,
                    hypothesis.branch_id,
                    hypothesis.hypothesis_id,
                ],
            ).fetchone()[0]
        )
        if branch_cards == 0 and branch_count >= budget.max_hypothesis_branches:
            raise MiningStateError("搜索预算已用尽：max_hypothesis_branches")
        if branch_cards > budget.max_refines_per_branch:
            raise MiningStateError("搜索预算已用尽：max_refines_per_branch")
        if count >= budget.max_search_turns:
            raise MiningStateError("搜索预算已用尽：假设数量")

    @staticmethod
    def _candidate_entry_hash(
        candidate_id: str,
        candidate_hash: str,
        semantic_hash: str,
        name: str,
        status: str,
        payload: Mapping[str, Any],
    ) -> str:
        return sha256_json(
            {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "semantic_hash": semantic_hash,
                "name": name,
                "payload": dict(payload),
            }
        )

    @staticmethod
    def _public_row(row: Mapping[str, Any] | None) -> dict[str, Any]:
        if row is None:
            raise MiningNotFoundError("请求的 V2 轨迹记录不存在")
        return {
            key: value
            for key, value in row.items()
            if key not in {"request_json", "context_payload_json"}
        }
