"""挖掘 V2 搜索轨迹读取和链式完整性校验。"""

from __future__ import annotations

import json

import duckdb

from mining.contract_types import sha256_json
from mining.errors import MiningNotFoundError, MiningStorageError
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evidence import EvidencePacket, SearchDecision
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_trials import TrialRecord
from mining.storage_v2_trace_states import ZERO_HASH


class V2TraceValidationMixin:
    """完整轨迹读取和篡改检测。"""

    def get_search_trace(self, run_id: str) -> dict:
        """读取并校验完整 V2 搜索轨迹。"""

        run = self.get_run(run_id)
        with self._connection() as conn:
            hypotheses = self._payload_rows_conn(
                conn,
                "SELECT payload_json FROM mining_v2.hypothesis "
                "WHERE run_id=? ORDER BY turn_index, hypothesis_id",
                run_id,
            )
            batches = self._payload_rows_conn(
                conn,
                "SELECT payload_json FROM mining_v2.experiment_batch "
                "WHERE run_id=? ORDER BY turn_index, batch_id",
                run_id,
            )
            candidates = self._candidate_rows_conn(conn, run_id)
            turn_rows = conn.execute(
                "SELECT * FROM mining_v2.turn WHERE run_id=? ORDER BY turn_index",
                [run_id],
            ).fetchall()
            turn_columns = [item[0] for item in conn.description]
            turns = [
                self._public_row(dict(zip(turn_columns, values, strict=True)))
                for values in turn_rows
            ]
            evidence = self._payload_rows_conn(
                conn,
                "SELECT payload_json FROM mining_v2.evidence_packet "
                "WHERE run_id=? ORDER BY turn_index",
                run_id,
            )
            decisions = self._payload_rows_conn(
                conn,
                "SELECT payload_json FROM mining_v2.search_decision "
                "WHERE run_id=? ORDER BY turn_index",
                run_id,
            )
            trial_count = conn.execute(
                "SELECT count(*) FROM mining_v2.trial_record WHERE run_id=?",
                [run_id],
            ).fetchone()[0]
        self.validate_search_trace(run_id)
        return {
            "run": run,
            "hypotheses": hypotheses,
            "experiment_batches": batches,
            "candidates": candidates,
            "turns": turns,
            "evidence": evidence,
            "decisions": decisions,
            "trial_count": int(trial_count),
            "trace_hash": run["trace_hash"],
        }

    def validate_search_trace(self, run_id: str) -> bool:
        """复算链式哈希并校验可重建对象的当前内容。"""

        with self._connection() as conn:
            run = self._row(
                conn,
                "SELECT trace_hash FROM mining_v2.run WHERE run_id=?",
                [run_id],
            )
            if run is None:
                raise MiningNotFoundError(f"V2 搜索运行不存在：{run_id}")
            entries = conn.execute(
                "SELECT sequence, entry_type, entry_id, entry_hash, "
                "previous_hash, trace_hash FROM mining_v2.trace_entry "
                "WHERE run_id=? ORDER BY sequence",
                [run_id],
            ).fetchall()
            previous = ZERO_HASH
            for sequence, entry_type, entry_id, entry_hash, stored_previous, stored_trace in entries:
                if stored_previous != previous:
                    raise MiningStorageError(
                        f"V2 轨迹第 {sequence} 条 previous_hash 不一致"
                    )
                current_hash = self._entry_hash_conn(
                    conn,
                    run_id,
                    entry_type,
                    entry_id,
                )
                if current_hash is not None and current_hash != entry_hash:
                    raise MiningStorageError(
                        f"V2 轨迹第 {sequence} 条内容哈希不一致"
                    )
                trace_hash = sha256_json(
                    {
                        "previous_hash": previous,
                        "entry_type": entry_type,
                        "entry_id": entry_id,
                        "entry_hash": entry_hash,
                    }
                )
                if trace_hash != stored_trace:
                    raise MiningStorageError(
                        f"V2 轨迹第 {sequence} 条 trace_hash 不一致"
                    )
                previous = trace_hash
            expected = previous if entries else None
            if run["trace_hash"] != expected:
                raise MiningStorageError("V2 运行 trace_hash 与轨迹链不一致")
        return True

    def _entry_hash_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        entry_type: str,
        entry_id: str,
    ) -> str | None:
        if entry_type == "hypothesis":
            row = self._require_hypothesis_conn(conn, run_id, entry_id)
            return HypothesisCard.from_dict(json.loads(row["payload_json"])).card_hash
        if entry_type == "experiment_batch":
            row = self._require_batch_conn(conn, run_id, entry_id)
            return ExperimentBatch.from_dict(json.loads(row["payload_json"])).batch_hash
        if entry_type == "evidence_packet":
            payload = self._entity_payload_conn(
                conn,
                "mining_v2.evidence_packet",
                "evidence_id",
                run_id,
                entry_id,
            )
            return EvidencePacket.from_dict(payload).evidence_hash
        if entry_type == "search_decision":
            payload = self._entity_payload_conn(
                conn,
                "mining_v2.search_decision",
                "decision_id",
                run_id,
                entry_id,
            )
            return SearchDecision.from_dict(payload).decision_hash
        if entry_type == "candidate_trace":
            row = self._require_candidate_conn(conn, run_id, entry_id)
            return self._candidate_entry_hash(
                row["candidate_id"],
                row["candidate_hash"],
                row["semantic_hash"],
                row["name"],
                row["status"],
                json.loads(row["payload_json"]),
            )
        if entry_type == "trial_record":
            payload = self._entity_payload_conn(
                conn,
                "mining_v2.trial_record",
                "trial_id",
                run_id,
                entry_id,
            )
            return TrialRecord.from_dict(payload).trial_hash
        if entry_type == "trial_invalidation":
            row = conn.execute(
                "SELECT trial_id, reason_code "
                "FROM mining_v2.trial_invalidation "
                "WHERE run_id=? AND invalidation_id=?",
                [run_id, entry_id],
            ).fetchone()
            if row is None:
                raise MiningStorageError(f"轨迹作废记录不存在：{entry_id}")
            return sha256_json(
                {
                    "trial_id": row[0],
                    "reason_code": row[1],
                }
            )
        if entry_type == "evaluation_evidence":
            payload = self._entity_payload_conn(
                conn,
                "mining_v2.evaluation_evidence",
                "evaluation_id",
                run_id,
                entry_id,
            )
            return SearchEvaluationEvidence.from_dict(payload).evidence_hash
        return None

    def _candidate_rows_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> list[dict]:
        rows = conn.execute(
            "SELECT c.*, l.parent_candidate_id, l.hypothesis_id, l.batch_id "
            "FROM mining_v2.candidate_trace c "
            "JOIN mining_v2.candidate_lineage l "
            "ON l.run_id=c.run_id AND l.candidate_id=c.candidate_id "
            "WHERE c.run_id=? ORDER BY c.created_at, c.candidate_id",
            [run_id],
        ).fetchall()
        columns = [item[0] for item in conn.description]
        result = []
        for values in rows:
            item = self._public_row(dict(zip(columns, values, strict=True)))
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    @staticmethod
    def _payload_rows_conn(
        conn: duckdb.DuckDBPyConnection,
        sql: str,
        run_id: str,
    ) -> list[dict]:
        return [json.loads(value) for (value,) in conn.execute(sql, [run_id]).fetchall()]

    @staticmethod
    def _entity_payload_conn(
        conn: duckdb.DuckDBPyConnection,
        table: str,
        id_column: str,
        run_id: str,
        entry_id: str,
    ) -> dict:
        row = conn.execute(
            f"SELECT payload_json FROM {table} WHERE run_id=? AND {id_column}=?",
            [run_id, entry_id],
        ).fetchone()
        if row is None:
            raise MiningStorageError(f"轨迹对象不存在：{entry_id}")
        return json.loads(row[0])
