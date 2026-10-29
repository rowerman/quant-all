"""挖掘 V2 假设、实验批次、turn、证据和决策写入。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.contract_types import canonical_json, identifier, sha256_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.logging_conf import format_event
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_contract_types import hash_text
from mining.search_evidence import EvidencePacket, SearchDecision
from mining.storage_lock import database_state_lock
from mining.storage_v2_trace_states import TURN_STATUSES, TURN_TRANSITIONS


class V2TurnStoreMixin:
    """搜索 turn、假设、批次、证据和决策仓库。"""

    def begin_turn(
        self,
        run_id: str,
        hypothesis: HypothesisCard,
        batch: ExperimentBatch,
        *,
        snapshot_hash: str | None = None,
    ) -> dict[str, Any]:
        """原子登记假设、实验批次和 turn，重复提交返回原记录。"""

        run_id = identifier(run_id, "run_id")
        if hypothesis.run_id != run_id or batch.run_id != run_id:
            raise MiningContractError("假设、实验批次和 turn 必须属于同一运行")
        if batch.hypothesis_id != hypothesis.hypothesis_id:
            raise MiningContractError("实验批次没有绑定当前假设")
        if (batch.turn_index, batch.branch_id) != (
            hypothesis.turn_index,
            hypothesis.branch_id,
        ):
            raise MiningContractError("实验批次与假设的 turn 或分支不一致")
        if snapshot_hash is not None:
            snapshot_hash = hash_text(snapshot_hash, "snapshot_hash")
        request_hash = sha256_json(
            {
                "hypothesis_hash": hypothesis.card_hash,
                "batch_hash": batch.batch_hash,
                "snapshot_hash": snapshot_hash,
            }
        )
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_run_conn(conn, run_id)
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.turn "
                            "WHERE run_id=? AND turn_index=?",
                            [run_id, hypothesis.turn_index],
                        )
                        if existing is not None:
                            if (
                                existing["request_hash"] != request_hash
                                or existing["hypothesis_id"] != hypothesis.hypothesis_id
                                or existing["experiment_batch_id"] != batch.batch_id
                            ):
                                raise MiningDuplicateError(
                                    "同一 turn 已绑定不同假设或实验批次"
                                )
                            reused = True
                        else:
                            self._ensure_turn_budget_conn(conn, hypothesis)
                            self._insert_hypothesis_conn(conn, hypothesis)
                            self._insert_batch_conn(conn, batch)
                            now = utc_now_iso()
                            conn.execute(
                                "INSERT INTO mining_v2.turn VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    hypothesis.turn_index,
                                    hypothesis.hypothesis_id,
                                    batch.batch_id,
                                    "requested",
                                    request_hash,
                                    None,
                                    None,
                                    snapshot_hash,
                                    now,
                                    now,
                                ],
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "hypothesis",
                                hypothesis.hypothesis_id,
                                hypothesis.card_hash,
                                now,
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "experiment_batch",
                                batch.batch_id,
                                batch.batch_hash,
                                now,
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "turn_state",
                                f"turn:{hypothesis.turn_index}:requested",
                                sha256_json({"status": "requested"}),
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                run_id,
                                f"turn_{hypothesis.turn_index}_requested",
                                "turn_requested",
                                {
                                    "turn_index": hypothesis.turn_index,
                                    "hypothesis_hash": hypothesis.card_hash,
                                    "batch_hash": batch.batch_hash,
                                    "request_hash": request_hash,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.get_turn(run_id, hypothesis.turn_index)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记 V2 turn 失败：{error}") from error
        result["reused"] = reused
        self.logger.info(
            format_event(
                {
                    "operation": "begin_turn",
                    "run_id": run_id,
                    "turn_index": hypothesis.turn_index,
                    "request_hash": request_hash,
                    "reused": reused,
                }
            )
        )
        return result

    def record_evidence(self, evidence: EvidencePacket) -> dict[str, Any]:
        """登记开发期证据并推进 turn。"""

        now = utc_now_iso()
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        turn = self._require_turn_conn(
                            conn,
                            evidence.run_id,
                            evidence.turn_index,
                        )
                        if (
                            turn["hypothesis_id"] != evidence.hypothesis_id
                            or turn["experiment_batch_id"] != evidence.batch_id
                        ):
                            raise MiningContractError("证据没有绑定当前 turn 的假设和批次")
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.evidence_packet "
                            "WHERE run_id=? AND turn_index=?",
                            [evidence.run_id, evidence.turn_index],
                        )
                        if existing is not None:
                            if (
                                existing["evidence_id"] != evidence.evidence_id
                                or existing["evidence_hash"] != evidence.evidence_hash
                            ):
                                raise MiningDuplicateError("同一 turn 已绑定不同证据")
                            reused = True
                        else:
                            conn.execute(
                                "INSERT INTO mining_v2.evidence_packet VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    evidence.run_id,
                                    evidence.turn_index,
                                    evidence.evidence_id,
                                    evidence.hypothesis_id,
                                    evidence.batch_id,
                                    evidence.evidence_hash,
                                    canonical_json(evidence.to_dict()),
                                    now,
                                ],
                            )
                            self._update_turn_conn(
                                conn,
                                evidence.run_id,
                                evidence.turn_index,
                                "evidence_ready",
                                evidence_id=evidence.evidence_id,
                                updated_at=now,
                            )
                            self._append_trace_conn(
                                conn,
                                evidence.run_id,
                                "evidence_packet",
                                evidence.evidence_id,
                                evidence.evidence_hash,
                                now,
                            )
                            self._append_trace_conn(
                                conn,
                                evidence.run_id,
                                "turn_state",
                                f"turn:{evidence.turn_index}:evidence_ready",
                                sha256_json({"status": "evidence_ready"}),
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                evidence.run_id,
                                f"turn_{evidence.turn_index}_evidence",
                                "turn_evidence_ready",
                                {
                                    "turn_index": evidence.turn_index,
                                    "evidence_hash": evidence.evidence_hash,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.get_evidence(evidence.run_id, evidence.turn_index)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记 V2 证据失败：{error}") from error
        result["reused"] = reused
        return result

    def record_decision(self, decision: SearchDecision) -> dict[str, Any]:
        """登记控制器决策并推进 turn。"""

        now = utc_now_iso()
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        turn = self._require_turn_conn(
                            conn,
                            decision.run_id,
                            decision.turn_index,
                        )
                        if turn["evidence_id"] != decision.evidence_id:
                            raise MiningContractError("决策没有绑定当前 turn 的证据")
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.search_decision "
                            "WHERE run_id=? AND turn_index=?",
                            [decision.run_id, decision.turn_index],
                        )
                        if existing is not None:
                            if (
                                existing["decision_id"] != decision.decision_id
                                or existing["decision_hash"] != decision.decision_hash
                            ):
                                raise MiningDuplicateError("同一 turn 已绑定不同搜索决策")
                            reused = True
                        else:
                            conn.execute(
                                "INSERT INTO mining_v2.search_decision VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    decision.run_id,
                                    decision.turn_index,
                                    decision.decision_id,
                                    decision.hypothesis_id,
                                    decision.evidence_id,
                                    decision.decision_hash,
                                    decision.action,
                                    canonical_json(decision.to_dict()),
                                    now,
                                ],
                            )
                            self._update_turn_conn(
                                conn,
                                decision.run_id,
                                decision.turn_index,
                                "decided",
                                decision_id=decision.decision_id,
                                updated_at=now,
                            )
                            self._append_trace_conn(
                                conn,
                                decision.run_id,
                                "search_decision",
                                decision.decision_id,
                                decision.decision_hash,
                                now,
                            )
                            self._append_trace_conn(
                                conn,
                                decision.run_id,
                                "turn_state",
                                f"turn:{decision.turn_index}:decided",
                                sha256_json({"status": "decided"}),
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                decision.run_id,
                                f"turn_{decision.turn_index}_decided",
                                "turn_decided",
                                {
                                    "turn_index": decision.turn_index,
                                    "decision_hash": decision.decision_hash,
                                    "action": decision.action,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.get_decision(decision.run_id, decision.turn_index)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记 V2 决策失败：{error}") from error
        result["reused"] = reused
        return result

    def transition_turn(
        self,
        run_id: str,
        turn_index: int,
        target: str,
    ) -> dict[str, Any]:
        """按固定状态机迁移一个 turn。"""

        if target not in TURN_STATUSES:
            raise MiningStateError(f"未知 turn 状态：{target!r}")
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._require_turn_conn(conn, run_id, turn_index)
                    previous = current["status"]
                    if previous == target:
                        reused = True
                    else:
                        if target not in TURN_TRANSITIONS[previous]:
                            raise MiningStateError(
                                f"turn 不能从 {previous} 变为 {target}"
                            )
                        now = utc_now_iso()
                        self._update_turn_conn(
                            conn,
                            run_id,
                            turn_index,
                            target,
                            updated_at=now,
                        )
                        self._append_trace_conn(
                            conn,
                            run_id,
                            "turn_state",
                            f"turn:{turn_index}:{target}",
                            sha256_json({"status": target}),
                            now,
                        )
                        self._append_event_conn(
                            conn,
                            run_id,
                            f"turn_{turn_index}_{target}",
                            "turn_state_changed",
                            {"turn_index": turn_index, "from": previous, "to": target},
                            created_at=now,
                        )
                        reused = False
        result = self.get_turn(run_id, turn_index)
        result["reused"] = reused
        return result

    def get_turn(self, run_id: str, turn_index: int) -> dict[str, Any]:
        """读取一个 turn，并在存在时附带假设和批次。"""

        with self._connection() as conn:
            row = self._require_turn_conn(conn, run_id, turn_index)
            result = self._public_row(row)
            if row["hypothesis_id"]:
                hypothesis = self._require_hypothesis_conn(
                    conn,
                    run_id,
                    row["hypothesis_id"],
                )
                result["hypothesis"] = json.loads(hypothesis["payload_json"])
            if row["experiment_batch_id"]:
                batch = self._require_batch_conn(
                    conn,
                    run_id,
                    row["experiment_batch_id"],
                )
                result["experiment_batch"] = json.loads(batch["payload_json"])
            return result

    def get_hypothesis(self, run_id: str, hypothesis_id: str) -> dict[str, Any]:
        """读取一条 V2 假设。"""

        with self._connection() as conn:
            row = self._require_hypothesis_conn(conn, run_id, hypothesis_id)
        return json.loads(row["payload_json"])

    def get_batch(self, run_id: str, batch_id: str) -> dict[str, Any]:
        """读取一个实验批次。"""

        with self._connection() as conn:
            row = self._require_batch_conn(conn, run_id, batch_id)
        return json.loads(row["payload_json"])

    def get_evidence(self, run_id: str, turn_index: int) -> dict[str, Any]:
        """读取一个 turn 的证据。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.evidence_packet "
                "WHERE run_id=? AND turn_index=?",
                [run_id, turn_index],
            )
        if row is None:
            raise MiningNotFoundError(f"V2 证据不存在：{run_id}/{turn_index}")
        result = self._public_row(row)
        result["packet"] = json.loads(result.pop("payload_json"))
        return result

    def get_decision(self, run_id: str, turn_index: int) -> dict[str, Any]:
        """读取一个 turn 的搜索决策。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.search_decision "
                "WHERE run_id=? AND turn_index=?",
                [run_id, turn_index],
            )
        if row is None:
            raise MiningNotFoundError(f"V2 决策不存在：{run_id}/{turn_index}")
        result = self._public_row(row)
        result["decision"] = json.loads(result.pop("payload_json"))
        return result
