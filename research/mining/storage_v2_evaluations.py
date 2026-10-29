"""V2 技术、开发、选择、完整和测试段评估证据存储。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.contract_types import canonical_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.storage_lock import database_state_lock


class V2EvaluationStoreMixin:
    """追加式 V2 评估证据仓库。"""

    def record_evaluation_evidence(
        self,
        evidence: SearchEvaluationEvidence,
    ) -> dict[str, Any]:
        """幂等登记单个阶段的脱敏评估证据。"""

        if not isinstance(evidence, SearchEvaluationEvidence):
            raise MiningContractError(
                "evidence 必须是 SearchEvaluationEvidence"
            )
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_candidate_conn(
                            conn,
                            evidence.run_id,
                            evidence.candidate_id,
                        )
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.evaluation_evidence "
                            "WHERE run_id=? AND evaluation_id=?",
                            [evidence.run_id, evidence.evaluation_id],
                        )
                        if existing is not None:
                            if existing["evidence_hash"] != evidence.evidence_hash:
                                raise MiningDuplicateError(
                                    "评估 ID 已绑定不同证据"
                                )
                            reused = True
                        else:
                            now = utc_now_iso()
                            conn.execute(
                                "INSERT INTO mining_v2.evaluation_evidence VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    evidence.run_id,
                                    evidence.evaluation_id,
                                    evidence.candidate_id,
                                    evidence.stage,
                                    evidence.input_hash,
                                    evidence.status,
                                    evidence.evidence_hash,
                                    canonical_json(evidence.to_dict()),
                                    evidence.created_at,
                                ],
                            )
                            self._append_trace_conn(
                                conn,
                                evidence.run_id,
                                "evaluation_evidence",
                                evidence.evaluation_id,
                                evidence.evidence_hash,
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                evidence.run_id,
                                f"evaluation_{evidence.evaluation_id}",
                                "evaluation_evidence_recorded",
                                {
                                    "candidate_id": evidence.candidate_id,
                                    "stage": evidence.stage,
                                    "status": evidence.status,
                                    "input_hash": evidence.input_hash,
                                    "evidence_hash": evidence.evidence_hash,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.get_evaluation_evidence(
                evidence.run_id,
                evidence.evaluation_id,
            )
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记 V2 评估证据失败：{error}") from error
        result["reused"] = reused
        return result

    def get_evaluation_evidence(
        self,
        run_id: str,
        evaluation_id: str,
    ) -> dict[str, Any]:
        """按 ID 读取评估证据。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.evaluation_evidence "
                "WHERE run_id=? AND evaluation_id=?",
                [run_id, evaluation_id],
            )
        if row is None:
            raise MiningNotFoundError(f"V2 评估证据不存在：{evaluation_id}")
        result = self._public_row(row)
        result["evidence"] = json.loads(result.pop("payload_json"))
        return result

    def list_evaluation_evidence(
        self,
        run_id: str,
        candidate_id: str | None = None,
        *,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """按创建顺序读取评估证据。"""

        sql = (
            "SELECT * FROM mining_v2.evaluation_evidence WHERE run_id=?"
        )
        params: list[Any] = [run_id]
        if candidate_id is not None:
            sql += " AND candidate_id=?"
            params.append(candidate_id)
        if stage is not None:
            sql += " AND stage=?"
            params.append(stage)
        sql += " ORDER BY created_at, evaluation_id"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            columns = [item[0] for item in conn.description]
        result = []
        for values in rows:
            item = self._public_row(dict(zip(columns, values, strict=True)))
            item["evidence"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result
