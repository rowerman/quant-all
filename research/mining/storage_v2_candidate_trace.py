"""挖掘 V2 候选节点和父子谱系存储。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contract_types import canonical_json, identifier, text, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.search_contract_types import hash_text
from mining.states import CANDIDATE_STATUSES
from mining.storage_lock import database_state_lock


class V2CandidateTraceMixin:
    """候选轨迹和父候选仓库。"""

    def record_candidate(
        self,
        run_id: str,
        *,
        candidate_id: str,
        candidate_hash: str,
        semantic_hash: str,
        name: str,
        status: str,
        hypothesis_id: str,
        batch_id: str,
        payload: Mapping[str, Any],
        parent_candidate_id: str | None = None,
    ) -> dict[str, Any]:
        """登记候选节点、归属批次和父候选。"""

        candidate_id = identifier(candidate_id, "candidate_id")
        candidate_hash = hash_text(candidate_hash, "candidate_hash")
        semantic_hash = hash_text(semantic_hash, "semantic_hash")
        name = text(name, "name", maximum=120)
        if status not in CANDIDATE_STATUSES:
            raise MiningContractError(f"未知候选状态：{status!r}")
        hypothesis_id = identifier(hypothesis_id, "hypothesis_id")
        batch_id = identifier(batch_id, "batch_id")
        parent_candidate_id = (
            None
            if parent_candidate_id is None
            else identifier(parent_candidate_id, "parent_candidate_id")
        )
        if not isinstance(payload, Mapping):
            raise MiningContractError("payload 必须是 JSON 对象")
        payload_dict = json.loads(canonical_json(dict(payload)))
        entry_hash = self._candidate_entry_hash(
            candidate_id,
            candidate_hash,
            semantic_hash,
            name,
            status,
            payload_dict,
        )
        now = utc_now_iso()
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_run_conn(conn, run_id)
                        batch = self._require_batch_conn(conn, run_id, batch_id)
                        self._require_hypothesis_conn(conn, run_id, hypothesis_id)
                        if batch["hypothesis_id"] != hypothesis_id:
                            raise MiningContractError("候选所属批次没有绑定该假设")
                        batch_payload = json.loads(batch["payload_json"])
                        if candidate_id not in batch_payload["candidate_ids"]:
                            raise MiningContractError("候选不在实验批次的候选集合中")
                        if parent_candidate_id is not None:
                            self._require_candidate_conn(
                                conn,
                                run_id,
                                parent_candidate_id,
                            )
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.candidate_trace "
                            "WHERE run_id=? AND candidate_id=?",
                            [run_id, candidate_id],
                        )
                        if existing is not None:
                            if (
                                existing["candidate_hash"] != candidate_hash
                                or existing["semantic_hash"] != semantic_hash
                                or existing["payload_json"] != canonical_json(payload_dict)
                            ):
                                raise MiningDuplicateError("候选 ID 已绑定不同轨迹内容")
                            lineage = self._row(
                                conn,
                                "SELECT * FROM mining_v2.candidate_lineage "
                                "WHERE run_id=? AND candidate_id=?",
                                [run_id, candidate_id],
                            )
                            if (
                                lineage is None
                                or lineage["parent_candidate_id"] != parent_candidate_id
                                or lineage["hypothesis_id"] != hypothesis_id
                                or lineage["batch_id"] != batch_id
                            ):
                                raise MiningDuplicateError("候选谱系与既有记录不一致")
                            reused = True
                        else:
                            conn.execute(
                                "INSERT INTO mining_v2.candidate_trace VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    candidate_id,
                                    candidate_hash,
                                    semantic_hash,
                                    name,
                                    status,
                                    canonical_json(payload_dict),
                                    now,
                                    now,
                                ],
                            )
                            conn.execute(
                                "INSERT INTO mining_v2.candidate_lineage VALUES "
                                "(?, ?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    candidate_id,
                                    parent_candidate_id,
                                    hypothesis_id,
                                    batch_id,
                                    now,
                                ],
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "candidate_trace",
                                candidate_id,
                                entry_hash,
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                run_id,
                                f"candidate_{candidate_id}_recorded",
                                "candidate_recorded",
                                {
                                    "candidate_id": candidate_id,
                                    "candidate_hash": candidate_hash,
                                    "semantic_hash": semantic_hash,
                                    "batch_id": batch_id,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.get_candidate(run_id, candidate_id)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            if "semantic_hash" in str(error):
                raise MiningDuplicateError("运行内已有相同语义的候选") from error
            raise MiningStorageError(f"登记 V2 候选失败：{error}") from error
        result["reused"] = reused
        return result

    def get_candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """读取候选轨迹和直接父候选。"""

        with self._connection() as conn:
            row = self._require_candidate_conn(conn, run_id, candidate_id)
            lineage = self._row(
                conn,
                "SELECT * FROM mining_v2.candidate_lineage "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        result = self._public_row(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["lineage"] = self._public_row(lineage) if lineage is not None else None
        return result
