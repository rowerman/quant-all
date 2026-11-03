"""候选定义及候选状态迁移存储。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contracts import CandidateDefinition, canonical_json, utc_now_iso
from mining.errors import (
    ApprovalConflictError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningRecoveryError,
    MiningStateError,
)
from mining.logging_conf import format_event
from mining.states import CandidateStateMachine


class CandidateStoreMixin:
    """候选定义的持久化方法。"""

    def add_candidate(
        self,
        run_id: str,
        definition: CandidateDefinition,
    ) -> dict[str, Any]:
        """幂等写入候选，并识别同运行内的语义重复。"""

        run_id = self._validate_id(run_id, "run_id")
        reused = False
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    run = self._row(
                        conn, "SELECT status FROM mining.run WHERE run_id=?", [run_id]
                    )
                    if run is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
                    if run["status"] not in {"queued", "generating"}:
                        raise MiningStateError(
                            f"运行状态 {run['status']} 不允许新增候选"
                        )
                    existing = self._row(
                        conn,
                        "SELECT candidate_id, content_hash, semantic_hash "
                        "FROM mining.candidate WHERE run_id=? AND "
                        "(content_hash=? OR semantic_hash=?)",
                        [run_id, definition.content_hash, definition.semantic_hash],
                    )
                    if existing is not None:
                        if (
                            existing["candidate_id"] == definition.candidate_id
                            and existing["content_hash"] == definition.content_hash
                        ):
                            reused = True
                        else:
                            raise MiningDuplicateError(
                                "同一运行内已存在相同或语义重复的候选"
                            )
                    else:
                        self._insert_candidate_conn(conn, run_id, definition)
            result = self.get_candidate(run_id, definition.candidate_id)
        result["reused"] = reused
        return result

    def _insert_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        definition: CandidateDefinition,
    ) -> None:
        """插入候选及发现事件。"""

        timestamp = utc_now_iso()
        payload = {
            **definition.content_material(),
            "canonical_expression": definition.canonical_expression,
            "direction": definition.direction,
            "lookback": definition.lookback,
            "dependencies": list(definition.dependencies),
            "output_type": definition.output_type,
            "parameters": definition.parameters,
            "discovery_round": definition.discovery_round,
            "discovery_index": definition.discovery_index,
        }
        conn.execute(
            "INSERT INTO mining.candidate VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                definition.candidate_id,
                definition.semantic_hash,
                definition.content_hash,
                definition.name,
                "proposed",
                definition.discovery_round,
                definition.discovery_index,
                canonical_json(payload),
                None,
                None,
                None,
                None,
                timestamp,
                timestamp,
                None,
            ],
        )
        self._append_event_conn(
            conn,
            run_id,
            f"candidate_proposed_{definition.candidate_id}",
            "generating",
            "candidate_proposed",
            {
                "candidate_id": definition.candidate_id,
                "content_hash": definition.content_hash,
                "semantic_hash": definition.semantic_hash,
            },
            created_at=timestamp,
        )

    def get_candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """读取运行内的候选。"""

        run_id = self._validate_id(run_id, "run_id")
        candidate_id = self._validate_id(candidate_id, "candidate_id")
        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.candidate "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        if row is None:
            raise MiningNotFoundError(f"候选不存在：{run_id}/{candidate_id}")
        row["definition"] = json.loads(row.pop("definition_json"))
        return row

    def list_candidates(self, run_id: str) -> list[dict[str, Any]]:
        """按发现顺序列出运行内候选。"""

        run_id = self._validate_id(run_id, "run_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mining.candidate WHERE run_id=? "
                "ORDER BY discovery_round, discovery_index, candidate_id",
                [run_id],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        result = []
        for row in rows:
            item = dict(zip(columns, row, strict=True))
            item["definition"] = json.loads(item.pop("definition_json"))
            result.append(item)
        return result

    def transition_candidate(
        self,
        run_id: str,
        candidate_id: str,
        target: str,
        *,
        event_key: str,
        stage: str,
        payload: Mapping[str, Any] | None = None,
        failure_stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """原子迁移候选状态并追加事件。"""

        run_id = self._validate_id(run_id, "run_id")
        candidate_id = self._validate_id(candidate_id, "candidate_id")
        event_key = self._validate_id(event_key, "event_key")
        event_payload = dict(payload or {})
        if target == "interrupted":
            raise MiningRecoveryError("候选中断状态只能由启动恢复流程写入")
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._row(
                        conn,
                        "SELECT status, interrupted_from FROM mining.candidate "
                        "WHERE run_id=? AND candidate_id=?",
                        [run_id, candidate_id],
                    )
                    if current is None:
                        raise MiningNotFoundError(f"候选不存在：{candidate_id}")
                    previous = current["status"]
                    if previous == target:
                        event = self._row(
                            conn,
                            "SELECT sequence FROM mining.event "
                            "WHERE run_id=? AND event_key=?",
                            [run_id, event_key],
                        )
                        if event is None:
                            raise MiningStateError(f"候选已经处于 {target}")
                    else:
                        CandidateStateMachine.ensure(
                            previous,
                            target,
                            interrupted_from=current["interrupted_from"],
                        )
                        self._transition_candidate_conn(
                            conn,
                            run_id,
                            candidate_id,
                            previous,
                            target,
                            event_key,
                            stage,
                            event_payload,
                            failure_stage,
                            error_code,
                            error_message,
                        )
            result = self.get_candidate(run_id, candidate_id)
        self.logger.info(
            format_event(
                {
                    "operation": "transition_candidate",
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "stage": stage,
                    "to": target,
                    "error_code": error_code,
                }
            )
        )
        return result

    def _transition_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
        previous: str,
        target: str,
        event_key: str,
        stage: str,
        event_payload: Mapping[str, Any],
        failure_stage: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> None:
        """执行已经过状态机校验的候选更新。"""

        timestamp = utc_now_iso()
        finished = timestamp if target in {
            "technical_failed",
            "screened_out",
            "holdout_failed",
            "published",
            "rejected",
            "failed",
            "cancelled",
            "stale",
        } else None
        updated = conn.execute(
            "UPDATE mining.candidate SET status=?, interrupted_from=NULL, "
            "failure_stage=coalesce(?, failure_stage), "
            "error_code=coalesce(?, error_code), "
            "error_message=coalesce(?, error_message), updated_at=?, "
            "finished_at=coalesce(?, finished_at) "
            "WHERE run_id=? AND candidate_id=? AND status=? "
            "RETURNING candidate_id",
            [
                target,
                failure_stage,
                error_code,
                error_message,
                timestamp,
                finished,
                run_id,
                candidate_id,
                previous,
            ],
        ).fetchall()
        if not updated:
            raise MiningStateError("候选状态已被其他请求修改")
        self._append_event_conn(
            conn,
            run_id,
            event_key,
            stage,
            "candidate_status_changed",
            {
                "candidate_id": candidate_id,
                "from": previous,
                "to": target,
                "error_code": error_code,
                **event_payload,
            },
            created_at=timestamp,
        )

    def _require_current_candidate(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        run_id: str,
        candidate_id: str,
        content_hash: str,
    ) -> dict[str, Any]:
        """确认报告绑定的是当前候选内容。"""

        row = self._row(
            conn,
            "SELECT status, content_hash FROM mining.candidate "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        )
        if row is None:
            raise MiningNotFoundError(f"候选不存在：{candidate_id}")
        if row["content_hash"] != content_hash:
            raise ApprovalConflictError("候选内容已变化，旧证据不再适用")
        return row
