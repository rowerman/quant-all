"""V2 搜索运行状态、取消标记和进度查询。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contract_types import (
    canonical_json,
    identifier,
    sha256_json,
    text,
    utc_now_iso,
)
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.search_states import (
    SEARCH_RUN_ACTIVE_STATUSES,
    SEARCH_RUN_STATUSES,
    SearchRunStateMachine,
)
from mining.states import CANDIDATE_TERMINAL_STATUSES, CandidateStateMachine
from mining.storage_lock import database_state_lock


class V2SearchControlMixin:
    """V2 运行控制与恢复状态。"""

    def _ensure_run_control_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> dict[str, Any]:
        self._require_run_conn(conn, run_id)
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.run_control WHERE run_id=?",
            [run_id],
        )
        if row is None:
            now = utc_now_iso()
            conn.execute(
                "INSERT INTO mining_v2.run_control VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, "queued", None, False, None, None, 0, None, now],
            )
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.run_control WHERE run_id=?",
                [run_id],
            )
        if row is None:
            raise MiningStorageError("V2 运行控制行写入后无法读取")
        return row

    def transition_search_run(
        self,
        run_id: str,
        target: str,
        *,
        event_key: str,
        stage: str,
        payload: Mapping[str, Any] | None = None,
        stop_reason: str | None = None,
        current_action: str | None = None,
    ) -> dict[str, Any]:
        """按 V2 状态机迁移运行并追加事件。"""

        run_id = identifier(run_id, "run_id")
        event_key = identifier(event_key, "event_key")
        stage = identifier(stage, "stage")
        if target not in SEARCH_RUN_STATUSES:
            raise MiningStateError(f"未知 V2 运行状态：{target!r}")
        event_payload = dict(payload or {})
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_run_conn(conn, run_id)
                        control = self._ensure_run_control_conn(conn, run_id)
                        previous = control["status"]
                        if previous == target:
                            reused = True
                        else:
                            SearchRunStateMachine.ensure(
                                previous,
                                target,
                                interrupted_from=control["interrupted_from"],
                            )
                            now = utc_now_iso()
                            interrupted_from = (
                                previous
                                if target == "interrupted"
                                else (
                                    None
                                    if previous == "interrupted"
                                    else control["interrupted_from"]
                                )
                            )
                            updated = conn.execute(
                                "UPDATE mining_v2.run_control SET status=?, updated_at=? "
                                "WHERE run_id=? AND status=? RETURNING run_id",
                                [target, now, run_id, previous],
                            ).fetchall()
                            if not updated:
                                raise MiningStateError("V2 运行状态已被其他请求修改")
                            conn.execute(
                                "UPDATE mining_v2.run_control SET "
                                "interrupted_from=?, stop_reason=coalesce(?, stop_reason), "
                                "current_action=coalesce(?, current_action), updated_at=? "
                                "WHERE run_id=?",
                                [
                                    interrupted_from,
                                    stop_reason,
                                    current_action,
                                    now,
                                    run_id,
                                ],
                            )
                            self._append_event_conn(
                                conn,
                                run_id,
                                event_key,
                                "search_run_state_changed",
                                {
                                    "from": previous,
                                    "to": target,
                                    "stage": stage,
                                    "stop_reason": stop_reason,
                                    "current_action": current_action,
                                    **event_payload,
                                },
                                created_at=now,
                            )
                            reused = False
            result = self.search_control_state(run_id)
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
            MiningStateError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"迁移 V2 运行状态失败：{error}") from error
        result["reused"] = reused
        return result

    def request_search_cancel(self, run_id: str) -> dict[str, Any]:
        """持久化取消请求，实际状态在安全 turn 边界迁移。"""

        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    self._require_run_conn(conn, run_id)
                    control = self._ensure_run_control_conn(conn, run_id)
                    if control["status"] not in SEARCH_RUN_ACTIVE_STATUSES:
                        return {
                            **self._public_row(control),
                            "reused": True,
                        }
                    if control["cancel_requested"]:
                        return {
                            **self._public_row(control),
                            "reused": True,
                        }
                    now = utc_now_iso()
                    conn.execute(
                        "UPDATE mining_v2.run_control SET cancel_requested=TRUE, "
                        "updated_at=? WHERE run_id=?",
                        [now, run_id],
                    )
                    self._append_event_conn(
                        conn,
                        run_id,
                        f"run_cancel_requested_{control['status']}",
                        "search_run_cancel_requested",
                        {"status": control["status"]},
                        created_at=now,
                    )
        return {
            **self.search_control_state(run_id),
            "reused": False,
        }

    def interrupt_active_search_runs(self) -> list[dict[str, Any]]:
        """把上次进程遗留的活动 V2 运行标记为中断。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT r.run_id, coalesce(ctl.status, r.status) AS status "
                "FROM mining_v2.run r LEFT JOIN mining_v2.run_control ctl "
                "ON ctl.run_id=r.run_id "
                "WHERE coalesce(ctl.status, r.status) IN "
                "('searching', 'selecting', 'evaluating', 'holding_out') "
                "ORDER BY r.run_id"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for run_id, status in rows:
            result.append(
                self.transition_search_run(
                    str(run_id),
                    "interrupted",
                    event_key=f"run_interrupted_{status}",
                    stage=str(status),
                    payload={"reason": "主服务重启"},
                )
            )
        return result

    def mark_search_run_stale(
        self,
        run_id: str,
        *,
        event_key: str,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        """原子把运行和未终态候选收口为 stale。"""

        run_id = identifier(run_id, "run_id")
        event_key = identifier(event_key, "event_key")
        error_code = identifier(error_code, "error_code")
        error_message = text(error_message, "error_message", maximum=2000)
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_run_conn(conn, run_id)
                        control = self._ensure_run_control_conn(conn, run_id)
                        previous = control["status"]
                        reused = previous in {"failed", "cancelled", "stale"}
                        changed_candidates = 0
                        if not reused:
                            SearchRunStateMachine.ensure(previous, "stale")
                            now = utc_now_iso()
                            rows = conn.execute(
                                "SELECT candidate_id, status "
                                "FROM mining_v2.candidate_trace "
                                "WHERE run_id=? ORDER BY candidate_id",
                                [run_id],
                            ).fetchall()
                            for candidate_id, candidate_status in rows:
                                if candidate_status in CANDIDATE_TERMINAL_STATUSES:
                                    continue
                                CandidateStateMachine.ensure(
                                    candidate_status,
                                    "stale",
                                )
                                updated = conn.execute(
                                    "UPDATE mining_v2.candidate_trace "
                                    "SET status='stale', updated_at=? "
                                    "WHERE run_id=? AND candidate_id=? "
                                    "AND status=? RETURNING candidate_id",
                                    [
                                        now,
                                        run_id,
                                        candidate_id,
                                        candidate_status,
                                    ],
                                ).fetchall()
                                if not updated:
                                    raise MiningStateError(
                                        "V2 候选状态已被其他请求修改"
                                    )
                                self._append_trace_conn(
                                    conn,
                                    run_id,
                                    "candidate_state",
                                    f"candidate:{candidate_id}:stale",
                                    sha256_json({"status": "stale"}),
                                    now,
                                )
                                self._append_event_conn(
                                    conn,
                                    run_id,
                                    f"candidate_{candidate_id}_stale",
                                    "candidate_state_changed",
                                    {
                                        "candidate_id": candidate_id,
                                        "from": candidate_status,
                                        "to": "stale",
                                        "error_code": error_code,
                                    },
                                    created_at=now,
                                )
                                changed_candidates += 1
                            updated = conn.execute(
                                "UPDATE mining_v2.run_control SET "
                                "status='stale', interrupted_from=NULL, "
                                "cancel_requested=FALSE, stop_reason=?, "
                                "updated_at=? "
                                "WHERE run_id=? AND status=? RETURNING run_id",
                                [error_message, now, run_id, previous],
                            ).fetchall()
                            if not updated:
                                raise MiningStateError(
                                    "V2 运行状态已被其他请求修改"
                                )
                            self._append_event_conn(
                                conn,
                                run_id,
                                event_key,
                                "search_run_state_changed",
                                {
                                    "from": previous,
                                    "to": "stale",
                                    "error_code": error_code,
                                    "candidate_count": changed_candidates,
                                },
                                created_at=now,
                            )
            result = self.search_control_state(run_id)
        except (
            MiningContractError,
            MiningNotFoundError,
            MiningStateError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(
                f"收口 V2 失效运行失败：{error}"
            ) from error
        result["reused"] = reused
        return result

    def search_control_state(self, run_id: str) -> dict[str, Any]:
        """读取运行状态和控制器可恢复字段。"""

        with self._connection() as conn:
            run = self._require_run_conn(conn, run_id)
            control = self._row(
                conn,
                "SELECT * FROM mining_v2.run_control WHERE run_id=?",
                [run_id],
            )
        return {
            "run_id": run_id,
            "status": (
                control["status"] if control is not None else run["status"]
            ),
            "updated_at": (
                control["updated_at"] if control is not None else run["updated_at"]
            ),
            "interrupted_from": (
                control["interrupted_from"] if control is not None else None
            ),
            "cancel_requested": (
                bool(control["cancel_requested"]) if control is not None else False
            ),
            "stop_reason": control["stop_reason"] if control is not None else None,
            "current_action": (
                control["current_action"] if control is not None else None
            ),
            "no_improvement_turns": (
                int(control["no_improvement_turns"])
                if control is not None
                else 0
            ),
            "previous_best": (
                tuple(json.loads(control["previous_best_json"]))
                if control is not None
                and control["previous_best_json"] is not None
                else None
            ),
        }

    def cancel_requested(self, run_id: str) -> bool:
        """读取取消请求标志。"""

        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            row = self._row(
                conn,
                "SELECT cancel_requested FROM mining_v2.run_control WHERE run_id=?",
                [run_id],
            )
        return bool(row is not None and row["cancel_requested"])

    def update_search_control_progress(
        self,
        run_id: str,
        *,
        no_improvement_turns: int,
        previous_best: tuple[float, float, float, float] | None,
    ) -> None:
        """保存控制器跨 turn 的无改善计数和最佳质量向量。"""

        if (
            isinstance(no_improvement_turns, bool)
            or not isinstance(no_improvement_turns, int)
            or no_improvement_turns < 0
        ):
            raise MiningContractError("no_improvement_turns 必须是非负整数")
        encoded = (
            None
            if previous_best is None
            else canonical_json(list(previous_best))
        )
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    self._ensure_run_control_conn(conn, run_id)
                    conn.execute(
                        "UPDATE mining_v2.run_control SET "
                        "no_improvement_turns=?, previous_best_json=?, updated_at=? "
                        "WHERE run_id=?",
                        [no_improvement_turns, encoded, utc_now_iso(), run_id],
                    )
