"""运行级失败、取消和快照失效的原子终止。"""

from __future__ import annotations

from typing import Any

import duckdb

from mining.contracts import utc_now_iso
from mining.errors import (
    MiningNotFoundError,
    MiningStateError,
)
from mining.logging_conf import format_event
from mining.states import (
    CANDIDATE_TERMINAL_STATUSES,
    RUN_TERMINAL_STATUSES,
    CandidateStateMachine,
    RunStateMachine,
)


class RunControlStoreMixin:
    """在一次事务内终止运行及其全部未终态候选。"""

    def terminate_run(
        self,
        run_id: str,
        target: str,
        *,
        event_key: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """按状态机原子终止运行和候选。"""

        run_id = self._validate_id(run_id, "run_id")
        event_key = self._validate_id(event_key, "event_key")
        if target not in {"failed", "cancelled", "stale"}:
            raise MiningStateError("终止目标只能是 failed、cancelled 或 stale")
        changed_candidates: list[str] = []
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._row(
                        conn,
                        "SELECT status FROM mining.run WHERE run_id=?",
                        [run_id],
                    )
                    if current is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
                    previous = current["status"]
                    if previous in RUN_TERMINAL_STATUSES:
                        return self.get_run(run_id)
                    RunStateMachine.ensure(previous, target)
                    timestamp = utc_now_iso()
                    self._terminate_candidates_conn(
                        conn,
                        run_id,
                        target,
                        timestamp,
                        error_code,
                        error_message,
                        changed_candidates,
                    )
                    updated = conn.execute(
                        "UPDATE mining.run SET status=?, interrupted_from=NULL, "
                        "error_code=coalesce(?, error_code), "
                        "error_message=coalesce(?, error_message), updated_at=?, "
                        "finished_at=? WHERE run_id=? AND status=? RETURNING run_id",
                        [
                            target,
                            error_code,
                            error_message,
                            timestamp,
                            timestamp,
                            run_id,
                            previous,
                        ],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError("运行状态已被其他请求修改")
                    self._append_event_conn(
                        conn,
                        run_id,
                        event_key,
                        previous,
                        f"run_{target}",
                        {
                            "from": previous,
                            "to": target,
                            "error_code": error_code,
                            "candidate_count": len(changed_candidates),
                        },
                        created_at=timestamp,
                    )
            result = self.get_run(run_id)
        self.logger.warning(
            format_event(
                {
                    "operation": "terminate_run",
                    "run_id": run_id,
                    "status": target,
                    "candidates": len(changed_candidates),
                    "error_code": error_code,
                }
            )
        )
        return result

    def _terminate_candidates_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        target: str,
        timestamp: str,
        error_code: str | None,
        error_message: str | None,
        changed: list[str],
    ) -> None:
        """终止运行内所有可以进入目标状态的候选。"""

        placeholders = ",".join("?" for _ in CANDIDATE_TERMINAL_STATUSES)
        rows = conn.execute(
            f"SELECT candidate_id, status FROM mining.candidate "
            f"WHERE run_id=? AND status NOT IN ({placeholders})",
            [run_id, *CANDIDATE_TERMINAL_STATUSES],
        ).fetchall()
        for candidate_id, previous in rows:
            candidate_target = target
            try:
                CandidateStateMachine.ensure(previous, candidate_target)
            except MiningStateError:
                candidate_target = "stale"
                CandidateStateMachine.ensure(previous, candidate_target)
            updated = conn.execute(
                "UPDATE mining.candidate SET status=?, interrupted_from=NULL, "
                "error_code=coalesce(?, error_code), "
                "error_message=coalesce(?, error_message), updated_at=?, "
                "finished_at=? WHERE run_id=? AND candidate_id=? AND status=? "
                "RETURNING candidate_id",
                [
                    candidate_target,
                    error_code,
                    error_message,
                    timestamp,
                    timestamp,
                    run_id,
                    candidate_id,
                    previous,
                ],
            ).fetchall()
            if not updated:
                raise MiningStateError("候选状态已被其他请求修改")
            changed.append(candidate_id)
            self._append_event_conn(
                conn,
                run_id,
                f"run_{target}_{candidate_id}",
                previous,
                "candidate_status_changed",
                {
                    "candidate_id": candidate_id,
                    "from": previous,
                    "to": candidate_target,
                    "error_code": error_code,
                },
                created_at=timestamp,
            )
