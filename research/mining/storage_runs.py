"""研究运行、冻结上下文和中断恢复存储。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contracts import (
    FrozenContext,
    MiningRunRequest,
    canonical_json,
    sha256_json,
    utc_now_iso,
)
from mining.errors import (
    MiningBusyError,
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningRecoveryError,
    MiningStateError,
    MiningStorageError,
)
from mining.logging_conf import format_event
from mining.states import (
    CANDIDATE_ACTIVE_STATUSES,
    RUN_ACTIVE_STATUSES,
    CandidateStateMachine,
    RunStateMachine,
)


class RunStoreMixin:
    """运行和冻结上下文的持久化方法。"""

    def create_run(
        self,
        request: MiningRunRequest,
        context: FrozenContext,
        *,
        worker_mode: str = "rdagent",
    ) -> dict[str, Any]:
        """幂等创建运行及其冻结上下文。"""

        self._check_context(request, context)
        worker_mode = self._validate_id(worker_mode, "worker_mode")
        created = utc_now_iso()
        reused = False
        try:
            with self._state_lock():
                with self._connection() as conn:
                    with self._transaction(conn):
                        existing = self._row(
                            conn,
                            "SELECT r.request_hash, r.worker_mode, c.context_hash "
                            "FROM mining.run r "
                            "JOIN mining.run_context c ON c.context_id=r.context_id "
                            "WHERE r.idempotency_key=?",
                            [request.idempotency_key],
                        )
                        if existing is not None:
                            if (
                                existing["request_hash"] != request.request_hash
                                or existing["context_hash"] != context.context_hash
                                or existing["worker_mode"] != worker_mode
                            ):
                                raise MiningDuplicateError(
                                    "幂等键已用于不同请求、Worker 模式或冻结上下文"
                                )
                            reused = True
                        else:
                            stored_context = self._row(
                                conn,
                                "SELECT context_hash FROM mining.run_context "
                                "WHERE context_id=?",
                                [context.context_id],
                            )
                            if stored_context is not None:
                                if stored_context["context_hash"] != context.context_hash:
                                    raise MiningDuplicateError(
                                        "上下文 ID 已绑定不同内容"
                                    )
                            else:
                                conn.execute(
                                    "INSERT INTO mining.run_context VALUES "
                                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                                    "?, ?, ?, ?)",
                                    [
                                        context.context_id,
                                        context.context_hash,
                                        context.data_version,
                                        context.data_fingerprint,
                                        context.label,
                                        context.universe,
                                        context.price_channel,
                                        context.research_start,
                                        context.research_end,
                                        context.full_eval_start,
                                        context.full_eval_end,
                                        context.holdout_start,
                                        context.holdout_end,
                                        context.llm_profile,
                                        context.llm_model,
                                        context.field_catalog_hash,
                                        context.operator_catalog_hash,
                                        sha256_json(list(context.existing_factors)),
                                        canonical_json(context.material()),
                                        context.created_at,
                                    ],
                                )
                            conn.execute(
                                "INSERT INTO mining.run VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    request.run_id,
                                    context.context_id,
                                    request.idempotency_key,
                                    request.request_hash,
                                    "queued",
                                    request.hypothesis,
                                    canonical_json(request.material()),
                                    worker_mode,
                                    0,
                                    None,
                                    None,
                                    None,
                                    None,
                                    created,
                                    created,
                                    None,
                                    None,
                                ],
                            )
                            self._append_event_conn(
                                conn,
                                request.run_id,
                                "run_created",
                                "queued",
                                "status_changed",
                                {
                                    "from": None,
                                    "to": "queued",
                                    "request_hash": request.request_hash,
                                    "context_hash": context.context_hash,
                                },
                                created_at=created,
                            )
        except (MiningContractError, MiningDuplicateError):
            raise
        except (duckdb.Error, OSError) as error:
            self.logger.error(
                format_event(
                    {
                        "operation": "create_run",
                        "run_id": request.run_id,
                        "error_code": "MINING-112",
                    }
                )
            )
            raise MiningStorageError(f"创建运行失败：{error}") from error
        result = self.get_run(request.run_id)
        result["reused"] = reused
        self.logger.info(
            format_event(
                {
                    "operation": "create_run",
                    "run_id": request.run_id,
                    "context_hash": context.context_hash,
                    "reused": reused,
                }
            )
        )
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        """读取运行及其冻结上下文。"""

        run_id = self._validate_id(run_id, "run_id")
        try:
            with self._connection() as conn:
                run = self._row(
                    conn, "SELECT * FROM mining.run WHERE run_id=?", [run_id]
                )
                context = self._context_for_run_conn(conn, run_id)
        except duckdb.Error as error:
            raise MiningStorageError(f"读取运行失败：{error}") from error
        if run is None or context is None:
            raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
        run["request"] = json.loads(run.pop("request_json"))
        context["payload"] = json.loads(context.pop("payload_json"))
        run["context"] = context
        return run

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """按创建时间倒序列出运行。"""

        limit = max(1, min(int(limit), 200))
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT r.run_id, r.idempotency_key, r.status, r.hypothesis, "
                    "r.created_at, r.updated_at, r.error_code, c.context_hash "
                    "FROM mining.run r "
                    "JOIN mining.run_context c ON c.context_id=r.context_id "
                    "ORDER BY r.created_at DESC LIMIT ?",
                    [limit],
                ).fetchall()
                columns = [item[0] for item in conn.description]
        except duckdb.Error as error:
            raise MiningStorageError(f"列出运行失败：{error}") from error
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def active_run(self) -> dict[str, Any] | None:
        """返回占用本机唯一执行槽的活动运行。"""

        placeholders = ",".join("?" for _ in RUN_ACTIVE_STATUSES)
        with self._connection() as conn:
            row = self._row(
                conn,
                f"SELECT * FROM mining.run WHERE status IN ({placeholders}) "
                "ORDER BY created_at LIMIT 1",
                list(RUN_ACTIVE_STATUSES),
            )
        if row is None:
            return None
        row.pop("request_json", None)
        return row

    def transition_run(
        self,
        run_id: str,
        target: str,
        *,
        event_key: str,
        stage: str,
        payload: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """原子迁移运行状态并追加事件。"""

        run_id = self._validate_id(run_id, "run_id")
        event_key = self._validate_id(event_key, "event_key")
        event_payload = dict(payload or {})
        if target == "interrupted":
            raise MiningRecoveryError("中断状态只能由启动恢复流程写入")
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._row(
                        conn,
                        "SELECT status, interrupted_from, last_completed_stage "
                        "FROM mining.run WHERE run_id=?",
                        [run_id],
                    )
                    if current is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
                    previous = current["status"]
                    if previous == target:
                        event = self._row(
                            conn,
                            "SELECT sequence FROM mining.event "
                            "WHERE run_id=? AND event_key=?",
                            [run_id, event_key],
                        )
                        if event is None:
                            raise MiningStateError(f"运行已经处于 {target}")
                    else:
                        RunStateMachine.ensure(
                            previous,
                            target,
                            interrupted_from=current["interrupted_from"],
                        )
                        if target == "generating":
                            self._ensure_no_other_active_conn(conn, run_id)
                        self._transition_run_conn(
                            conn,
                            run_id,
                            previous,
                            target,
                            event_key,
                            stage,
                            event_payload,
                            error_code,
                            error_message,
                        )
            result = self.get_run(run_id)
        self.logger.info(
            format_event(
                {
                    "operation": "transition_run",
                    "run_id": run_id,
                    "stage": stage,
                    "to": target,
                    "error_code": error_code,
                }
            )
        )
        return result

    @staticmethod
    def _ensure_no_other_active_conn(conn, run_id: str) -> None:
        """确认没有其他运行占用本机执行槽。"""

        placeholders = ",".join("?" for _ in RUN_ACTIVE_STATUSES)
        active = conn.execute(
            f"SELECT run_id FROM mining.run WHERE run_id<>? "
            f"AND status IN ({placeholders}) LIMIT 1",
            [run_id, *RUN_ACTIVE_STATUSES],
        ).fetchone()
        if active is not None:
            raise MiningBusyError()

    def _transition_run_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        previous: str,
        target: str,
        event_key: str,
        stage: str,
        event_payload: Mapping[str, Any],
        error_code: str | None,
        error_message: str | None,
    ) -> None:
        """执行已经过状态机校验的运行更新。"""

        timestamp = utc_now_iso()
        finished = timestamp if target in {
            "completed",
            "failed",
            "cancelled",
            "stale",
        } else None
        updated = conn.execute(
            "UPDATE mining.run SET status=?, interrupted_from=NULL, "
            "error_code=coalesce(?, error_code), "
            "error_message=coalesce(?, error_message), updated_at=?, "
            "started_at=coalesce(started_at, ?), "
            "finished_at=coalesce(?, finished_at) "
            "WHERE run_id=? AND status=? RETURNING run_id",
            [
                target,
                error_code,
                error_message,
                timestamp,
                timestamp if target == "generating" else None,
                finished,
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
            stage,
            "status_changed",
            {
                "from": previous,
                "to": target,
                "error_code": error_code,
                **event_payload,
            },
            created_at=timestamp,
        )

    def interrupt_active_runs(self) -> list[str]:
        """把上次进程遗留的活动运行标记为中断，不自动继续执行。"""

        interrupted: list[str] = []
        interrupted_candidates: list[tuple[str, str]] = []
        placeholders = ",".join("?" for _ in RUN_ACTIVE_STATUSES)
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    rows = conn.execute(
                        f"SELECT run_id, status, interruption_count FROM mining.run "
                        f"WHERE status IN ({placeholders}) ORDER BY created_at",
                        list(RUN_ACTIVE_STATUSES),
                    ).fetchall()
                    for run_id, previous, count in rows:
                        timestamp = utc_now_iso()
                        conn.execute(
                            "UPDATE mining.run SET status='interrupted', "
                            "interrupted_from=?, interruption_count=?, updated_at=? "
                            "WHERE run_id=? AND status=?",
                            [previous, int(count) + 1, timestamp, run_id, previous],
                        )
                        candidates = conn.execute(
                            "SELECT candidate_id, status FROM mining.candidate "
                            "WHERE run_id=? AND status IN (?, ?, ?, ?, ?)",
                            [
                                run_id,
                                *CANDIDATE_ACTIVE_STATUSES,
                            ],
                        ).fetchall()
                        conn.execute(
                            "UPDATE mining.candidate SET status='interrupted', "
                            "interrupted_from=status, updated_at=? "
                            "WHERE run_id=? AND status IN (?, ?, ?, ?, ?)",
                            [
                                timestamp,
                                run_id,
                                *CANDIDATE_ACTIVE_STATUSES,
                            ],
                        )
                        for candidate_id, candidate_previous in candidates:
                            interrupted_candidates.append((run_id, candidate_id))
                            self._append_event_conn(
                                conn,
                                run_id,
                                f"candidate_interrupted_{count + 1}_{candidate_id}",
                                candidate_previous,
                                "candidate_status_changed",
                                {
                                    "candidate_id": candidate_id,
                                    "from": candidate_previous,
                                    "to": "interrupted",
                                    "error_code": "MINING-113",
                                },
                                created_at=timestamp,
                            )
                        self._append_event_conn(
                            conn,
                            run_id,
                            f"interrupted_{int(count) + 1}",
                            previous,
                            "run_interrupted",
                            {"from": previous, "to": "interrupted"},
                            created_at=timestamp,
                        )
                        interrupted.append(run_id)
        for run_id in interrupted:
            self.logger.warning(
                format_event(
                    {
                        "operation": "interrupt_active_runs",
                        "run_id": run_id,
                        "error_code": "MINING-113",
                    }
                )
            )
        for run_id, candidate_id in interrupted_candidates:
            self.logger.warning(
                format_event(
                    {
                        "operation": "interrupt_candidate",
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "error_code": "MINING-113",
                    }
                )
            )
        return interrupted

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """把显式中断的运行恢复到中断前阶段。"""

        run_id = self._validate_id(run_id, "run_id")
        resumed_candidates: list[str] = []
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    current = self._row(
                        conn,
                        "SELECT status, interrupted_from, interruption_count "
                        "FROM mining.run WHERE run_id=?",
                        [run_id],
                    )
                    if current is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
                    if current["status"] != "interrupted":
                        raise MiningStateError("只有 interrupted 运行可以显式恢复")
                    target = current["interrupted_from"]
                    RunStateMachine.ensure(
                        "interrupted", target, interrupted_from=target
                    )
                    if target == "generating":
                        self._ensure_no_other_active_conn(conn, run_id)
                    timestamp = utc_now_iso()
                    candidates = conn.execute(
                        "SELECT candidate_id, interrupted_from FROM mining.candidate "
                        "WHERE run_id=? AND status='interrupted'",
                        [run_id],
                    ).fetchall()
                    for candidate_id, previous in candidates:
                        resumed_candidates.append(candidate_id)
                        CandidateStateMachine.ensure(
                            "interrupted",
                            previous,
                            interrupted_from=previous,
                        )
                        conn.execute(
                            "UPDATE mining.candidate SET status=?, interrupted_from=NULL, "
                            "updated_at=? WHERE run_id=? AND candidate_id=? "
                            "AND status='interrupted'",
                            [previous, timestamp, run_id, candidate_id],
                        )
                        self._append_event_conn(
                            conn,
                            run_id,
                            f"candidate_resumed_{int(current['interruption_count'])}_{candidate_id}",
                            previous,
                            "candidate_status_changed",
                            {
                                "candidate_id": candidate_id,
                                "from": "interrupted",
                                "to": previous,
                            },
                            created_at=timestamp,
                        )
                    updated = conn.execute(
                        "UPDATE mining.run SET status=?, interrupted_from=NULL, "
                        "updated_at=? WHERE run_id=? AND status='interrupted' "
                        "RETURNING run_id",
                        [target, timestamp, run_id],
                    ).fetchall()
                    if not updated:
                        raise MiningStateError("运行状态已被其他请求修改")
                    self._append_event_conn(
                        conn,
                        run_id,
                        f"resume_{int(current['interruption_count'])}",
                        target,
                        "run_resumed",
                        {"from": "interrupted", "to": target},
                        created_at=timestamp,
                    )
            result = self.get_run(run_id)
        self.logger.info(
            format_event(
                {
                    "operation": "resume_run",
                    "run_id": run_id,
                    "to": result["status"],
                }
            )
        )
        for candidate_id in resumed_candidates:
            self.logger.info(
                format_event(
                    {
                        "operation": "resume_candidate",
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                    }
                )
            )
        return result
