"""挖掘 V2 不可删除尝试台账、预算和候选状态事务。"""

from __future__ import annotations

from typing import Any

import duckdb

from mining.contract_types import canonical_json, sha256_json, stable_id, text, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStateError,
    MiningStorageError,
)
from mining.search_budget import TrialCounts
from mining.search_trials import TrialRecord
from mining.states import CANDIDATE_STATUSES, CandidateStateMachine
from mining.storage_lock import database_state_lock

_COUNT_COLUMNS = set(TrialCounts().to_dict())


class V2TrialStoreMixin:
    """追加式尝试记录和冻结预算执行。"""

    def record_trial(self, trial: TrialRecord) -> dict[str, Any]:
        """幂等追加一条尝试，并在同一事务更新分类计数。"""

        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        _, reused = self._record_trial_conn(conn, trial)
            result = self.get_trial(trial.run_id, trial.trial_id)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"写入 V2 尝试记录失败：{error}") from error
        result["reused"] = reused
        return result

    def record_research_seed_trial(self, trial: TrialRecord) -> dict[str, Any]:
        """记录资料种子尝试；只免除每轮上限，不绕过每运行预算。"""

        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        _, reused = self._record_trial_conn(
                            conn,
                            trial,
                            enforce_turn_budget=False,
                        )
            result = self.get_trial(trial.run_id, trial.trial_id)
        except (MiningContractError, MiningDuplicateError, MiningNotFoundError):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(
                f"写入资料种子尝试记录失败：{error}"
            ) from error
        result["reused"] = reused
        return result

    def transition_candidate(
        self,
        run_id: str,
        candidate_id: str,
        target: str,
        *,
        trial: TrialRecord | None = None,
    ) -> dict[str, Any]:
        """在同一事务中迁移候选状态并追加相关尝试。"""

        if target not in CANDIDATE_STATUSES:
            raise MiningStateError(f"未知候选状态：{target!r}")
        if trial is not None:
            if trial.run_id != run_id or trial.candidate_id != candidate_id:
                raise MiningContractError("尝试记录没有绑定当前候选")
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        candidate = self._require_candidate_conn(
                            conn,
                            run_id,
                            candidate_id,
                        )
                        previous = candidate["status"]
                        if previous == target:
                            reused = True
                        else:
                            CandidateStateMachine.ensure(previous, target)
                            reused = False
                        trial_result_id = None
                        if trial is not None:
                            trial_result, trial_reused = self._record_trial_conn(
                                conn,
                                trial,
                            )
                            trial_result_id = trial_result["trial_id"]
                            reused = reused and trial_reused
                        if previous != target:
                            now = utc_now_iso()
                            conn.execute(
                                "UPDATE mining_v2.candidate_trace "
                                "SET status=?, updated_at=? "
                                "WHERE run_id=? AND candidate_id=? AND status=?",
                                [
                                    target,
                                    now,
                                    run_id,
                                    candidate_id,
                                    previous,
                                ],
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "candidate_state",
                                f"candidate:{candidate_id}:{target}",
                                sha256_json({"status": target}),
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                run_id,
                                f"candidate_{candidate_id}_{target}",
                                "candidate_state_changed",
                                {
                                    "candidate_id": candidate_id,
                                    "from": previous,
                                    "to": target,
                                    "trial_id": trial_result_id,
                                },
                                created_at=now,
                            )
            result = self.get_candidate(run_id, candidate_id)
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
            MiningStateError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"迁移 V2 候选状态失败：{error}") from error
        result["reused"] = reused
        result["trial_id"] = trial_result_id
        return result

    def invalidate_trial(
        self,
        run_id: str,
        trial_id: str,
        *,
        reason_code: str,
    ) -> dict[str, Any]:
        """追加无效标记，保留原尝试及其计数。"""

        reason_code = text(reason_code, "reason_code", maximum=500)
        invalidation_id = stable_id(
            "trialinv",
            {
                "run_id": run_id,
                "trial_id": trial_id,
                "reason_code": reason_code,
            },
        )
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        trial = self._row(
                            conn,
                            "SELECT * FROM mining_v2.trial_record "
                            "WHERE run_id=? AND trial_id=?",
                            [run_id, trial_id],
                        )
                        if trial is None:
                            raise MiningNotFoundError(
                                f"V2 尝试记录不存在：{trial_id}"
                            )
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.trial_invalidation "
                            "WHERE run_id=? AND trial_id=?",
                            [run_id, trial_id],
                        )
                        if existing is not None:
                            if existing["invalidation_id"] != invalidation_id:
                                raise MiningDuplicateError(
                                    "尝试记录已有不同作废原因"
                                )
                            reused = True
                        else:
                            now = utc_now_iso()
                            conn.execute(
                                "INSERT INTO mining_v2.trial_invalidation VALUES "
                                "(?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    invalidation_id,
                                    trial_id,
                                    reason_code,
                                    now,
                                ],
                            )
                            self._increment_counter_conn(
                                conn,
                                run_id,
                                "invalidated",
                                now,
                            )
                            self._append_trace_conn(
                                conn,
                                run_id,
                                "trial_invalidation",
                                invalidation_id,
                                sha256_json(
                                    {
                                        "trial_id": trial_id,
                                        "reason_code": reason_code,
                                    }
                                ),
                                now,
                            )
                            self._append_event_conn(
                                conn,
                                run_id,
                                f"trial_{trial_id}_invalidated",
                                "trial_invalidated",
                                {
                                    "trial_id": trial_id,
                                    "reason_code": reason_code,
                                },
                                created_at=now,
                            )
                            reused = False
            result = {
                "invalidation_id": invalidation_id,
                "run_id": run_id,
                "trial_id": trial_id,
                "reason_code": reason_code,
                "reused": reused,
            }
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"作废 V2 尝试记录失败：{error}") from error
        return result

    def _record_trial_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        trial: TrialRecord,
        *,
        enforce_turn_budget: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        existing = self._row(
            conn,
            "SELECT * FROM mining_v2.trial_record "
            "WHERE run_id=? AND trial_key=?",
            [trial.run_id, trial.trial_key],
        )
        if existing is not None:
            if (
                existing["trial_id"] != trial.trial_id
                or existing["trial_hash"] != trial.trial_hash
            ):
                raise MiningDuplicateError("trial_key 已绑定不同尝试内容")
            return existing, True
        self._validate_trial_references_conn(conn, trial)
        self._ensure_trial_budget_conn(
            conn,
            trial,
            enforce_turn_budget=enforce_turn_budget,
        )
        now = utc_now_iso()
        conn.execute(
            "INSERT INTO mining_v2.trial_record VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                trial.run_id,
                trial.trial_id,
                trial.trial_key,
                trial.kind,
                trial.turn_index,
                trial.hypothesis_id,
                trial.batch_id,
                trial.candidate_id,
                trial.trial_hash,
                canonical_json(trial.to_dict()),
                trial.occurred_at,
            ],
        )
        self._increment_counter_conn(conn, trial.run_id, trial.kind, now)
        self._append_trace_conn(
            conn,
            trial.run_id,
            "trial_record",
            trial.trial_id,
            trial.trial_hash,
            now,
        )
        self._append_event_conn(
            conn,
            trial.run_id,
            f"trial_{trial.trial_id}_recorded",
            "trial_recorded",
            {
                "trial_id": trial.trial_id,
                "kind": trial.kind,
                "turn_index": trial.turn_index,
                "candidate_id": trial.candidate_id,
            },
            created_at=now,
        )
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.trial_record "
            "WHERE run_id=? AND trial_id=?",
            [trial.run_id, trial.trial_id],
        )
        if row is None:
            raise MiningStorageError("尝试记录写入后无法读取")
        return row, False

    def _validate_trial_references_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        trial: TrialRecord,
    ) -> None:
        self._require_run_conn(conn, trial.run_id)
        turn = self._require_turn_conn(conn, trial.run_id, trial.turn_index)
        if trial.hypothesis_id is None:
            raise MiningContractError("尝试记录必须绑定假设")
        self._require_hypothesis_conn(conn, trial.run_id, trial.hypothesis_id)
        if trial.batch_id is not None:
            batch = self._require_batch_conn(conn, trial.run_id, trial.batch_id)
            if batch["hypothesis_id"] != trial.hypothesis_id:
                raise MiningContractError("尝试记录的批次与假设不一致")
        if trial.candidate_id is not None:
            candidate = self._require_candidate_conn(
                conn,
                trial.run_id,
                trial.candidate_id,
            )
            lineage = self._row(
                conn,
                "SELECT * FROM mining_v2.candidate_lineage "
                "WHERE run_id=? AND candidate_id=?",
                [trial.run_id, trial.candidate_id],
            )
            if (
                lineage is None
                or lineage["hypothesis_id"] != trial.hypothesis_id
                or (
                    trial.batch_id is not None
                    and lineage["batch_id"] != trial.batch_id
                )
            ):
                raise MiningContractError("尝试记录的候选谱系不一致")
            if candidate["status"] == "stale":
                raise MiningStateError("stale 候选不能追加新尝试")
        if turn["hypothesis_id"] != trial.hypothesis_id:
            raise MiningContractError("尝试记录没有绑定当前 turn 假设")

    def _ensure_trial_budget_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        trial: TrialRecord,
        *,
        enforce_turn_budget: bool = True,
    ) -> None:
        budget = self._budget_for_run_conn(conn, trial.run_id)
        if trial.kind == "proposed":
            total = int(
                conn.execute(
                    "SELECT count(*) FROM mining_v2.trial_record "
                    "WHERE run_id=? AND kind='proposed'",
                    [trial.run_id],
                ).fetchone()[0]
            )
            turn_total = int(
                conn.execute(
                    "SELECT count(*) FROM mining_v2.trial_record "
                    "WHERE run_id=? AND turn_index=? AND kind='proposed'",
                    [trial.run_id, trial.turn_index],
                ).fetchone()[0]
            )
            if total >= budget.max_candidates_per_run:
                raise MiningStateError("搜索预算已用尽：max_candidates_per_run")
            if enforce_turn_budget and turn_total >= budget.max_candidates_per_turn:
                raise MiningStateError("搜索预算已用尽：max_candidates_per_turn")
        if trial.kind == "full_evaluated":
            total = int(
                conn.execute(
                    "SELECT count(*) FROM mining_v2.trial_record "
                    "WHERE run_id=? AND kind='full_evaluated'",
                    [trial.run_id],
                ).fetchone()[0]
            )
            if total >= budget.max_full_evaluations:
                raise MiningStateError("搜索预算已用尽：max_full_evaluations")
        if trial.kind == "holdout_claimed":
            total = int(
                conn.execute(
                    "SELECT count(*) FROM mining_v2.trial_record "
                    "WHERE run_id=? AND kind='holdout_claimed'",
                    [trial.run_id],
                ).fetchone()[0]
            )
            if total >= 1:
                raise MiningStateError("模块测试段已经领取")

    def _counter_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> dict[str, Any]:
        self._require_run_conn(conn, run_id)
        row = self._row(
            conn,
            "SELECT * FROM mining_v2.trial_counter WHERE run_id=?",
            [run_id],
        )
        if row is None:
            raise MiningStorageError("V2 运行缺少尝试计数行")
        return row

    def _increment_counter_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        column: str,
        updated_at: str,
    ) -> None:
        if column not in _COUNT_COLUMNS:
            raise MiningContractError(f"未知尝试计数列：{column}")
        updated = conn.execute(
            f"UPDATE mining_v2.trial_counter SET {column}={column}+1, updated_at=? "
            "WHERE run_id=? RETURNING run_id",
            [updated_at, run_id],
        ).fetchall()
        if not updated:
            raise MiningStorageError("V2 运行缺少尝试计数行")
