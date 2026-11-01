"""挖掘 V2 运行、冻结上下文和基础事件存储。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.contract_types import canonical_json, sha256_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.search_budget import TrialCounts
from mining.search_context import FrozenSearchContext
from mining.research_run_contracts import ResearchRunInput
from mining.search_run import SearchRunRequest
from mining.storage_lock import database_state_lock


class V2RunStoreMixin:
    """运行与冻结上下文仓库。"""

    def create_run(
        self,
        request: SearchRunRequest,
        context: FrozenSearchContext,
        *,
        research_input: ResearchRunInput | None = None,
    ) -> dict[str, Any]:
        """幂等创建 V2 运行、上下文、计数器和可选资料输入。"""

        if request.budget != context.budget:
            raise MiningContractError("请求预算与 V2 冻结上下文预算不一致")
        if request.llm_profile != context.llm_profile:
            raise MiningContractError("请求模型配置与 V2 冻结上下文不一致")
        if research_input is not None:
            if not isinstance(research_input, ResearchRunInput):
                raise MiningContractError(
                    "research_input 必须是 ResearchRunInput"
                )
            if research_input.run_id != request.run_id:
                raise MiningContractError("资料运行输入没有绑定当前运行")
        created = utc_now_iso()
        reused = False
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        existing = self._row(
                            conn,
                            "SELECT r.request_hash, c.context_hash "
                            "FROM mining_v2.run r JOIN mining_v2.context c "
                            "ON c.context_id=r.context_id "
                            "WHERE r.idempotency_key=?",
                            [request.idempotency_key],
                        )
                        if existing is not None:
                            if (
                                existing["request_hash"] != request.request_hash
                                or existing["context_hash"] != context.context_hash
                            ):
                                raise MiningDuplicateError(
                                    "幂等键已用于不同 V2 请求或冻结上下文"
                                )
                            stored_research = self._row(
                                conn,
                                "SELECT input_hash FROM mining_v2.research_run "
                                "WHERE run_id=?",
                                [request.run_id],
                            )
                            expected_hash = (
                                None
                                if research_input is None
                                else research_input.input_hash
                            )
                            if (
                                stored_research is None
                                and expected_hash is not None
                            ) or (
                                stored_research is not None
                                and stored_research["input_hash"] != expected_hash
                            ):
                                raise MiningDuplicateError(
                                    "幂等键已用于不同的资料运行输入"
                                )
                            reused = True
                        else:
                            stored = self._row(
                                conn,
                                "SELECT context_hash FROM mining_v2.context "
                                "WHERE context_id=?",
                                [context.context_id],
                            )
                            if stored is not None:
                                if stored["context_hash"] != context.context_hash:
                                    raise MiningDuplicateError(
                                        "V2 上下文 ID 已绑定不同内容"
                                    )
                            else:
                                windows = context.windows
                                conn.execute(
                                    "INSERT INTO mining_v2.context VALUES "
                                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                                    "?, ?, ?, ?, ?, ?, ?, ?)",
                                    [
                                        context.context_id,
                                        context.context_hash,
                                        context.data_version,
                                        context.data_fingerprint,
                                        context.label,
                                        context.universe,
                                        context.price_channel,
                                        windows.development_start,
                                        windows.development_end,
                                        windows.selection_start,
                                        windows.selection_end,
                                        windows.final_start,
                                        windows.final_end,
                                        windows.holdout_start,
                                        windows.holdout_end,
                                        context.field_catalog_hash,
                                        context.operator_catalog_hash,
                                        sha256_json(list(context.existing_factors)),
                                        context.budget.budget_hash,
                                        context.windows_hash,
                                        canonical_json(context.to_dict()),
                                        context.created_at,
                                    ],
                                )
                            conn.execute(
                                "INSERT INTO mining_v2.run VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    request.run_id,
                                    context.context_id,
                                    request.idempotency_key,
                                    request.request_hash,
                                    "queued",
                                    request.hypothesis,
                                    canonical_json(request.to_dict()),
                                    None,
                                    created,
                                    created,
                                ],
                            )
                            conn.execute(
                                "INSERT INTO mining_v2.run_control VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    request.run_id,
                                    "queued",
                                    None,
                                    False,
                                    None,
                                    None,
                                    0,
                                    None,
                                    created,
                                ],
                            )
                            if research_input is not None:
                                conn.execute(
                                    "INSERT INTO mining_v2.research_run VALUES "
                                    "(?, ?, ?, ?, ?, ?, ?, ?)",
                                    [
                                        request.run_id,
                                        research_input.input_hash,
                                        research_input.manifest.manifest_hash,
                                        "processing",
                                        canonical_json(research_input.to_dict()),
                                        None,
                                        created,
                                        created,
                                    ],
                                )
                            counts = TrialCounts().to_dict()
                            conn.execute(
                                "INSERT INTO mining_v2.trial_counter VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    request.run_id,
                                    counts["proposed"],
                                    counts["rejected"],
                                    counts["duplicate"],
                                    counts["invalid"],
                                    counts["gate_failed"],
                                    counts["evaluated"],
                                    counts["full_evaluated"],
                                    counts["holdout_claimed"],
                                    counts["invalidated"],
                                    created,
                                ],
                            )
                            self._append_event_conn(
                                conn,
                                request.run_id,
                                "run_created",
                                "run_created",
                                {
                                    "request_hash": request.request_hash,
                                    "context_hash": context.context_hash,
                                },
                                created_at=created,
                            )
        except (MiningContractError, MiningDuplicateError):
            raise
        except (duckdb.Error, OSError) as error:
            raise MiningStorageError(f"创建 V2 运行失败：{error}") from error
        result = self.get_run(request.run_id)
        result["reused"] = reused
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        """读取 V2 运行及冻结上下文。"""

        try:
            with self._connection() as conn:
                run = self._row(
                    conn,
                    "SELECT r.*, c.context_hash, c.payload_json AS context_payload_json, "
                    "ctl.status AS control_status, ctl.updated_at AS control_updated_at "
                    "FROM mining_v2.run r JOIN mining_v2.context c "
                    "ON c.context_id=r.context_id "
                    "LEFT JOIN mining_v2.run_control ctl ON ctl.run_id=r.run_id "
                    "WHERE r.run_id=?",
                    [run_id],
                )
        except duckdb.Error as error:
            raise MiningStorageError(f"读取 V2 运行失败：{error}") from error
        if run is None:
            raise MiningNotFoundError(f"V2 搜索运行不存在：{run_id}")
        run["request"] = json.loads(run.pop("request_json"))
        run["context"] = json.loads(run.pop("context_payload_json"))
        run["context_hash"] = run.pop("context_hash")
        run["status"] = run.pop("control_status") or run["status"]
        run["updated_at"] = run.pop("control_updated_at") or run["updated_at"]
        return run

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """按创建时间倒序列出 V2 运行。"""

        limit = max(1, min(int(limit), 200))
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT r.run_id, r.idempotency_key, "
                "coalesce(ctl.status, r.status) AS status, r.hypothesis, "
                "r.trace_hash, r.created_at, "
                "coalesce(ctl.updated_at, r.updated_at) AS updated_at "
                "FROM mining_v2.run r LEFT JOIN mining_v2.run_control ctl "
                "ON ctl.run_id=r.run_id "
                "ORDER BY r.created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            columns = [item[0] for item in conn.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]
