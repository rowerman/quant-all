"""V2 分析工具调用缓存和审计存储。"""

from __future__ import annotations

import json
from typing import Any, Mapping

import duckdb

from mining.contract_types import canonical_json, identifier, sha256_json, text, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningStorageError,
)
from mining.search_contract_types import bounded_integer, hash_text, strict_fields
from mining.storage_lock import database_state_lock


class V2AnalysisToolStoreMixin:
    """保存运行内工具缓存并追加每次调用的审计记录。"""

    def get_cached_analysis_tool_result(
        self,
        run_id: str,
        cache_key: str,
    ) -> dict[str, Any] | None:
        """读取当前运行的工具缓存；不存在时返回 None。"""

        run_id = identifier(run_id, "run_id")
        cache_key = hash_text(cache_key, "cache_key")
        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.analysis_tool_cache "
                "WHERE run_id=? AND cache_key=?",
                [run_id, cache_key],
            )
        if row is None:
            return None
        result = self._public_row(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def record_analysis_tool_call(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_index: int,
        cache_key: str,
        context_hash: str,
        turn_request_hash: str,
        candidate_hash: str,
        tool_name: str,
        tool_version: str,
        request_hash: str,
        status: str,
        cache_hit: bool,
        duration_ms: float,
        payload: Mapping[str, Any] | None,
        result_hash: str | None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """在同一事务内写缓存和本次调用审计。"""

        run_id = identifier(run_id, "run_id")
        turn_index = bounded_integer(
            turn_index,
            "turn_index",
            minimum=1,
            maximum=1000,
        )
        call_index = bounded_integer(
            call_index,
            "call_index",
            minimum=1,
            maximum=4,
        )
        cache_key = hash_text(cache_key, "cache_key")
        context_hash = hash_text(context_hash, "context_hash")
        turn_request_hash = hash_text(turn_request_hash, "turn_request_hash")
        candidate_hash = hash_text(candidate_hash, "candidate_hash")
        request_hash = hash_text(request_hash, "request_hash")
        tool_name = identifier(tool_name, "tool_name")
        tool_version = text(tool_version, "tool_version", maximum=20)
        status = identifier(status, "status")
        if not isinstance(cache_hit, bool):
            raise MiningContractError("cache_hit 必须是布尔值")
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or duration_ms < 0
        ):
            raise MiningContractError("duration_ms 必须是非负数值")
        if result_hash is not None:
            result_hash = hash_text(result_hash, "result_hash")
        encoded_payload = (
            canonical_json(dict(payload)) if payload is not None else None
        )
        if encoded_payload is not None and len(encoded_payload.encode("utf-8")) > 65536:
            raise MiningContractError("工具缓存结果超过 65536 字节")
        if error_code is not None:
            error_code = identifier(error_code, "error_code")
        if error_message is not None:
            error_message = str(error_message)[:500]
        audit_id = sha256_json(
            {
                "run_id": run_id,
                "turn_index": turn_index,
                "call_index": call_index,
            }
        )
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        self._require_run_conn(conn, run_id)
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v2.analysis_tool_audit "
                            "WHERE run_id=? AND turn_index=? AND call_index=?",
                            [run_id, turn_index, call_index],
                        )
                        if existing is not None:
                            if existing["cache_key"] != cache_key:
                                raise MiningDuplicateError(
                                    "同一工具调用序号已绑定不同缓存键"
                                )
                            reused = not cache_hit
                        else:
                            now = utc_now_iso()
                            if payload is not None and result_hash is not None:
                                cached = self._row(
                                    conn,
                                    "SELECT * FROM mining_v2.analysis_tool_cache "
                                    "WHERE run_id=? AND cache_key=?",
                                    [run_id, cache_key],
                                )
                                if cached is None:
                                    conn.execute(
                                        "INSERT INTO mining_v2.analysis_tool_cache "
                                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                        [
                                            run_id,
                                            cache_key,
                                            context_hash,
                                            turn_request_hash,
                                            candidate_hash,
                                            tool_name,
                                            tool_version,
                                            request_hash,
                                            result_hash,
                                            encoded_payload,
                                            now,
                                        ],
                                    )
                                elif (
                                    cached["result_hash"] != result_hash
                                    or cached["payload_json"] != encoded_payload
                                ):
                                    raise MiningDuplicateError(
                                        "工具缓存键已绑定不同结果"
                                    )
                            conn.execute(
                                "INSERT INTO mining_v2.analysis_tool_audit "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    turn_index,
                                    call_index,
                                    audit_id,
                                    cache_key,
                                    tool_name,
                                    tool_version,
                                    candidate_hash,
                                    status,
                                    cache_hit,
                                    float(duration_ms),
                                    error_code,
                                    error_message,
                                    result_hash,
                                    now,
                                ],
                            )
                            reused = False
            return {
                "run_id": run_id,
                "turn_index": turn_index,
                "call_index": call_index,
                "audit_id": audit_id,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "status": status,
                "reused": reused,
            }
        except (
            MiningContractError,
            MiningDuplicateError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记分析工具调用失败：{error}") from error

    def list_analysis_tool_calls(
        self,
        run_id: str,
        *,
        turn_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """按 turn 和调用序号读取工具审计摘要。"""

        sql = (
            "SELECT run_id, turn_index, call_index, audit_id, cache_key, tool_name, "
            "tool_version, candidate_hash, status, cache_hit, duration_ms, "
            "error_code, error_message, result_hash, created_at "
            "FROM mining_v2.analysis_tool_audit WHERE run_id=?"
        )
        params: list[Any] = [identifier(run_id, "run_id")]
        if turn_index is not None:
            sql += " AND turn_index=?"
            params.append(
                bounded_integer(
                    turn_index,
                    "turn_index",
                    minimum=1,
                    maximum=1000,
                )
            )
        sql += " ORDER BY turn_index, call_index"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            columns = [item[0] for item in conn.description]
        return [
            self._public_row(dict(zip(columns, row, strict=True)))
            for row in rows
        ]


__all__ = ["V2AnalysisToolStoreMixin"]
