"""主服务侧受控分析工具网关。"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from mining.analysis_tool_contracts import (
    ANALYSIS_TOOL_MAX_CALLS_PER_TURN,
    ANALYSIS_TOOL_TIMEOUT_SECONDS,
    TOOL_BY_NAME,
    AnalysisToolRequest,
    AnalysisToolResult,
    finite_float,
    tool_catalog,
)
from mining.contract_types import canonical_json, sha256_json
from mining.errors import (
    AnalysisToolArgumentError,
    AnalysisToolExecutionError,
    AnalysisToolNotFoundError,
    AnalysisToolResultError,
    AnalysisToolTimeoutError,
    AnalysisToolVisibilityError,
    MiningError,
)
from mining.expression_ast import BinaryNode
from mining.expression_parser import parse_expression
from mining.logging_conf import format_event, get_mining_logger
from mining.search_context import FrozenSearchContext, ensure_agent_visible
from mining.search_stability import generate_primary_window_variants
from mining.storage_v2 import MiningV2Store

_LOGGER = get_mining_logger("mining.analysis_tools")


def _safe_text(value: Any, maximum: int = 500) -> str:
    """把异常摘要限制为脱敏短文本。"""

    result = str(value)
    for marker in (str(Path.home()), "/home/", "/data/", "/app/"):
        result = result.replace(marker, "[PATH]")
    return result[:maximum]


def _run_scope_hash() -> str:
    return sha256_json({"scope": "run"})


def _top_level_components(expression: str) -> tuple[str, ...]:
    """按确定性顺序返回加法树的顶层组成项。"""

    node = parse_expression(expression, max_depth=8)
    components: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, BinaryNode) and item.operator == "+":
            visit(item.left)
            visit(item.right)
        else:
            components.append(item.canonical())

    visit(node)
    return tuple(sorted(components))


class AnalysisToolGateway:
    """校验并执行当前运行的白名单分析工具。"""

    def __init__(
        self,
        *,
        store: MiningV2Store,
        timeout_seconds: float = ANALYSIS_TOOL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise AnalysisToolArgumentError("工具超时必须大于 0")
        self.store = store
        self.timeout_seconds = float(timeout_seconds)
        self.clock = clock

    def catalog(self) -> tuple[Mapping[str, Any], ...]:
        """返回 Worker 可请求的固定工具目录。"""

        return tool_catalog()

    def execute_batch(
        self,
        *,
        run_id: str,
        context: FrozenSearchContext,
        turn_index: int,
        turn_request_hash: str,
        requests: Sequence[AnalysisToolRequest],
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[AnalysisToolResult, ...]:
        """顺序执行一轮工具请求，并持久化缓存与审计。"""

        if len(requests) > ANALYSIS_TOOL_MAX_CALLS_PER_TURN:
            raise AnalysisToolArgumentError(
                f"每轮最多调用 {ANALYSIS_TOOL_MAX_CALLS_PER_TURN} 次工具"
            )
        if len({item.call_id for item in requests}) != len(requests):
            raise AnalysisToolArgumentError("工具 call_id 不得重复")
        results: list[AnalysisToolResult] = []
        for call_index, request in enumerate(requests, start=1):
            if should_cancel is not None and should_cancel():
                raise AnalysisToolTimeoutError("分析工具调用已取消")
            results.append(
                self._execute_one(
                    run_id=run_id,
                    context=context,
                    turn_index=turn_index,
                    turn_request_hash=turn_request_hash,
                    call_index=call_index,
                    request=request,
                )
            )
        return tuple(results)

    def _execute_one(
        self,
        *,
        run_id: str,
        context: FrozenSearchContext,
        turn_index: int,
        turn_request_hash: str,
        call_index: int,
        request: AnalysisToolRequest,
    ) -> AnalysisToolResult:
        definition = TOOL_BY_NAME.get(request.tool_name)
        if definition is None:
            raise AnalysisToolNotFoundError(
                f"未知分析工具：{request.tool_name!r}"
            )
        candidate_hash = _run_scope_hash()
        candidate_id = request.arguments.get("candidate_id")
        candidate_lookup_error: Exception | None = None
        if candidate_id is not None:
            try:
                candidate = self.store.get_candidate(run_id, str(candidate_id))
                candidate_hash = str(candidate["candidate_hash"])
            except Exception as exc:
                candidate_lookup_error = exc
        cache_key = sha256_json(
            {
                "run_id": run_id,
                "context_hash": context.context_hash,
                "turn_request_hash": turn_request_hash,
                "candidate_hash": candidate_hash,
                "tool_name": request.tool_name,
                "tool_version": definition.version,
                "arguments": request.arguments,
            }
        )
        cached = self.store.get_cached_analysis_tool_result(run_id, cache_key)
        if cached is not None:
            cached_result = AnalysisToolResult.from_dict(cached["payload"])
            result = AnalysisToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                tool_version=definition.version,
                payload=cached_result.payload,
            )
            self.store.record_analysis_tool_call(
                run_id=run_id,
                turn_index=turn_index,
                call_index=call_index,
                cache_key=cache_key,
                context_hash=context.context_hash,
                turn_request_hash=turn_request_hash,
                candidate_hash=candidate_hash,
                tool_name=request.tool_name,
                tool_version=definition.version,
                candidate_id=(
                    str(candidate_id) if candidate_id is not None else None
                ),
                request_hash=request.request_hash,
                arguments_summary=request.arguments,
                status="cache_hit",
                cache_hit=True,
                duration_ms=0.0,
                payload=None,
                result_hash=result.result_hash,
            )
            return result

        if candidate_lookup_error is not None:
            error = AnalysisToolExecutionError(
                f"候选证据查询失败：{_safe_text(candidate_lookup_error)}"
            )
            self._record_failure(
                run_id=run_id,
                context=context,
                turn_index=turn_index,
                turn_request_hash=turn_request_hash,
                call_index=call_index,
                cache_key=cache_key,
                candidate_hash=candidate_hash,
                request=request,
                definition_version=definition.version,
                duration_ms=0.0,
                error=error,
            )
            raise error from candidate_lookup_error

        started = self.clock()
        try:
            payload = self._dispatch(
                run_id=run_id,
                context=context,
                request=request,
            )
            ensure_agent_visible(payload, path="tool_result.payload")
            result = AnalysisToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                tool_version=definition.version,
                payload=payload,
            )
            result_hash = result.result_hash
            status = "succeeded"
            error_code = None
            error_message = None
        except AnalysisToolTimeoutError:
            duration_ms = (self.clock() - started) * 1000
            self._record_failure(
                run_id=run_id,
                context=context,
                turn_index=turn_index,
                turn_request_hash=turn_request_hash,
                call_index=call_index,
                cache_key=cache_key,
                candidate_hash=candidate_hash,
                request=request,
                definition_version=definition.version,
                duration_ms=duration_ms,
                error=AnalysisToolTimeoutError("分析工具调用超时"),
            )
            raise
        except Exception as exc:
            duration_ms = (self.clock() - started) * 1000
            error = (
                exc
                if isinstance(exc, MiningError)
                else AnalysisToolExecutionError(
                    f"分析工具执行失败：{_safe_text(exc)}"
                )
            )
            if not isinstance(
                error,
                (
                    AnalysisToolArgumentError,
                    AnalysisToolNotFoundError,
                    AnalysisToolResultError,
                    AnalysisToolVisibilityError,
                ),
            ):
                error = AnalysisToolExecutionError(_safe_text(error))
            self._record_failure(
                run_id=run_id,
                context=context,
                turn_index=turn_index,
                turn_request_hash=turn_request_hash,
                call_index=call_index,
                cache_key=cache_key,
                candidate_hash=candidate_hash,
                request=request,
                definition_version=definition.version,
                duration_ms=duration_ms,
                error=error,
            )
            raise error from exc
        duration_ms = (self.clock() - started) * 1000
        if duration_ms / 1000 > self.timeout_seconds:
            error = AnalysisToolTimeoutError("分析工具调用超过时限")
            self._record_failure(
                run_id=run_id,
                context=context,
                turn_index=turn_index,
                turn_request_hash=turn_request_hash,
                call_index=call_index,
                cache_key=cache_key,
                candidate_hash=candidate_hash,
                request=request,
                definition_version=definition.version,
                duration_ms=duration_ms,
                error=error,
            )
            raise error
        self.store.record_analysis_tool_call(
            run_id=run_id,
            turn_index=turn_index,
            call_index=call_index,
            cache_key=cache_key,
            context_hash=context.context_hash,
            turn_request_hash=turn_request_hash,
            candidate_hash=candidate_hash,
            tool_name=request.tool_name,
            tool_version=definition.version,
            candidate_id=(
                str(candidate_id) if candidate_id is not None else None
            ),
            request_hash=request.request_hash,
            arguments_summary=request.arguments,
            status=status,
            cache_hit=False,
            duration_ms=duration_ms,
            payload=result.to_dict(),
            result_hash=result_hash,
            error_code=error_code,
            error_message=error_message,
        )
        _LOGGER.info(
            format_event(
                {
                    "event": "analysis_tool_succeeded",
                    "run_id": run_id,
                    "turn_index": turn_index,
                    "call_index": call_index,
                    "tool_name": request.tool_name,
                    "tool_version": definition.version,
                    "result_hash": result_hash,
                    "duration_ms": round(duration_ms, 1),
                    "cache_hit": False,
                }
            )
        )
        return result

    def _record_failure(
        self,
        *,
        run_id: str,
        context: FrozenSearchContext,
        turn_index: int,
        turn_request_hash: str,
        call_index: int,
        cache_key: str,
        candidate_hash: str,
        request: AnalysisToolRequest,
        definition_version: str,
        duration_ms: float,
        error: MiningError,
    ) -> None:
        self.store.record_analysis_tool_call(
            run_id=run_id,
            turn_index=turn_index,
            call_index=call_index,
            cache_key=cache_key,
            context_hash=context.context_hash,
            turn_request_hash=turn_request_hash,
            candidate_hash=candidate_hash,
            tool_name=request.tool_name,
            tool_version=definition_version,
            candidate_id=(
                str(request.arguments["candidate_id"])
                if request.arguments.get("candidate_id") is not None
                else None
            ),
            request_hash=request.request_hash,
            arguments_summary=request.arguments,
            status="failed",
            cache_hit=False,
            duration_ms=duration_ms,
            payload=None,
            result_hash=None,
            error_code=error.code,
            error_message=_safe_text(error.message),
        )
        _LOGGER.warning(
            format_event(
                {
                    "event": "analysis_tool_failed",
                    "run_id": run_id,
                    "turn_index": turn_index,
                    "call_index": call_index,
                    "tool_name": request.tool_name,
                    "tool_version": definition_version,
                    "error_code": error.code,
                    "duration_ms": round(duration_ms, 1),
                }
            )
        )

    def _dispatch(
        self,
        *,
        run_id: str,
        context: FrozenSearchContext,
        request: AnalysisToolRequest,
    ) -> Mapping[str, Any]:
        handlers = {
            "lookup_fields": self._lookup_fields,
            "lookup_operators": self._lookup_operators,
            "run_summary": self._run_summary,
            "candidate_evidence": self._candidate_evidence,
            "candidate_correlation": self._candidate_correlation,
            "candidate_ablation": self._candidate_ablation,
            "window_neighborhood": self._window_neighborhood,
        }
        return handlers[request.tool_name](run_id, context, request.arguments)

    @staticmethod
    def _lookup_fields(
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del run_id
        query = str(arguments["query"]).casefold()
        matches = [
            dict(item)
            for item in context.field_catalog
            if query in canonical_json(item).casefold()
        ]
        matches.sort(key=lambda item: str(item.get("name", "")))
        limit = int(arguments["limit"])
        return {
            "scope": "development_catalog",
            "catalog_hash": context.field_catalog_hash,
            "match_count": len(matches),
            "items": matches[:limit],
            "truncated": len(matches) > limit,
        }

    @staticmethod
    def _lookup_operators(
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del run_id
        query = str(arguments["query"]).casefold()
        matches = [
            dict(item)
            for item in context.operator_catalog
            if query in canonical_json(item).casefold()
        ]
        matches.sort(key=lambda item: str(item.get("name", "")))
        limit = int(arguments["limit"])
        return {
            "scope": "development_catalog",
            "catalog_hash": context.operator_catalog_hash,
            "match_count": len(matches),
            "items": matches[:limit],
            "truncated": len(matches) > limit,
        }

    def _run_summary(
        self,
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del arguments
        trace = self.store.get_search_trace(run_id)
        counts = self.store.trial_counts(run_id).to_dict()
        return {
            "scope": "current_run_development",
            "context_hash": context.context_hash,
            "hypotheses": len(trace["hypotheses"]),
            "branches": len(
                {item["branch_id"] for item in trace["hypotheses"]}
            ),
            "turns": len(trace["turns"]),
            "candidates": len(trace["candidates"]),
            "failed_candidates": sum(
                item["status"]
                in {
                    "technical_failed",
                    "screened_out",
                    "holdout_failed",
                    "failed",
                }
                for item in trace["candidates"]
            ),
            "trial_counts": {
                key: value
                for key, value in counts.items()
                if key != "holdout_claimed"
            },
        }

    def _candidate_record(
        self,
        run_id: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.store.get_candidate(run_id, str(arguments["candidate_id"]))

    def _candidate_evidence(
        self,
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        candidate = self._candidate_record(run_id, arguments)
        rows = self.store.list_evaluation_evidence(
            run_id,
            str(candidate["candidate_id"]),
        )
        by_stage = {
            row["evidence"]["stage"]: row["evidence"]
            for row in rows
            if row["evidence"]["stage"] in {"technical", "development"}
        }
        technical = by_stage.get("technical", {})
        development = by_stage.get("development", {})
        metrics = development.get("metrics", {})
        values = technical.get("metrics", {}).get("values", {})
        return {
            "scope": "candidate_development_evidence",
            "context_hash": context.context_hash,
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
            "status": candidate["status"],
            "technical_status": technical.get("status"),
            "development_status": development.get("status"),
            "metrics": {
                "rank_ic": finite_float(metrics.get("rank_ic_mean")),
                "rank_ic_ir": finite_float(metrics.get("rank_ic_ir")),
                "year_sign_consistency": finite_float(
                    metrics.get("year_sign_consistency")
                ),
                "coverage": finite_float(values.get("coverage")),
                "valid_days": metrics.get("valid_days"),
                "sample_count": metrics.get("sample_count"),
            },
        }

    def _candidate_correlation(
        self,
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        candidate_id = str(arguments["candidate_id"])
        self._candidate_record(run_id, arguments)
        trace = self.store.get_search_trace(run_id)
        matches = [
            cluster
            for packet in trace["evidence"]
            for cluster in packet.get("correlation_clusters", ())
            if candidate_id
            in {str(item) for item in cluster.get("candidate_ids", ())}
        ]
        if not matches:
            return {
                "scope": "candidate_development_correlation",
                "context_hash": context.context_hash,
                "candidate_id": candidate_id,
                "status": "not_computable",
                "reason": "当前运行尚未登记开发期相关摘要",
                "cluster": None,
            }
        return {
            "scope": "candidate_development_correlation",
            "context_hash": context.context_hash,
            "candidate_id": candidate_id,
            "status": "available",
            "reason": None,
            "cluster": dict(matches[-1]),
        }

    def _candidate_ablation(
        self,
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        candidate = self._candidate_record(run_id, arguments)
        expression = str(candidate["payload"]["canonical_expression"])
        components = _top_level_components(expression)
        evidence = self._candidate_evidence(run_id, context, arguments)
        return {
            "scope": "candidate_development_ablation",
            "context_hash": context.context_hash,
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
            "status": "available" if len(components) > 1 else "not_applicable",
            "reason": (
                None
                if len(components) > 1
                else "候选没有可拆分的顶层加项"
            ),
            "base_rank_ic": evidence["metrics"]["rank_ic"],
            "components": [
                {
                    "index": index,
                    "component_hash": sha256_json({"component": item}),
                    "component": item,
                }
                for index, item in enumerate(components, start=1)
            ],
        }

    def _window_neighborhood(
        self,
        run_id: str,
        context: FrozenSearchContext,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from mining.search_orchestrator_support import candidate_from_v2_record

        candidate_record = self._candidate_record(run_id, arguments)
        candidate = candidate_from_v2_record(candidate_record)
        plan = generate_primary_window_variants(candidate)
        stability_rows = self.store.list_evaluation_evidence(
            run_id,
            candidate.candidate_id,
            stage="stability",
        )
        stability = (
            stability_rows[-1]["evidence"]["metrics"]
            if stability_rows
            else None
        )
        return {
            "scope": "candidate_development_neighborhood",
            "context_hash": context.context_hash,
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.content_hash,
            "status": plan.status,
            "reason": plan.reason,
            "base_window": plan.base_window,
            "variants": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "candidate_hash": item.candidate.content_hash,
                    "window": item.variant_window,
                    "delta": item.delta,
                }
                for item in plan.variants
            ],
            "stability": stability,
        }


__all__ = ["AnalysisToolGateway"]
