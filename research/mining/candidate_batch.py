"""同轮候选的 DAG 批量物化、失败隔离和单候选回退。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
import uuid

from calculate import open_source
from common.paths import FACTOR_DIR
from mining.candidate_materialization import (
    CandidateComputation,
    MaterializedCandidate,
    _candidate_base,
    _computation,
    _identity,
    _log_materialized,
    _publish_staging,
    materialize_candidate,
)
from mining.catalog import MiningCatalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext
from mining.errors import (
    BatchMaterializationError,
    CatalogDriftError,
)
from mining.expression_batch import compile_candidate_batch
from mining.expression_dag import (
    CandidateExpressionPlan,
    build_candidate_expression_plan,
)
from mining.expression_validation import (
    ValidatedExpression,
    validate_candidate_expression,
)
from mining.logging_conf import format_event, get_mining_logger

_LOGGER = get_mining_logger("mining.evaluation.materialization")


@dataclass(frozen=True)
class CandidateBatchItem:
    """批量物化中的一个候选及其隔离根目录。"""

    candidate: CandidateDefinition
    candidate_root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_root", Path(self.candidate_root))


@dataclass(frozen=True)
class CandidateBatchFailure:
    """单个候选物化失败，不影响同批其他候选。"""

    candidate_id: str
    error_code: str
    message: str


@dataclass(frozen=True)
class CandidateBatchDiagnostics:
    """批量物化的确定性和资源诊断。"""

    candidate_count: int
    reused_count: int
    written_count: int
    shared_node_count: int
    materialized_node_count: int
    analysis_sql_count: int
    daily_source_reference_count: int
    temporary_bytes: int
    elapsed_ms: float
    fallback_reason: str | None


@dataclass(frozen=True)
class CandidateBatchResult:
    """批量物化的成功项、候选级失败和诊断。"""

    materialized: tuple[MaterializedCandidate, ...]
    failures: tuple[CandidateBatchFailure, ...]
    diagnostics: CandidateBatchDiagnostics


def _batch_failure(candidate_id: str, exc: Exception) -> CandidateBatchFailure:
    """把候选级异常转换为稳定失败记录。"""

    return CandidateBatchFailure(
        candidate_id=candidate_id,
        error_code=getattr(exc, "code", "MINING-106"),
        message=str(exc)[:2000],
    )


def _has_shared_computation(plan: CandidateExpressionPlan) -> bool:
    """判断 DAG 是否包含可复用的计算节点，而非只重复引用字段或常量。"""

    return any(
        node.reference_count > 1 and node.node_kind not in {"number", "field"}
        for node in plan.nodes
    )


def _copy_combined_batch_query(
    connection,
    sql: str,
    combined: Path,
) -> tuple[int, ...]:
    """执行共享查询并将宽结果写入临时 Parquet。

    参数:
        connection: 当前 DuckDB 连接。
        sql: DAG 批量查询。
        combined: 临时宽 Parquet 路径。
    """

    quoted = str(combined).replace("'", "''")
    connection.execute(
        f"COPY ({sql}) TO '{quoted}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
    )
    return tuple(
        int(row[0])
        for row in connection.execute(
            "SELECT DISTINCT year(trade_date) FROM read_parquet(?) ORDER BY 1",
            [str(combined)],
        ).fetchall()
    )


def _publish_from_combined(
    item: CandidateBatchItem,
    validated: ValidatedExpression,
    computation: CandidateComputation,
    context: FrozenContext,
    *,
    combined: Path,
    years: tuple[int, ...],
    start: str,
    end: str,
    connection,
) -> MaterializedCandidate:
    """把一个候选列从批量宽结果拆入独立 staging 并原子发布。

    参数:
        item: 候选及其目标根目录。
        validated: 已验证表达式。
        computation: 单候选兼容计算定义。
        context: 当前冻结上下文。
        combined/years: 临时宽结果及其年份。
        start/end: 物化日期区间。
        connection: 复用当前 DuckDB 连接。
    """

    factor_root = item.candidate_root / "factor"
    staging = factor_root.parent / f".{factor_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    quoted = str(combined).replace("'", "''")
    try:
        for year in years:
            target = staging / f"year={year}" / "part.parquet"
            target.parent.mkdir(parents=True)
            target_sql = str(target).replace("'", "''")
            connection.execute(
                f"COPY (SELECT stock_code, trade_date, {item.candidate.name} "
                f"FROM read_parquet('{quoted}') WHERE year(trade_date)={year}) "
                f"TO '{target_sql}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
            )
        materialized = _publish_staging(
            staging,
            factor_root,
            identity=_identity(validated, item.candidate, context, start, end),
            candidate=item.candidate,
            computation=computation,
            validated=validated,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _log_materialized(
        item.candidate,
        materialized,
        operation="materialize_candidate_batch_item",
    )
    return materialized


def _finish_batch_result(
    *,
    started: float,
    candidate_count: int,
    reused_count: int,
    written_count: int,
    materialized: list[MaterializedCandidate],
    failures: list[CandidateBatchFailure],
    shared_node_count: int,
    materialized_node_count: int,
    analysis_sql_count: int,
    daily_source_reference_count: int,
    temporary_bytes: int,
    fallback_reason: str | None,
) -> CandidateBatchResult:
    """构造批量结果并记录统一诊断日志。"""

    diagnostics = CandidateBatchDiagnostics(
        candidate_count=candidate_count,
        reused_count=reused_count,
        written_count=written_count,
        shared_node_count=shared_node_count,
        materialized_node_count=materialized_node_count,
        analysis_sql_count=analysis_sql_count,
        daily_source_reference_count=daily_source_reference_count,
        temporary_bytes=temporary_bytes,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        fallback_reason=fallback_reason,
    )
    _LOGGER.info(
        format_event(
            {
                "operation": "materialize_candidate_batch",
                "candidate_count": candidate_count,
                "reused_count": reused_count,
                "written_count": written_count,
                "failed_count": len(failures),
                "shared_node_count": shared_node_count,
                "analysis_sql_count": analysis_sql_count,
                "daily_source_reference_count": daily_source_reference_count,
                "temporary_bytes": temporary_bytes,
                "elapsed_ms": diagnostics.elapsed_ms,
                "fallback_reason": fallback_reason,
            }
        )
    )
    return CandidateBatchResult(
        materialized=tuple(materialized),
        failures=tuple(failures),
        diagnostics=diagnostics,
    )


def _fallback_candidates(
    pending,
    context: FrozenContext,
    *,
    start: str,
    end: str,
    started: float,
    candidate_count: int,
    reused_count: int,
    materialized: list[MaterializedCandidate],
    failures: list[CandidateBatchFailure],
    reason: str,
    temporary_bytes: int,
) -> CandidateBatchResult:
    """逐候选执行兼容路径，并保留候选级失败。

    参数:
        pending: 尚未物化的候选及其校验结果。
        context: 当前冻结上下文。
        start/end: 物化日期区间。
        started: 批次计时起点。
        reused_count: 批量前已经复用的候选数。
        materialized/failures: 已累计结果。
        reason: 回退原因。
        temporary_bytes: 批量临时文件大小。
    """

    before = len(materialized)
    for item, _validated, _plan in pending:
        try:
            candidate = materialize_candidate(
                item.candidate,
                context,
                item.candidate_root,
                start=start,
                end=end,
            )
        except Exception as exc:
            failures.append(_batch_failure(item.candidate.candidate_id, exc))
        else:
            materialized.append(candidate)
    if pending and len(materialized) == before and not materialized:
        details = "; ".join(
            f"{failure.candidate_id}: {failure.error_code}" for failure in failures
        )
        raise BatchMaterializationError(
            f"批量物化与逐候选回退均未成功：{reason}；{details}"
        )
    return _finish_batch_result(
        started=started,
        candidate_count=candidate_count,
        reused_count=reused_count,
        written_count=len(materialized) - reused_count,
        materialized=materialized,
        failures=failures,
        shared_node_count=0,
        materialized_node_count=0,
        analysis_sql_count=len(pending),
        daily_source_reference_count=len(pending) * 2,
        temporary_bytes=temporary_bytes,
        fallback_reason=reason,
    )


def materialize_candidate_batch(
    items,
    context: FrozenContext,
    *,
    start: str,
    end: str,
    catalog: MiningCatalog | None = None,
) -> CandidateBatchResult:
    """批量物化同轮候选；无共享子树时自动使用单候选兼容路径。

    参数:
        items: 同一运行和冻结上下文中的候选及根目录。
        context: 当前冻结上下文。
        start/end: 物化日期区间。
        catalog: 可选冻结目录。
    """

    started = time.perf_counter()
    batch_items = tuple(items)
    if not batch_items:
        raise BatchMaterializationError("候选批量不能为空")
    if not all(isinstance(item, CandidateBatchItem) for item in batch_items):
        raise BatchMaterializationError("候选批量只能包含 CandidateBatchItem")
    candidate_ids = [item.candidate.candidate_id for item in batch_items]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise BatchMaterializationError("候选批量包含重复 candidate_id")
    bases = tuple(_candidate_base(item.candidate_root) for item in batch_items)
    normalized = tuple(
        CandidateBatchItem(item.candidate, base)
        for item, base in zip(batch_items, bases, strict=True)
    )
    roots = tuple(item.candidate_root.resolve() for item in normalized)
    if len(roots) != len(set(roots)):
        raise BatchMaterializationError("候选批量包含重复根目录")
    batch_parent = Path(os.path.commonpath(roots))
    if batch_parent == Path(batch_parent.anchor):
        raise BatchMaterializationError("同批候选缺少共同的运行目录")
    formal = FACTOR_DIR.resolve()
    if batch_parent == formal or batch_parent.is_relative_to(formal):
        raise BatchMaterializationError("批量临时目录不能位于正式因子目录内")

    materialized: list[MaterializedCandidate] = []
    failures: list[CandidateBatchFailure] = []
    pending: list[
        tuple[CandidateBatchItem, ValidatedExpression, CandidateComputation]
    ] = []
    for item in normalized:
        try:
            validated = validate_candidate_expression(
                item.candidate,
                context,
                catalog=catalog,
            )
        except Exception as exc:
            failures.append(_batch_failure(item.candidate.candidate_id, exc))
            continue
        computation = _computation(item.candidate, validated)
        factor_root = item.candidate_root / "factor"
        if not factor_root.exists():
            pending.append((item, validated, computation))
            continue
        try:
            existing = materialize_candidate(
                item.candidate,
                context,
                item.candidate_root,
                start=start,
                end=end,
            )
        except Exception as exc:
            failures.append(_batch_failure(item.candidate.candidate_id, exc))
        else:
            materialized.append(existing)

    reused_count = len(materialized)
    if len(pending) < 2:
        return _fallback_candidates(
            pending,
            context,
            start=start,
            end=end,
            started=started,
            candidate_count=len(normalized),
            reused_count=reused_count,
            materialized=materialized,
            failures=failures,
            reason="insufficient_pending_candidates",
            temporary_bytes=0,
        )

    try:
        plan = build_candidate_expression_plan(
            (item.candidate for item, _validated, _computation in pending),
            context,
            catalog=catalog,
        )
        if not _has_shared_computation(plan):
            return _fallback_candidates(
                pending,
                context,
                start=start,
                end=end,
                started=started,
                candidate_count=len(normalized),
                reused_count=reused_count,
                materialized=materialized,
                failures=failures,
                reason="no_shared_nodes",
                temporary_bytes=0,
            )
        compiled = compile_candidate_batch(
            plan,
            (item.candidate for item, _validated, _computation in pending),
            context,
            start=start,
            end=end,
            catalog=catalog,
        )
    except CatalogDriftError as exc:
        raise BatchMaterializationError(
            f"批量表达式目录与冻结上下文不一致：{exc}"
        ) from exc
    except Exception as exc:
        return _fallback_candidates(
            pending,
            context,
            start=start,
            end=end,
            started=started,
            candidate_count=len(normalized),
            reused_count=reused_count,
            materialized=materialized,
            failures=failures,
            reason=f"batch_compile_failed:{type(exc).__name__}",
            temporary_bytes=0,
        )

    batch_root = batch_parent / f".batch-materialization.staging-{uuid.uuid4().hex}"
    combined = batch_root / "combined.parquet"
    batch_root.mkdir(parents=True)
    connection = open_source()
    temporary_bytes = 0
    try:
        try:
            years = _copy_combined_batch_query(connection, compiled.sql, combined)
            temporary_bytes = combined.stat().st_size
        except Exception as exc:
            return _fallback_candidates(
                pending,
                context,
                start=start,
                end=end,
                started=started,
                candidate_count=len(normalized),
                reused_count=reused_count,
                materialized=materialized,
                failures=failures,
                reason=f"batch_query_failed:{type(exc).__name__}",
                temporary_bytes=combined.stat().st_size if combined.exists() else 0,
            )
        for item, validated, computation in pending:
            try:
                candidate = _publish_from_combined(
                    item,
                    validated,
                    computation,
                    context,
                    combined=combined,
                    years=years,
                    start=start,
                    end=end,
                    connection=connection,
                )
            except Exception as exc:
                failures.append(_batch_failure(item.candidate.candidate_id, exc))
            else:
                materialized.append(candidate)
    finally:
        connection.close()
        shutil.rmtree(batch_root, ignore_errors=True)

    return _finish_batch_result(
        started=started,
        candidate_count=len(normalized),
        reused_count=reused_count,
        written_count=len(materialized) - reused_count,
        materialized=materialized,
        failures=failures,
        shared_node_count=plan.diagnostics.shared_node_count,
        materialized_node_count=compiled.materialized_node_count,
        analysis_sql_count=compiled.analysis_sql_count,
        daily_source_reference_count=compiled.daily_source_reference_count,
        temporary_bytes=temporary_bytes,
        fallback_reason=None,
    )
