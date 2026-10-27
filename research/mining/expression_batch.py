"""把候选表达式 DAG 编译为一次共享计算的 DuckDB 查询。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from factors.ops import WINDOW_DEF
from factors.sources import SOURCES, join_clause
from mining.catalog import MiningCatalog, load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext
from mining.errors import CandidateError, CatalogDriftError
from mining.expression_compiler import CompiledValue, ExpressionCompiler
from mining.expression_dag import CandidateExpressionPlan
from mining.expression_parser import parse_expression
from scripts.build_factors.sql import base_cte, date_literal


@dataclass(frozen=True)
class CompiledCandidateBatch:
    """一条可物化全部候选根列的共享节点查询。"""

    plan_hash: str
    candidate_names: tuple[str, ...]
    materialized_node_ids: tuple[str, ...]
    materialized_node_count: int
    analysis_sql_count: int
    daily_source_reference_count: int
    sql: str


def compile_candidate_batch(
    plan: CandidateExpressionPlan,
    candidates: Iterable[CandidateDefinition],
    context: FrozenContext,
    *,
    start: str,
    end: str,
    catalog: MiningCatalog | None = None,
    db: str = "src",
) -> CompiledCandidateBatch:
    """按 DAG 后序生成共享 CTE，并输出每个候选的独立根列。

    参数:
        plan: 已绑定冻结上下文的候选 DAG。
        candidates: 与计划根节点一致的候选定义。
        context: 当前冻结上下文。
        start/end: 输出行集的首尾日期。
        catalog: 可选冻结目录。
        db: DuckDB 主库别名。
    """

    catalog = catalog or load_catalog()
    if plan.context_hash != context.context_hash:
        raise CatalogDriftError("表达式计划与冻结上下文不一致")
    if plan.catalog_hash != catalog.catalog_hash:
        raise CatalogDriftError("表达式计划与目录不一致")
    candidate_list = tuple(candidates)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_list}
    if len(candidate_by_id) != len(candidate_list):
        raise CandidateError("批量编译包含重复候选")
    planned_ids = {candidate.candidate_id for candidate in plan.candidates}
    if set(candidate_by_id) != planned_ids:
        raise CandidateError("批量候选与表达式计划不一致")
    candidate_names = tuple(
        candidate_by_id[root.candidate_id].name for root in plan.candidates
    )
    if len(candidate_names) != len(set(candidate_names)):
        raise CandidateError("批量候选名重复")

    source_keys = sorted(
        {
            source
            for candidate in candidate_list
            for source in _validated_sources(candidate, context, catalog)
        }
    )
    source_ctes = [
        f"src_{key} AS ({SOURCES[key].sql.format(prefix=f'{db}.')})"
        for key in source_keys
    ]
    source_select = "".join(
        f", src_{key}.{column}" for key in source_keys for column in SOURCES[key].columns
    )
    source_join = " ".join(join_clause(key) for key in source_keys)
    ctes = [
        "spine AS (SELECT stock_code, trade_date FROM "
        f"{db}.md.daily_hfq WHERE trade_date BETWEEN {date_literal(start)} "
        f"AND {date_literal(end)})",
        *source_ctes,
        base_cte(end, db),
    ]
    relation = "base"
    if source_keys:
        ctes.append(f"enriched AS (SELECT base.*{source_select} FROM base {source_join})")
        relation = "enriched"

    node_by_expression = {
        node.canonical_expression: node for node in plan.nodes
    }
    if len(node_by_expression) != len(plan.nodes):
        raise CandidateError("表达式计划包含重复规范表达式")
    values: dict[str, CompiledValue] = {}
    materialized_node_ids: list[str] = []
    compiler = ExpressionCompiler(
        catalog,
        excluded_fields=context.budget.excluded_fields,
    )

    for node_index, node in enumerate(plan.nodes, start=1):
        ast = parse_expression(
            node.canonical_expression,
            max_depth=context.budget.max_expression_depth,
        )

        def resolve_child(child):
            child_node = node_by_expression.get(child.canonical())
            if child_node is None:
                raise CandidateError("表达式 DAG 子节点缺失")
            try:
                return values[child_node.node_id]
            except KeyError as exc:
                raise CandidateError("表达式 DAG 节点顺序无效") from exc

        value = compiler.compile_value(ast, resolve_child)
        shared_computation = (
            node.reference_count > 1
            and node.node_kind not in {"number", "field"}
        )
        if value.needs_layer or shared_computation:
            alias = f"cse_{node.structural_hash[:24]}"
            cte_name = f"batch_{node_index}"
            ctes.append(
                f"{cte_name} AS (SELECT *, {value.sql} AS {alias} "
                f"FROM {relation} WINDOW w AS {WINDOW_DEF})"
            )
            relation = cte_name
            value = CompiledValue(alias, value.dimension, value.lookback)
            materialized_node_ids.append(node.node_id)
        values[node.node_id] = value

    root_columns = ["spine.stock_code", "spine.trade_date"]
    for root, name in zip(plan.candidates, candidate_names, strict=True):
        value = values[root.root_node_id]
        root_columns.append(
            f"CAST(CASE WHEN isfinite({value.sql}) THEN {value.sql} ELSE NULL END "
            f"AS FLOAT) AS {name}"
        )
    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        f"SELECT {', '.join(root_columns)}\n"
        f"FROM spine LEFT JOIN {relation} USING (stock_code, trade_date)\n"
        "ORDER BY spine.trade_date, spine.stock_code"
    )
    return CompiledCandidateBatch(
        plan_hash=plan.plan_hash,
        candidate_names=candidate_names,
        materialized_node_ids=tuple(materialized_node_ids),
        materialized_node_count=len(materialized_node_ids),
        analysis_sql_count=1,
        daily_source_reference_count=2,
        sql=sql,
    )


def _validated_sources(
    candidate: CandidateDefinition,
    context: FrozenContext,
    catalog: MiningCatalog,
) -> tuple[str, ...]:
    """复用单候选校验结果，避免批量查询绕过字段安全规则。

    参数:
        candidate: 待校验候选。
        context: 当前冻结上下文。
        catalog: 当前字段和算子目录。
    """

    from mining.expression_validation import validate_candidate_expression

    return validate_candidate_expression(
        candidate,
        context,
        catalog=catalog,
    ).extra_sources
