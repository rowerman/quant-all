"""挖掘 DSL 原生执行器决策使用的纯分析函数。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import statistics
from typing import Any, Iterable

from mining.catalog import MiningCatalog
from mining.expression_ast import (
    BinaryNode,
    CallNode,
    ExpressionNode,
    UnaryNode,
    expression_depth,
    expression_node_count,
)
from mining.expression_parser import parse_expression
from mining.expression_validation import validate_expression

EXPRESSION_SHARE_THRESHOLD = 0.50
ZERO_COST_IMPROVEMENT_THRESHOLD = 0.20

OPERATOR_COVERAGE_EXPRESSIONS = (
    "close + open",
    "close - open",
    "close * ret",
    "close / open",
    "-close",
    "abs(ret)",
    "ln(close)",
    "sqrt(close)",
    "sign(ret)",
    "least(close, open)",
    "greatest(close, open)",
    "lag(close, 2)",
    "delta(close, 2)",
    "mean(close, 5)",
    "sum(ret, 5)",
    "min(close, 5)",
    "max(close, 5)",
    "median(close, 5)",
    "std(ret, 5)",
    "skew(ret, 5)",
    "kurt(ret, 5)",
    "product(ret + 1, 5)",
    "count(close, 5)",
    "corr(ret, amount_cny, 5)",
    "cov(close, open, 5)",
    "slope(close, open, 5)",
    "r2(close, open, 5)",
    "downside_std(ret, 5)",
    "ts_rank(close, 5)",
    "rank(close)",
    "zscore(close)",
    "demean(close)",
    "scale(close)",
    "signed_power(ret, 2)",
    "mean(is_st, 5)",
)

FIXTURE_EXPRESSIONS = (
    "mean(close,5)",
    "mean(volume_share,5)",
    "mean(turnover_rate,5)",
    "mean(volume_share,10)",
)

ROUTE_COSTS = {
    "continue_sql": {
        "version_lock": 0,
        "build": 0,
        "debug": 0,
        "crash_isolation": 0,
        "test": 0,
        "relative_cost": 0,
        "summary": "复用现有 DuckDB，无新增版本和构建链。",
    },
    "duckdb_table_function": {
        "version_lock": 2,
        "build": 2,
        "debug": 2,
        "crash_isolation": 2,
        "test": 2,
        "relative_cost": 10,
        "summary": "依赖 DuckDB 扩展 ABI，仍难覆盖全部窗口和截面语义。",
    },
    "external_cpp_vm": {
        "version_lock": 1,
        "build": 2,
        "debug": 2,
        "crash_isolation": 2,
        "test": 2,
        "relative_cost": 9,
        "summary": "需自建扫描、窗口、截面、时点连接和 Parquet 写出链路。",
    },
}


def _operator_names(node: ExpressionNode) -> tuple[str, ...]:
    if isinstance(node, UnaryNode):
        return ("neg",)
    if isinstance(node, BinaryNode):
        return (
            {"+": "add", "-": "sub", "*": "mul", "/": "div"}[node.operator],
        )
    if isinstance(node, CallNode):
        return (node.name,)
    return ()


def _iter_nodes(node: ExpressionNode) -> Iterable[ExpressionNode]:
    yield node
    if isinstance(node, UnaryNode):
        yield from _iter_nodes(node.operand)
    elif isinstance(node, BinaryNode):
        yield from _iter_nodes(node.left)
        yield from _iter_nodes(node.right)
    elif isinstance(node, CallNode):
        for argument in node.arguments:
            yield from _iter_nodes(argument)


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return {
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p95": round(p95, 6),
        "max": ordered[-1],
    }


def summarize_corpus(
    source_expressions: dict[str, Iterable[str]],
    catalog: MiningCatalog,
) -> dict[str, Any]:
    """统计去重后的候选语料和全部冻结算子覆盖。"""

    by_canonical: dict[str, dict[str, Any]] = {}
    for source, expressions in source_expressions.items():
        for expression in expressions:
            validated = validate_expression(expression, direction=1, catalog=catalog)
            item = by_canonical.setdefault(
                validated.canonical_expression,
                {
                    "canonical_expression": validated.canonical_expression,
                    "sources": set(),
                    "depth": 0,
                    "node_count": 0,
                    "lookback": validated.lookback,
                    "operators": [],
                },
            )
            item["sources"].add(source)

    operator_counts: Counter[str] = Counter()
    rows = []
    for canonical in sorted(by_canonical):
        item = by_canonical[canonical]
        node = parse_expression(canonical, max_depth=8)
        operators = [
            name
            for current in _iter_nodes(node)
            for name in _operator_names(current)
        ]
        operator_counts.update(operators)
        item["depth"] = expression_depth(node)
        item["node_count"] = expression_node_count(node)
        item["operators"] = operators
        item["sources"] = sorted(item["sources"])
        rows.append(item)

    known_operators = {operator.name for operator in catalog.operators}
    missing = sorted(known_operators - set(operator_counts))
    unknown = sorted(set(operator_counts) - known_operators)
    if missing:
        raise ValueError(f"候选语料未覆盖算子：{', '.join(missing)}")
    if unknown:
        raise ValueError(f"候选语料包含目录外算子：{', '.join(unknown)}")

    kind_by_name = {operator.name: operator.kind for operator in catalog.operators}
    kind_counts: Counter[str] = Counter()
    for name, count in operator_counts.items():
        kind_counts[kind_by_name[name]] += count

    return {
        "source_counts": {
            source: sum(source in item["sources"] for item in rows)
            for source in source_expressions
        },
        "candidate_count": len(rows),
        "operator_counts": dict(sorted(operator_counts.items())),
        "operator_kind_counts": dict(sorted(kind_counts.items())),
        "coverage": {
            "catalog_operator_count": len(known_operators),
            "covered_operator_count": len(operator_counts),
            "missing": missing,
            "unknown": unknown,
        },
        "depth": _distribution([int(item["depth"]) for item in rows]),
        "node_count": _distribution([int(item["node_count"]) for item in rows]),
        "lookback": _distribution([int(item["lookback"]) for item in rows]),
        "expressions": [
            {
                "canonical_expression": item["canonical_expression"],
                "sources": item["sources"],
                "depth": item["depth"],
                "node_count": item["node_count"],
                "lookback": item["lookback"],
                "operators": item["operators"],
            }
            for item in rows
        ],
    }


def classify_profile_operator(
    operator_type: str,
    extra_info: dict[str, Any] | None,
) -> str:
    """把一个 DuckDB profile 节点归入扫描、表达式或物化类别。"""

    operator = operator_type.upper()
    if "SCAN" in operator or operator in {"READ_PARQUET", "READ_CSV"}:
        return "scan"
    if operator in {"WINDOW", "FILTER"}:
        return "expression"
    if operator == "PROJECTION":
        projections = (extra_info or {}).get("Projections") or []
        for projection in projections:
            text = str(projection).strip()
            if (
                text.startswith("#")
                or text.startswith("__internal_")
                or text in {"stock_code", "trade_date"}
            ):
                continue
            return "expression"
        return "materialization"
    if any(
        token in operator
        for token in (
            "COPY",
            "WRITE",
            "ORDER_BY",
            "JOIN",
            "GROUP_BY",
            "AGGREGATE",
            "TOP_N",
            "MATERIALIZE",
        )
    ):
        return "materialization"
    return "other"


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """汇总 DuckDB JSON profile 的算子分类耗时。"""

    category_times = Counter()
    operator_times: Counter[str] = Counter()

    def visit(node: dict[str, Any]) -> None:
        operator_type = str(node.get("operator_type") or "UNKNOWN")
        elapsed = float(node.get("operator_timing") or 0.0)
        category_times[
            classify_profile_operator(operator_type, node.get("extra_info"))
        ] += elapsed
        operator_times[operator_type] += elapsed
        for child in node.get("children", []):
            visit(child)

    visit(profile)
    query_s = float(profile.get("latency") or 0.0)
    query_cpu_s = float(profile.get("cpu_time") or 0.0)
    categorized_cpu_s = sum(category_times.values())
    categories_cpu = {
        name: round(category_times.get(name, 0.0), 6)
        for name in ("scan", "expression", "materialization", "other")
    }
    categories_cpu["unattributed"] = round(
        max(query_cpu_s - categorized_cpu_s, 0.0),
        6,
    )
    wall_factor = query_s / query_cpu_s if query_cpu_s else 1.0
    categories_wall = {
        name: round(value * wall_factor, 6)
        for name, value in categories_cpu.items()
    }
    return {
        "query_wall_s": round(query_s, 6),
        "query_cpu_s": round(query_cpu_s, 6),
        "category_cpu_s": categories_cpu,
        "category_wall_equivalent_s": categories_wall,
        "operator_s": dict(
            sorted(
                (name, round(value, 6))
                for name, value in operator_times.items()
            )
        ),
    }


def decision_from_measurements(
    expression_share_of_pipeline: float,
    zero_cost_improvement: float,
) -> dict[str, Any]:
    """按固定 50% 和 20% 门槛给出后续动作。"""

    expression_majority = (
        expression_share_of_pipeline >= EXPRESSION_SHARE_THRESHOLD
    )
    upper_bound_met = zero_cost_improvement >= ZERO_COST_IMPROVEMENT_THRESHOLD
    reasons = []
    if not expression_majority:
        reasons.append("expression_share_below_50pct")
    if not upper_bound_met:
        reasons.append("zero_cost_improvement_below_20pct")
    return {
        "expression_share_threshold": EXPRESSION_SHARE_THRESHOLD,
        "zero_cost_improvement_threshold": ZERO_COST_IMPROVEMENT_THRESHOLD,
        "expression_is_majority": expression_majority,
        "zero_cost_upper_bound_met": upper_bound_met,
        "decision": (
            "create_followup_design"
            if not reasons
            else "do_not_implement"
        ),
        "reasons": reasons,
        "python_expression_vm_implemented": False,
        "cpp_modules_replaced": 0,
    }


def stable_digest(payload: Any) -> str:
    """返回稳定 JSON 的 SHA-256。"""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
