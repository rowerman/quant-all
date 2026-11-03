"""候选 DAG 批量 SQL 的共享、确定性和金标准对照测试。"""

from __future__ import annotations

import duckdb
import numpy as np

from mining.candidate_materialization import CandidateComputation
from mining.catalog import load_catalog
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.expression_batch import compile_candidate_batch
from mining.expression_dag import build_candidate_expression_plan
from mining.expression_validation import validate_expression
from scripts.build_factors.sql import family_query


def _context() -> FrozenContext:
    catalog = load_catalog()
    return FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=MiningConstraints(
            max_rounds=1,
            max_candidates=4,
            full_eval_top_k=1,
            max_llm_calls=1,
            max_failure_rate=1.0,
            min_quality_checks_for_failure_stop=1,
        ),
    )


def _candidate(expression: str, index: int):
    return validate_expression(expression, direction=1).to_candidate(
        mechanism="批量编译测试",
        description=f"测试候选 {index}",
        steps=("执行受控表达式",),
        discovery_round=1,
        discovery_index=index,
    )


def _daily_connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("ATTACH ':memory:' AS src")
    connection.execute("CREATE SCHEMA src.md")
    connection.execute(
        """
        CREATE TABLE src.md.daily_hfq AS
        SELECT
          code AS stock_code,
          DATE '2024-01-01' + CAST(day AS INTEGER) AS trade_date,
          'trading' AS status,
          10.0 + day + CASE WHEN code = 'A' THEN 0.5 ELSE 0.0 END AS open,
          11.0 + day + CASE WHEN code = 'A' THEN 0.2 ELSE 0.0 END AS high,
          9.0 + day + CASE WHEN code = 'A' THEN 0.1 ELSE 0.0 END AS low,
          10.5 + day + CASE WHEN code = 'A' THEN 0.3 ELSE 0.0 END AS close,
          10.2 + day + CASE WHEN code = 'A' THEN 0.2 ELSE 0.0 END AS vwap,
          1000.0 + day * 10 AS volume_share,
          100000.0 + day * 1000 AS amount_cny,
          0.01 + day / 1000.0 AS ret,
          day % 3 = 0 AS is_st
        FROM range(1, 21) AS days(day)
        CROSS JOIN (VALUES ('A'), ('B')) AS codes(code)
        """
    )
    return connection


def _assert_rows_equal(left, right) -> None:
    assert len(left) == len(right)
    for left_row, right_row in zip(left, right, strict=True):
        assert left_row[:2] == tuple(right_row[:2])
        assert np.allclose(
            np.asarray(left_row[2:], dtype=float),
            np.asarray(right_row[2:], dtype=float),
            equal_nan=True,
        )


def test_dag_batch_matches_single_candidate_queries():
    """共享均值节点的一次查询与四个单候选查询逐值一致。"""

    context = _context()
    catalog = load_catalog()
    expressions = (
        "mean(close, 5) + rank(open)",
        "mean(close, 5) - open",
        "rank(mean(close, 5))",
        "mean(close, 5) + close",
    )
    candidates = tuple(
        _candidate(expression, index)
        for index, expression in enumerate(expressions, start=1)
    )
    plan = build_candidate_expression_plan(candidates, context, catalog=catalog)
    compiled = compile_candidate_batch(
        plan,
        candidates,
        context,
        start="2024-01-01",
        end="2024-01-20",
        catalog=catalog,
    )
    single_sql = {}
    for candidate in candidates:
        validated = validate_expression(candidate.canonical_expression, direction=1)
        single_sql[candidate.candidate_id] = family_query(
            [CandidateComputation(candidate.name, validated.layers, validated.extra_sources)],
            spine_start="2024-01-01",
            spine_end="2024-01-20",
        )

    connection = _daily_connection()
    batch_rows = connection.execute(compiled.sql).fetchall()
    ordered_ids = [root.candidate_id for root in plan.candidates]
    single_rows = []
    for candidate_id in ordered_ids:
        values = connection.execute(single_sql[candidate_id]).fetchall()
        if not single_rows:
            single_rows = [[row[0], row[1], row[2]] for row in values]
            continue
        for combined, item in zip(single_rows, values, strict=True):
            combined.append(item[2])
    connection.close()

    _assert_rows_equal(batch_rows, single_rows)
    assert compiled.analysis_sql_count == 1
    assert compiled.daily_source_reference_count == 2
    assert compiled.materialized_node_count >= 3
    assert compiled.sql.count("avg(close)") == 1


def test_batch_sql_is_independent_of_candidate_input_order():
    """候选换序不改变 DAG 查询和根列顺序。"""

    context = _context()
    catalog = load_catalog()
    candidates = (
        _candidate("mean(close, 5) + open", 1),
        _candidate("mean(close, 5) - close", 2),
    )
    plan = build_candidate_expression_plan(candidates, context, catalog=catalog)
    first = compile_candidate_batch(
        plan,
        candidates,
        context,
        start="2024-01-01",
        end="2024-01-20",
        catalog=catalog,
    )
    second = compile_candidate_batch(
        plan,
        tuple(reversed(candidates)),
        context,
        start="2024-01-01",
        end="2024-01-20",
        catalog=catalog,
    )

    assert first == second


def test_all_frozen_operators_match_single_compiler():
    """批量编译器覆盖全部冻结算子且与单候选 SQL 一致。"""

    expressions = (
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
    context = _context()
    catalog = load_catalog()
    candidates = tuple(
        _candidate(expression, index)
        for index, expression in enumerate(expressions, start=1)
    )
    plan = build_candidate_expression_plan(candidates, context, catalog=catalog)
    compiled = compile_candidate_batch(
        plan,
        candidates,
        context,
        start="2024-01-01",
        end="2024-01-20",
        catalog=catalog,
    )
    candidate_by_id = {item.candidate_id: item for item in candidates}
    roots = [root.candidate_id for root in plan.candidates]
    connection = _daily_connection()
    batch_rows = connection.execute(compiled.sql).fetchall()
    single_rows = []
    for candidate_id in roots:
        candidate = candidate_by_id[candidate_id]
        validated = validate_expression(candidate.canonical_expression, direction=1)
        query = family_query(
            [CandidateComputation(candidate.name, validated.layers, validated.extra_sources)],
            spine_start="2024-01-01",
            spine_end="2024-01-20",
        )
        values = connection.execute(query).fetchall()
        if not single_rows:
            single_rows = [[row[0], row[1], row[2]] for row in values]
            continue
        for combined, item in zip(single_rows, values, strict=True):
            combined.append(item[2])
    connection.close()

    _assert_rows_equal(batch_rows, single_rows)
