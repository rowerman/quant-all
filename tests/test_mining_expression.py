"""S3 受控 DSL 的解析、安全、编译和时点测试。"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from factors.ops import WINDOW_DEF
from factors.sources import SOURCES, join_clause
from backend.errors import error_payload
from mining import validate_expression
from mining.errors import CatalogDriftError, ExpressionValidationError


def _run_layers(
    connection: duckdb.DuckDBPyConnection,
    layers,
    *,
    seed: str = "daily_seed",
) -> None:
    """按编译器输出逐层执行，确保窗口之间没有非法嵌套。"""

    ctes = [f"base AS (SELECT * FROM {seed})"]
    previous = "base"
    for index, layer in enumerate(layers, start=1):
        current = f"v{index}"
        ctes.append(
            f"{current} AS (SELECT *, {layer} AS {current} FROM {previous} "
            f"WINDOW w AS {WINDOW_DEF})"
        )
        previous = current
    connection.execute("WITH " + ",\n".join(ctes) + f" SELECT * FROM {previous}").fetchall()


@pytest.fixture
def daily_panel():
    """构造足够覆盖窗口算子的双股票小面板。"""

    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE daily_seed AS
        SELECT
          code AS stock_code,
          DATE '2024-01-01' + CAST(day AS INTEGER) AS trade_date,
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
    yield connection
    connection.close()


@pytest.mark.parametrize(
    "expression",
    [
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
    ],
)
def test_every_frozen_operator_compiles_and_executes(daily_panel, expression):
    """全部冻结算子都能在 DuckDB 中执行。"""

    validated = validate_expression(expression, direction=1)
    _run_layers(daily_panel, validated.layers)


def test_nested_windows_compile_to_separate_layers():
    """窗口套窗口自动拆层，并保守计算回看长度。"""

    validated = validate_expression("rank(mean(close, 5) + lag(close, 1))", direction=1)
    assert len(validated.layers) == 3
    assert "avg(close)" in validated.layers[0]
    assert "lag(close, 1)" in validated.layers[1]
    assert "percent_rank()" in validated.layers[2]
    assert validated.lookback == 4

    nested = validate_expression("mean(mean(close, 5), 5)", direction=1)
    assert len(nested.layers) == 2
    assert nested.lookback == 8


def test_canonicalization_aliases_and_stable_semantics():
    """空白、数字格式、交换顺序和函数别名得到同一语义。"""

    first = validate_expression("close * 1.00 + open", direction=1)
    second = validate_expression("open + 1 * close", direction=1)
    assert first.canonical_expression == second.canonical_expression
    assert first.semantic_hash == second.semantic_hash
    assert first.factor_name == second.factor_name
    assert first.factor_name.startswith("MINE_")
    assert len(first.factor_name) == 29

    alias = validate_expression("delay(close, 5)", direction=1)
    canonical = validate_expression("lag(close, 5)", direction=1)
    assert alias.canonical_expression == canonical.canonical_expression
    assert alias.semantic_hash == canonical.semantic_hash


def test_direction_changes_semantic_identity():
    """方向属于候选语义，取反后稳定名不同。"""

    positive = validate_expression("mean(close, 5)", direction=1)
    negative = validate_expression("mean(close, 5)", direction=-1)
    assert positive.semantic_hash != negative.semantic_hash
    assert positive.factor_name != negative.factor_name


def test_validated_expression_builds_matching_s2_candidate():
    """S3 输出可直接构造 S2 候选且语义哈希一致。"""

    validated = validate_expression("mean(close, 5)", direction=1)
    candidate = validated.to_candidate(
        mechanism="短期价格趋势",
        description="五日收盘价均值",
        steps=("取五日收盘价均值",),
        discovery_round=1,
        discovery_index=1,
    )
    assert candidate.name == validated.factor_name
    assert candidate.semantic_hash == validated.semantic_hash
    assert candidate.canonical_expression == "mean(close, 5)"
    assert candidate.lookback == 4


def test_s3_errors_keep_http_contract():
    """监测站通用挖掘错误转换保留 S3 新错误码和状态。"""

    validation = ExpressionValidationError("表达式不合法", reason="syntax")
    status, payload = error_payload(validation, "request-s3")
    assert status == 400
    assert payload["code"] == "MINING-114"
    assert payload["request_id"] == "request-s3"

    status, payload = error_payload(CatalogDriftError("目录漂移"), "request-s3")
    assert status == 500
    assert payload["code"] == "MINING-115"


@pytest.mark.parametrize(
    "expression",
    [
        "fwd_ret_open_5",
        "stock_code",
        "trade_date",
        "status",
        "change",
        "pre_close",
        "unknown_field",
    ],
)
def test_forbidden_and_unknown_fields_fail_before_execution(expression):
    """标签、键、审计列和未知字段不会进入 SQL。"""

    with pytest.raises(ExpressionValidationError, match="字段") as error:
        validate_expression(expression, direction=1)
    assert error.value.code == "MINING-114"
    assert error.value.reason in {"forbidden_field", "unknown_field"}


@pytest.mark.parametrize(
    "expression, reason",
    [
        ("mean(close, 0)", "invalid_window"),
        ("mean(close, 253)", "invalid_window"),
        ("mean(close, 5.5)", "invalid_window"),
        ("mean(close, close)", "invalid_parameter"),
        ("mean(close)", "invalid_arity"),
        ("mean(close, 5, 7)", "invalid_arity"),
        ("signed_power(close, 5)", "invalid_parameter"),
        ("close + 1e-999999", "invalid_number"),
        ("close + amount_cny", "dimension_mismatch"),
        ("least(close, amount_cny)", "dimension_mismatch"),
        ("unknown_operator(close)", "unknown_operator"),
    ],
)
def test_invalid_parameters_and_dimensions_are_rejected(expression, reason):
    """参数、元数和维度在编译阶段失败。"""

    with pytest.raises(ExpressionValidationError) as error:
        validate_expression(expression, direction=1)
    assert error.value.reason == reason


@pytest.mark.parametrize(
    "expression",
    [
        "close; DROP TABLE md.daily_hfq",
        "close -- comment",
        "close /* comment */",
        "`close`",
        "md.close",
        "__import__('os')",
        "close[0]",
        "close\\n+ open",
    ],
)
def test_sql_and_python_injection_is_rejected(expression):
    """原始 SQL、注释、限定名和 Python 语法全部拒绝。"""

    with pytest.raises(ExpressionValidationError) as error:
        validate_expression(expression, direction=1)
    assert error.value.code == "MINING-114"


def test_excluded_field_depth_length_and_node_limits_are_enforced():
    """运行排除字段和资源上限在解析与编译阶段生效。"""

    with pytest.raises(ExpressionValidationError, match="排除") as error:
        validate_expression("mean(close, 5)", direction=1, excluded_fields=("close",))
    assert error.value.reason == "excluded_field"

    deep = "mean(" * 7 + "close" + ", 1)" * 7
    with pytest.raises(ExpressionValidationError, match="深度"):
        validate_expression(deep, direction=1, max_depth=6)

    recursive = "-" * 2000 + "close"
    with pytest.raises(ExpressionValidationError, match="嵌套过深") as recursion_error:
        validate_expression(recursive, direction=1, max_depth=8)
    assert recursion_error.value.reason == "too_deep"

    with pytest.raises(ExpressionValidationError, match="字节"):
        validate_expression(" " * 12001, direction=1)

    too_many = "greatest(" + ", ".join(["close"] * 257) + ")"
    with pytest.raises(ExpressionValidationError, match="节点数"):
        validate_expression(too_many, direction=1)


def test_financial_source_uses_next_trading_day():
    """财务字段公告当日不可见，次一交易日才进入 ASOF 面板。"""

    validated = validate_expression("mean(fi_roe, 1)", direction=1)
    assert validated.extra_sources == ("fin_indicator",)

    connection = duckdb.connect()
    connection.execute("CREATE SCHEMA fin")
    connection.execute(
        """
        CREATE TABLE fin.fina_indicator(
          stock_code VARCHAR, available_date DATE, report_date DATE,
          update_flag VARCHAR, profit_dedt DOUBLE, roe DOUBLE, roe_waa DOUBLE,
          roe_dt DOUBLE, roe_yearly DOUBLE, roa DOUBLE, roa_yearly DOUBLE,
          roa2_yearly DOUBLE, roic DOUBLE, grossprofit_margin DOUBLE,
          netprofit_margin DOUBLE, debt_to_assets DOUBLE, current_ratio DOUBLE,
          quick_ratio DOUBLE, cash_ratio DOUBLE, assets_turn DOUBLE,
          or_yoy DOUBLE, tr_yoy DOUBLE, netprofit_yoy DOUBLE, op_yoy DOUBLE,
          ocf_yoy DOUBLE, assets_yoy DOUBLE, eqt_yoy DOUBLE,
          q_sales_yoy DOUBLE, q_op_qoq DOUBLE
        )
        """
    )
    connection.execute(
        """
        INSERT INTO fin.fina_indicator
        SELECT 'A', DATE '2024-01-02', DATE '2023-12-31', '1',
          1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1
        """
    )
    connection.execute(
        """
        CREATE TABLE base_seed AS
        SELECT 'A' AS stock_code, DATE '2024-01-02' AS trade_date
        UNION ALL
        SELECT 'A', DATE '2024-01-03'
        """
    )
    source_sql = SOURCES["fin_indicator"].sql.format(prefix="")
    ctes = [
        f"src_fin_indicator AS ({source_sql})",
        "base AS (SELECT * FROM base_seed)",
        f"enriched AS (SELECT base.*, src_fin_indicator.fi_roe "
        f"FROM base {join_clause('fin_indicator')})",
    ]
    previous = "enriched"
    for index, layer in enumerate(validated.layers, start=1):
        current = f"v{index}"
        ctes.append(
            f"{current} AS (SELECT *, {layer} AS {current} FROM {previous} "
            f"WINDOW w AS {WINDOW_DEF})"
        )
        previous = current
    rows = connection.execute(
        "WITH " + ",\n".join(ctes) + f" SELECT trade_date, v1 FROM {previous} ORDER BY trade_date"
    ).fetchall()
    connection.close()
    assert rows == [(date(2024, 1, 2), None), (date(2024, 1, 3), 10.0)]
