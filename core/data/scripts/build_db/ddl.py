"""表结构定义与派生表 SQL。"""

from __future__ import annotations

from quant.schema import DAILY_COLUMNS

SCHEMA_DDL = [
    "CREATE SCHEMA IF NOT EXISTS md",
    "CREATE SCHEMA IF NOT EXISTS ref",
    "CREATE SCHEMA IF NOT EXISTS fin",
    "CREATE SCHEMA IF NOT EXISTS evt",
    "CREATE SCHEMA IF NOT EXISTS meta",
]

FLOAT_COLUMNS = {
    "open", "high", "low", "close", "volume_share", "amount_cny", "vwap",
    "factor", "change", "pre_close", "ret",
    "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "turnover_rate",
    "limit_ratio", "limit_up", "limit_down",
}
BOOL_COLUMNS = {
    "is_suspended", "is_resume", "is_ex_div", "is_st", "is_delisting",
    "is_limit_up", "is_limit_down", "touched_limit_up", "touched_limit_down",
    "is_one_word_board", "is_price_limit_applicable",
    "can_buy_close", "can_sell_close", "can_buy_open_next", "can_sell_open_next",
}
DATE_COLUMNS = {"trade_date", "list_date", "delist_date"}


def daily_select(source: str, table_alias: str = "") -> str:
    """生成日线表的取数列清单，把日期与布尔列转成合适的类型。

    参数:
        source: 来源关系，例如 read_parquet('...')。
        table_alias: 子查询别名前缀，形如 "b."。
    """
    pieces = []
    for name in DAILY_COLUMNS:
        column = f"{table_alias}{name}"
        if name in DATE_COLUMNS:
            pieces.append(f"CAST({column} AS DATE) AS {name}")
        elif name in FLOAT_COLUMNS:
            pieces.append(f"CAST({column} AS DOUBLE) AS {name}")
        elif name in BOOL_COLUMNS:
            pieces.append(f"CAST({column} AS BOOLEAN) AS {name}")
        else:
            pieces.append(f"{column} AS {name}")
    return ",\n    ".join(pieces)


def hfq_sql(source: str) -> str:
    """构造后复权表：四价与 vwap 乘以当日复权因子。

    参数:
        source: 来源关系或表名。
    """
    scaled = ", ".join(f"CAST(b.{name} * b.factor AS DOUBLE) AS {name}"
                       for name in ("open", "high", "low", "close", "vwap"))
    return f"""
    CREATE OR REPLACE TABLE md.daily_hfq AS
    SELECT
      b.stock_code, b.trade_date,
      {scaled},
      {", ".join(f"b.{name}" for name in DAILY_COLUMNS if name not in ("stock_code", "trade_date", "open", "high", "low", "close", "vwap"))}
    FROM {source} b
    """


def qfq_sql(source: str) -> str:
    """构造前复权表：四价与 vwap 乘以「当日因子 / 该股最后一个因子」。

    参数:
        source: 来源关系或表名。
    """
    scaled = ", ".join(
        f"CAST(CASE WHEN base.factor_last IS NULL OR b.factor IS NULL THEN NULL "
        f"ELSE b.{name} * b.factor / base.factor_last END AS DOUBLE) AS {name}"
        for name in ("open", "high", "low", "close", "vwap")
    )
    return f"""
    CREATE OR REPLACE TABLE md.daily_qfq AS
    WITH base AS (
      SELECT stock_code, arg_max(factor, trade_date) AS factor_last
      FROM {source} WHERE factor IS NOT NULL GROUP BY stock_code
    )
    SELECT
      b.stock_code, b.trade_date,
      {scaled},
      {", ".join(f"b.{name}" for name in DAILY_COLUMNS if name not in ("stock_code", "trade_date", "open", "high", "low", "close", "vwap"))}
    FROM {source} b
    LEFT JOIN base ON base.stock_code = b.stock_code
    """
