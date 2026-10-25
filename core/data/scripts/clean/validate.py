"""清洗结果校验：直接在 data/clean/daily 上跑 DuckDB 聚合。

校验项编号对应 cleaning_spec.md 第 10 节的 A1~A8。
"""

from __future__ import annotations

import duckdb
import pandas as pd

from scripts.config import CLEAN_DIR, LIMIT_EPS, RET_TOLERANCE
from scripts.logging_conf import get_logger

logger = get_logger("clean.validate")

DAILY_GLOB = str(CLEAN_DIR / "daily" / "year=*" / "part.parquet")


def connect() -> duckdb.DuckDBPyConnection:
    """建立 DuckDB 连接并把日线分区与交易日历注册成视图。"""
    conn = duckdb.connect()
    conn.execute(
        f"CREATE VIEW daily AS SELECT * FROM read_parquet('{DAILY_GLOB}', hive_partitioning=false)"
    )
    conn.execute(
        f"CREATE VIEW calendar AS SELECT * FROM read_parquet('{CLEAN_DIR / 'calendar.parquet'}')"
    )
    return conn


def status_checks(conn: duckdb.DuckDBPyConnection) -> dict:
    """A1：状态列无缺失、取值只有 trading/suspended，且上市期内交易日无空洞。

    空洞 = 每只股票在 `[list_date, delist_date] ∩ 数据区间` 内的交易日数
    减去它在 `data/clean/daily` 里的实际行数（和为正才算空洞）。

    参数:
        conn: DuckDB 连接。
    """
    frame = conn.execute(
        "SELECT status, count(*) AS rows FROM daily GROUP BY status ORDER BY status"
    ).df()
    counts = {str(row["status"]): int(row["rows"]) for row in frame.to_dict("records")}
    nulls = int(conn.execute("SELECT count(*) FROM daily WHERE status IS NULL").fetchone()[0])
    unknown = int(
        conn.execute(
            "SELECT count(*) FROM daily WHERE status NOT IN ('trading', 'suspended')"
        ).fetchone()[0]
    )
    gaps_row = conn.execute(
        """
        WITH bounds AS (
          SELECT stock_code,
                 min(CAST(list_date AS DATE)) AS list_date,
                 min(CAST(delist_date AS DATE)) AS delist_date,
                 count(*) AS n_rows
          FROM daily GROUP BY stock_code
        ),
        data_window AS (
          SELECT min(CAST(trade_date AS DATE)) AS d0, max(CAST(trade_date AS DATE)) AS d1
          FROM daily
        ),
        expected AS (
          SELECT b.stock_code, count(*) AS n_expected
          FROM bounds b
          CROSS JOIN data_window w
          JOIN calendar c
            ON CAST(c.trade_date AS DATE) >= greatest(b.list_date, w.d0)
           AND CAST(c.trade_date AS DATE) <= least(coalesce(b.delist_date, w.d1), w.d1)
          GROUP BY b.stock_code
        )
        SELECT count(*) AS stocks_checked,
               coalesce(sum(greatest(e.n_expected - b.n_rows, 0)), 0) AS gaps
        FROM expected e JOIN bounds b USING (stock_code)
        """
    ).df().to_dict("records")[0]
    return {
        "A1_status_counts": counts,
        "A1_status_null": nulls,
        "A1_unknown_status": unknown,
        "A1_stocks_checked": int(gaps_row["stocks_checked"]),
        "A1_calendar_gaps": int(gaps_row["gaps"]),
    }


def ex_dividend_checks(conn: duckdb.DuckDBPyConnection) -> dict:
    """A2：除权日与复权因子跳变的一一对应关系。

    只在有行情的交易日上比较：Tushare 的 adj_factor 在停牌期间照样逐日更新，
    但停牌日没有收盘价可比，因子变化应归到复牌日一起看，否则会出现
    「跳变但没有除权」的假阳性。

    参数:
        conn: DuckDB 连接。
    """
    row = conn.execute(
        """
        WITH t AS (
          SELECT stock_code, trade_date, factor, is_ex_div,
                 lag(factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_factor
          FROM daily WHERE close IS NOT NULL AND factor IS NOT NULL
        )
        SELECT
          sum(CASE WHEN is_ex_div THEN 1 ELSE 0 END) AS ex_days,
          sum(CASE WHEN prev_factor IS NOT NULL AND abs(factor/prev_factor - 1) > 1e-9 THEN 1 ELSE 0 END) AS jump_days,
          sum(CASE WHEN is_ex_div AND (prev_factor IS NULL OR abs(factor/prev_factor - 1) <= 1e-9) THEN 1 ELSE 0 END) AS ex_without_jump,
          sum(CASE WHEN prev_factor IS NOT NULL AND abs(factor/prev_factor - 1) > 1e-9 AND NOT is_ex_div THEN 1 ELSE 0 END) AS jump_without_ex
        FROM t
        """
    ).df().to_dict("records")[0]
    return {
        "A2_ex_div_days": int(row["ex_days"]),
        "A2_factor_jump_days": int(row["jump_days"]),
        "A2_ex_without_jump": int(row["ex_without_jump"]),
        "A2_jump_without_ex_div": int(row["jump_without_ex"]),
    }


def return_consistency(conn: duckdb.DuckDBPyConnection) -> dict:
    """A3：官方涨跌额口径与因子口径收益的差异。

    参数:
        conn: DuckDB 连接。
    """
    row = conn.execute(
        f"""
        WITH t AS (
          SELECT stock_code, trade_date, ret, close * factor AS adj,
                 lag(close * factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_adj
          FROM daily WHERE factor IS NOT NULL AND close IS NOT NULL
        ),
        c AS (SELECT ret, adj / prev_adj - 1 AS factor_ret FROM t WHERE prev_adj IS NOT NULL AND ret IS NOT NULL)
        SELECT count(*) AS compared,
               count(*) FILTER (WHERE abs(ret - factor_ret) > {RET_TOLERANCE}) AS mismatch
        FROM c
        """
    ).df().to_dict("records")[0]
    compared = int(row["compared"])
    mismatch = int(row["mismatch"])
    return {
        "A3_compared_rows": compared,
        "A3_mismatch_rows": mismatch,
        "A3_mismatch_ratio": round(mismatch / compared, 6) if compared else 0.0,
    }


def limit_checks(conn: duckdb.DuckDBPyConnection) -> dict:
    """A4：涨跌停越界为 0，并统计涨停家数占比。

    参数:
        conn: DuckDB 连接。
    """
    over = conn.execute(
        f"""
        SELECT count(*) FROM daily
        WHERE is_price_limit_applicable AND close IS NOT NULL
          AND (close > limit_up + 0.011 OR close < limit_down - 0.011)
        """
    ).fetchone()[0]
    ratio = conn.execute(
        """
        SELECT trade_date,
               count(*) FILTER (WHERE status = 'trading') AS n,
               count(*) FILTER (WHERE is_limit_up) AS up
        FROM daily GROUP BY trade_date HAVING n > 0
        """
    ).df()
    daily_ratio = (ratio["up"] / ratio["n"]).astype(float)
    return {
        "A4_over_limit_rows": int(over),
        "A4_limit_up_daily_ratio_mean": round(float(daily_ratio.mean()), 6),
        "A4_limit_up_daily_ratio_min": round(float(daily_ratio.min()), 6),
        "A4_limit_up_daily_ratio_max": round(float(daily_ratio.max()), 6),
    }


def suspension_checks(conn: duckdb.DuckDBPyConnection) -> dict:
    """A6：停牌天数分布。

    参数:
        conn: DuckDB 连接。
    """
    frame = conn.execute(
        """
        SELECT stock_code, count(*) FILTER (WHERE status = 'suspended') AS n_suspend,
               count(*) FILTER (WHERE status = 'trading') AS n_trading
        FROM daily GROUP BY stock_code
        """
    ).df()
    total = int(frame["n_suspend"].sum() + frame["n_trading"].sum())
    return {
        "A6_suspended_stock_days": int(frame["n_suspend"].sum()),
        "A6_suspended_ratio": round(float(frame["n_suspend"].sum()) / max(total, 1), 6),
        "A6_per_stock_max_days": int(frame["n_suspend"].max()),
        "A6_most_suspended_code": str(
            frame.loc[frame["n_suspend"].idxmax(), "stock_code"]
        ),
    }


def collect_anomalies(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """把各类异常汇总成长表，定位到 stock_code + trade_date。

    与 A2/A3 校验保持同一比较口径：窗口函数只看有行情的交易日
    （`close` 与 `factor` 都非空），否则复牌日的收益差异会被漏掉。

    参数:
        conn: DuckDB 连接。
    """
    sql = f"""
    WITH t AS (
      SELECT stock_code, trade_date, close, ret, factor, is_ex_div, limit_up, limit_down,
             is_price_limit_applicable,
             lag(factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_factor,
             lag(close * factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_adj,
             close * factor AS adj
      FROM daily WHERE close IS NOT NULL AND factor IS NOT NULL
    )
    SELECT stock_code::VARCHAR AS stock_code, trade_date::VARCHAR AS trade_date, kind FROM (
      SELECT stock_code, trade_date, 'limit_out_of_range' AS kind FROM t
        WHERE is_price_limit_applicable AND close IS NOT NULL
          AND (close > limit_up + 0.011 OR close < limit_down - 0.011)
      UNION ALL
      SELECT stock_code, trade_date, 'return_mismatch' FROM t
        WHERE prev_adj IS NOT NULL AND ret IS NOT NULL AND abs(ret - (adj/prev_adj - 1)) > {RET_TOLERANCE}
      UNION ALL
      SELECT stock_code, trade_date, 'ex_div_without_factor_jump' FROM t
        WHERE is_ex_div AND (prev_factor IS NULL OR abs(factor/prev_factor - 1) <= 1e-9)
      UNION ALL
      SELECT stock_code, trade_date, 'factor_jump_without_ex_div' FROM t
        WHERE prev_factor IS NOT NULL AND abs(factor/prev_factor - 1) > 1e-9 AND NOT is_ex_div
      UNION ALL
      SELECT stock_code, trade_date, 'non_positive_price' FROM t WHERE close IS NOT NULL AND close <= 0
    )
    """
    return conn.execute(sql).df()


def build_suspend_segments(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """由日线状态列还原跨年连续的停牌区间。

    参数:
        conn: DuckDB 连接。
    """
    return conn.execute(
        """
        WITH s AS (
          SELECT stock_code, trade_date, status,
                 row_number() OVER (PARTITION BY stock_code ORDER BY trade_date) AS rn
          FROM daily
        ),
        marked AS (
          SELECT stock_code, trade_date, status, rn,
                 rn - row_number() OVER (PARTITION BY stock_code, status ORDER BY trade_date) AS grp
          FROM s
        ),
        seg AS (
          SELECT stock_code, min(trade_date) AS start_date, max(trade_date) AS end_date,
                 count(*) AS n_days
          FROM marked WHERE status = 'suspended' GROUP BY stock_code, grp
        )
        SELECT stock_code, start_date, end_date, n_days,
               n_days > 20 AS is_long
        FROM seg ORDER BY stock_code, start_date
        """
    ).df()


def run_all() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """执行全部校验。

    返回:
        (校验结果字典, 异常表, 停牌区间表)。
    """
    conn = connect()
    checks = {}
    checks.update(status_checks(conn))
    checks.update(ex_dividend_checks(conn))
    checks.update(return_consistency(conn))
    checks.update(limit_checks(conn))
    checks.update(suspension_checks(conn))
    anomalies = collect_anomalies(conn)
    checks["A7_anomaly_rows"] = int(len(anomalies))
    segments = build_suspend_segments(conn)
    conn.close()
    logger.info("校验完成：异常 %d 行，停牌区间 %d 段", len(anomalies), len(segments))
    return checks, anomalies, segments
