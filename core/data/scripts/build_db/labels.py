"""未来收益标签的生成与校验。

口径：t 日收盘后决策，赚 t 之后 N 个交易日的钱。

* close 通道：买入 t 日收盘价、卖出 t+N 日收盘价；
* open 通道：买入 t+1 日开盘价、卖出 t+1+N 日开盘价，持有交易日数与 close 一致。

停牌段用「上一有效交易日价格」填平，复牌跳空一次性计入终点价；
非交易状态的行与尾部不足 N 个交易日的行保持 NULL。
行集与 ``md.daily_hfq`` 完全一致，不做股票池过滤。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

from quant.schema import LABEL_HORIZONS, LABEL_PRICES, label_columns
from scripts.config import DB_FILE, ensure_dirs
from scripts.logging_conf import get_logger

logger = get_logger("build_db.labels")

# 与 md.daily_hfq.ret 错位比较时允许的绝对差异
RET_TOLERANCE = 1e-4
# 两个口径的差异占比超过这个比例就判标签异常（实测 0.165%）
MISMATCH_RATIO_LIMIT = 0.005
# 收益上限：实测最大 10.42（BJ920799 2024 年 10 月连续暴涨），留一倍余量，
# 超过说明价格口径写错或复权乘错
EXTREME_ABS_RETURN = 20.0
# 做多不可能亏掉超过 100%，跌破下界说明口径写反了
INVALID_RETURN_FLOOR = -1.0

_LABEL_COLUMNS = [name for price in LABEL_PRICES for name in label_columns(price)]


def _tail_offset(price: str, horizon: int) -> int:
    """返回某通道某个周期需要的未来交易日数。

    参数:
        price: close / open。
        horizon: 预测周期。
    """
    return horizon if price == "close" else horizon + 1


def _filled_price_columns() -> str:
    """生成把停牌段填平的价格列 SQL 片段。"""
    pieces = [
        f'''last_value("{name}" IGNORE NULLS) OVER (
             PARTITION BY stock_code ORDER BY trade_date
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {alias}'''
        for name, alias in (("close", "px_close"), ("open", "px_open"))
    ]
    return ",\n                 ".join(pieces)


def build_labels(conn: duckdb.DuckDBPyConnection) -> int:
    """从 md.daily_hfq 生成 md.fwd_return。

    参数:
        conn: 可写的 DuckDB 连接。

    返回:
        生成的行数。
    """
    selects = []
    for horizon in LABEL_HORIZONS:
        selects.append(
            f"CASE WHEN status = 'trading' THEN lead(px_close, {horizon}) OVER w / px_close - 1 "
            f"END AS fwd_ret_close_{horizon}"
        )
    for horizon in LABEL_HORIZONS:
        # 开盘通道从 t+1 起算，卖出价取 t+1+N，持有长度才和收盘通道一致
        selects.append(
            f"CASE WHEN status = 'trading' "
            f"THEN lead(px_open, {horizon + 1}) OVER w / lead(px_open, 1) OVER w - 1 "
            f"END AS fwd_ret_open_{horizon}"
        )
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE md.fwd_return AS
        WITH px AS (
          SELECT stock_code, trade_date, status,
                 {_filled_price_columns()}
          FROM md.daily_hfq
        )
        SELECT stock_code, trade_date,
               {", ".join(selects)}
        FROM px
        WINDOW w AS (PARTITION BY stock_code ORDER BY trade_date)
        """
    )
    rows = int(conn.execute("SELECT count(*) FROM md.fwd_return").fetchone()[0])
    logger.info("md.fwd_return 生成 %d 行", rows)
    return rows


def _verify_sql() -> str:
    """拼装一次扫描完成的校验 SQL。"""
    any_not_null = " OR ".join(f"l.{name} IS NOT NULL" for name in _LABEL_COLUMNS)
    any_extreme = " OR ".join(f"abs(l.{name}) > {EXTREME_ABS_RETURN}" for name in _LABEL_COLUMNS)
    any_below_floor = " OR ".join(f"l.{name} < {INVALID_RETURN_FLOOR}" for name in _LABEL_COLUMNS)
    pieces = [
        "count(*) AS rows",
        "count(*) FILTER (WHERE base.status = 'suspended') AS suspended_rows",
        "count(*) FILTER (WHERE base.status = 'trading') AS trading_rows",
        f"count(*) FILTER (WHERE base.status = 'suspended' AND ({any_not_null})) AS suspended_with_value",
        f"count(*) FILTER (WHERE {any_extreme}) AS extreme_rows",
        f"count(*) FILTER (WHERE {any_below_floor}) AS below_floor_rows",
        f"count(*) FILTER (WHERE abs(base.ret_next - l.fwd_ret_close_1) > {RET_TOLERANCE}) AS ret_mismatch",
    ]
    for price in LABEL_PRICES:
        for horizon in LABEL_HORIZONS:
            column = f"fwd_ret_{price}_{horizon}"
            offset = _tail_offset(price, horizon)
            pieces.append(
                f"count(*) FILTER (WHERE base.status = 'trading' AND l.{column} IS NULL) AS null_{column}"
            )
            pieces.append(
                f"count(*) FILTER (WHERE base.status = 'trading' AND base.rn_desc <= {offset}) AS tail_{column}"
            )
    return f"""
    WITH base AS (
      SELECT stock_code, trade_date, status, ret,
             row_number() OVER (PARTITION BY stock_code ORDER BY trade_date DESC) AS rn_desc,
             lead(ret, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) AS ret_next
      FROM md.daily_hfq
    )
    SELECT {", ".join(pieces)}
    FROM base JOIN md.fwd_return l USING (stock_code, trade_date)
    """


def verify_labels(conn: duckdb.DuckDBPyConnection) -> dict:
    """校验标签表的行数对齐、停牌与尾部缺失、以及与官方收益口径的一致性。

    参数:
        conn: DuckDB 连接。

    返回:
        校验结果字典，``passed`` 为假时必须当作建库失败处理。
    """
    expected_rows = int(conn.execute("SELECT count(*) FROM md.daily_hfq").fetchone()[0])
    actual_rows = int(conn.execute("SELECT count(*) FROM md.fwd_return").fetchone()[0])
    raw = conn.execute(_verify_sql()).df().iloc[0].to_dict()
    stats = {key: (int(value) if key != "rows" else int(value)) for key, value in raw.items()}

    failures: list[str] = []
    if actual_rows != expected_rows:
        failures.append(f"行数不一致：fwd_return {actual_rows} 行，daily_hfq {expected_rows} 行")
    if stats["suspended_with_value"] != 0:
        failures.append(f"停牌行仍有标签值 {stats['suspended_with_value']} 行")
    if stats["extreme_rows"] != 0:
        failures.append(f"收益绝对值超过 {EXTREME_ABS_RETURN} 的行有 {stats['extreme_rows']} 行")
    if stats["below_floor_rows"] != 0:
        failures.append(f"收益小于 {INVALID_RETURN_FLOOR:.0%} 的行有 {stats['below_floor_rows']} 行")
    for column in _LABEL_COLUMNS:
        got = stats[f"null_{column}"]
        want = stats[f"tail_{column}"]
        if got != want:
            failures.append(f"{column} 交易日上的缺失 {got} 行，尾部应有 {want} 行")
    mismatch_ratio = stats["ret_mismatch"] / max(stats["trading_rows"], 1)
    if mismatch_ratio > MISMATCH_RATIO_LIMIT:
        failures.append(
            f"与 ret 口径的差异 {stats['ret_mismatch']} 行（{mismatch_ratio:.4%}）"
            f"超过上限 {MISMATCH_RATIO_LIMIT:.2%}"
        )

    result = {
        "passed": not failures,
        "failures": failures,
        "rows": actual_rows,
        "trading_rows": stats["trading_rows"],
        "suspended_rows": stats["suspended_rows"],
        "ret_mismatch_rows": stats["ret_mismatch"],
        "ret_mismatch_ratio": round(mismatch_ratio, 6),
    }
    if failures:
        logger.error("标签校验未通过：%s", "；".join(failures))
    else:
        logger.info("标签校验通过 %s", result)
    return result


def run(db_path: Path | str | None = None) -> dict:
    """在已有数据库上重建标签并校验。

    参数:
        db_path: 数据库路径，默认 data/databases/quant.duckdb。
    """
    ensure_dirs()
    conn = duckdb.connect(str(db_path or DB_FILE))
    try:
        rows = build_labels(conn)
        result = verify_labels(conn)
        conn.execute("CHECKPOINT")
    finally:
        conn.close()
    return {"rows": rows, **result}


def main() -> int:
    """命令行入口。"""
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
