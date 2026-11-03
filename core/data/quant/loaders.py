"""清洗后数据的高层读取接口。

统一从这里取数据，避免下游直接拼 SQL 时选错价格口径。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from quant.db import get_conn
from quant.schema import (
    ADJUST_MODES,
    DAILY_COLUMNS,
    DAILY_TABLES,
    KEY_COLUMNS,
    LABEL_HORIZONS,
    LABEL_PRICES,
    label_columns,
)

# 标的清单状态：对外用 listed/delisted，库里存 Tushare 的 L/D/P
STATUS_CODES = {"listed": "L", "delisted": "D", "suspended": "P", "l": "L", "d": "D", "p": "P"}

FINANCIAL_TABLES = (
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "fina_indicator",
)
FINANCIAL_KEYS = (
    "stock_code",
    "ts_code",
    "report_date",
    "ann_date",
    "f_ann_date",
    "available_date",
    "report_type",
    "update_flag",
    "comp_type",
)


def _status_code(status: str) -> str:
    """把对外的上市状态写法映射成 Tushare 的 list_status。

    参数:
        status: ``listed`` / ``delisted`` / ``suspended``，或直接给 ``L`` / ``D`` / ``P``。

    返回:
        单字母的 ``list_status``；无法识别时抛 ValueError。
    """
    code = STATUS_CODES.get(str(status).strip().lower())
    if code is None:
        raise ValueError(f"status 只支持 listed/delisted/L/D/P，收到 {status!r}")
    return code


def _require_table(conn: duckdb.DuckDBPyConnection, schema: str, table: str) -> None:
    """确认表存在，缺失时报出可用表清单而不是 DuckDB 的 Catalog Error。

    参数:
        conn: DuckDB 连接。
        schema: schema 名。
        table: 表名。
    """
    available = [
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = ? ORDER BY 1",
            [schema],
        ).fetchall()
    ]
    if table not in available:
        raise ValueError(
            f"{schema}.{table} 不在当前数据库里；{schema} 下可用表：{available or '（该 schema 未建）'}"
        )


def load_panel(
    start: str | None = None,
    end: str | None = None,
    fields: list[str] | None = None,
    adjust: str = "raw",
    stocks: list[str] | None = None,
    drop_suspended: bool = False,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """读取日线面板。

    参数:
        start: 起始交易日，闭区间。
        end: 截止交易日，闭区间。
        fields: 需要的列；默认返回全部列。``stock_code``/``trade_date`` 始终返回。
        adjust: ``raw`` 原始价 / ``hfq`` 后复权 / ``qfq`` 前复权，对应三张同构表。
        stocks: 股票池；None 表示全部。
        drop_suspended: 为真时剔除 ``status != 'trading'`` 的行。
        db_path: 数据库路径，默认 data/databases/quant.duckdb。

    返回:
        长表 DataFrame，按 ``trade_date``、``stock_code`` 排序。
    """
    if adjust not in ADJUST_MODES:
        raise ValueError(f"adjust 只能是 {ADJUST_MODES}，收到 {adjust!r}")
    table = DAILY_TABLES[adjust]
    selected = _resolve_fields(fields)

    conditions = []
    params: list = []
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if stocks:
        placeholders = ", ".join(["?"] * len(stocks))
        conditions.append(f"stock_code IN ({placeholders})")
        params.extend(stocks)
    if drop_suspended:
        conditions.append("status = 'trading'")

    sql = f"SELECT {', '.join(selected)} FROM {table}"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY trade_date, stock_code"

    with get_conn(db_path) as conn:
        return conn.execute(sql, params).df()


def load_panel_wide(
    field: str,
    start: str | None = None,
    end: str | None = None,
    adjust: str = "hfq",
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """把单个字段透视成「交易日 × 股票」的宽表，便于做横截面运算。

    参数:
        field: 需要透视的字段。
        start: 起始交易日。
        end: 截止交易日。
        adjust: 价格口径。
        db_path: 数据库路径。
    """
    frame = load_panel(start=start, end=end, fields=[field], adjust=adjust, db_path=db_path)
    return frame.pivot(index="trade_date", columns="stock_code", values=field)


def load_calendar(start: str | None = None, end: str | None = None,
                  db_path: Path | str | None = None) -> pd.DataFrame:
    """读取交易日历。

    参数:
        start: 起始日期。
        end: 截止日期。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM md.calendar"
    conditions = []
    params: list = []
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY trade_date"
    with get_conn(db_path) as conn:
        return conn.execute(sql, params).df()


def load_stock_basic(status: str | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    """读取股票基础信息。

    参数:
        status: 上市状态过滤，``"listed"`` / ``"delisted"``（也接受 Tushare 的 ``"L"`` / ``"D"`` / ``"P"``）；
            None 表示全部。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM ref.stock_basic"
    params: list = []
    if status:
        sql += " WHERE list_status = ?"
        params.append(_status_code(status))
    with get_conn(db_path) as conn:
        return conn.execute(sql, params).df()


def load_suspend_segments(db_path: Path | str | None = None) -> pd.DataFrame:
    """读取停牌区间表。

    参数:
        db_path: 数据库路径。
    """
    with get_conn(db_path) as conn:
        return conn.execute("SELECT * FROM md.suspend_segment ORDER BY stock_code, start_date").df()


def load_corporate_actions(
    start: str | None = None,
    end: str | None = None,
    stocks: list[str] | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """读取公司行动账本，包含送转、现金分红、配股比例与配股价。

    参数:
        start: 除权日下界，闭区间。
        end: 除权日上界，闭区间。
        stocks: 股票池；None 表示全部。
        db_path: 数据库路径。
    """
    conditions = []
    params: list = []
    if start:
        conditions.append("ex_date >= ?")
        params.append(start)
    if end:
        conditions.append("ex_date <= ?")
        params.append(end)
    if stocks:
        placeholders = ", ".join(["?"] * len(stocks))
        conditions.append(f"stock_code IN ({placeholders})")
        params.extend(stocks)
    sql = "SELECT * FROM ref.corporate_action"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY ex_date, stock_code"
    with get_conn(db_path) as conn:
        _require_table(conn, "ref", "corporate_action")
        return conn.execute(sql, params).df()


def load_index_weight(index_code: str | None = None,
                      start: str | None = None,
                      end: str | None = None,
                      db_path: Path | str | None = None) -> pd.DataFrame:
    """读取指数历史成分权重（``ref.index_weight``）。

    参数:
        index_code: 指数代码，如 ``000300.SH``；None 表示全部。
        start: 起始交易日。
        end: 截止交易日。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM ref.index_weight"
    conditions = []
    params: list = []
    if index_code:
        conditions.append("index_code = ?")
        params.append(index_code)
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY index_code, trade_date, con_code"
    with get_conn(db_path) as conn:
        return conn.execute(sql, params).df()


def load_sw_industry_member(
    start: str | None = None,
    end: str | None = None,
    stocks: list[str] | None = None,
    db_path: Path | str | None = None,
    classification: str | None = None,
) -> pd.DataFrame:
    """读取与日期区间相交的时点申万历史行业成员。

    参数:
        start: 区间起始日；保留尚未在该日之前退出的记录。
        end: 区间截止日；保留不晚于该日生效的记录。
        stocks: 可选项目口径股票代码列表。
        db_path: 数据库路径。
        classification: 可选 ``SW2014`` / ``SW2021``；None 表示读取全部版本。
    """
    sql = "SELECT * FROM ref.sw_industry_member"
    conditions = []
    params: list = []
    if start:
        conditions.append("(out_date IS NULL OR out_date >= ?)")
        params.append(start)
    if end:
        conditions.append("in_date <= ?")
        params.append(end)
    if stocks:
        placeholders = ", ".join(["?"] * len(stocks))
        conditions.append(f"stock_code IN ({placeholders})")
        params.extend(stocks)
    if classification:
        conditions.append("classification = ?")
        params.append(classification)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY stock_code, in_date, l3_code"
    with get_conn(db_path) as conn:
        _require_table(conn, "ref", "sw_industry_member")
        return conn.execute(sql, params).df()


def load_index_constituents(index_code: str | None = None,
                            db_path: Path | str | None = None) -> pd.DataFrame:
    """读取指数历史成分权重（`load_index_weight` 的旧名，保留兼容）。

    参数:
        index_code: 指数代码，如 ``000300.SH``。
        db_path: 数据库路径。
    """
    return load_index_weight(index_code=index_code, db_path=db_path)


def load_index_daily(index_code: str | None = None,
                     start: str | None = None,
                     end: str | None = None,
                     db_path: Path | str | None = None) -> pd.DataFrame:
    """读取指数日线行情（``md.index_daily``）。

    参数:
        index_code: 指数代码，如 ``000300.SH``；None 表示全部。
        start: 起始交易日。
        end: 截止交易日。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM md.index_daily"
    conditions = []
    params: list = []
    if index_code:
        conditions.append("index_code = ?")
        params.append(index_code)
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY index_code, trade_date"
    with get_conn(db_path) as conn:
        _require_table(conn, "md", "index_daily")
        return conn.execute(sql, params).df()


def load_sw_index_daily(index_code: str | None = None,
                        start: str | None = None,
                        end: str | None = None,
                        db_path: Path | str | None = None) -> pd.DataFrame:
    """读取申万行业指数日线（``md.sw_index_daily``）。

    申万指数走 ``sw_daily`` 接口，字段比 ``md.index_daily`` 多出 ``pe`` / ``pb`` / 市值列。

    参数:
        index_code: 行业指数代码，如 ``801010.SI``；None 表示全部。
        start: 起始交易日。
        end: 截止交易日。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM md.sw_index_daily"
    conditions = []
    params: list = []
    if index_code:
        conditions.append("index_code = ?")
        params.append(index_code)
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY index_code, trade_date"
    with get_conn(db_path) as conn:
        _require_table(conn, "md", "sw_index_daily")
        return conn.execute(sql, params).df()


def load_index_basic(index_code: str | None = None,
                     market: str | None = None,
                     category: str | None = None,
                     db_path: Path | str | None = None) -> pd.DataFrame:
    """读取指数目录（``ref.index_basic``）。

    参数:
        index_code: 指数代码。
        market: 指数来源，``SSE`` / ``SZSE`` / ``CSI`` / ``SW``。
        category: 指数类别，如 ``规模指数``。
        db_path: 数据库路径。
    """
    sql = "SELECT * FROM ref.index_basic"
    conditions = []
    params: list = []
    if index_code:
        conditions.append("index_code = ?")
        params.append(index_code)
    if market:
        conditions.append("market = ?")
        params.append(market)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY index_code"
    with get_conn(db_path) as conn:
        _require_table(conn, "ref", "index_basic")
        return conn.execute(sql, params).df()


def load_label(start: str | None = None,
               end: str | None = None,
               horizons: list[int] | None = None,
               price: str = "close",
               stocks: list[str] | None = None,
               db_path: Path | str | None = None) -> pd.DataFrame:
    """读取未来收益标签（``md.fwd_return``）。

    参数:
        start: 起始交易日，闭区间。
        end: 截止交易日，闭区间。
        horizons: 预测周期（交易日数），默认六档；只能取 ``LABEL_HORIZONS`` 中的值。
        price: ``close`` 表示 t 日收盘价买入、t+N 日收盘价卖出；
            ``open`` 表示 t+1 日开盘价买入、t+1+N 日开盘价卖出。
        stocks: 股票池；None 表示全部。
        db_path: 数据库路径。

    返回:
        长表，列含 ``stock_code``、``trade_date`` 与所选周期的标签，按交易日、代码排序。
    """
    if price not in LABEL_PRICES:
        raise ValueError(f"price 只能是 {list(LABEL_PRICES)}，收到 {price!r}")
    chosen = list(horizons) if horizons else list(LABEL_HORIZONS)
    unknown = [value for value in chosen if value not in LABEL_HORIZONS]
    if unknown:
        raise ValueError(f"未知的预测周期 {unknown}；可用周期：{list(LABEL_HORIZONS)}")

    conditions = []
    params: list = []
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)
    if stocks:
        placeholders = ", ".join(["?"] * len(stocks))
        conditions.append(f"stock_code IN ({placeholders})")
        params.extend(stocks)

    selected = KEY_COLUMNS + label_columns(price, chosen)
    sql = f"SELECT {', '.join(selected)} FROM md.fwd_return"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY trade_date, stock_code"
    with get_conn(db_path) as conn:
        _require_table(conn, "md", "fwd_return")
        return conn.execute(sql, params).df()


def load_label_wide(horizon: int,
                    start: str | None = None,
                    end: str | None = None,
                    price: str = "close",
                    db_path: Path | str | None = None) -> pd.DataFrame:
    """把某一档标签透视成「交易日 × 股票」宽表，便于做横截面运算。

    参数:
        horizon: 预测周期（交易日数）。
        start: 起始交易日。
        end: 截止交易日。
        price: ``close`` / ``open``，含义同 ``load_label``。
        db_path: 数据库路径。
    """
    frame = load_label(start=start, end=end, horizons=[horizon], price=price, db_path=db_path)
    column = label_columns(price, [horizon])[0]
    return frame.pivot(index="trade_date", columns="stock_code", values=column)


def load_members(table: str, name: str | None = None,
                 db_path: Path | str | None = None) -> pd.DataFrame:
    """读取板块/概念成分。

    参数:
        table: ``industry_member`` / ``concept_member`` / ``block_member``。
        name: 板块或概念名称，None 表示全部。
        db_path: 数据库路径。
    """
    if table == "industry_member":
        sql = (
            "SELECT stock_code, ts_code, l1_code AS block_code, l1_name AS block_name, "
            "l2_code, l2_name, l3_code, l3_name, classification, in_date, out_date, "
            "source_api, carry_forward FROM ref.sw_industry_member"
        )
        params: list = []
        if name:
            sql += " WHERE l1_name = ?"
            params.append(name)
        with get_conn(db_path) as conn:
            _require_table(conn, "ref", "sw_industry_member")
            return conn.execute(sql + " ORDER BY stock_code, in_date", params).df()
    if table in ("concept_member", "block_member"):
        raise ValueError(f"{table} 尚未入库；当前只支持 industry_member（申万历史行业）")
    raise ValueError(f"不支持的成分表: {table}")


def load_financial(stock_code: str, table: str = "income_statement",
                   start: str | None = None, end: str | None = None,
                   fields: list[str] | None = None,
                   db_path: Path | str | None = None) -> pd.DataFrame:
    """读取某只股票的财务报表。

    参数:
        stock_code: 项目口径代码。
        table: ``balance_sheet`` / ``income_statement`` / ``cash_flow`` / ``fina_indicator``。
        start: 报告期起始日，闭区间。
        end: 报告期截止日，闭区间。
        fields: 需要的列；默认返回全部列。
        db_path: 数据库路径。
    """
    if table not in FINANCIAL_TABLES:
        raise ValueError(f"不支持的财务表: {table}；可选：{list(FINANCIAL_TABLES)}")
    with get_conn(db_path) as conn:
        _require_table(conn, "fin", table)
        available = [row[0] for row in conn.execute(f"DESCRIBE fin.{table}").fetchall()]
        selected = available if fields is None else list(fields)
        if not selected:
            raise ValueError("fields 不能为空")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"{table} 没有字段 {unknown}")
        conditions = ["stock_code = ?"]
        params: list = [stock_code]
        if start:
            conditions.append("report_date >= ?")
            params.append(start)
        if end:
            conditions.append("report_date <= ?")
            params.append(end)
        return conn.execute(
            f"SELECT {', '.join(selected)} FROM fin.{table} "
            f"WHERE {' AND '.join(conditions)} ORDER BY report_date, ann_date",
            params,
        ).df()


def load_financial_panel(table: str = "fina_indicator",
                         start: str | None = None,
                         end: str | None = None,
                         fields: list[str] | None = None,
                         stocks: list[str] | None = None,
                         drop_suspended: bool = True,
                         db_path: Path | str | None = None) -> pd.DataFrame:
    """读取按公告日时点对齐的财务日面板。

    公告日次一交易日起才可见，即 ``trade_date > available_date``。

    参数:
        table: 财务表名。
        start: 起始交易日，闭区间。
        end: 截止交易日，闭区间。
        fields: 需要的指标列；默认返回该表全部指标。
        stocks: 股票池；None 表示全部。
        drop_suspended: 是否剔除停牌行。
        db_path: 数据库路径。
    """
    if table not in FINANCIAL_TABLES:
        raise ValueError(f"不支持的财务表: {table}；可选：{list(FINANCIAL_TABLES)}")
    with get_conn(db_path) as conn:
        _require_table(conn, "fin", table)
        columns = [row[0] for row in conn.execute(f"DESCRIBE fin.{table}").fetchall()]
        metric_columns = [name for name in columns if name not in FINANCIAL_KEYS]
        selected = metric_columns if fields is None else list(fields)
        if not selected:
            raise ValueError("fields 不能为空")
        unknown = sorted(set(selected) - set(metric_columns))
        if unknown:
            raise ValueError(f"{table} 没有可读取的指标字段 {unknown}")
        metric_sql = ", ".join(selected)
        conditions = []
        params: list = []
        if start:
            conditions.append("b.trade_date >= ?")
            params.append(start)
        if end:
            conditions.append("b.trade_date <= ?")
            params.append(end)
        if stocks:
            placeholders = ", ".join(["?"] * len(stocks))
            conditions.append(f"b.stock_code IN ({placeholders})")
            params.extend(stocks)
        if drop_suspended:
            conditions.append("b.status = 'trading'")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
        WITH src AS (
          SELECT stock_code, available_date, report_date, {metric_sql}
          FROM (
            SELECT *, row_number() OVER (
              PARTITION BY stock_code, available_date
              ORDER BY report_date DESC, update_flag DESC
            ) AS rn
            FROM fin.{table}
            WHERE available_date IS NOT NULL
          )
          WHERE rn = 1
        )
        SELECT b.trade_date, b.stock_code, f.report_date, f.available_date,
               {', '.join('f.' + name for name in selected)}
        FROM md.daily_bar b
        ASOF LEFT JOIN src f
          ON b.stock_code = f.stock_code AND b.trade_date > f.available_date
        {where}
        ORDER BY b.trade_date, b.stock_code
        """
        return conn.execute(sql, params).df()


def load_financial_manifest(db_path: Path | str | None = None) -> pd.DataFrame:
    """读取财务入库清单。"""
    with get_conn(db_path) as conn:
        _require_table(conn, "meta", "financial_manifest")
        return conn.execute(
            "SELECT * FROM meta.financial_manifest ORDER BY table_name"
        ).df()


def _resolve_fields(fields: list[str] | None) -> list[str]:
    """校验请求列是否存在于日线表中，并补上主键列。

    参数:
        fields: 调用方请求的列。
    """
    if not fields:
        return list(DAILY_COLUMNS)
    unknown = [name for name in fields if name not in DAILY_COLUMNS]
    if unknown:
        raise ValueError(f"未知字段: {unknown}")
    ordered = list(KEY_COLUMNS)
    ordered.extend(name for name in fields if name not in KEY_COLUMNS)
    return ordered
