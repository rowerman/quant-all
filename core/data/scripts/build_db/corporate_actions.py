"""把公司行动原始记录整理成回测账本。

gbbq 类别 1 记录送转、现金分红和配股条款，是主数据源；Tushare 分红表用于补缺
和交叉校验。两边都缺失时，只有总股本与复权因子同步变化才按股本变动兜底。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quant.codes import normalize_code
from scripts.config import DB_FILE, EXTRA_DIR
from scripts.errors import DbWriteError
from scripts.logging_conf import get_logger
from scripts.sources import gbbq

logger = get_logger("build_db.corporate_actions")

GBBQ_FILE = gbbq.GBBQ_FILE
DIVIDEND_GLOB = str(EXTRA_DIR / "tushare" / "raw_events" / "dividend" / "*.parquet")
DAILY_BASIC_GLOB = str(EXTRA_DIR / "tushare" / "raw" / "daily_basic" / "year=*" / "month=*.parquet")
TABLE_NAME = "corporate_action"
MANIFEST_NAME = "corporate_action_manifest"

CORPORATE_ACTION_COLUMNS = (
    "stock_code",
    "ex_date",
    "record_date",
    "pay_date",
    "share_multiplier",
    "cash_div_per_share",
    "rights_ratio",
    "rights_price",
    "action_type",
    "source",
    "source_event_key",
)


def _date(value) -> pd.Timestamp | None:
    """把 YYYYMMDD 或整数日期转成 Timestamp；空值返回 None。"""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _source_key(row: pd.Series) -> str:
    """为归一事件生成稳定来源键。"""
    rights_price = row.get("rights_price")
    payload = "|".join(
        [
            str(row["stock_code"]),
            str(row["ex_date"]),
            f"{float(row['share_multiplier']):.10f}",
            f"{float(row['cash_div_per_share']):.10f}",
            f"{float(row['rights_ratio']):.10f}",
            "nan" if pd.isna(rights_price) else f"{float(rights_price):.10f}",
            str(row["source"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _previous_trade_date_map(conn: duckdb.DuckDBPyConnection) -> dict[pd.Timestamp, pd.Timestamp]:
    """读取交易日历，返回每个交易日的前一交易日。"""
    try:
        frame = conn.execute(
            "SELECT trade_date FROM md.calendar WHERE is_open = 1 ORDER BY trade_date"
        ).df()
    except Exception as error:
        raise DbWriteError(f"读取交易日历失败，无法补齐公司行动登记日：{error}") from error
    days = [pd.Timestamp(value).normalize() for value in frame["trade_date"].tolist()]
    return {day: days[position - 1] for position, day in enumerate(days) if position}


def _action_type(share_multiplier: float, cash: float, rights_ratio: float) -> str:
    """按送转、现金和配股组合给出事件类型。"""
    share_changed = abs(float(share_multiplier) - 1.0) > 1e-12
    has_cash = float(cash) > 1e-12
    has_rights = float(rights_ratio) > 1e-12
    if has_rights:
        return "rights_mixed" if share_changed or has_cash else "rights"
    if share_changed and has_cash:
        return "mixed"
    if share_changed:
        return "consolidation" if float(share_multiplier) < 1.0 else "stock"
    return "cash"


def _load_gbbq_events(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """读取 gbbq 类别 1 并归一成公司行动。"""
    path = Path(GBBQ_FILE)
    if not path.exists():
        raise DbWriteError(
            f"gbbq 文件不存在：{path}；请先运行 python -m scripts.sources.fetch_all gbbq"
        )
    digest = gbbq.verify_file(path)
    try:
        raw = conn.execute(
            """
            SELECT market, code, datetime, hongli_panqianliutong,
                   peigujia_qianzongguben, songgu_qianzongguben,
                   peigu_houzongguben
            FROM read_parquet(?)
            WHERE category = 1
            """,
            [str(path)],
        ).df()
    except Exception as error:
        raise DbWriteError(f"读取 gbbq 原始分片失败：{error}") from error
    source_rows = len(raw)
    suffix = raw["market"].map({0: "SZ", 1: "SH"})
    raw = raw.loc[suffix.notna()].copy()
    raw["ts_code"] = raw["code"].astype(str).str.zfill(6) + "." + suffix.loc[raw.index]
    raw["stock_code"] = [normalize_code(value) for value in raw["ts_code"]]
    raw = raw.loc[raw["stock_code"].notna()].copy()
    raw["ex_date"] = [_date(value) for value in raw["datetime"]]
    null_ex_date = int(raw["ex_date"].isna().sum())
    raw = raw.loc[raw["ex_date"].notna()].copy()
    for name in (
        "hongli_panqianliutong",
        "peigujia_qianzongguben",
        "songgu_qianzongguben",
        "peigu_houzongguben",
    ):
        raw[name] = pd.to_numeric(raw[name], errors="coerce").fillna(0.0)
    raw["share_multiplier"] = 1.0 + raw["songgu_qianzongguben"] / 10.0
    raw["cash_div_per_share"] = raw["hongli_panqianliutong"] / 10.0
    raw["rights_ratio"] = raw["peigu_houzongguben"].clip(lower=0.0) / 10.0
    raw["rights_price"] = raw["peigujia_qianzongguben"].where(
        raw["rights_ratio"] > 1e-12
    )
    invalid_rights = raw["rights_ratio"].gt(1e-12) & raw["rights_price"].le(0)
    if invalid_rights.any():
        sample = raw.loc[invalid_rights, ["stock_code", "ex_date"]].head(5).to_dict("records")
        logger.warning(
            "gbbq 有 %d 条配股记录缺少正数配股价，保留其余条款但不执行配股：%s",
            int(invalid_rights.sum()),
            sample,
        )
        raw.loc[invalid_rights, "rights_ratio"] = 0.0
        raw.loc[invalid_rights, "rights_price"] = float("nan")

    selected = raw[
        ["stock_code", "ex_date", "share_multiplier", "cash_div_per_share",
         "rights_ratio", "rights_price"]
    ].copy()
    selected = selected.drop_duplicates().reset_index(drop=True)
    conflicts = selected.duplicated(["stock_code", "ex_date"], keep=False)
    if conflicts.any():
        sample = selected.loc[conflicts, ["stock_code", "ex_date"]].head(5).to_dict("records")
        raise DbWriteError(f"gbbq 同一股票同一除权日存在冲突条款：{sample}")
    previous = _previous_trade_date_map(conn)
    selected["record_date"] = [
        previous.get(pd.Timestamp(value)) for value in selected["ex_date"]
    ]
    selected["pay_date"] = selected["ex_date"]
    selected["action_type"] = [
        _action_type(share, cash, rights)
        for share, cash, rights in zip(
            selected["share_multiplier"],
            selected["cash_div_per_share"],
            selected["rights_ratio"],
        )
    ]
    selected["source"] = "gbbq"
    selected["source_event_key"] = selected.apply(_source_key, axis=1)
    selected = selected[list(CORPORATE_ACTION_COLUMNS)]
    stats = {
        "gbbq_rows": source_rows,
        "gbbq_null_ex_date_rows": null_ex_date,
        "gbbq_events": len(selected),
        "gbbq_stocks": int(selected["stock_code"].nunique()),
        "gbbq_rights_missing_price": int(invalid_rights.sum()),
    }
    return selected, stats, digest


def _load_dividend_events(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取 Tushare 分红实施记录并归一成公司行动。"""
    dividend_dir = Path(DIVIDEND_GLOB).parent
    if not dividend_dir.exists():
        raise DbWriteError(f"分红原始目录不存在：{dividend_dir}")
    try:
        raw = conn.execute(
            f"""
            SELECT ts_code, end_date, ann_date, div_proc, stk_div,
                   stk_bo_rate, stk_co_rate, cash_div, cash_div_tax,
                   record_date, ex_date, pay_date, div_listdate
            FROM read_parquet('{DIVIDEND_GLOB}')
            WHERE trim(div_proc) LIKE '实施%'
            """
        ).df()
    except Exception as error:
        raise DbWriteError(f"读取分红原始分片失败：{error}") from error

    source_rows = len(raw)
    raw["ts_code"] = raw["ts_code"].astype(str)
    raw = raw.loc[raw["ts_code"].str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")].copy()
    raw["stock_code"] = [normalize_code(value) for value in raw["ts_code"]]
    raw = raw.loc[raw["stock_code"].notna()].copy()
    raw["ex_date"] = [_date(value) for value in raw["ex_date"]]
    null_ex_date = int(raw["ex_date"].isna().sum())
    raw = raw.loc[raw["ex_date"].notna()].copy()
    raw["record_date"] = [_date(value) for value in raw["record_date"]]
    raw["pay_date"] = [_date(value) for value in raw["pay_date"]]
    raw["stk_div"] = pd.to_numeric(raw["stk_div"], errors="coerce").fillna(0.0).clip(lower=0.0)
    raw["cash_div_tax"] = (
        pd.to_numeric(raw["cash_div_tax"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    events = (
        raw.groupby(["stock_code", "ex_date"], as_index=False)
        .agg(
            stk_div=("stk_div", "max"),
            cash_div_tax=("cash_div_tax", "max"),
            record_date=("record_date", "max"),
            pay_date=("pay_date", "max"),
        )
        .sort_values(["stock_code", "ex_date"])
        .reset_index(drop=True)
    )
    previous = _previous_trade_date_map(conn)
    events["record_date"] = [
        row.record_date
        if row.record_date is not None and not pd.isna(row.record_date)
        else previous.get(pd.Timestamp(row.ex_date))
        for row in events.itertuples(index=False)
    ]
    events["pay_date"] = [
        row.pay_date if row.pay_date is not None and not pd.isna(row.pay_date) else row.ex_date
        for row in events.itertuples(index=False)
    ]
    events["share_multiplier"] = 1.0 + events["stk_div"].astype(float)
    events["cash_div_per_share"] = events["cash_div_tax"].astype(float)
    events["rights_ratio"] = 0.0
    events["rights_price"] = float("nan")
    events["action_type"] = [
        _action_type(share, cash, 0.0)
        for share, cash in zip(events["share_multiplier"], events["cash_div_per_share"])
    ]
    events = events.loc[
        (events["share_multiplier"].sub(1.0).abs() > 1e-12)
        | (events["cash_div_per_share"] > 1e-12)
    ].copy()
    events["source"] = "dividend"
    events["source_event_key"] = events.apply(_source_key, axis=1)
    events = events[list(CORPORATE_ACTION_COLUMNS)]
    stats = {
        "dividend_source_rows": source_rows,
        "dividend_null_ex_date_rows": null_ex_date,
        "dividend_events": len(events),
        "dividend_stocks": int(events["stock_code"].nunique()),
    }
    return events, stats


def _crosscheck(
    gbbq_events: pd.DataFrame,
    dividends: pd.DataFrame,
) -> tuple[int, list[str]]:
    """比较 gbbq 与 Tushare 重叠事件，返回冲突数量和样例。"""
    left = gbbq_events[
        ["stock_code", "ex_date", "share_multiplier", "cash_div_per_share"]
    ]
    right = dividends[
        ["stock_code", "ex_date", "share_multiplier", "cash_div_per_share"]
    ]
    merged = left.merge(
        right,
        on=["stock_code", "ex_date"],
        suffixes=("_gbbq", "_ts"),
    )
    if merged.empty:
        return 0, []
    share_diff = (merged["share_multiplier_gbbq"] - merged["share_multiplier_ts"]).abs()
    cash_diff = (merged["cash_div_per_share_gbbq"] - merged["cash_div_per_share_ts"]).abs()
    conflict = (share_diff > 0.005) | (cash_diff > 0.01)
    samples = [
        (
            f"{row.stock_code}:{pd.Timestamp(row.ex_date).date()} "
            f"gbbq(share={row.share_multiplier_gbbq:.6f},cash={row.cash_div_per_share_gbbq:.6f}) "
            f"tushare(share={row.share_multiplier_ts:.6f},cash={row.cash_div_per_share_ts:.6f})"
        )
        for row in merged.loc[conflict].head(20).itertuples(index=False)
    ]
    return int(conflict.sum()), samples


def _share_count_fallback(
    conn: duckdb.DuckDBPyConnection,
    known: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """在两边都缺事件时，用总股本与复权因子严格交叉验证拆并。"""
    try:
        candidates = conn.execute(
            """
            WITH d AS (
              SELECT stock_code, trade_date, factor, is_ex_div,
                     lag(factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_factor
              FROM md.daily_bar
              WHERE factor IS NOT NULL
            ), jumps AS (
              SELECT stock_code, trade_date AS ex_date, factor / prev_factor AS factor_ratio
              FROM d
              WHERE is_ex_div AND prev_factor > 0
            ), basic AS (
              SELECT CASE
                       WHEN ts_code LIKE '%.SH' THEN 'SH' || substr(ts_code, 1, 6)
                       WHEN ts_code LIKE '%.SZ' THEN 'SZ' || substr(ts_code, 1, 6)
                       WHEN ts_code LIKE '%.BJ' THEN 'BJ' || substr(ts_code, 1, 6)
                     END AS stock_code,
                     CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date,
                     total_share
              FROM read_parquet(?)
              WHERE total_share IS NOT NULL
            ), shares AS (
              SELECT stock_code, trade_date, total_share,
                     lag(total_share) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_share
              FROM basic
            )
            SELECT j.stock_code, j.ex_date, j.factor_ratio,
                   s.total_share / s.prev_share AS share_ratio
            FROM jumps j
            JOIN shares s
              ON s.stock_code = j.stock_code
             AND s.trade_date = j.ex_date
            WHERE s.prev_share > 0
              AND abs(s.total_share / s.prev_share - 1.0) > 0.005
              AND abs(j.factor_ratio / (s.total_share / s.prev_share) - 1.0) <= 0.05
            """,
            [DAILY_BASIC_GLOB],
        ).df()
    except Exception as error:
        raise DbWriteError(f"读取股本变动兜底事件失败：{error}") from error
    candidates["ex_date"] = pd.to_datetime(candidates["ex_date"]).dt.normalize()
    known_keys = set(zip(known["stock_code"], pd.to_datetime(known["ex_date"])))
    candidates = candidates.loc[
        [
            (row.stock_code, pd.Timestamp(row.ex_date)) not in known_keys
            for row in candidates.itertuples(index=False)
        ]
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS), {
            "share_count_events": 0,
            "share_count_stocks": 0,
        }
    previous = _previous_trade_date_map(conn)
    candidates["record_date"] = [
        previous.get(pd.Timestamp(value)) for value in candidates["ex_date"]
    ]
    candidates["pay_date"] = candidates["ex_date"]
    candidates["share_multiplier"] = candidates["share_ratio"].astype(float)
    candidates["cash_div_per_share"] = 0.0
    candidates["rights_ratio"] = 0.0
    candidates["rights_price"] = float("nan")
    candidates["action_type"] = [
        "consolidation" if value < 1.0 else "stock"
        for value in candidates["share_multiplier"]
    ]
    candidates["source"] = "share_count"
    candidates["source_event_key"] = candidates.apply(_source_key, axis=1)
    candidates = candidates[list(CORPORATE_ACTION_COLUMNS)]
    return candidates, {
        "share_count_events": len(candidates),
        "share_count_stocks": int(candidates["stock_code"].nunique()),
    }


def _unexplained_jumps(conn: duckdb.DuckDBPyConnection, known: pd.DataFrame) -> int:
    """统计已知事件之外超过 5% 的复权因子跳变。"""
    frame = conn.execute(
        """
        WITH d AS (
          SELECT stock_code, trade_date, factor, is_ex_div,
                 lag(factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_factor
          FROM md.daily_bar
          WHERE factor IS NOT NULL
        )
        SELECT stock_code, trade_date, factor / prev_factor AS ratio
        FROM d
        WHERE is_ex_div AND prev_factor > 0
          AND abs(factor / prev_factor - 1.0) > 0.05
        """
    ).df()
    known_keys = set(zip(known["stock_code"], pd.to_datetime(known["ex_date"])))
    return int(
        sum(
            (row.stock_code, pd.Timestamp(row.trade_date)) not in known_keys
            for row in frame.itertuples(index=False)
        )
    )


def build_frame(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造公司行动账本及摘要，不写库。"""
    gbbq_events, stats, gbbq_digest = _load_gbbq_events(conn)
    dividends, dividend_stats = _load_dividend_events(conn)
    stats.update(dividend_stats)
    overlap = gbbq_events.merge(
        dividends[["stock_code", "ex_date"]],
        on=["stock_code", "ex_date"],
        how="inner",
    )
    conflicts, samples = _crosscheck(gbbq_events, dividends)
    stats["tushare_overlap_events"] = len(overlap)
    stats["tushare_conflict_events"] = conflicts
    stats["tushare_conflict_samples"] = samples

    gbbq_keys = set(zip(gbbq_events["stock_code"], pd.to_datetime(gbbq_events["ex_date"])))
    extra_dividends = dividends.loc[
        [
            (row.stock_code, pd.Timestamp(row.ex_date)) not in gbbq_keys
            for row in dividends.itertuples(index=False)
        ]
    ].copy()
    primary_parts = [
        item for item in (gbbq_events, extra_dividends) if not item.empty
    ]
    primary = (
        pd.concat(primary_parts, ignore_index=True)
        if primary_parts
        else gbbq_events.copy()
    )
    fallback, fallback_stats = _share_count_fallback(conn, primary)
    frame_parts = [item for item in (primary, fallback) if not item.empty]
    frame = (
        pd.concat(frame_parts, ignore_index=True)
        if frame_parts
        else primary.copy()
    )
    frame = frame.sort_values(["stock_code", "ex_date"]).reset_index(drop=True)
    stats.update(fallback_stats)
    stats["unexplained_factor_jumps"] = _unexplained_jumps(conn, frame)
    stats["events"] = len(frame)
    stats["stocks"] = int(frame["stock_code"].nunique())
    stats["rights_events"] = int(frame["rights_ratio"].gt(1e-12).sum())
    stats["rights_stocks"] = int(frame.loc[
        frame["rights_ratio"].gt(1e-12), "stock_code"
    ].nunique())
    stats["min_ex_date"] = str(frame["ex_date"].min().date()) if len(frame) else None
    stats["max_ex_date"] = str(frame["ex_date"].max().date()) if len(frame) else None
    stats["gbbq_sha256"] = gbbq_digest
    stats["digest"] = hashlib.sha256(
        pd.util.hash_pandas_object(frame.fillna("").astype(str), index=False).values.tobytes()
    ).hexdigest()
    return frame, stats


def _write_manifest(conn: duckdb.DuckDBPyConnection, stats: dict[str, Any]) -> None:
    """写入公司行动摘要表。"""
    conn.execute(
        """
        CREATE OR REPLACE TABLE meta.corporate_action_manifest AS
        SELECT ?::VARCHAR AS source,
               ?::BIGINT AS source_rows,
               ?::BIGINT AS null_ex_date_rows,
               ?::BIGINT AS events,
               ?::BIGINT AS stocks,
               ?::DATE AS min_ex_date,
               ?::DATE AS max_ex_date,
               ?::BIGINT AS gbbq_rows,
               ?::BIGINT AS gbbq_events,
               ?::BIGINT AS gbbq_rights_missing_price,
               ?::BIGINT AS dividend_events,
               ?::BIGINT AS share_count_events,
               ?::BIGINT AS tushare_overlap_events,
               ?::BIGINT AS tushare_conflict_events,
               ?::BIGINT AS unexplained_factor_jumps,
               ?::BIGINT AS rights_events,
               ?::BIGINT AS rights_stocks,
               ?::VARCHAR AS gbbq_sha256,
               ?::VARCHAR AS conflict_samples,
               ?::VARCHAR AS digest,
               current_timestamp AS built_at
        """,
        [
            "gbbq category1 + tushare dividend + share_count fallback",
            stats.get("dividend_source_rows", 0),
            stats.get("dividend_null_ex_date_rows", 0),
            stats.get("events", 0),
            stats.get("stocks", 0),
            stats.get("min_ex_date"),
            stats.get("max_ex_date"),
            stats.get("gbbq_rows", 0),
            stats.get("gbbq_events", 0),
            stats.get("gbbq_rights_missing_price", 0),
            stats.get("dividend_events", 0),
            stats.get("share_count_events", 0),
            stats.get("tushare_overlap_events", 0),
            stats.get("tushare_conflict_events", 0),
            stats.get("unexplained_factor_jumps", 0),
            stats.get("rights_events", 0),
            stats.get("rights_stocks", 0),
            stats.get("gbbq_sha256"),
            json.dumps(stats.get("tushare_conflict_samples", []), ensure_ascii=False),
            stats.get("digest"),
        ],
    )


def _update_dictionary(conn: duckdb.DuckDBPyConnection) -> None:
    """在数据字典中登记公司行动表，保留其它表条目。"""
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta.data_dictionary (
            table_name VARCHAR,
            column_name VARCHAR,
            unit VARCHAR,
            description VARCHAR
        )
        """
    )
    conn.execute("DELETE FROM meta.data_dictionary WHERE table_name = 'ref.corporate_action'")
    rows = [
        ("ref.corporate_action", "stock_code", "字符串", "统一股票代码"),
        ("ref.corporate_action", "ex_date", "日期", "除权除息日，开盘前生效"),
        ("ref.corporate_action", "record_date", "日期", "上一交易日，供追溯"),
        ("ref.corporate_action", "pay_date", "日期", "现金分红到账日；缺失按除权日"),
        ("ref.corporate_action", "share_multiplier", "倍", "送转/拆并后的每股持股倍数"),
        ("ref.corporate_action", "cash_div_per_share", "元/股", "税前每股现金分红"),
        ("ref.corporate_action", "rights_ratio", "倍", "每股可配新股数"),
        ("ref.corporate_action", "rights_price", "元/股", "配股认购价；无配股为空"),
        ("ref.corporate_action", "action_type", "枚举",
         "cash/stock/mixed/rights/rights_mixed/consolidation"),
        ("ref.corporate_action", "source", "字符串", "gbbq、dividend 或 share_count"),
        ("ref.corporate_action", "source_event_key", "SHA-256 前缀", "来源事件稳定键"),
    ]
    conn.executemany("INSERT INTO meta.data_dictionary VALUES (?, ?, ?, ?)", rows)


def build(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """在现有连接上增量重建公司行动表。"""
    frame, stats = build_frame(conn)
    if frame.empty:
        raise DbWriteError("公司行动账本没有有效事件")
    if frame.duplicated(["stock_code", "ex_date"]).any():
        raise DbWriteError("公司行动账本存在重复股票和除权日")
    if (frame["share_multiplier"] <= 0).any():
        raise DbWriteError("公司行动账本存在非正股数乘数")
    if (frame["cash_div_per_share"] < 0).any():
        raise DbWriteError("公司行动账本存在负现金分红")
    if (frame["rights_ratio"] < 0).any():
        raise DbWriteError("公司行动账本存在负配股比例")
    rights = frame["rights_ratio"] > 1e-12
    if (rights & (~frame["rights_price"].gt(0))).any():
        raise DbWriteError("公司行动账本存在缺失配股价")

    conn.execute("CREATE SCHEMA IF NOT EXISTS ref")
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.begin()
    try:
        conn.register("corporate_action_frame", frame)
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE corporate_action_new AS "
            "SELECT * FROM corporate_action_frame"
        )
        conn.unregister("corporate_action_frame")
        conn.execute("DROP TABLE IF EXISTS ref.corporate_action")
        conn.execute(
            "CREATE TABLE ref.corporate_action AS "
            "SELECT stock_code, CAST(ex_date AS DATE) AS ex_date, "
            "CAST(record_date AS DATE) AS record_date, CAST(pay_date AS DATE) AS pay_date, "
            "CAST(share_multiplier AS DOUBLE) AS share_multiplier, "
            "CAST(cash_div_per_share AS DOUBLE) AS cash_div_per_share, "
            "CAST(rights_ratio AS DOUBLE) AS rights_ratio, "
            "CAST(rights_price AS DOUBLE) AS rights_price, "
            "action_type, source, source_event_key "
            "FROM corporate_action_new"
        )
        conn.execute("DROP TABLE corporate_action_new")
        _write_manifest(conn, stats)
        _update_dictionary(conn)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    logger.info(
        "ref.corporate_action 写入 %d 个事件、%d 只股票；gbbq %d，Tushare %d，"
        "股本兜底 %d，配股 %d，Tushare 冲突 %d，未解释跳变 %d",
        stats["events"],
        stats["stocks"],
        stats["gbbq_events"],
        stats["dividend_events"],
        stats["share_count_events"],
        stats["rights_events"],
        stats["tushare_conflict_events"],
        stats["unexplained_factor_jumps"],
    )
    return stats


def main() -> int:
    """命令行入口：增量更新现有主库。"""
    if not DB_FILE.exists():
        raise SystemExit(f"主库不存在：{DB_FILE}")
    conn = duckdb.connect(str(DB_FILE))
    try:
        stats = build(conn)
    finally:
        conn.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
