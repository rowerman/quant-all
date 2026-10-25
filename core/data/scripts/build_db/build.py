"""把清洗产物与 Tushare 参照 / 事件数据导入 DuckDB。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time

import duckdb
import pandas as pd

from quant.codes import board_of, infer_exchange, normalize_code, normalize_codes
from scripts.build_db import corporate_actions, financial, labels
from scripts.build_db.ddl import SCHEMA_DDL, daily_select, hfq_sql, qfq_sql
from scripts.config import CLEAN_DIR, DB_DIR, DB_FILE, EXTRA_DIR, ensure_dirs
from scripts.errors import DbWriteError
from scripts.logging_conf import get_logger

logger = get_logger("build_db.build")

DAILY_GLOB = str(CLEAN_DIR / "daily" / "year=*" / "part.parquet")
TUSHARE_DIR = EXTRA_DIR / "tushare"
SW_INDUSTRY_ROOT = TUSHARE_DIR / "raw_industry"
SW_INDUSTRY_DIRS = {
    "SW2014": SW_INDUSTRY_ROOT / "sw2014",
    "SW2021": SW_INDUSTRY_ROOT / "sw2021",
}
# 兼容旧测试与外部引用。
SW_INDUSTRY_DIR = SW_INDUSTRY_DIRS["SW2021"]
SW_INDUSTRY_REPAIR_DIR = SW_INDUSTRY_ROOT / "repair"
SW_CLASSIFICATION_SWITCH = pd.Timestamp("2021-12-13")
_TS_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def to_iso(value) -> str | None:
    """把 20200727 形式的日期转成 2020-07-27，空值返回 None。

    参数:
        value: 原始日期值。
    """
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat"):
        return None
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def build_daily(conn: duckdb.DuckDBPyConnection) -> None:
    """导入三张同构日线表。

    参数:
        conn: DuckDB 连接。
    """
    source = f"read_parquet('{DAILY_GLOB}')"
    conn.execute(
        f"CREATE OR REPLACE TABLE md.daily_bar AS SELECT {daily_select(source)} FROM {source}"
    )
    rows = conn.execute("SELECT count(*) FROM md.daily_bar").fetchone()[0]
    logger.info("md.daily_bar 导入 %d 行", rows)

    conn.execute(hfq_sql("md.daily_bar"))
    conn.execute(qfq_sql("md.daily_bar"))
    for table in ("md.daily_hfq", "md.daily_qfq"):
        logger.info("%s 导入 %d 行", table, conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def build_reference(conn: duckdb.DuckDBPyConnection) -> None:
    """导入股票基础表、曾用名与交易日历。

    参数:
        conn: DuckDB 连接。
    """
    basic = duckdb.read_parquet(str(TUSHARE_DIR / "stock_basic_all.parquet")).df()
    basic["stock_code"] = [normalize_code(value) for value in basic["ts_code"]]
    basic["exchange"] = [infer_exchange(code[2:]) for code in basic["stock_code"]]
    basic["board"] = [board_of(code[2:]) for code in basic["stock_code"]]
    for column in ("list_date", "delist_date"):
        basic[column] = [to_iso(value) for value in basic[column].tolist()]
    basic["_rank"] = (basic["list_status"] != "L").astype(int)
    basic = (
        basic.sort_values(["stock_code", "_rank"])
        .drop_duplicates(subset=["stock_code"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    conn.register("basic_frame", basic)
    conn.execute(
        """
        CREATE OR REPLACE TABLE ref.stock_basic AS
        SELECT
          stock_code,
          ts_code,
          name,
          area,
          industry,
          market,
          exchange,
          board,
          list_status,
          CAST(list_date AS DATE) AS list_date,
          CAST(delist_date AS DATE) AS delist_date,
          CASE WHEN list_status = 'D' THEN 'delisted' ELSE 'listed' END AS lifecycle
        FROM basic_frame
        """
    )
    conn.unregister("basic_frame")
    logger.info("ref.stock_basic 导入 %d 行", len(basic))

    change = duckdb.read_parquet(str(TUSHARE_DIR / "namechange_all.parquet")).df()
    change["stock_code"] = [normalize_code(value) for value in change["ts_code"]]
    for column in ("start_date", "end_date", "ann_date"):
        if column in change.columns:
            change[column] = [to_iso(value) for value in change[column].tolist()]
    conn.register("change_frame", change)
    conn.execute(
        """
        CREATE OR REPLACE TABLE ref.namechange AS
        SELECT stock_code, ts_code, name, change_reason,
               CAST(start_date AS DATE) AS start_date,
               CAST(end_date AS DATE) AS end_date,
               CAST(ann_date AS DATE) AS ann_date
        FROM change_frame
        """
    )
    conn.unregister("change_frame")

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE md.calendar AS
        SELECT CAST(trade_date AS DATE) AS trade_date, CAST(is_open AS INTEGER) AS is_open
        FROM read_parquet('{CLEAN_DIR / "calendar.parquet"}')
        """
    )
    logger.info("md.calendar 导入 %d 行",
                conn.execute("SELECT count(*) FROM md.calendar").fetchone()[0])


def build_derived_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """导入停牌区间与异常表。

    参数:
        conn: DuckDB 连接。
    """
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE md.suspend_segment AS
        SELECT stock_code, CAST(start_date AS DATE) AS start_date,
               CAST(end_date AS DATE) AS end_date, CAST(n_days AS INTEGER) AS n_days,
               CAST(is_long AS BOOLEAN) AS is_long
        FROM read_parquet('{CLEAN_DIR / "suspend_segments.parquet"}')
        """
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE meta.anomaly AS
        SELECT * FROM read_parquet('{CLEAN_DIR / "anomalies.parquet"}')
        """
    )
    counts = {
        "md.suspend_segment": conn.execute("SELECT count(*) FROM md.suspend_segment").fetchone()[0],
        "meta.anomaly": conn.execute("SELECT count(*) FROM meta.anomaly").fetchone()[0],
    }
    logger.info("衍生表导入 %s", counts)


def build_index_daily(conn: duckdb.DuckDBPyConnection) -> None:
    """导入指数日线行情。

    参数:
        conn: DuckDB 连接。
    """
    path = TUSHARE_DIR / "index_daily.parquet"
    if not path.exists():
        logger.warning("跳过 md.index_daily：%s 不存在", path)
        return
    frame = duckdb.read_parquet(str(path)).df()
    frame["trade_date"] = [to_iso(value) for value in frame["trade_date"].tolist()]
    conn.register("index_daily_frame", frame)
    conn.execute(
        """
        CREATE OR REPLACE TABLE md.index_daily AS
        SELECT
          ts_code AS index_code,
          index_name,
          CAST(trade_date AS DATE) AS trade_date,
          CAST(open AS DOUBLE) AS open,
          CAST(high AS DOUBLE) AS high,
          CAST(low AS DOUBLE) AS low,
          CAST(close AS DOUBLE) AS close,
          CAST(pre_close AS DOUBLE) AS pre_close,
          CAST(change AS DOUBLE) AS change,
          CAST(pct_chg AS DOUBLE) AS pct_chg,
          CAST(vol AS DOUBLE) AS vol,
          CAST(amount AS DOUBLE) AS amount
        FROM index_daily_frame
        """
    )
    conn.unregister("index_daily_frame")
    logger.info("md.index_daily 导入 %d 行",
                conn.execute("SELECT count(*) FROM md.index_daily").fetchone()[0])


def build_index_basic(conn: duckdb.DuckDBPyConnection) -> None:
    """导入指数目录。

    参数:
        conn: DuckDB 连接。
    """
    path = TUSHARE_DIR / "index_basic.parquet"
    if not path.exists():
        logger.warning("跳过 ref.index_basic：%s 不存在", path)
        return
    frame = duckdb.read_parquet(str(path)).df()
    # 中转服务不同批次返回的 index_basic 可选列不完全一致，缺失列按 NULL 入库。
    for column in (
        "fullname", "publisher", "category", "index_type", "base_point",
        "base_date", "list_date", "exp_date",
    ):
        if column not in frame.columns:
            frame[column] = None
    for column in ("base_date", "list_date", "exp_date"):
        frame[column] = [to_iso(value) for value in frame[column].tolist()]
    conn.register("index_basic_frame", frame)
    conn.execute(
        """
        CREATE OR REPLACE TABLE ref.index_basic AS
        SELECT
          ts_code AS index_code,
          CAST(name AS VARCHAR) AS index_name,
          CAST(fullname AS VARCHAR) AS fullname,
          CAST(market AS VARCHAR) AS market,
          CAST(publisher AS VARCHAR) AS publisher,
          CAST(category AS VARCHAR) AS category,
          CAST(index_type AS VARCHAR) AS index_type,
          CAST(base_point AS DOUBLE) AS base_point,
          CAST(base_date AS DATE) AS base_date,
          CAST(list_date AS DATE) AS list_date,
          CAST(exp_date AS DATE) AS exp_date
        FROM index_basic_frame
        """
    )
    conn.unregister("index_basic_frame")
    logger.info("ref.index_basic 导入 %d 行",
                conn.execute("SELECT count(*) FROM ref.index_basic").fetchone()[0])


def build_sw_index_daily(conn: duckdb.DuckDBPyConnection) -> None:
    """导入申万行业指数日线。

    申万指数来自 ``sw_daily``，字段比 ``index_daily`` 多出估值与市值列，单独建表。

    参数:
        conn: DuckDB 连接。
    """
    path = TUSHARE_DIR / "sw_index_daily.parquet"
    if not path.exists():
        logger.warning("跳过 md.sw_index_daily：%s 不存在", path)
        return
    frame = duckdb.read_parquet(str(path)).df()
    frame["trade_date"] = [to_iso(value) for value in frame["trade_date"].tolist()]
    conn.register("sw_index_frame", frame)
    conn.execute(
        """
        CREATE OR REPLACE TABLE md.sw_index_daily AS
        SELECT
          ts_code AS index_code,
          CAST(index_name AS VARCHAR) AS index_name,
          CAST(trade_date AS DATE) AS trade_date,
          CAST(open AS DOUBLE) AS open,
          CAST(low AS DOUBLE) AS low,
          CAST(high AS DOUBLE) AS high,
          CAST(close AS DOUBLE) AS close,
          CAST(change AS DOUBLE) AS change,
          CAST(pct_change AS DOUBLE) AS pct_change,
          CAST(vol AS DOUBLE) AS vol,
          CAST(amount AS DOUBLE) AS amount,
          CAST(pe AS DOUBLE) AS pe,
          CAST(pb AS DOUBLE) AS pb,
          CAST(float_mv AS DOUBLE) AS float_mv,
          CAST(total_mv AS DOUBLE) AS total_mv
        FROM sw_index_frame
        """
    )
    conn.unregister("sw_index_frame")
    logger.info("md.sw_index_daily 导入 %d 行",
                conn.execute("SELECT count(*) FROM md.sw_index_daily").fetchone()[0])


def build_index_weight(conn: duckdb.DuckDBPyConnection) -> None:
    """导入指数历史成分权重，并把成分代码归一成项目口径。

    原始 con_code 是 Tushare 写法（600050.SH），与日线表的 SH600050 对不上，
    直接 join 会命中 0 行，因此新增 stock_code 列，原列保留以便溯源；
    trade_date 同时统一成 DATE。

    参数:
        conn: DuckDB 连接。
    """
    path = TUSHARE_DIR / "index_weight.parquet"
    if not path.exists():
        logger.warning("跳过 ref.index_weight：%s 不存在", path)
        return
    frame = duckdb.read_parquet(str(path)).df()
    frame["stock_code"] = normalize_codes(
        frame["con_code"].tolist(), "指数权重 index_weight", logger
    )
    frame["trade_date"] = [to_iso(value) for value in frame["trade_date"].tolist()]
    conn.register("index_weight_frame", frame)
    conn.execute(
        "CREATE OR REPLACE TABLE ref.index_weight AS "
        "SELECT * EXCLUDE (stock_code, trade_date), "
        "CAST(trade_date AS DATE) AS trade_date, stock_code "
        "FROM index_weight_frame"
    )
    conn.unregister("index_weight_frame")
    rows = conn.execute("SELECT count(*) FROM ref.index_weight").fetchone()[0]
    matched = conn.execute(
        "SELECT count(*) FROM ref.index_weight w "
        "WHERE EXISTS (SELECT 1 FROM md.daily_bar b WHERE b.stock_code = w.stock_code)"
    ).fetchone()[0]
    logger.info("ref.index_weight 导入 %d 行，归一后能对上日线表 %d 行", rows, matched)


def _validate_sw_industry_intervals(frame) -> None:
    """校验同一股票的申万一级行业区间不重叠。

    参数:
        frame: 已补齐一级行业和日期列的历史成员表。
    """
    invalid = frame.loc[
        frame["out_date"].notna() & (frame["out_date"] < frame["in_date"])
    ]
    if not invalid.empty:
        row = invalid.iloc[0]
        raise DbWriteError(
            f"申万行业日期倒置：{row['stock_code']} {row['l3_code']} "
            f"{row['in_date']}~{row['out_date']}"
        )
    far_future = pd.Timestamp.max.normalize()
    for stock_code, group in frame.groupby("stock_code", sort=False):
        rows = group.sort_values(["in_date", "out_date"], na_position="last").reset_index(drop=True)
        for left in range(len(rows)):
            left_end = rows.at[left, "out_date"]
            left_end = far_future if pd.isna(left_end) else left_end
            for right in range(left + 1, len(rows)):
                if rows.at[right, "in_date"] > left_end:
                    break
                if rows.at[left, "l1_code"] != rows.at[right, "l1_code"]:
                    raise DbWriteError(
                        f"申万行业区间冲突：{stock_code} 在 {rows.at[right, 'in_date'].date()} "
                        f"同时命中 {rows.at[left, 'l1_code']} 与 {rows.at[right, 'l1_code']}"
                    )


def _canonicalize_sw_industry_intervals(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, int]:
    """把来源重叠区间解析成每只股票唯一的时点行业。

    同日存在多条记录时，先取 ``in_date`` 最新者；同日起优先有明确 ``out_date`` 的
    历史记录，再取较早结束者。该规则避免把当前开放区间从上市日起回填到历史。
    有 ``classification`` 列时按股票和分类版本分别规范化。

    参数:
        frame: 带 L1/L2/L3 和来源日期的历史成员表。
    """
    resolved = 0
    one_day = pd.Timedelta(days=1)
    has_classification = "classification" in frame.columns
    group_columns = ["stock_code", "classification"] if has_classification else ["stock_code"]
    segments: list[dict] = []
    for _, group in frame.groupby(group_columns, sort=False):
        boundaries = set(group["in_date"].tolist())
        boundaries.update((group["out_date"].dropna() + one_day).tolist())
        ordered = sorted(boundaries)
        for position, start in enumerate(ordered):
            active = group.loc[
                (group["in_date"] <= start)
                & (group["out_date"].isna() | (group["out_date"] >= start))
            ]
            if active.empty:
                continue
            if len(active) > 1:
                resolved += 1
            chosen = active.loc[active["in_date"] == active["in_date"].max()]
            finite = chosen.loc[chosen["out_date"].notna()]
            if not finite.empty:
                chosen = finite.loc[finite["out_date"] == finite["out_date"].min()]
            if chosen["l1_code"].nunique() > 1:
                codes = sorted(chosen["l1_code"].astype(str).unique())
                raise DbWriteError(
                    f"申万行业来源冲突无法解析：{group.iloc[0]['stock_code']} "
                    f"在 {start.date()} 命中 {codes}"
                )
            row = chosen.sort_values("l3_code").iloc[0].to_dict()
            source_end = row["out_date"]
            end = ordered[position + 1] - one_day if position + 1 < len(ordered) else source_end
            if pd.notna(source_end) and (pd.isna(end) or end > source_end):
                end = source_end
            row["in_date"] = start
            row["out_date"] = end
            segments.append(row)
    canonical = pd.DataFrame(segments)
    if canonical.empty:
        return canonical, resolved
    merged: list[dict] = []
    filled_segments = 0
    filled_days = 0
    identity = tuple(
        name for name in ("stock_code", "classification", "l1_code", "l2_code", "l3_code")
        if name in canonical.columns
    )
    for _, group in canonical.groupby(group_columns, sort=False):
        group_merged: list[dict] = []
        for row in group.sort_values(["in_date", "l3_code"]).to_dict("records"):
            previous = group_merged[-1] if group_merged else None
            same_industry = (
                previous is not None
                and all(previous[name] == row[name] for name in identity)
                and pd.notna(previous["out_date"])
            )
            if same_industry:
                gap_days = max(int((row["in_date"] - previous["out_date"]).days) - 1, 0)
                if gap_days:
                    filled_segments += 1
                    filled_days += gap_days
                    previous["source_api"] = "same_industry_gap_fill"
                previous["out_date"] = row["out_date"]
            else:
                group_merged.append(row)
        merged.extend(group_merged)
    return pd.DataFrame(merged), resolved, filled_segments, filled_days


def _normalized_industry_name(value) -> str:
    """规范化行业名称，用于跨版本代码别名匹配。"""
    text = "" if pd.isna(value) else str(value)
    text = text.replace("申万", "")
    return re.sub(r"[ⅠⅡⅢⅣⅤIVX\s\-_（）()·]+", "", text)


def _load_sw_catalogs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """加载 SW2014/SW2021 目录并生成三级行业层级。"""
    frames = []
    for classification, directory in SW_INDUSTRY_DIRS.items():
        path = directory / "classify.parquet"
        if not path.exists():
            raise DbWriteError(f"缺少 {classification} 申万行业目录：{path}")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise DbWriteError(f"{classification} 申万行业目录为空：{path}")
        required = {
            "index_code", "industry_name", "parent_code", "level", "industry_code", "src"
        }
        missing = required - set(frame.columns)
        if missing:
            raise DbWriteError(f"{classification} 申万行业目录缺少字段 {sorted(missing)}")
        frame = frame.copy()
        frame["classification"] = classification
        if frame["src"].astype(str).nunique() != 1 or frame["src"].astype(str).iloc[0] != classification:
            raise DbWriteError(f"{classification} 目录的 src 与分类版本不一致")
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["classification", "index_code"]
    )

    def part(level: str) -> pd.DataFrame:
        return catalog.loc[catalog["level"].astype(str) == level].copy()

    l1 = part("L1")[
        ["classification", "index_code", "industry_name", "industry_code"]
    ].rename(columns={
        "index_code": "l1_code", "industry_name": "l1_name",
        "industry_code": "l1_industry_code",
    })
    l2 = part("L2")[
        ["classification", "index_code", "industry_name", "industry_code", "parent_code"]
    ].rename(columns={
        "index_code": "l2_code", "industry_name": "l2_name",
        "industry_code": "l2_industry_code", "parent_code": "l1_industry_code",
    })
    l3 = part("L3")[
        ["classification", "index_code", "industry_name", "parent_code", "is_pub"]
    ].rename(columns={
        "index_code": "l3_code", "industry_name": "l3_name",
        "parent_code": "l2_industry_code",
    })
    hierarchy = l3.merge(
        l2, on=["classification", "l2_industry_code"], how="left", validate="many_to_one"
    ).merge(
        l1, on=["classification", "l1_industry_code"], how="left", validate="many_to_one"
    )
    if hierarchy[["l1_code", "l2_code"]].isna().any().any():
        raise DbWriteError("申万分类目录存在无法关联到一级行业的三级行业")

    names: dict[str, str] = {}
    basic_path = TUSHARE_DIR / "index_basic.parquet"
    if basic_path.exists():
        basic = pd.read_parquet(basic_path)
        if {"ts_code", "name"} <= set(basic.columns):
            names = dict(zip(basic["ts_code"].astype(str), basic["name"]))
    return catalog, hierarchy, names


def _resolve_sw_l3_code(
    code: str,
    classification: str,
    hierarchy: pd.DataFrame,
    names: dict[str, str],
) -> str | None:
    """把来源代码解析到指定分类版本的三级行业代码。"""
    exact = hierarchy.loc[
        (hierarchy["classification"] == classification)
        & (hierarchy["l3_code"].astype(str) == str(code))
    ]
    if not exact.empty:
        return str(exact.iloc[0]["l3_code"])
    normalized = _normalized_industry_name(names.get(str(code), ""))
    if not normalized:
        return None
    candidates = hierarchy.loc[
        (hierarchy["classification"] == classification)
        & (hierarchy["l3_name"].map(_normalized_industry_name) == normalized)
    ]
    codes = sorted(candidates["l3_code"].astype(str).unique())
    return codes[0] if len(codes) == 1 else None


def _infer_sw_classification(
    row: pd.Series,
    hierarchy: pd.DataFrame,
    names: dict[str, str],
) -> str | None:
    """为逐股修复记录判断所属分类版本。"""
    code = str(row["index_code"])
    exact = {
        str(item) for item in hierarchy.loc[
            hierarchy["l3_code"].astype(str) == code, "classification"
        ].unique()
    }
    in_date = row["in_date"]
    post_switch = pd.notna(in_date) and in_date >= SW_CLASSIFICATION_SWITCH
    if post_switch:
        alias = _resolve_sw_l3_code(code, "SW2021", hierarchy, names)
        if alias is not None:
            return "SW2021"
    if len(exact) == 1:
        return next(iter(exact))
    if not post_switch:
        for classification in ("SW2014", "SW2021"):
            if _resolve_sw_l3_code(code, classification, hierarchy, names) is not None:
                return classification
    return None


def _load_sw_members(
    hierarchy: pd.DataFrame,
    names: dict[str, str],
) -> pd.DataFrame:
    """读取两套历史分片、当前快照与逐股修复记录。"""
    frames: list[pd.DataFrame] = []
    for classification, directory in SW_INDUSTRY_DIRS.items():
        history_dir = directory / "history_members"
        for path in sorted(history_dir.glob("l3_code=*.parquet")):
            frame = pd.read_parquet(path)
            frame["classification"] = classification
            frame["source_api"] = "index_member"
            frame["source_l3_code"] = frame["index_code"].astype(str)
            frames.append(frame)

    current_paths = sorted((SW_INDUSTRY_DIRS["SW2021"] / "members").glob("l3_code=*.parquet"))
    for path in current_paths:
        frame = pd.read_parquet(path)
        frame = frame.rename(columns={"l3_code": "index_code", "ts_code": "con_code"})
        frame["classification"] = "SW2021"
        frame["source_api"] = "index_member_all"
        frame["source_l3_code"] = frame["index_code"].astype(str)
        frames.append(frame)

    for path in sorted(SW_INDUSTRY_REPAIR_DIR.glob("ts_code=*.parquet")):
        frame = pd.read_parquet(path)
        frame["source_api"] = "index_member_ts_code"
        frame["source_l3_code"] = frame["index_code"].astype(str)
        frame["in_date"] = pd.to_datetime(frame["in_date"], format="%Y%m%d", errors="coerce")
        frame["out_date"] = pd.to_datetime(frame["out_date"], format="%Y%m%d", errors="coerce")
        frame["classification"] = None
        for position, row in frame.iterrows():
            frame.at[position, "classification"] = _infer_sw_classification(row, hierarchy, names)
        frames.append(frame)

    if not frames:
        raise DbWriteError("没有申万历史成员分片")
    members = pd.concat(frames, ignore_index=True)
    required = {
        "index_code", "con_code", "in_date", "out_date", "is_new",
        "classification", "source_api", "source_l3_code",
    }
    missing = required - set(members.columns)
    if missing:
        raise DbWriteError(f"申万历史成员缺少字段 {sorted(missing)}")
    members["in_date"] = pd.to_datetime(members["in_date"], format="%Y%m%d", errors="coerce")
    members["out_date"] = pd.to_datetime(members["out_date"], format="%Y%m%d", errors="coerce")
    members = members.loc[
        members["classification"].notna()
        & members["con_code"].astype(str).str.match(_TS_CODE_RE)
    ].copy()
    members["ts_code"] = members["con_code"].astype(str)
    members["stock_code"] = normalize_codes(
        members["con_code"].tolist(), "申万历史行业 index_member", logger
    )
    members = members.dropna(subset=["stock_code", "in_date"]).copy()

    resolved_codes = []
    for row in members.itertuples(index=False):
        resolved_codes.append(
            _resolve_sw_l3_code(row.index_code, row.classification, hierarchy, names)
        )
    members["l3_code"] = resolved_codes
    unresolved = members.loc[members["l3_code"].isna()]
    if not unresolved.empty:
        sample = ", ".join(unresolved["index_code"].astype(str).unique()[:10])
        raise DbWriteError(f"申万历史成员存在无法映射的三级行业代码：{sample}")
    members = members.drop(
        columns=[
            name for name in ("l1_code", "l1_name", "l2_code", "l2_name", "l3_name", "is_pub")
            if name in members.columns
        ]
    )
    members = members.merge(
        hierarchy, on=["classification", "l3_code"], how="left", validate="many_to_one"
    )
    if members["l1_code"].isna().any():
        raise DbWriteError("申万历史成员存在目录外三级行业")
    members["source_in_date"] = members["in_date"]
    members["source_out_date"] = members["out_date"]
    members["_source_rank"] = members["source_api"].map(
        {"index_member": 0, "index_member_ts_code": 1, "index_member_all": 2}
    ).fillna(3).astype(int)
    members = (
        members.sort_values(
            ["_source_rank", "classification", "l3_code", "stock_code", "in_date"]
        )
        .drop_duplicates(
            subset=["classification", "l3_code", "stock_code", "in_date", "out_date"],
            keep="first",
        )
        .drop(columns="_source_rank")
    )
    return members


def _apply_sw_version_windows(
    members: pd.DataFrame,
    list_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """应用分类版本窗口，并在缺失时沿用最近已知行业。

    切换日前优先 SW2014，切换日后优先 SW2021；某日没有任何精确记录时，
    沿用最近一条已知行业并标记 carry_forward，避免把真实数据空档误当成无行业。
    """
    one_day = pd.Timedelta(days=1)
    switch = SW_CLASSIFICATION_SWITCH
    rows: list[dict] = []
    for stock_code, group in members.groupby("stock_code", sort=False):
        records = group.sort_values(["in_date", "classification"]).to_dict("records")
        first_start = min(record["in_date"] for record in records)
        stock_start = None if list_dates is None else list_dates.get(stock_code)
        if stock_start is None or pd.isna(stock_start) or stock_start > first_start:
            stock_start = first_start
        boundaries = {switch}
        if stock_start < first_start:
            boundaries.add(stock_start)
        for record in records:
            boundaries.add(record["in_date"])
            if pd.notna(record["out_date"]):
                boundaries.add(record["out_date"] + one_day)
        ordered = sorted(boundaries)
        for position, start in enumerate(ordered):
            if start > pd.Timestamp.max.normalize() - one_day:
                continue
            end = ordered[position + 1] - one_day if position + 1 < len(ordered) else None
            active = [
                record for record in records
                if record["in_date"] <= start
                and (pd.isna(record["out_date"]) or record["out_date"] >= start)
            ]
            chosen = None
            carry = False
            source_api = None
            if start < switch:
                old = [record for record in active if record["classification"] == "SW2014"]
                new = [record for record in active if record["classification"] == "SW2021"]
                if old:
                    chosen = sorted(old, key=lambda record: record["in_date"])[-1]
                elif new:
                    chosen = sorted(new, key=lambda record: record["in_date"])[-1]
                    source_api = "early_sw2021"
            else:
                new = [record for record in active if record["classification"] == "SW2021"]
                old = [record for record in active if record["classification"] == "SW2014"]
                if new:
                    chosen = sorted(new, key=lambda record: record["in_date"])[-1]
                elif old:
                    chosen = sorted(old, key=lambda record: record["in_date"])[-1]
                    carry = True
                    source_api = "carry_forward_sw2014"
            if chosen is None:
                previous = [record for record in records if record["in_date"] < start]
                if previous:
                    chosen = sorted(
                        previous, key=lambda record: (record["in_date"], record["classification"])
                    )[-1]
                    carry = True
                    source_api = (
                        "carry_forward_sw2014"
                        if chosen["classification"] == "SW2014"
                        else "carry_forward_sw2021"
                    )
                elif stock_start <= start < first_start:
                    chosen = records[0]
                    carry = True
                    source_api = (
                        "backfill_earliest_sw2014"
                        if chosen["classification"] == "SW2014"
                        else "backfill_earliest_sw2021"
                    )
            if chosen is None:
                continue
            item = dict(chosen)
            item["in_date"] = start
            item["out_date"] = end
            item["carry_forward"] = carry
            if source_api:
                item["source_api"] = source_api
            rows.append(item)

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline
    identity = (
        "stock_code", "classification", "l1_code", "l2_code", "l3_code",
        "carry_forward", "source_api",
    )
    merged: list[dict] = []
    for _, group in timeline.groupby(["stock_code", "classification", "carry_forward"], sort=False):
        current: list[dict] = []
        for row in group.sort_values(["in_date", "l3_code"]).to_dict("records"):
            previous = current[-1] if current else None
            if (
                previous is not None
                and all(previous[name] == row[name] for name in identity)
                and pd.notna(previous["out_date"])
                and row["in_date"] == previous["out_date"] + one_day
            ):
                previous["out_date"] = row["out_date"]
            else:
                current.append(row)
        merged.extend(current)
    return pd.DataFrame(merged)


def _empty_sw_source_rows() -> int:
    """统计清单中空响应的申万历史成员分片数。"""
    manifest_path = TUSHARE_DIR / "manifests" / "core_resources.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sum(
        item.get("status") == "empty"
        for item in payload.get("items", {}).values()
        if item.get("api") == "index_member"
    )


def _stock_list_dates() -> dict[str, pd.Timestamp]:
    """读取股票上市日，用于把最早行业记录向前补到上市日。"""
    path = TUSHARE_DIR / "stock_basic_all.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if not {"ts_code", "list_date"} <= set(frame.columns):
        return {}
    frame = frame.loc[frame["ts_code"].astype(str).str.match(_TS_CODE_RE)].copy()
    frame["stock_code"] = normalize_codes(
        frame["ts_code"].tolist(), "申万行业上市日", logger
    )
    frame = frame.dropna(subset=["stock_code"])
    values = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce")
    return dict(zip(frame["stock_code"], values))


def build_sw_industry(conn: duckdb.DuckDBPyConnection) -> None:
    """导入时点申万分类目录、历史成员和逐股修复记录。

    参数:
        conn: DuckDB 连接。
    """
    catalog, hierarchy, names = _load_sw_catalogs()
    conn.register("sw_classify_frame", catalog)
    conn.execute(
        """
        CREATE OR REPLACE TABLE ref.sw_industry_classify AS
        SELECT CAST(index_code AS VARCHAR) AS index_code,
               CAST(industry_name AS VARCHAR) AS industry_name,
               CAST(parent_code AS VARCHAR) AS parent_code,
               CAST(level AS VARCHAR) AS level,
               CAST(industry_code AS VARCHAR) AS industry_code,
               CAST(is_pub AS BOOLEAN) AS is_pub,
               CAST(src AS VARCHAR) AS src
        FROM sw_classify_frame
        """
    )
    conn.unregister("sw_classify_frame")

    members = _load_sw_members(hierarchy, names)
    source_rows = len(members)
    history_rows = int((members["source_api"] == "index_member").sum())
    current_rows = int((members["source_api"] == "index_member_all").sum())
    repair_rows = int((members["source_api"] == "index_member_ts_code").sum())
    members, overlap_segments, gap_segments, gap_days = _canonicalize_sw_industry_intervals(members)
    members = _apply_sw_version_windows(members, _stock_list_dates())
    if members.empty:
        raise DbWriteError("申万行业规范化后没有有效区间")
    columns = [
        "stock_code", "ts_code", "classification", "l1_code", "l1_name",
        "l2_code", "l2_name", "l3_code", "l3_name", "in_date", "out_date",
        "source_in_date", "source_out_date", "source_l3_code",
        "source_api", "carry_forward", "is_new",
    ]
    members = members[columns].drop_duplicates().sort_values(
        ["stock_code", "in_date", "classification", "l3_code"]
    )
    _validate_sw_industry_intervals(members)
    conn.register("sw_member_frame", members)
    conn.execute(
        """
        CREATE OR REPLACE TABLE ref.sw_industry_member AS
        SELECT stock_code, ts_code, CAST(classification AS VARCHAR) AS classification,
               l1_code, l1_name, l2_code, l2_name, l3_code, l3_name,
               CAST(in_date AS DATE) AS in_date, CAST(out_date AS DATE) AS out_date,
               CAST(source_in_date AS DATE) AS source_in_date,
               CAST(source_out_date AS DATE) AS source_out_date,
               CAST(source_l3_code AS VARCHAR) AS source_l3_code,
               CAST(source_api AS VARCHAR) AS source_api,
               CAST(carry_forward AS BOOLEAN) AS carry_forward,
               CAST(is_new AS VARCHAR) AS is_new
        FROM sw_member_frame
        """
    )
    conn.unregister("sw_member_frame")
    digest_columns = members.fillna("").astype(str)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(digest_columns, index=False).values.tobytes()
    ).hexdigest()
    manifest_path = TUSHARE_DIR / "manifests" / "core_resources.json"
    source_updated_at = None
    if manifest_path.exists():
        source_updated_at = json.loads(manifest_path.read_text(encoding="utf-8")).get("updated_at")
    carry_forward = members["carry_forward"].fillna(False).astype(bool)
    backfill = members["source_api"].astype(str).str.startswith("backfill_earliest_")
    conn.execute(
        """
        CREATE OR REPLACE TABLE meta.sw_industry_manifest AS
        SELECT 'SW2014+SW2021' AS source,
               'index_member+index_member_ts_code+index_member_all' AS api,
               ? AS digest, ?::INTEGER AS rows, ?::INTEGER AS source_rows,
               ?::INTEGER AS stocks, ?::INTEGER AS history_rows,
               ?::INTEGER AS current_snapshot_rows,
               ?::INTEGER AS overlap_segments_resolved,
               ?::INTEGER AS gap_segments_filled, ?::INTEGER AS gap_days_filled,
               ?::DATE AS min_date, ?::DATE AS max_date,
               ? AS source_updated_at, current_timestamp AS built_at,
               'SW2014,SW2021' AS classifications,
               ?::DATE AS switch_date,
               ?::INTEGER AS repair_rows,
               ?::INTEGER AS carry_forward_rows,
               ?::INTEGER AS carry_forward_stocks,
               ?::INTEGER AS empty_source_rows,
               ?::INTEGER AS backfill_rows,
               ?::INTEGER AS backfill_stocks
        """,
        [
            digest, len(members), source_rows, int(members["stock_code"].nunique()),
            history_rows, current_rows, overlap_segments, gap_segments, gap_days,
            members["in_date"].min().date(),
            members["out_date"].max().date() if members["out_date"].notna().any() else None,
            source_updated_at,
            SW_CLASSIFICATION_SWITCH.date(),
            repair_rows,
            int(carry_forward.sum()),
            int(members.loc[carry_forward, "stock_code"].nunique()),
            _empty_sw_source_rows(),
            int(backfill.sum()),
            int(members.loc[backfill, "stock_code"].nunique()),
        ],
    )
    logger.info(
        "ref.sw_industry_member 导入 %d 个规范区间、%d 只股票，解析重叠 %d、"
        "填平同业空档 %d 段/%d 天、carry-forward %d 段、摘要 %s",
        len(members), members["stock_code"].nunique(), overlap_segments,
        gap_segments, gap_days, int(carry_forward.sum()), digest[:12],
    )


def build_suspend_daily(conn: duckdb.DuckDBPyConnection) -> None:
    """导入官方停牌记录明细。

    trade_date 统一成 DATE：原始 Tushare 写法是 ``20240131`` 字符串，
    按 ``trade_date >= '2024-01-02'`` 过滤时字符串比较永远不成立，会静默返回空集。

    参数:
        conn: DuckDB 连接。
    """
    path = TUSHARE_DIR / "suspend_d.parquet"
    if not path.exists():
        logger.warning("跳过 evt.suspend_daily：%s 不存在", path)
        return
    frame = duckdb.read_parquet(str(path)).df()
    frame["trade_date"] = [to_iso(value) for value in frame["trade_date"].tolist()]
    conn.register("suspend_daily_frame", frame)
    conn.execute(
        "CREATE OR REPLACE TABLE evt.suspend_daily AS "
        "SELECT * EXCLUDE (trade_date), CAST(trade_date AS DATE) AS trade_date "
        "FROM suspend_daily_frame"
    )
    conn.unregister("suspend_daily_frame")
    logger.info("evt.suspend_daily 导入 %d 行",
                conn.execute("SELECT count(*) FROM evt.suspend_daily").fetchone()[0])


def build_extra(conn: duckdb.DuckDBPyConnection) -> None:
    """导入指数目录、指数日线、申万行业日线、指数权重与官方停牌等可选表。

    参数:
        conn: DuckDB 连接。
    """
    build_suspend_daily(conn)
    build_index_basic(conn)
    build_index_daily(conn)
    build_sw_index_daily(conn)
    build_index_weight(conn)
    build_sw_industry(conn)
    corporate_actions.build(conn)


def write_manifest_table(conn: duckdb.DuckDBPyConnection) -> None:
    """把清洗 manifest 与数据字典写进 meta schema。

    参数:
        conn: DuckDB 连接。
    """
    path = CLEAN_DIR / "manifest.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        conn.execute(
            "CREATE OR REPLACE TABLE meta.manifest AS SELECT ? AS data_version, ? AS generated_at, ? AS payload",
            [payload.get("data_version"), payload.get("generated_at"), json.dumps(payload, ensure_ascii=False)],
        )
    dictionary = [
        ("daily_bar", "stock_code", "字符串", "统一代码，SH/SZ/BJ + 6 位数字"),
        ("daily_bar", "trade_date", "日期", "交易日"),
        ("daily_bar", "open/high/low/close", "元", "未复权四价"),
        ("daily_bar", "volume_share", "股", "成交量，Tushare vol × 100"),
        ("daily_bar", "amount_cny", "元", "成交额，Tushare amount × 1000"),
        ("daily_bar", "vwap", "元", "amount_cny / volume_share，未复权口径"),
        ("daily_bar", "factor", "无量纲", "后复权因子，后复权价 = 原始价 × factor"),
        ("daily_bar", "change", "元", "官方涨跌额"),
        ("daily_bar", "pre_close", "元", "前收盘价 = close - change"),
        ("daily_bar", "ret", "小数", "官方日收益 = change / pre_close"),
        ("daily_bar", "status", "枚举", "trading / suspended（上市期外的行不入库）"),
        ("daily_bar", "is_st", "布尔", "当日是否 ST，来源 namechange"),
        ("daily_bar", "is_delisting", "布尔", "当日是否处于退市整理期"),
        ("daily_bar", "limit_up/limit_down", "元", "涨跌停价，优先官方 stk_limit，缺失时规则推算"),
        ("daily_bar", "can_buy_*/can_sell_*", "布尔", "保守口径的可交易性"),
        ("daily_bar", "total_mv/circ_mv", "万元", "总市值 / 流通市值"),
        ("daily_bar", "turnover_rate/dv_ttm", "%", "换手率 / 股息率"),
        ("daily_hfq", "open/high/low/close/vwap", "元", "后复权口径，其余列与 daily_bar 一致"),
        ("daily_qfq", "open/high/low/close/vwap", "元", "前复权口径，基准为各股最后一个因子"),
        ("ref.stock_basic", "-", "-", "股票基础表，含退市股的上市日与退市日"),
        ("ref.namechange", "-", "-", "证券简称变更历史，用于推 ST 与退市整理期区间"),
        ("ref.index_weight", "-", "-", "指数历史成分权重；con_code 为 Tushare 原写法，stock_code 为归一后的项目口径"),
        ("ref.sw_industry_classify", "src", "字符串", "SW2014 / SW2021 一级、二级、三级行业目录"),
        ("ref.sw_industry_member", "in_date/out_date", "日期", "规范化后的申万行业生效闭区间；out_date 为空表示持续有效"),
        ("ref.sw_industry_member", "classification", "字符串", "该区间采用的申万分类版本：SW2014 或 SW2021"),
        ("ref.sw_industry_member", "source_in_date/source_out_date", "日期", "Tushare 原始成员区间，用于追溯重叠解析"),
        ("ref.sw_industry_member", "source_l3_code", "字符串", "映射前的原始三级行业代码，便于追溯代码别名"),
        ("ref.sw_industry_member", "source_api", "字符串", "来源：历史 index_member、逐股修复 index_member_ts_code、快照 index_member_all 或 carry_forward_sw2014"),
        ("ref.sw_industry_member", "carry_forward", "布尔", "SW2021 缺失时是否沿用最近有效的 SW2014 行业"),
        ("ref.corporate_action", "stock_code", "字符串", "统一股票代码"),
        ("ref.corporate_action", "ex_date", "日期", "除权除息日"),
        ("ref.corporate_action", "share_multiplier", "倍", "除权日开盘前生效的每股新股数"),
        ("ref.corporate_action", "cash_div_per_share", "元/股", "税前每股现金分红"),
        ("ref.corporate_action", "rights_ratio", "倍", "每股可配新股数"),
        ("ref.corporate_action", "rights_price", "元/股", "配股认购价；无配股为空"),
        ("ref.corporate_action", "record_date", "日期", "分红登记日"),
        ("ref.corporate_action", "pay_date", "日期", "现金分红到账日"),
        ("ref.corporate_action", "action_type", "枚举",
         "cash/stock/mixed/rights/rights_mixed/consolidation"),
        ("ref.corporate_action", "source", "字符串", "gbbq、dividend 或 share_count"),
        ("ref.corporate_action", "source_event_key", "SHA-256", "来源事件稳定键"),
        ("meta.sw_industry_manifest", "digest", "SHA-256", "申万历史成员内容摘要，用于回测复现"),
        ("meta.sw_industry_manifest", "switch_date", "日期", "SW2021 分类生效日，当前为 2021-12-13"),
        ("meta.sw_industry_manifest", "backfill_rows/backfill_stocks", "整数", "最早行业记录向前补到上市日的区间数与股票数"),
        ("fin.income_statement", "report_date/ann_date/available_date", "日期", "利润表报告期、公告日与可用日；available_date 次一交易日起可用于信号"),
        ("fin.income_statement", "total_revenue/revenue/oper_cost/operate_profit/n_income", "元", "营业收入、营业成本、营业利润与净利润"),
        ("fin.balance_sheet", "total_assets/total_liab/total_hldr_eqy_exc_min_int", "元", "总资产、总负债与归母股东权益"),
        ("fin.cash_flow", "n_cashflow_act/c_pay_acq_const_fiolta", "元", "经营活动现金流净额与购建固定资产等支付的现金"),
        ("fin.fina_indicator", "roe_yearly/roa_yearly/roic", "%", "年化 ROE、ROA 与 ROIC；入库保留 Tushare 百分数口径"),
        ("fin.fina_indicator", "or_yoy/netprofit_yoy/op_yoy/ocf_yoy", "%", "营收、净利润、营业利润与经营现金流的同比增速"),
        ("meta.financial_manifest", "pit_rule", "字符串", "财务时点口径，当前为 trade_date > available_date"),
        ("meta.financial_manifest", "source_digest", "SHA-256", "财务原始分片内容摘要，用于复现"),
        ("index_daily", "index_code", "字符串", "指数代码，保留 Tushare 写法 000300.SH"),
        ("index_daily", "open/high/low/close", "点", "指数行情点位"),
        ("index_daily", "pct_chg", "%", "指数涨跌幅"),
        ("index_daily", "vol/amount", "手/千元", "指数成交量与成交额，与 Tushare 原始口径一致"),
        ("sw_index_daily", "index_code", "字符串", "申万行业指数代码，如 801010.SI"),
        ("sw_index_daily", "open/high/low/close", "点", "申万行业指数点位"),
        ("sw_index_daily", "pct_change", "%", "申万行业指数涨跌幅"),
        ("sw_index_daily", "pe/pb", "倍", "行业整体估值，Tushare sw_daily 口径"),
        ("sw_index_daily", "float_mv/total_mv", "万元", "行业流通市值与总市值"),
        ("index_basic", "index_code", "字符串", "指数代码，保留 Tushare 写法"),
        ("index_basic", "market/category", "字符串", "指数来源（SSE/SZSE/CSI/SW）与类别"),
        ("index_basic", "list_date", "日期", "指数发布日"),
        ("fwd_return", "fwd_ret_close_N", "小数", "未来 N 日收益，N 取 1/3/5/10/15/20；t 日收盘买入、t+N 日收盘卖出；仅 t 日为交易状态时才有值"),
        ("fwd_return", "fwd_ret_open_N", "小数", "未来 N 日收益；t+1 日开盘买入、t+1+N 日开盘卖出，持有交易日数与收盘口径一致"),
        ("fwd_return", "-", "-", "停牌段用上一有效交易日价格填平，复牌跳空计入终点价；停牌行与尾部不足 N 的行留空"),
        ("meta.anomaly", "-", "-", "清洗异常明细"),
        ("meta.manifest", "-", "-", "清洗输入、校验结果与已知限制"),
    ]
    values = ", ".join(["(?, ?, ?, ?)"] * len(dictionary))
    frame = conn.execute(
        f"SELECT * FROM (VALUES {values})", [item for row in dictionary for item in row]
    ).df()
    frame.columns = ["table_name", "column_name", "unit", "description"]
    conn.register("dict_frame", frame)
    conn.execute("CREATE OR REPLACE TABLE meta.data_dictionary AS SELECT * FROM dict_frame")
    conn.unregister("dict_frame")


def run(skip_financial: bool = False) -> dict:
    """建库总入口。

    参数:
        skip_financial: 跳过财务表导入，仅用于磁盘不足时的临时排查。
    """
    ensure_dirs()
    started = time.time()
    if not skip_financial:
        space = financial.preflight(DB_FILE, DB_DIR)
        logger.info("建库空间预检通过：%s", space)
    if DB_FILE.exists():
        DB_FILE.unlink()
    conn = duckdb.connect(str(DB_FILE))
    try:
        for statement in SCHEMA_DDL:
            conn.execute(statement)
        build_reference(conn)
        build_daily(conn)
        if not skip_financial:
            build_stats = financial.build_financial(conn)
            logger.info("财务表摘要：%s", build_stats["source_digest"][:12])
        labels.build_labels(conn)
        label_check = labels.verify_labels(conn)
        if not label_check["passed"]:
            raise DbWriteError("标签校验未通过：" + "；".join(label_check["failures"]))
        build_derived_tables(conn)
        build_extra(conn)
        write_manifest_table(conn)
        conn.execute("CHECKPOINT")
    except Exception as exc:
        conn.close()
        raise DbWriteError(f"建库失败: {exc}") from exc
    tables = conn.execute(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables ORDER BY 1"
    ).fetchall()
    conn.close()
    payload = {
        "database": str(DB_FILE),
        "size_mb": round(DB_FILE.stat().st_size / 1e6, 1),
        "tables": [row[0] for row in tables],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    logger.info("建库完成 %s", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="把清洗产物与 Tushare 数据导入 DuckDB")
    parser.add_argument(
        "--skip-financial",
        action="store_true",
        help="跳过财务表导入；正式建库不要使用",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(run(skip_financial=args.skip_financial), ensure_ascii=False, indent=2, default=str),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
