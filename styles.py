"""风格因子：Barra 近似口径的 7 个截面风格值，以及行业数据源协议。

口径：

- 窗口按**有效交易日**推进（``status = 'trading'``，停牌不占窗口），与数据项目的
  因子构建脚本一致，这样停牌大半年的股票也算得出自己的窗口；
- 风格值只算到评估区间结束为止，为了拿到 ``t-250`` 的收盘价，取数从区间起始日
  往前回溯 260 个市场交易日，并为长期停牌股补足此前 250 次有效交易；
- 这里产出的是**原始值**（不做截面标准化）：标准化发生在 ``exposure`` 模块里，
  用的是同一个回归样本，避免「风格按全市场标准化、回归用池内样本」的口径错位；
- 风格值不塞进评估面板（面板已经几百万行），由 ``EvalContext.styles()`` 懒加载。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import duckdb
import pandas as pd

from calculate.config import DB_FILE
from calculate.errors import IndustryError, StyleFactorError
from calculate.logging_conf import get_logger
from calculate.params import EvalParams

logger = get_logger("calculate.styles")

# 取数时往前回溯的交易日数：最长窗口是 MOMENTUM 的 250 个交易日
STYLE_LOOKBACK_DAYS = 260

# BETA_60 的市场基准，与库里已实现的 BETA_60 因子保持一致
BETA_BENCHMARK = "000300.SH"


@dataclass(frozen=True)
class StyleSpec:
    """一个风格因子的定义。

    参数:
        key: 风格名，也是输出列名。
        label: 中文名。
        definition: 一句话口径。
        window: 需要的交易日窗口长度。
        min_obs: 窗口内最少有效观测数，不够就置空。
        note: 展示给用户的补充说明。
    """

    key: str
    label: str
    definition: str
    window: int
    min_obs: int
    note: str = ""


STYLE_FACTORS: tuple[StyleSpec, ...] = (
    StyleSpec("SIZE", "规模", "ln(总市值)", 1, 1, "总市值越大取值越大"),
    StyleSpec(
        "VALUE", "估值", "z(1/pb) 与 z(1/pe_ttm) 的均值", 1, 1,
        "账面市值比与盈利收益率各做截面标准化后取均值，缺一个用另一个；pb/pe 非正视为缺失",
    ),
    StyleSpec(
        "MOMENTUM", "动量", "close[t-20] / close[t-250] - 1", 250, 251,
        "12 个月动量、跳过最近 1 个月；窗口内有效交易日不足则置空",
    ),
    StyleSpec("SHORT_REV", "短期反转", "1 - close[t] / close[t-20]", 20, 21,
              "最近 1 个月收益的反向"),
    StyleSpec("VOL", "波动", "std(ret, 60)", 60, 40, "60 个有效交易日的日收益标准差"),
    StyleSpec("LIQUIDITY", "流动性", "mean(turnover_rate, 20)", 20, 15,
              "20 个有效交易日的平均换手率，越大越活跃"),
    StyleSpec(
        "BETA_60", "贝塔", f"regr_slope(ret, idx_ret, 60)，基准 {BETA_BENCHMARK}", 60, 40,
        "对沪深300 日收益做一元回归的斜率；有效配对不足 40 天则置空，避免次新股伪贝塔",
    ),
)

STYLE_KEYS: tuple[str, ...] = tuple(spec.key for spec in STYLE_FACTORS)

# 参与回归的原始列：VALUE 由 BP、EP 两列合成，因此比风格数多一列
RAW_COLUMNS: tuple[str, ...] = ("SIZE", "BP", "EP", "MOMENTUM", "SHORT_REV", "VOL",
                                "LIQUIDITY", "BETA_60")


def lookback_start(conn: duckdb.DuckDBPyConnection, start: str,
                   days: int = STYLE_LOOKBACK_DAYS) -> str:
    """取区间起始日往前第 ``days`` 个交易日的日期。

    参数:
        conn: 已挂载主库 ``src`` 的连接。
        start: 评估区间起始日。
        days: 需要回溯的交易日数。
    """
    row = conn.execute(
        "SELECT min(trade_date) FROM ("
        " SELECT trade_date FROM src.md.calendar"
        f" WHERE trade_date < DATE '{start}' ORDER BY trade_date DESC LIMIT {int(days)})"
    ).fetchone()
    day = row[0] if row else None
    if day is None:  # 区间从头开始，能取多少取多少
        earliest = conn.execute("SELECT min(trade_date) FROM src.md.calendar").fetchone()[0]
        if earliest is None:
            raise StyleFactorError("md.calendar 是空的，无法定位风格因子的回看起点")
        return pd.Timestamp(earliest).strftime("%Y-%m-%d")
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def style_sql(params: EvalParams, window_start: str) -> str:
    """生成风格因子的取数 SQL。

    参数:
        params: 评估参数，取它的 ``start`` / ``end``。
        window_start: 市场回看起点（含）；个股有效历史不足时向前补足 250 次交易。
    """
    return (
        "WITH warmup AS ("
        " SELECT stock_code, min(trade_date) AS first_date FROM ("
        " SELECT stock_code, trade_date FROM src.md.daily_hfq"
        f" WHERE status = 'trading' AND trade_date < DATE '{params.start}'"
        " QUALIFY row_number() OVER (PARTITION BY stock_code ORDER BY trade_date DESC) <= 250"
        " ) GROUP BY stock_code), index_returns AS ("
        " SELECT trade_date, CAST(pct_chg AS DOUBLE) / 100.0 AS idx_ret"
        f" FROM src.md.index_daily WHERE index_code = '{BETA_BENCHMARK}'), "
        "base AS ("
        " SELECT b.stock_code, b.trade_date, CAST(b.close AS DOUBLE) AS px,"
        " CAST(b.ret AS DOUBLE) AS ret, CAST(b.total_mv AS DOUBLE) AS total_mv,"
        " CAST(b.turnover_rate AS DOUBLE) AS turnover_rate,"
        " CAST(b.pe_ttm AS DOUBLE) AS pe_ttm, CAST(b.pb AS DOUBLE) AS pb,"
        " ir.idx_ret"
        " FROM src.md.daily_hfq b"
        " LEFT JOIN warmup history ON history.stock_code = b.stock_code"
        " LEFT JOIN index_returns ir ON ir.trade_date = b.trade_date"
        " WHERE b.status = 'trading'"
        f" AND b.trade_date BETWEEN least(DATE '{window_start}', history.first_date)"
        f" AND DATE '{params.end}'), "
        "features AS ("
        " SELECT *,"
        " count(turnover_rate) OVER w20 AS n_turnover,"
        " avg(turnover_rate) OVER w20 AS liq,"
        " count(ret) OVER w60 AS n_ret,"
        " stddev_samp(ret) OVER w60 AS vol,"
        " regr_count(ret, idx_ret) OVER w60 AS n_pair,"
        " regr_slope(ret, idx_ret) OVER w60 AS beta,"
        " lag(px, 20) OVER w252 AS px_20,"
        " lag(px, 250) OVER w252 AS px_250"
        " FROM base"
        " WINDOW w20 AS (PARTITION BY stock_code ORDER BY trade_date"
        "   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),"
        " w60 AS (PARTITION BY stock_code ORDER BY trade_date"
        "   ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),"
        " w252 AS (PARTITION BY stock_code ORDER BY trade_date"
        "   ROWS BETWEEN 251 PRECEDING AND CURRENT ROW))"
        " SELECT stock_code, trade_date,"
        " ln(nullif(total_mv, 0)) AS SIZE,"
        " CASE WHEN pb > 0 THEN 1.0 / pb END AS BP,"
        " CASE WHEN pe_ttm > 0 THEN 1.0 / pe_ttm END AS EP,"
        " CASE WHEN px_20 IS NOT NULL AND px_250 IS NOT NULL"
        "   THEN px_20 / px_250 - 1 END AS MOMENTUM,"
        " CASE WHEN px_20 IS NOT NULL AND px_20 > 0 THEN 1.0 - px / px_20 END AS SHORT_REV,"
        " CASE WHEN n_ret >= 40 THEN vol END AS VOL,"
        " CASE WHEN n_turnover >= 15 THEN liq END AS LIQUIDITY,"
        " CASE WHEN n_pair >= 40 THEN beta END AS BETA_60"
        " FROM features"
        f" WHERE trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'"
    )


def build_styles(params: EvalParams, *, db_file: Path | str = DB_FILE,
                 conn: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """取评估区间内全部股票的风格因子原始值。

    参数:
        params: 评估参数。
        db_file: 日线主库路径。
        conn: 已有的主库连接；留空则自行打开并在结束时关闭。
    """
    owns_connection = conn is None
    if owns_connection:
        target = Path(db_file)
        if not target.exists():
            raise StyleFactorError(f"主库不存在: {target}")
        conn = duckdb.connect()
        conn.execute(f"ATTACH '{target}' AS src (READ_ONLY)")
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'md'"
        ).fetchall()}
        if "index_daily" not in tables:
            raise StyleFactorError(
                f"主库缺少 md.index_daily，算不出 BETA_60（基准 {BETA_BENCHMARK}）；"
                f"md schema 现有表：{sorted(tables)}"
            )
        window_start = lookback_start(conn, params.start)
        frame = conn.execute(style_sql(params, window_start)).df()
    finally:
        if owns_connection and conn is not None:
            conn.close()
    if frame.empty:
        raise StyleFactorError(
            f"{params.start} ~ {params.end} 没有可用的交易日行，算不出风格因子"
        )
    frame["stock_code"] = frame["stock_code"].astype(str)
    for column in RAW_COLUMNS:
        frame[column] = frame[column].astype("float64")
    logger.info(
        "风格因子就绪：行=%d 交易日=%d 股票=%d 回看起点=%s",
        len(frame), frame["trade_date"].nunique(), frame["stock_code"].nunique(), window_start,
    )
    return frame


def style_specs(keys=None) -> tuple[StyleSpec, ...]:
    """按名字取风格定义，留空返回全部。

    参数:
        keys: 风格名序列。
    """
    if keys is None:
        return STYLE_FACTORS
    wanted = list(keys)
    specs = tuple(spec for spec in STYLE_FACTORS if spec.key in wanted)
    known = {spec.key for spec in specs}
    missing = [key for key in wanted if key not in known]
    if missing:
        raise StyleFactorError(f"未知风格因子 {missing}；可选：{list(STYLE_KEYS)}")
    return specs


@runtime_checkable
class IndustrySource(Protocol):
    """行业数据源协议：给拆分阶段提供「股票 → 行业」的映射。"""

    key: str
    label: str
    note: str

    def load(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """返回含 ``stock_code`` 与 ``industry`` 两列的表。

        参数:
            conn: 已挂载主库 ``src`` 的连接。
        """
        ...


@dataclass(frozen=True)
class StockBasicIndustry:
    """静态行业分类：``ref.stock_basic.industry``（Tushare 股票基础表快照）。

    这是**静态**分类：用今天的行业去解释历史，无法反映行业变更；
    退市股不在表里，覆盖不全；与中证/申万行业口径也不一致。
    所以默认关闭，需要时由指标参数显式打开。
    """

    key: str = "stock_basic"
    label: str = "Tushare 静态行业"
    note: str = (
        "ref.stock_basic.industry 的静态快照：非时点数据（用今天的行业解释历史）、"
        "退市股缺失、与中证/申万行业口径不一致，因此默认关闭"
    )

    def load(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """读静态行业映射。

        参数:
            conn: 已挂载主库 ``src`` 的连接。
        """
        try:
            frame = conn.execute(
                "SELECT stock_code, industry FROM src.ref.stock_basic"
            ).df()
        except duckdb.Error as error:  # 表或列不存在
            raise IndustryError(
                f"读不到 ref.stock_basic.industry，无法启用行业层：{error}"
            ) from error
        if frame.empty:
            raise IndustryError("ref.stock_basic 是空的，无法启用行业层")
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame["industry"] = frame["industry"].fillna("未知").replace("", "未知")
        return frame


INDUSTRY_SOURCES: dict[str, IndustrySource] = {
    StockBasicIndustry().key: StockBasicIndustry(),
}

# 关闭行业层时指标参数取这个值
INDUSTRY_OFF = "off"


def list_industry_sources() -> tuple[IndustrySource, ...]:
    """返回可用的行业数据源，供指标参数枚举。"""
    return tuple(INDUSTRY_SOURCES.values())


def resolve_industry_source(key: str) -> IndustrySource | None:
    """按标识取行业数据源；``off`` 返回 None。

    参数:
        key: 数据源标识，或 ``off`` 表示关闭。
    """
    if key in (None, "", INDUSTRY_OFF):
        return None
    source = INDUSTRY_SOURCES.get(key)
    if source is None:
        raise IndustryError(
            f"未知行业数据源 {key!r}；可选：{INDUSTRY_OFF}、{sorted(INDUSTRY_SOURCES)}"
        )
    return source


def industry_choices() -> tuple[str, ...]:
    """指标参数里行业开关的可选值。"""
    return (INDUSTRY_OFF, *sorted(INDUSTRY_SOURCES))
