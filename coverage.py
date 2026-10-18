"""回测区间与基准覆盖检查。

预检只读主库，回答三个问题：行情实际覆盖到哪一天、所选基准覆盖哪一段、
请求区间内是否每个交易日都有基准行情。检查失败时不读面板、不创建运行记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.db import query as db_query
from backtest.errors import DataAlignmentError


@dataclass(frozen=True)
class DateCoverage:
    """一段数据的起止日与行数。"""

    start: str | None
    end: str | None
    trading_days: int


def _iso(value) -> str | None:
    """把 DuckDB 的日期值转成 ISO 字符串。"""
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _query(sql: str, params: list, db_path: Path | str | None) -> pd.DataFrame:
    """执行预检查询，把缺表等底层错误转成可读的 BT-201。"""
    try:
        return db_query(sql, params=params, path=db_path)
    except Exception as error:
        raise DataAlignmentError(f"读取行情或基准覆盖失败：{error}") from error


def market_coverage(db_path: Path | str | None = None) -> DateCoverage:
    """返回行情实际可用的起止日。

    交易日历决定第一段可用日期，日线表的最大交易日决定最后一天；
    两者取交集，避免日历已经更新但日线尚未入库时把结束日算多。

    参数:
        db_path: 主库路径；None 表示默认库。
    """
    calendar = _query(
        "SELECT min(trade_date) AS start, max(trade_date) AS end, count(*) AS days "
        "FROM md.calendar WHERE is_open = 1",
        [],
        db_path,
    )
    bars = _query(
        "SELECT min(trade_date) AS start, max(trade_date) AS end FROM md.daily_bar",
        [],
        db_path,
    )
    cal_start, cal_end = _iso(calendar.iloc[0]["start"]), _iso(calendar.iloc[0]["end"])
    bar_start, bar_end = _iso(bars.iloc[0]["start"]), _iso(bars.iloc[0]["end"])
    if cal_start is None or cal_end is None or bar_start is None or bar_end is None:
        raise DataAlignmentError("行情库没有可用的交易日历或日线数据")
    start = max(cal_start, bar_start)
    end = min(cal_end, bar_end)
    if start > end:
        raise DataAlignmentError(f"交易日历与日线没有重叠区间：{cal_start}~{cal_end} / {bar_start}~{bar_end}")
    days = _query(
        "SELECT count(*) AS days FROM md.calendar WHERE is_open = 1 "
        "AND trade_date BETWEEN ? AND ?",
        [start, end],
        db_path,
    )
    return DateCoverage(start, end, int(days.iloc[0]["days"]))


def benchmark_coverages(
    codes: list[str],
    db_path: Path | str | None = None,
) -> dict[str, DateCoverage]:
    """批量读取指数日线覆盖，缺失的代码返回空覆盖。

    参数:
        codes: 指数代码列表。
        db_path: 主库路径；None 表示默认库。
    """
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    frame = _query(
        f"SELECT index_code, min(trade_date) AS start, max(trade_date) AS end, "
        f"count(*) AS days FROM md.index_daily WHERE index_code IN ({placeholders}) "
        f"GROUP BY index_code",
        list(codes),
        db_path,
    )
    result = {code: DateCoverage(None, None, 0) for code in codes}
    for row in frame.itertuples(index=False):
        result[str(row.index_code)] = DateCoverage(
            _iso(row.start), _iso(row.end), int(row.days)
        )
    return result


def benchmark_coverage(code: str, db_path: Path | str | None = None) -> DateCoverage:
    """返回单个基准的覆盖；没有行情时抛 BT-201。"""
    coverage = benchmark_coverages([code], db_path).get(code)
    if coverage is None or coverage.start is None or coverage.end is None:
        raise DataAlignmentError(f"基准指数 {code} 在 md.index_daily 中没有行情")
    return coverage


def default_run_end(
    benchmark_code: str | None,
    db_path: Path | str | None = None,
) -> str:
    """返回行情与指定基准共同覆盖的最后一天。

    参数:
        benchmark_code: 基准指数代码；None 表示只看行情覆盖。
        db_path: 主库路径；None 表示默认库。
    """
    market = market_coverage(db_path)
    if benchmark_code is None:
        return str(market.end)
    benchmark = benchmark_coverage(benchmark_code, db_path)
    end = min(str(market.end), str(benchmark.end))
    start = max(str(market.start), str(benchmark.start))
    if start > end:
        raise DataAlignmentError(
            f"行情覆盖 {market.start}~{market.end} 与基准 {benchmark_code} "
            f"覆盖 {benchmark.start}~{benchmark.end} 没有重叠区间"
        )
    return end


def _missing_benchmark_days(
    code: str,
    start: str,
    end: str,
    db_path: Path | str | None,
) -> list[str]:
    """列出请求区间内基准缺失的开市日。"""
    frame = _query(
        "SELECT c.trade_date FROM md.calendar c "
        "LEFT JOIN md.index_daily i "
        "ON i.index_code = ? AND i.trade_date = c.trade_date "
        "WHERE c.is_open = 1 AND c.trade_date BETWEEN ? AND ? "
        "AND i.trade_date IS NULL ORDER BY c.trade_date",
        [code, start, end],
        db_path,
    )
    return [_iso(value) for value in frame["trade_date"].tolist()]


def validate_window(
    benchmark_code: str | None,
    start: str,
    end: str,
    db_path: Path | str | None = None,
) -> None:
    """检查请求区间是否落在行情与基准的覆盖范围内。

    参数:
        benchmark_code: 基准指数代码；None 表示本次关闭基准。
        start: 请求起始日。
        end: 请求截止日。
        db_path: 主库路径；None 表示默认库。
    """
    market = market_coverage(db_path)
    days = _query(
        "SELECT min(trade_date) AS start, max(trade_date) AS end, count(*) AS days "
        "FROM md.calendar WHERE is_open = 1 AND trade_date BETWEEN ? AND ?",
        [start, end],
        db_path,
    )
    first, last = _iso(days.iloc[0]["start"]), _iso(days.iloc[0]["end"])
    if first is None or last is None:
        raise DataAlignmentError(
            f"请求区间 {start} ~ {end} 内没有交易日；行情覆盖 {market.start} ~ {market.end}"
        )
    if first < str(market.start) or last > str(market.end):
        raise DataAlignmentError(
            f"请求区间 {start} ~ {end} 超出行情覆盖 {market.start} ~ {market.end}；"
            "请把 start/end 调整到覆盖范围内，或先补齐行情数据"
        )
    if benchmark_code is None:
        return

    coverage = benchmark_coverage(benchmark_code, db_path)
    missing = _missing_benchmark_days(benchmark_code, start, end, db_path)
    if not missing:
        return
    examples = "、".join(missing[:5])
    options = []
    if start < str(coverage.start):
        options.append(f"将 start 改为不早于 {coverage.start}")
    if end > str(coverage.end):
        options.append(f"将 end 改为不晚于 {coverage.end}")
    options.append("换用覆盖该区间的基准")
    options.append("补齐基准缺失交易日")
    options.append("显式设置 benchmark=None 关闭基准")
    raise DataAlignmentError(
        f"基准指数 {benchmark_code} 覆盖 {coverage.start} ~ {coverage.end}，"
        f"请求区间 {start} ~ {end} 缺少 {len(missing)} 个交易日"
        f"（示例：{examples}）。可选方案："
        + "；".join(f"{number}) {text}" for number, text in enumerate(options, start=1))
    )
