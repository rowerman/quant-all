"""基准与行情覆盖预检。"""

import duckdb
import pandas as pd
import pytest

from backtest.coverage import (
    benchmark_coverage,
    benchmark_coverages,
    default_run_end,
    market_coverage,
    validate_window,
)
from backtest.errors import DataAlignmentError


def _database(
    tmp_path,
    *,
    dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
    bench_dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
    bar_end="2024-01-05",
):
    """构造只含覆盖检查所需三张表的临时主库。"""
    path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE SCHEMA md")
    conn.execute("CREATE TABLE md.calendar(trade_date DATE, is_open INTEGER)")
    conn.executemany(
        "INSERT INTO md.calendar VALUES (?, 1)",
        [(pd.Timestamp(value).date(),) for value in dates],
    )
    conn.execute("CREATE TABLE md.daily_bar(stock_code VARCHAR, trade_date DATE)")
    conn.executemany(
        "INSERT INTO md.daily_bar VALUES ('SH600000', ?)",
        [(pd.Timestamp(dates[0]).date(),), (pd.Timestamp(bar_end).date(),)],
    )
    conn.execute(
        "CREATE TABLE md.index_daily(index_code VARCHAR, trade_date DATE, close DOUBLE)"
    )
    if bench_dates:
        conn.executemany(
            "INSERT INTO md.index_daily VALUES ('H00300.CSI', ?, 1000.0)",
            [(pd.Timestamp(value).date(),) for value in bench_dates],
        )
    conn.close()
    return path


def test_market_and_benchmark_coverages(tmp_path):
    """覆盖查询按交易日历与日线交集给出起止日。"""
    path = _database(tmp_path)
    market = market_coverage(path)
    assert (market.start, market.end, market.trading_days) == (
        "2024-01-02", "2024-01-05", 4
    )
    coverage = benchmark_coverage("H00300.CSI", path)
    assert (coverage.start, coverage.end, coverage.trading_days) == (
        "2024-01-02", "2024-01-05", 4
    )
    assert benchmark_coverages(["H00300.CSI", "H00905.CSI"], path)["H00905.CSI"].start is None
    assert default_run_end("H00300.CSI", path) == "2024-01-05"
    assert default_run_end(None, path) == "2024-01-05"


def test_missing_benchmark_day_reports_options(tmp_path):
    """区间内缺日必须列出日期和可选方案。"""
    path = _database(tmp_path, bench_dates=("2024-01-02", "2024-01-03", "2024-01-05"))
    with pytest.raises(DataAlignmentError) as caught:
        validate_window("H00300.CSI", "2024-01-02", "2024-01-05", path)
    message = str(caught.value)
    assert "缺少 1 个交易日" in message
    assert "2024-01-04" in message
    assert "benchmark=None" in message
    assert "补齐基准缺失交易日" in message


def test_start_before_benchmark_suggests_start(tmp_path):
    """请求早于基准发布日时直接给出 start 下限。"""
    path = _database(tmp_path, bench_dates=("2024-01-03", "2024-01-04", "2024-01-05"))
    with pytest.raises(DataAlignmentError, match="将 start 改为不早于 2024-01-03"):
        validate_window("H00300.CSI", "2024-01-02", "2024-01-05", path)


def test_end_after_benchmark_suggests_end(tmp_path):
    """请求晚于基准最后一天时直接给出 end 上限。"""
    path = _database(tmp_path, bench_dates=("2024-01-02", "2024-01-03", "2024-01-04"))
    with pytest.raises(DataAlignmentError, match="将 end 改为不晚于 2024-01-04"):
        validate_window("H00300.CSI", "2024-01-02", "2024-01-05", path)


def test_no_benchmark_skips_index_check(tmp_path):
    """显式关闭基准时只校验行情。"""
    path = _database(tmp_path, bench_dates=())
    validate_window(None, "2024-01-02", "2024-01-05", path)


def test_request_beyond_bar_coverage_is_rejected(tmp_path):
    """日历已更新但日线未入库时不能静默截断结束日。"""
    path = _database(
        tmp_path,
        dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"),
        bench_dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"),
        bar_end="2024-01-05",
    )
    with pytest.raises(DataAlignmentError, match="超出行情覆盖"):
        validate_window("H00300.CSI", "2024-01-02", "2024-01-08", path)


def test_missing_benchmark_table_row_is_an_error(tmp_path):
    """基准完全没有行情时给出明确错误。"""
    path = _database(tmp_path, bench_dates=())
    with pytest.raises(DataAlignmentError, match="没有行情"):
        benchmark_coverage("H00300.CSI", path)
