"""指数取数的分片与覆盖度计算单元测试。

中转服务对 index_weight 的返回上限与粒度都不稳定，这些纯函数决定了怎么分片、
怎么判断某个区间是不是拉全了，必须单独锁住。
"""

from datetime import date

import pandas as pd

from scripts.sources import tushare_src
from scripts.sources.fetch_all import INDEX_CODES, INDEX_WEIGHT_CODES


def test_to_date_and_add_months():
    """日期解析与按月平移（含跨年、跨负月）要正确。"""
    assert tushare_src._to_date("20240131") == date(2024, 1, 31)
    assert tushare_src._add_months(date(2024, 1, 1), 3) == date(2024, 4, 1)
    assert tushare_src._add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)
    assert tushare_src._add_months(date(2024, 1, 1), -1) == date(2023, 12, 1)
    assert tushare_src._add_months(date(2024, 1, 1), -14) == date(2022, 11, 1)


def test_month_span_counts_both_ends():
    """月份跨度含首尾，用来判断覆盖比例。"""
    assert tushare_src._month_span("20240131", "20240131") == 1
    assert tushare_src._month_span("20240131", "20240331") == 3
    assert tushare_src._month_span("20231130", "20240229") == 4


def test_weight_step_keeps_window_within_call_budget():
    """分片步长要保证「期数 × 成分数」不超单次预算，同时能整除 12 以对齐日历。"""
    cases = {50: 12, 100: 12, 300: 12, 500: 12, 800: 6, 1000: 6, 2000: 3, 5000: 1}
    for members, expected in cases.items():
        step = tushare_src._weight_step(members)
        assert step == expected, (members, step)
        assert step * members <= tushare_src.INDEX_WEIGHT_CALL_BUDGET
        assert 12 % step == 0


def test_weight_step_handles_empty_index():
    """成分数未知时退化成按年拉，不能除零。"""
    assert tushare_src._weight_step(0) == 12


def _frame(periods):
    """构造一份只含期数的成分权重表。"""
    return pd.DataFrame({
        "index_code": "000300.SH",
        "con_code": "600000.SH",
        "trade_date": periods,
        "weight": 1.0,
    })


def test_coverage_ratio_flags_missing_months():
    """缺月份的区间覆盖率要低于 1，全了就正好是 1。"""
    full = _frame(["20240131", "20240229", "20240331"])
    missing = _frame(["20240131", "20240331"])
    assert tushare_src._coverage_ratio(full) == 1.0
    assert tushare_src._coverage_ratio(missing) < 0.7
    assert tushare_src._coverage_ratio(pd.DataFrame()) == 0.0


def test_min_coverage_matches_documented_threshold():
    """覆盖率门槛是产物过滤的依据，改它要同步改文档。"""
    assert tushare_src.INDEX_WEIGHT_MIN_COVERAGE == 0.98
    assert tushare_src.INDEX_WEIGHT_HARD_CAP == 7000
    assert tushare_src.INDEX_WEIGHT_CALL_BUDGET < tushare_src.INDEX_WEIGHT_HARD_CAP


def test_periods_in_only_counts_window():
    """窗口期数只统计落在区间内的快照。"""
    frame = _frame(["20240131", "20240229", "20240331"])
    assert tushare_src._periods_in(frame, date(2024, 1, 1), date(2024, 3, 31)) == 3
    assert tushare_src._periods_in(frame, date(2024, 2, 1), date(2024, 2, 29)) == 1
    assert tushare_src._periods_in(frame, date(2025, 1, 1), date(2025, 3, 31)) == 0


def test_total_return_indices_are_prices_not_membership_pools():
    """全收益指数只拉行情，股票池成分仍使用对应价格指数。"""
    assert INDEX_CODES["H00300.CSI"] == "沪深300全收益"
    assert INDEX_CODES["H00905.CSI"] == "中证500全收益"
    assert "H00300.CSI" not in INDEX_WEIGHT_CODES
    assert "H00905.CSI" not in INDEX_WEIGHT_CODES


def test_merge_index_daily_keeps_other_codes_and_replaces_overlap():
    """定向补拉只替换指定指数的重叠日期，不能丢掉其他指数。"""
    existing = pd.DataFrame(
        {
            "ts_code": ["H00300.CSI", "H00300.CSI", "000300.SH"],
            "trade_date": ["20261014", "20261015", "20261015"],
            "close": [100.0, 101.0, 4000.0],
            "index_name": ["沪深300全收益", "沪深300全收益", "沪深300"],
        }
    )
    incoming = pd.DataFrame(
        {
            "ts_code": ["H00300.CSI", "H00300.CSI"],
            "trade_date": ["20261015", "20261016"],
            "close": [102.0, 103.0],
            "index_name": ["沪深300全收益", "沪深300全收益"],
        }
    )
    merged = tushare_src.merge_index_daily(existing, incoming)
    assert len(merged) == 4
    assert set(merged["ts_code"]) == {"H00300.CSI", "000300.SH"}
    h00300 = merged[merged["ts_code"] == "H00300.CSI"].set_index("trade_date")
    assert h00300.loc["20261015", "close"] == 102.0
    assert h00300.loc["20261016", "close"] == 103.0
    assert merged[merged["ts_code"] == "000300.SH"]["close"].tolist() == [4000.0]
