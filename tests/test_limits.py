"""涨跌停规则与取整的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from scripts.clean import limits


def test_round_half_up_matches_exchange_convention():
    """交易所是四舍五入，4.045 必须进位成 4.05，不能走 numpy 的银行家舍入。"""
    values = np.array([4.045, 3.675 * 1.1, 7.326, 1.001, np.nan])
    result = limits.round_half_up(values)
    assert result[0] == pytest.approx(4.05)
    assert result[1] == pytest.approx(4.04)
    assert result[2] == pytest.approx(7.33)
    assert result[3] == pytest.approx(1.00)
    assert np.isnan(result[4])


def _reference(rows):
    """构造 limits 需要的股票基础表。"""
    return pd.DataFrame(rows, columns=["stock_code", "name", "list_date", "delist_date", "board"])


def test_main_board_st_uses_5_percent():
    """主板 ST 是 5%，普通主板是 10%，创业板 ST 仍是 20%。"""
    codes = ["SH600000", "SH600001", "SZ300001"]
    reference = _reference(
        [
            ("SH600000", "普通", "1999-01-01", None, "main_sh"),
            ("SH600001", "ST股", "1999-01-01", None, "main_sh"),
            ("SZ300001", "创业板", "2011-01-01", None, "chinext"),
        ]
    )
    dates = ["2023-06-01"]
    is_st = np.array([[False], [True], [True]])
    is_delisting = np.zeros((3, 1), dtype=bool)
    up, down, applicable = limits.compute_limit_ratios(codes, dates, reference, is_st, is_delisting)
    assert up[0, 0] == pytest.approx(0.10)
    assert up[1, 0] == pytest.approx(0.05)
    assert up[2, 0] == pytest.approx(0.20)
    assert applicable.all()


def test_new_listing_five_day_window():
    """注册制板块上市前 5 个交易日不设涨跌幅，第 6 日起适用。"""
    codes = ["SH688001"]
    reference = _reference([("SH688001", "科创", "2023-06-01", None, "star")])
    dates = ["2023-06-01", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08"]
    is_st = np.zeros((1, len(dates)), dtype=bool)
    is_delisting = np.zeros((1, len(dates)), dtype=bool)
    _, _, applicable = limits.compute_limit_ratios(codes, dates, reference, is_st, is_delisting)
    assert not applicable[0, :5].any()
    assert applicable[0, 5]


def test_legacy_first_day_44_percent():
    """2014 年起、注册制之前，主板新股上市首日是 +44% / -36%。"""
    codes = ["SH601999"]
    reference = _reference([("SH601999", "新股", "2019-01-02", None, "main_sh")])
    dates = ["2019-01-02", "2019-01-03"]
    is_st = np.zeros((1, 2), dtype=bool)
    is_delisting = np.zeros((1, 2), dtype=bool)
    up, down, applicable = limits.compute_limit_ratios(codes, dates, reference, is_st, is_delisting)
    assert up[0, 0] == pytest.approx(0.44)
    assert down[0, 0] == pytest.approx(0.36)
    assert up[0, 1] == pytest.approx(0.10)
    assert applicable.all()


def test_delisting_period_first_day_no_limit():
    """退市整理期首日不设涨跌幅，其后按 10%。"""
    codes = ["SZ000004"]
    reference = _reference([("SZ000004", "退市", "1990-12-01", "2026-07-14", "main_sz")])
    dates = ["2026-06-23", "2026-06-24"]
    is_st = np.ones((1, 2), dtype=bool)
    is_delisting = np.ones((1, 2), dtype=bool)
    up, _, applicable = limits.compute_limit_ratios(codes, dates, reference, is_st, is_delisting)
    assert up[0, 0] == pytest.approx(0.10)
    assert not applicable[0, 0]
    assert applicable[0, 1]


def test_prefer_official_over_rule():
    """官方涨跌停价优先，缺失时才回退到规则值。"""
    official_up = np.array([[10.0, np.nan]])
    official_down = np.array([[8.0, np.nan]])
    rule_up = np.array([[9.9, 9.9]])
    rule_down = np.array([[8.1, 8.1]])
    up, down, used = limits.prefer_official(official_up, official_down, rule_up, rule_down)
    assert up[0, 0] == pytest.approx(10.0)
    assert up[0, 1] == pytest.approx(9.9)
    assert down[0, 0] == pytest.approx(8.0)
    assert bool(used[0, 0]) and not bool(used[0, 1])


def test_one_word_board_requires_full_day_lock():
    """一字板要求开高低收全部贴在涨停价上。"""
    close = np.array([[11.0, 11.0]])
    high = np.array([[11.0, 11.2]])
    low = np.array([[11.0, 10.5]])
    open_price = np.array([[11.0, 10.9]])
    limit_up = np.array([[11.0, 11.0]])
    limit_down = np.array([[9.0, 9.0]])
    applicable = np.ones((1, 2), dtype=bool)
    marks = limits.mark_limits(close, high, low, open_price, limit_up, limit_down, applicable)
    assert bool(marks["is_one_word_board"][0, 0])
    assert not bool(marks["is_one_word_board"][0, 1])


def test_tradeable_is_conservative_on_limit_up():
    """涨停封板当日不可买、跌停封板当日不可卖。"""
    status = np.array([[1, 1]])
    close = np.array([[11.0, 9.0]])
    open_price = np.array([[11.0, 9.0]])
    limit_up = np.array([[11.0, 11.0]])
    limit_down = np.array([[9.0, 9.0]])
    applicable = np.ones((1, 2), dtype=bool)
    result = limits.mark_tradeable(status, close, open_price, limit_up, limit_down, applicable)
    assert not bool(result["can_buy_close"][0, 0])
    assert not bool(result["can_sell_close"][0, 1])
