"""生命周期与 ST 判定单元测试。"""

import numpy as np
import pandas as pd

from scripts.clean import lifecycle


def test_is_st_name():
    """ST / *ST / SST / S*ST 都算 ST，普通名称与 G 开头不算。"""
    assert lifecycle.is_st_name("ST星源")
    assert lifecycle.is_st_name("*ST国华")
    assert lifecycle.is_st_name("S*ST前锋")
    assert lifecycle.is_st_name("SST天海")
    assert not lifecycle.is_st_name("平安银行")
    assert not lifecycle.is_st_name("G浦发")
    assert not lifecycle.is_st_name(None)


def test_is_delisting_name():
    """退市整理期的简称以「退」结尾或含退市字样。"""
    assert lifecycle.is_delisting_name("国华退")
    assert lifecycle.is_delisting_name("退市新亿")
    assert not lifecycle.is_delisting_name("平安银行")


def test_intervals_to_matrix_clips_to_calendar():
    """区间超出日历范围要被裁剪，不能越界写。"""
    codes = ["SZ000004"]
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    intervals = pd.DataFrame(
        {"stock_code": ["SZ000004"], "start_date": ["2023-01-01"], "end_date": ["2024-01-03"]}
    )
    matrix = lifecycle.intervals_to_matrix(codes, dates, intervals)
    assert matrix[0].tolist() == [True, True, False]


def test_intervals_to_matrix_ignores_unknown_code():
    """区间里出现不在股票池里的代码要跳过。"""
    matrix = lifecycle.intervals_to_matrix(
        ["SH600000"], ["2024-01-02"], pd.DataFrame(
            {"stock_code": ["SZ000001"], "start_date": ["2020-01-01"], "end_date": ["2025-01-01"]}
        )
    )
    assert not matrix.any()


def test_build_status_uses_list_and_delist_dates():
    """上市前是 not_listed，退市后是 ended，中间按是否有行情区分交易与停牌。"""
    dates = ["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    reference = pd.DataFrame(
        {
            "stock_code": ["SZ000001"],
            "list_date": ["2020-01-02"],
            "delist_date": ["2020-01-06"],
        }
    )
    close = np.array([[np.nan, 10.0, np.nan, 11.0, np.nan]])
    status = lifecycle.build_status(dates, reference, close)
    assert status[0].tolist() == [
        lifecycle.STATUS_NOT_LISTED,
        lifecycle.STATUS_TRADING,
        lifecycle.STATUS_SUSPENDED,
        lifecycle.STATUS_TRADING,
        lifecycle.STATUS_ENDED,
    ]


def test_build_resume_flags():
    """停牌后恢复交易的第一天标记为复牌日。"""
    status = np.array(
        [[lifecycle.STATUS_TRADING, lifecycle.STATUS_SUSPENDED, lifecycle.STATUS_TRADING]],
        dtype=np.int8,
    )
    resume = lifecycle.build_resume_flags(status)
    assert resume[0].tolist() == [False, False, True]
