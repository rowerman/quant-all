"""复权与收益计算单元测试。"""

import numpy as np
import pytest

from scripts.clean import adjust


def test_shift_down_fills_first_column_with_nan():
    """按股票下移一行，首列没有前值，必须是 NaN 而不是 0。"""
    array = np.array([[1.0, 2.0, 3.0]])
    result = adjust.shift_down(array)
    assert np.isnan(result[0, 0])
    assert result[0, 1:].tolist() == [1.0, 2.0]


def test_pre_close_and_return_follow_exchange_convention():
    """ret = change / (close - change)，与交易所涨跌幅口径一致。"""
    close = np.array([[10.9, 10.3]])
    change = np.array([[0.6, -0.6]])
    pre_close = adjust.compute_pre_close(close, change)
    ret = adjust.compute_return(pre_close, change)
    assert pre_close[0].tolist() == pytest.approx([10.3, 10.9])
    assert ret[0, 0] == pytest.approx(0.6 / 10.3, rel=1e-12)
    assert ret[0, 1] == pytest.approx(-0.6 / 10.9, rel=1e-12)


def test_mark_ex_dividend_detects_price_adjustment():
    """昨收与「今收－涨跌额」对不上就是除权除息日。"""
    close = np.array([[8.0, 10.0]])
    change = np.array([[0.5, 0.0]])
    prev_close = np.array([[7.5, 10.5]])
    marked = adjust.mark_ex_dividend(close, change, prev_close)
    assert not bool(marked[0, 0])
    assert bool(marked[0, 1])


def test_adjust_scale_modes():
    """raw 不缩放；hfq 乘因子；qfq 再除以基准因子。"""
    factor = np.array([[2.0, 4.0]])
    base = np.array([[4.0]])
    assert adjust.adjust_scale("raw", factor, base).tolist() == [[1.0, 1.0]]
    assert adjust.adjust_scale("hfq", factor, base).tolist() == [[2.0, 4.0]]
    assert adjust.adjust_scale("qfq", factor, base).tolist() == [[0.5, 1.0]]
    with pytest.raises(Exception):
        adjust.adjust_scale("bad", factor, base)


def test_last_valid_factor_uses_last_trading_day():
    """前复权基准取最后一个有效交易日的因子，停牌日不参与。"""
    factor = np.array([[2.0, 3.0, np.nan]])
    close = np.array([[1.0, 1.1, np.nan]])
    result = adjust.last_valid_factor(factor, close)
    assert result[0, 0] == pytest.approx(3.0)


def test_return_mismatch_threshold():
    """超过 1e-4 的相对差异才算不一致。"""
    official = np.array([[0.01, 0.01]])
    factor_based = np.array([[0.01 + 5e-5, 0.0105]])
    result = adjust.return_mismatch(official, factor_based)
    assert not bool(result[0, 0])
    assert bool(result[0, 1])


def test_forward_fill_close_skips_suspension():
    """停牌日没有价格，昨收要跨过它取上一有效交易日。"""
    close = np.array([[np.nan, 10.0, np.nan, np.nan, 11.0]])
    filled = adjust.forward_fill_close(close)
    assert np.isnan(filled[0, 0])
    assert filled[0, 1:].tolist() == [10.0, 10.0, 10.0, 11.0]
    prev_close = adjust.shift_down(filled)
    assert prev_close[0, 2] == pytest.approx(10.0)
    assert prev_close[0, 4] == pytest.approx(10.0)


def test_mark_ex_dividend_across_suspension():
    """复牌首日除权：昨收取停牌前的收盘价才能标出来。"""
    close = np.array([[np.nan, np.nan, 6.0]])
    change = np.array([[np.nan, np.nan, 2.0]])
    prev_close = adjust.shift_down(adjust.forward_fill_close(np.array([[10.0, np.nan, np.nan]])))
    assert bool(adjust.mark_ex_dividend(close, change, prev_close)[0, 2])


def test_last_valid_close_picks_last_trading_day():
    """跨年传递的收盘价取最后一个有效交易日，末尾停牌不算。"""
    close = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]])
    result = adjust.last_valid_close(close)
    assert result[0] == pytest.approx(2.0)
    assert np.isnan(result[1])
