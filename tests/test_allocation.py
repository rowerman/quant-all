"""执行预算重建与目标股数分配。"""

import numpy as np
import pandas as pd
import pytest

from backtest.allocation import allocate_shares, execution_budget, execution_cash, execution_price
from backtest.errors import ExecutionError

DATES = pd.date_range("2024-01-29", periods=3, freq="B")


def _frame(values, columns, index=DATES):
    """按位置构造宽表，``None`` 表示 NaN。"""
    return pd.DataFrame(np.asarray(values, dtype="float64"), index=index, columns=columns)


def _allocate(targets, budgets, prices, columns, baseline=None, buffer=0.0, cash=None):
    """跑一次分配，返回 ``(股数, 明细, 统计)``；开盘现金默认与预算同值。"""
    shares, records, _, stats = allocate_shares(
        _frame(targets, columns),
        pd.Series(budgets, index=DATES),
        pd.Series(budgets if cash is None else cash, index=DATES),
        _frame(prices, columns),
        _frame(baseline, columns) if baseline is not None else _frame([[0.0] * len(columns)] * 3, columns),
        100,
        buffer,
    )
    return shares, records, stats


def test_execution_budget_uses_previous_state_and_execution_price():
    columns = ["A", "B"]
    cash = pd.Series([100_000.0, 20_000.0, 20_000.0], index=DATES)
    assets = _frame([[0.0, 0.0], [1_000.0, 500.0], [1_000.0, 500.0]], columns)
    price = _frame([[10.0, 20.0], [12.0, 25.0], [12.0, 25.0]], columns)
    budget = execution_budget(cash, assets, price, 100_000.0)
    # 首日没有上一交易日状态，用初始资金
    assert budget.iloc[0] == pytest.approx(100_000.0)
    # 第二日：第一日收盘现金 + 第一日持股按第二日开盘价估值
    assert budget.iloc[1] == pytest.approx(100_000.0)
    assert budget.iloc[2] == pytest.approx(20_000.0 + 1_000.0 * 12.0 + 500.0 * 25.0)


def test_execution_budget_falls_back_to_valuation_when_open_missing():
    columns = ["A"]
    cash = pd.Series([50_000.0, 50_000.0, 50_000.0], index=DATES)
    assets = _frame([[1_000.0], [1_000.0], [1_000.0]], columns)
    open_price = _frame([[10.0], [np.nan], [np.nan]], columns)
    valuation = _frame([[10.0], [11.0], [11.0]], columns)
    budget = execution_budget(cash, assets, execution_price(open_price, valuation), 50_000.0)
    assert budget.iloc[1] == pytest.approx(50_000.0 + 1_000.0 * 11.0)


def test_execution_cash_uses_previous_close_and_initial_cash():
    cash = pd.Series([10_000.0, 20_000.0, 30_000.0], index=DATES)
    values = execution_cash(cash, 100_000.0)
    assert values.tolist() == [100_000.0, 10_000.0, 20_000.0]


def test_allocate_shares_floors_to_lot_and_keeps_cash_buffer():
    shares, _, stats = _allocate(
        [[np.nan], [0.5], [np.nan]], [100_000.0] * 3, [[10.0]] * 3, ["A"], buffer=0.02
    )
    # 可用预算 98% × 10 万 × 0.5 = 49000 元，10 元价 -> 4900 股
    assert shares.loc[DATES[1], "A"] == 4900.0
    assert np.isnan(shares.loc[DATES[0], "A"])
    assert stats["allocated_cells"] == 1
    assert stats["no_price_cells"] == 0


def test_allocate_shares_bumps_one_lot_from_rounding_residue():
    """一手装不下的格子，用其他格子的取整余量补到一手。"""
    columns = [f"S{position}" for position in range(5)]
    prices = [[130.0, 130.0, 130.0, 130.0, 250.0]] * 3
    weights = [[np.nan] * 5, [0.2] * 5, [np.nan] * 5]
    shares, records, stats = _allocate(weights, [100_000.0] * 3, prices, columns)
    assert shares.loc[DATES[1], "S4"] == 100.0
    assert stats["bumped_one_lot_cells"] == 1
    assert stats["unaffordable_lot_cells"] == 0
    bumped = records.loc[records["status"] == "bumped"].iloc[0]
    assert bumped["stock_code"] == "S4"


def test_allocate_shares_reports_unaffordable_when_residue_is_not_enough():
    columns = [f"S{position}" for position in range(5)]
    prices = [[130.0, 130.0, 130.0, 130.0, 600.0]] * 3
    weights = [[np.nan] * 5, [0.2] * 5, [np.nan] * 5]
    shares, records, stats = _allocate(weights, [100_000.0] * 3, prices, columns)
    assert shares.loc[DATES[1], "S4"] == 0.0
    assert stats["bumped_one_lot_cells"] == 0
    assert stats["unaffordable_lot_cells"] == 1
    assert records.loc[records["status"] == "unaffordable"].iloc[0]["stock_code"] == "S4"


def test_allocate_shares_keeps_held_position_within_one_lot():
    """已有持仓只在差够一手时调整，避免预算微动带来的来回调仓。"""
    baseline = [[0.0], [100.0], [100.0]]
    weights = [[np.nan], [1.0], [1.0]]
    prices = [[10.0]] * 3
    shares, _, stats = _allocate(
        weights, [1_000.0, 1_050.0, 2_500.0], prices, ["A"], baseline=baseline
    )
    # 理想 105 股：差不够一手 -> 保持 100 股
    assert shares.loc[DATES[1], "A"] == 100.0
    # 理想 250 股：差够一手 -> 调到 200 股，仍向下取整
    assert shares.loc[DATES[2], "A"] == 200.0
    assert stats["allocated_cells"] == 2


def test_allocate_shares_sells_everything_when_target_is_zero():
    baseline = [[0.0], [500.0], [500.0]]
    weights = [[np.nan], [0.0], [np.nan]]
    shares, records, _ = _allocate(
        weights, [50_000.0] * 3, [[10.0]] * 3, ["A"], baseline=baseline
    )
    assert shares.loc[DATES[1], "A"] == 0.0
    assert records.iloc[0]["status"] == "flat"
    assert records.iloc[0]["baseline_shares"] == 500.0


def test_allocate_shares_excludes_blocked_holdings_from_budget():
    """被规则拦下的持仓当天卖不掉，它的市值不能算进可动用现金。"""
    # A 被拦下（目标为 NaN）且持有 2000 股 × 50 元 = 10 万，现金为 0
    baseline = [[0.0, 0.0], [2_000.0, 0.0], [2_000.0, 0.0]]
    weights = [[np.nan, np.nan], [np.nan, 1.0], [np.nan, np.nan]]
    prices = [[50.0, 50.0]] * 3
    shares, records, stats = _allocate(
        weights, [100_000.0, 100_000.0, 100_000.0], prices, ["A", "B"], baseline=baseline,
        cash=[0.0, 0.0, 0.0],
    )
    assert shares.loc[DATES[1], "B"] == 0.0
    assert stats["unaffordable_lot_cells"] == 1
    assert records.loc[records["status"] == "unaffordable"].iloc[0]["stock_code"] == "B"


def test_allocate_shares_keeps_full_weight_on_retry_row():
    """重试日只有少数格子下单，目标仍按整个组合价值算，不按这些格子自己的市值缩小。"""
    # 组合价值 10 万 = 现金 5 万 + B 持仓 5 万；A 没持仓，只有 A 有目标权重 0.5
    baseline = [[0.0, 0.0], [0.0, 5_000.0], [0.0, 5_000.0]]
    weights = [[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan]]
    prices = [[10.0, 10.0]] * 3
    shares, records, _ = _allocate(
        weights, [100_000.0] * 3, prices, ["A", "B"], baseline=baseline, buffer=0.02
    )
    # 0.98 × 10 万 × 0.5 = 4.9 万，10 元价 -> 4900 股
    assert shares.loc[DATES[1], "A"] == 4900.0
    assert records.loc[records["stock_code"] == "A"].iloc[0]["target_weight"] == pytest.approx(0.5)


def test_allocate_shares_caps_buys_by_open_cash():
    """买单按开盘现金限制：参考持仓是上一遍的执行后持仓，不能当成已有的钱。"""
    # 目标 0.5 × 10 万 = 4.9 万，但开盘只有 2 万现金，且没有可卖出的持仓
    baseline = [[0.0], [4_900.0], [4_900.0]]
    weights = [[np.nan], [0.5], [np.nan]]
    prices = [[10.0]] * 3
    shares, _, _ = _allocate(
        weights, [100_000.0] * 3, prices, ["A"], baseline=baseline, buffer=0.02,
        cash=[0.0, 20_000.0, 20_000.0],
    )
    assert shares.loc[DATES[1], "A"] == 2_000.0


def test_allocate_shares_handles_empty_budget_and_missing_price():
    weights = [[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan]]
    prices = [[10.0, np.nan], [10.0, np.nan], [10.0, np.nan]]
    shares, records, stats = _allocate(weights, [0.0, -1.0, 0.0], prices, ["A", "B"])
    assert np.isnan(shares.loc[DATES[1], "B"])  # 缺价格子不下单
    assert shares.loc[DATES[1], "A"] == 0.0
    assert stats["no_price_cells"] == 1
    assert stats["unaffordable_lot_cells"] == 1
    assert set(records["status"]) == {"no_price", "unaffordable"}


def test_allocate_shares_rejects_impossible_buffer():
    columns = ["A"]
    with pytest.raises(ExecutionError, match="现金缓冲"):
        allocate_shares(
            _frame([[np.nan], [1.0], [np.nan]], columns),
            pd.Series(10_000.0, index=DATES),
            pd.Series(10_000.0, index=DATES),
            _frame([[10.0]] * 3, columns),
            _frame([[0.0]] * 3, columns),
            100,
            1.0,
        )
