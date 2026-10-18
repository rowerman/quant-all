"""执行预算重建与目标股数分配。"""

import numpy as np
import pandas as pd
import pytest

from backtest.allocation import (
    allocate_shares,
    cash_carry,
    cash_retry_filled,
    execution_budget,
    execution_cash,
    execution_price,
    extend_carry,
)
from backtest.errors import ExecutionError
from backtest.trading_rules import BOARD_MAIN, BOARD_STAR, BOARD_BSE

DATES = pd.date_range("2024-01-29", periods=3, freq="B")


def _frame(values, columns, index=DATES):
    """按位置构造宽表，``None`` 表示 NaN。"""
    return pd.DataFrame(np.asarray(values, dtype="float64"), index=index, columns=columns)


def _boards(columns, boards=None):
    """按股票列构造板块序列，默认主板。"""
    if boards is not None:
        return pd.Series(boards, index=columns)
    return pd.Series(BOARD_MAIN, index=columns)


def _allocate(
    targets, budgets, prices, columns, baseline=None, buffer=0.0, cash=None,
    boards=None,
):
    """跑一次分配，返回 ``(股数, 明细, 统计)``；开盘现金默认与预算同值。"""
    shares, records, _, stats = allocate_shares(
        _frame(targets, columns),
        pd.Series(budgets, index=DATES),
        pd.Series(budgets if cash is None else cash, index=DATES),
        _frame(prices, columns),
        _frame(baseline, columns) if baseline is not None else _frame([[0.0] * len(columns)] * 3, columns),
        _boards(columns, boards),
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
    assert stats["bumped_min_order_cells"] == 1
    assert stats["unaffordable_min_order_cells"] == 0
    bumped = records.loc[records["status"] == "bumped"].iloc[0]
    assert bumped["stock_code"] == "S4"


def test_allocate_shares_reports_unaffordable_when_residue_is_not_enough():
    columns = [f"S{position}" for position in range(5)]
    prices = [[130.0, 130.0, 130.0, 130.0, 600.0]] * 3
    weights = [[np.nan] * 5, [0.2] * 5, [np.nan] * 5]
    shares, records, stats = _allocate(weights, [100_000.0] * 3, prices, columns)
    assert shares.loc[DATES[1], "S4"] == 0.0
    assert stats["bumped_min_order_cells"] == 0
    assert stats["unaffordable_min_order_cells"] == 1
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
    assert stats["unaffordable_min_order_cells"] == 1
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
    """买单按开盘现金限制，所有现金缩水格子都进补回。"""
    # 目标 0.5 × 10 万 = 4.9 万，但开盘只有 2 万现金，且没有可卖出的持仓
    baseline = [[0.0], [4_900.0], [4_900.0]]
    weights = [[np.nan], [0.5], [np.nan]]
    prices = [[10.0]] * 3
    shares, records, stats = _allocate(
        weights, [100_000.0] * 3, prices, ["A"], baseline=baseline, buffer=0.02,
        cash=[0.0, 20_000.0, 20_000.0],
    )
    assert shares.loc[DATES[1], "A"] == 2_000.0
    # 从 4900 股缩到 2000 股：建仓只到目标的 41%，进补回
    assert stats["cash_shrunk_cells"] == 1
    assert stats["cash_limited_cells"] == 1
    assert stats["cash_limited_value"] == pytest.approx(29_000.0)
    assert bool(records.iloc[0]["cash_limited"])
    assert records.iloc[0]["cash_shrunk_shares"] == pytest.approx(2_900.0)


def test_allocate_shares_retries_small_cash_shortfall():
    """即使只差一手，只要确实触发现金缩放，也进入补回。"""
    # 目标 0.49 × 10 万 = 4.9 万，开盘 4.8 万现金：现金缩放后只差 100 股
    weights = [[np.nan], [0.49], [np.nan]]
    shares, records, stats = _allocate(
        weights, [100_000.0] * 3, [[10.0]] * 3, ["A"], cash=[0.0, 48_000.0, 48_000.0],
    )
    assert shares.loc[DATES[1], "A"] == 4_800.0
    assert stats["cash_shrunk_cells"] == 1
    assert stats["cash_limited_cells"] == 1
    assert stats["cash_limited_value"] == pytest.approx(1_000.0)
    assert bool(records.iloc[0]["cash_limited"])
    assert records.iloc[0]["cash_shrunk_shares"] == pytest.approx(100.0)


def test_allocate_shares_without_cash_pressure_marks_nothing():
    weights = [[np.nan], [0.5], [np.nan]]
    _, records, stats = _allocate(weights, [100_000.0] * 3, [[10.0]] * 3, ["A"], buffer=0.02)
    assert stats["cash_shrunk_cells"] == 0
    assert stats["cash_limited_cells"] == 0
    assert stats["cash_limited_value"] == 0.0
    assert not records["cash_limited"].any()
    assert not records["cash_shrunk_shares"].any()


def test_allocate_shares_handles_empty_budget_and_missing_price():
    weights = [[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan]]
    prices = [[10.0, np.nan], [10.0, np.nan], [10.0, np.nan]]
    shares, records, stats = _allocate(weights, [0.0, -1.0, 0.0], prices, ["A", "B"])
    assert np.isnan(shares.loc[DATES[1], "B"])  # 缺价格子不下单
    assert shares.loc[DATES[1], "A"] == 0.0
    assert stats["no_price_cells"] == 1
    assert stats["unaffordable_min_order_cells"] == 1
    assert set(records["status"]) == {"no_price", "unaffordable"}


def test_allocate_shares_limits_buy_by_daily_amount_participation():
    targets = _frame([[np.nan], [1.0], [np.nan]], ["A"])
    amount = _frame([[1_000.0], [1_000.0], [1_000.0]], ["A"])
    shares, records, _, stats = allocate_shares(
        targets,
        pd.Series([100_000.0] * 3, index=DATES),
        pd.Series([100_000.0] * 3, index=DATES),
        _frame([[10.0], [10.0], [10.0]], ["A"]),
        _frame([[0.0], [0.0], [0.0]], ["A"]),
        _boards(["A"]),
        0.0,
        amount_cny=amount,
        max_participation_rate=0.1,
    )
    assert shares.loc[DATES[1], "A"] == 0.0
    row = records.loc[records["trade_date"] == DATES[1]].iloc[0]
    assert row["capacity_limited"]
    assert row["capacity_shortfall_shares"] > 0
    assert stats["liquidity_limited_cells"] == 1


def test_allocate_shares_retries_when_amount_is_unavailable():
    targets = _frame([[np.nan], [1.0], [np.nan]], ["A"])
    amount = _frame([[1_000.0], [np.nan], [1_000.0]], ["A"])
    _, records, _, stats = allocate_shares(
        targets,
        pd.Series([100_000.0] * 3, index=DATES),
        pd.Series([100_000.0] * 3, index=DATES),
        _frame([[10.0], [10.0], [10.0]], ["A"]),
        _frame([[0.0], [0.0], [0.0]], ["A"]),
        _boards(["A"]),
        0.0,
        amount_cny=amount,
        max_participation_rate=0.1,
    )
    row = records.loc[records["trade_date"] == DATES[1]].iloc[0]
    assert row["capacity_unavailable"]
    assert stats["liquidity_unavailable_cells"] == 1


def test_cash_carry_moves_limited_buys_to_the_next_day():
    """被现金缩小的买单落到下一交易日，权重沿用当天的目标权重。"""
    columns = ["A", "B"]
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[0], DATES[0]],
            "stock_code": ["A", "B"],
            "target_weight": [0.3, 0.4],
            "cash_limited": [True, False],
            "cash_shrunk_shares": [500.0, 0.0],
            "allocated_shares": [100.0, 4_000.0],
        }
    )
    targets = _frame([[np.nan, np.nan]] * 3, columns)
    carry = cash_carry(allocation, targets)
    assert carry.loc[DATES[1], "A"] == pytest.approx(0.3)
    assert np.isnan(carry.loc[DATES[1], "B"])
    assert carry.iloc[0].isna().all() and carry.iloc[2].isna().all()
    assert cash_carry(None, targets).isna().to_numpy().all()


def test_cash_carry_moves_capacity_limited_buy_to_the_next_day():
    """成交额容量造成的缺口沿同一补回状态机推进。"""
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[0]],
            "stock_code": ["A"],
            "target_weight": [0.3],
            "cash_limited": [False],
            "cash_shrunk_shares": [0.0],
            "capacity_limited": [True],
            "capacity_unavailable": [False],
            "capacity_shortfall_shares": [200.0],
        }
    )
    targets = _frame([[np.nan]] * 3, ["A"])
    carry = cash_carry(allocation, targets)
    assert carry.loc[DATES[1], "A"] == pytest.approx(0.3)


def test_cash_carry_continues_unfilled_retry_without_creating_duplicate_event():
    """已有补回未成交时继续传播，不能把重试再次当成新的补回事件。"""
    columns = ["A"]
    long_dates = pd.date_range("2024-01-29", periods=4, freq="B")
    targets = _frame([[np.nan]] * 4, columns, long_dates)
    previous = _frame([[np.nan], [0.3], [np.nan], [np.nan]], columns, long_dates)
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[1]],
            "stock_code": ["A"],
            "target_weight": [0.3],
            "cash_limited": [True],
            "allocated_shares": [100.0],
        }
    )
    next_carry = cash_carry(allocation, targets, previous)
    assert next_carry.loc[long_dates[2], "A"] == pytest.approx(0.3)
    assert np.isnan(next_carry.loc[long_dates[3], "A"])


def test_cash_carry_stops_after_existing_retry_is_filled():
    """已有补回拿到目标仓位后停止传播。"""
    columns = ["A"]
    long_dates = pd.date_range("2024-01-29", periods=4, freq="B")
    targets = _frame([[np.nan]] * 4, columns, long_dates)
    previous = _frame([[np.nan], [0.3], [np.nan], [np.nan]], columns, long_dates)
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[1]],
            "stock_code": ["A"],
            "target_weight": [0.3],
            "cash_limited": [False],
            "allocated_shares": [3_000.0],
        }
    )
    next_carry = cash_carry(allocation, targets, previous)
    assert next_carry.isna().all().all()


def test_cash_carry_yields_to_real_target_and_drops_last_day():
    """下一交易日有真实调仓目标时补回让位；末日被缩小的格子没有下一交易日。"""
    columns = ["A"]
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[0], DATES[-1]],
            "stock_code": ["A", "A"],
            "target_weight": [0.3, 0.3],
            "cash_limited": [True, True],
            "cash_shrunk_shares": [500.0, 500.0],
        }
    )
    targets = _frame([[np.nan], [0.25], [np.nan]], columns)
    carry = cash_carry(allocation, targets)
    assert carry.isna().to_numpy().all()


def test_extend_carry_keeps_retrying_until_the_next_target():
    """现金没到账的日子要天天补，种子把补回目标铺到下一次真实调仓之前。"""
    columns = ["A"]
    carry = _frame([[np.nan], [0.3], [np.nan]], columns)
    # 第 2 天有真实调仓目标：补回到此为止
    extended = extend_carry(carry, _frame([[np.nan], [np.nan], [0.2]], columns))
    assert extended.loc[DATES[1], "A"] == pytest.approx(0.3)
    assert np.isnan(extended.loc[DATES[2], "A"])
    # 第 2 天没有真实目标：补回继续铺下去
    extended = extend_carry(carry, _frame([[np.nan]] * 3, columns))
    assert extended.loc[DATES[2], "A"] == pytest.approx(0.3)
    assert np.isnan(extended.loc[DATES[0], "A"])


def test_extend_carry_spans_more_than_iteration_limit_days():
    """补回种子一次铺完整条长链，不依赖逐日增加迭代次数。"""
    long_dates = pd.date_range("2024-01-02", periods=30, freq="B")
    columns = ["A"]
    carry = _frame([[np.nan], [0.3]] + [[np.nan]] * 28, columns, long_dates)
    targets = _frame([[np.nan]] * 29 + [[0.2]], columns, long_dates)
    extended = extend_carry(carry, targets)
    assert extended.loc[long_dates[1]:long_dates[-2], "A"].eq(0.3).all()
    assert np.isnan(extended.loc[long_dates[0], "A"])
    assert np.isnan(extended.loc[long_dates[-1], "A"])


def test_cash_retry_filled_counts_only_topped_up_cells():
    """真正补齐要求当天拿到仓位且没有再被现金缩小。"""
    columns = ["A", "B", "C"]
    carry = _frame([[0.3, 0.3, 0.3]] * 3, columns)
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[1], DATES[1]],
            "stock_code": ["A", "B"],
            "cash_limited": [False, True],
            "allocated_shares": [1_000.0, 200.0],
        }
    )
    # A 补齐；B 有仓位但又被缩小，只能算尝试；C 当天没有分配明细（被规则拦下）
    assert cash_retry_filled(carry, allocation) == 1
    empty = _frame([[np.nan] * 3] * 3, columns)
    assert cash_retry_filled(empty, allocation) == 0


def test_allocate_shares_rejects_impossible_buffer():
    columns = ["A"]
    with pytest.raises(ExecutionError, match="现金缓冲"):
        allocate_shares(
            _frame([[np.nan], [1.0], [np.nan]], columns),
            pd.Series(10_000.0, index=DATES),
            pd.Series(10_000.0, index=DATES),
            _frame([[10.0]] * 3, columns),
            _frame([[0.0]] * 3, columns),
            _boards(columns),
            1.0,
        )


def test_allocate_shares_uses_star_minimum_order():
    columns = ["SH688001"]
    weights = [[np.nan], [0.1], [np.nan]]
    shares, records, stats = _allocate(
        weights,
        [2_000.0] * 3,
        [[10.0]] * 3,
        columns,
        boards=[BOARD_STAR],
    )

    assert shares.loc[DATES[1], "SH688001"] == 200
    assert records.iloc[0]["board"] == BOARD_STAR
    assert records.iloc[0]["min_buy_shares"] == 200
    assert stats["bumped_min_order_cells"] == 1


def test_allocate_shares_keeps_bse_single_share_increment():
    columns = ["BJ920000"]
    weights = [[np.nan], [0.0785], [np.nan]]
    shares, records, stats = _allocate(
        weights,
        [20_000.0] * 3,
        [[10.0]] * 3,
        columns,
        boards=[BOARD_BSE],
    )

    assert shares.loc[DATES[1], "BJ920000"] == 157
    assert records.iloc[0]["board"] == BOARD_BSE
    assert records.iloc[0]["order_shares"] == 157
    assert bool(records.iloc[0]["order_valid"])
    assert stats["bumped_min_order_cells"] == 0


def test_allocate_shares_does_not_shrink_star_buy_below_200():
    columns = ["SH688001"]
    targets = _frame([[np.nan], [1.0], [np.nan]], columns)
    amount = _frame([[1_000.0]] * 3, columns)
    shares, records, _, stats = allocate_shares(
        targets,
        pd.Series([10_000.0] * 3, index=DATES),
        pd.Series([10_000.0] * 3, index=DATES),
        _frame([[10.0]] * 3, columns),
        _frame([[0.0]] * 3, columns),
        pd.Series([BOARD_STAR], index=columns),
        0.0,
        amount_cny=amount,
        max_participation_rate=0.1,
    )

    assert shares.loc[DATES[1], "SH688001"] == 0
    assert stats["liquidity_limited_cells"] == 1
    assert bool(records.iloc[0]["order_valid"])
