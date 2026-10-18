"""订单平移、规则过滤与订单计划。"""

import numpy as np
import pandas as pd
import pytest

import quant
from backtest.orders import (
    apply_rules,
    build_order_plan,
    execution_targets,
    liquidation_cells,
    probe_matrices,
    WRITEOFF_COLUMNS,
)
from backtest.specs import ExecutionSpec

INDEX = pd.date_range("2024-01-29", periods=4, freq="B")
COLUMNS = pd.Index(["SH600000", "SZ000001"], name="stock_code")


def _targets(values):
    """构造目标权重矩阵，None 表示 NaN。"""
    return pd.DataFrame(values, index=INDEX, columns=COLUMNS, dtype="float64")


@pytest.fixture
def panel(make_data):
    """固定用两列股票的面板，便于与测试矩阵对齐。"""

    def factory(**overrides):
        return make_data(stocks=COLUMNS, **overrides)

    return factory


def test_execution_targets_shift_to_next_trading_day():
    weights = _targets([[0.5, 0.5], [np.nan, np.nan], [0.0, 1.0], [np.nan, np.nan]])
    targets = execution_targets(weights)
    assert targets.iloc[0].isna().all()
    assert targets.loc[INDEX[1], "SH600000"] == 0.5
    assert targets.loc[INDEX[3], "SH600000"] == 0.0
    assert targets.iloc[2].isna().all()


def test_probe_matrices_scatter_order_records(panel):
    probe = pd.DataFrame(
        {
            "trade_date": [INDEX[1], INDEX[3]],
            "stock_code": ["SH600000", "SZ000001"],
            "side": [1, -1],
            "value": [5000.0, 2500.0],
        }
    )
    side, value = probe_matrices(probe, INDEX, COLUMNS)
    assert side[1, 0] == 1
    assert side[3, 1] == -1
    assert value[1, 0] == pytest.approx(5000.0)
    assert side.sum() == 0


def test_suspended_execution_day_is_retried_next_day(panel):
    """停牌当天不下单、只记一条拒单，下一个交易日自动补单。"""
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[1], "SH600000"] = True
    data = panel(is_suspended=suspended)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, direction, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[1], "SH600000"])
    assert filtered.loc[INDEX[2], "SH600000"] == 0.5
    assert rejection.to_dict("records") == [
        {"trade_date": INDEX[1], "stock_code": "SH600000", "reason": "suspended"}
    ]
    assert direction[1, 0] == 0 or direction[1, 0] == 1
    assert stats == {
        "retry_attempts": 1, "retried_filled_cells": 1, "pending_at_end_cells": 0,
        "cash_retry_attempts": 0,
    }


def test_limit_up_blocks_buy_and_keeps_position(panel):
    can_buy = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_buy.loc[INDEX[0], "SH600000"] = False  # 信号日次一交易日开盘涨停
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] != filtered.loc[INDEX[1], "SH600000"]  # NaN
    assert rejection.iloc[0]["reason"] == "limit_up_or_not_buyable"
    # 涨停只挡一天，次日开盘恢复可买时补单
    assert filtered.loc[INDEX[2], "SH600000"] == 0.5
    assert stats["retried_filled_cells"] == 1


def test_limit_down_blocks_sell_only(panel):
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[2], "SH600000"] = False  # INDEX[3] 执行日不能卖
    data = panel(can_sell_open_next=can_sell)
    targets = _targets(
        [[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [0.0, np.nan]]
    )
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[3], "SH600000"])
    assert rejection.iloc[0]["reason"] == "limit_down_or_not_sellable"
    # 区间最后一个交易日仍卖不掉，格子留在待重试状态里
    assert stats == {
        "retry_attempts": 0, "retried_filled_cells": 0, "pending_at_end_cells": 1,
        "cash_retry_attempts": 0,
    }


def test_rule_switches_can_be_turned_off(panel):
    can_buy = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    execution = ExecutionSpec(reject_limit_up_buy=False)
    filtered, rejection, _, _, _, _ = apply_rules(data, targets, execution)
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert rejection.empty


def test_blocked_sell_is_retried_every_day_until_filled(panel):
    """被拦下的卖出在每个交易日重试，直到卖掉为止，仓位不会永久留在组合里。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 执行日跌停卖不掉
    data = panel(can_sell_open_next=can_sell)
    targets = _targets(
        [[np.nan, np.nan], [0.5, np.nan], [0.0, np.nan], [0.0, np.nan]]
    )
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert rejection.iloc[0]["reason"] == "limit_down_or_not_sellable"
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[3] 自己也有清仓目标，属于新事件而不是重试
    assert filtered.loc[INDEX[3], "SH600000"] == 0.0
    assert len(rejection) == 1


def test_pending_sell_is_retried_on_a_day_without_new_target(panel):
    """调仓执行日之后没有新目标时，待重试的卖出靠每日重试补上。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 执行日跌停卖不掉
    data = panel(can_sell_open_next=can_sell)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [0.0, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    assert filtered.loc[INDEX[3], "SH600000"] == 0.0
    assert stats == {
        "retry_attempts": 1, "retried_filled_cells": 1, "pending_at_end_cells": 0,
        "cash_retry_attempts": 0,
    }


def test_repeatedly_blocked_cell_writes_single_rejection(panel):
    """同一事件的多次重试失败不再重复写拒单，只计尝试次数。"""
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[1], "SH600000"] = True
    suspended.loc[INDEX[2], "SH600000"] = True
    data = panel(is_suspended=suspended)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert len(rejection) == 1
    assert rejection.iloc[0]["trade_date"] == INDEX[1]
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5
    assert stats == {
        "retry_attempts": 2, "retried_filled_cells": 1, "pending_at_end_cells": 0,
        "cash_retry_attempts": 0,
    }


def test_new_rebalance_target_replaces_pending_retry(panel):
    """待重试的买入被新一期目标覆盖：股票始终没买进，不会再补单。"""
    can_buy = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_buy.loc[INDEX[0], "SH600000"] = False  # INDEX[1] 执行日涨停买不到
    can_buy.loc[INDEX[1], "SH600000"] = False  # 次日的重试仍然涨停
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [0.0, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[1], "SH600000"])
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[3] 的新目标是不持有，本来就空仓，因此没有订单也不留待重试
    assert np.isnan(filtered.loc[INDEX[3], "SH600000"])
    assert set(rejection["reason"]) == {"limit_up_or_not_buyable"}
    assert len(rejection) == 1
    assert stats == {
        "retry_attempts": 1, "retried_filled_cells": 0, "pending_at_end_cells": 0,
        "cash_retry_attempts": 0,
    }


def test_rebalance_cell_is_skipped_when_a_side_is_blocked(panel):
    """方向未知的再平衡在买卖任一侧受限时不下单。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 的再平衡一侧受限
    data = panel(can_sell_open_next=can_sell)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [0.5, np.nan], [0.5, np.nan]])
    filtered, _, _, _, _, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[2] 的信号日两侧都正常，再平衡可以提交
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5


def test_flat_position_with_zero_target_is_not_submitted(panel):
    data = panel()
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, _, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.notna().sum().sum() == 2
    assert rejection.empty


def _carry(values):
    """构造补回目标矩阵，``None`` 表示 NaN。"""
    return pd.DataFrame(values, index=INDEX, columns=COLUMNS, dtype="float64")


def test_cash_carry_cell_is_submitted_on_the_retry_day(panel):
    """被现金缩小的买单靠补回目标在后续交易日下单，不产生新的拒单行。"""
    data = panel()
    targets = _targets([[np.nan, np.nan]] * 4)
    carry = _carry([[np.nan, np.nan], [np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec(), None, carry)
    assert rejection.empty
    assert filtered.loc[INDEX[2], "SH600000"] == 0.5
    assert stats["cash_retry_attempts"] == 1
    assert stats["pending_at_end_cells"] == 0


def test_blocked_cash_carry_cell_writes_no_new_rejection(panel):
    """补回目标被停牌拦下时算重试：不写新拒单，回到规则层的待重试状态。"""
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[2], "SH600000"] = True  # 补回当天停牌
    data = panel(is_suspended=suspended)
    targets = _targets([[np.nan, np.nan]] * 4)
    carry = _carry([[np.nan, np.nan], [np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec(), None, carry)
    assert rejection.empty
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # 次日沿用同一条待重试状态补上，但已经不算现金补回的尝试
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5
    assert stats["cash_retry_attempts"] == 1
    assert stats["retry_attempts"] == 1
    assert stats["retried_filled_cells"] == 1


def test_real_target_wins_over_cash_carry(panel):
    """同一天既有真实调仓目标又有补回目标时，以真实目标为准。"""
    data = panel()
    targets = _targets([[np.nan, np.nan], [np.nan, np.nan], [0.25, np.nan], [np.nan, np.nan]])
    carry = _carry([[np.nan, np.nan], [np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats, _ = apply_rules(data, targets, ExecutionSpec(), None, carry)
    assert filtered.loc[INDEX[2], "SH600000"] == 0.25
    assert rejection.empty
    assert stats["cash_retry_attempts"] == 0


def test_build_order_plan_collects_stats(panel):
    data = panel()
    targets = _targets([[np.nan, np.nan], [0.5, 0.0], [np.nan, np.nan], [np.nan, np.nan]])
    probe = pd.DataFrame(
        {
            "trade_date": [INDEX[1]],
            "stock_code": ["SH600000"],
            "side": [1],
            "value": [50000.0],
        }
    )
    budget = pd.Series(100_000.0, index=INDEX)
    held = pd.DataFrame(0.0, index=INDEX, columns=COLUMNS)
    plan = build_order_plan(data, targets, ExecutionSpec(), probe, budget, budget, held)
    assert plan.plan_stats["probe_orders"] == 1
    assert plan.plan_stats["planned_orders"] == 1
    assert plan.plan_stats["rejected_orders"] == 0
    assert plan.plan_stats["forced_liquidations"] == 0
    # 本来就空仓、目标也是 0，不需要下单
    assert np.isnan(plan.target_percent.loc[INDEX[1], "SZ000001"])
    # 目标 0.5、预算 10 万、缓冲 2%：可用 9.8 万 → 49000 元 → 10 元价 4900 股
    assert plan.shares.loc[INDEX[1], "SH600000"] == 4900
    assert plan.plan_stats["allocated_cells"] == 1


def test_fee_estimate_uses_previous_close_shares(panel):
    """费率估计的参考持股取「上一交易日收盘的目标持股」。

    基准遍执行完当天，它的持股数就等于最终一遍要下的目标股数；如果拿这份当日收盘持股
    算股数差，差会系统性变成 0，费率又退回兜底口径（A5）。
    """
    data = panel()
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    budget = pd.Series(100_000.0, index=INDEX)
    baseline = pd.DataFrame(0.0, index=INDEX, columns=COLUMNS)
    baseline.loc[INDEX[1], "SH600000"] = 4900.0
    probe = pd.DataFrame(columns=["trade_date", "stock_code", "side", "value"])
    plan = build_order_plan(data, targets, ExecutionSpec(), probe, budget, budget, baseline)
    assert plan.shares.loc[INDEX[1], "SH600000"] == 4900
    # 目标股数 - 上一交易日收盘持股 = 4900 - 0，方向是买入；下一日没有订单
    assert plan.estimate_side[1, 0] == 1
    assert plan.estimate_side[2, 0] == 0


def _delist_panel(panel, stock: str = "SZ000001", **overrides):
    """构造一只股票在 INDEX[2] 退市、之后没有行情的面板。"""
    prices = pd.DataFrame(10.0, index=INDEX, columns=COLUMNS)
    prices.loc[INDEX[3], stock] = np.nan
    delist = pd.Series(pd.NaT, index=COLUMNS)
    delist[stock] = INDEX[2]
    return panel(open=prices.astype("float32"), delist_date=delist, **overrides)


def test_writeoff_cells_pick_delist_date_and_reference_close(panel):
    data = _delist_panel(panel, "SZ000001")
    cells = liquidation_cells(data)
    assert list(cells.columns) == [
        "trade_date", "stock_code", "delist_date", "reference_price",
    ]
    assert cells["stock_code"].tolist() == ["SZ000001"]
    assert cells["trade_date"].tolist() == [INDEX[2]]
    assert cells["reference_price"].tolist() == [pytest.approx(10.5)]
    assert cells["delist_date"].tolist() == [INDEX[2]]


def test_writeoff_cells_include_delist_on_window_end(panel):
    """退市日等于面板最后一天时仍要留下核销候选。"""
    delist = pd.Series(pd.NaT, index=COLUMNS)
    delist["SZ000001"] = INDEX[3]
    cells = liquidation_cells(panel(delist_date=delist))
    assert cells["trade_date"].tolist() == [INDEX[3]]
    # 没有退市日序列时整体不生效
    assert liquidation_cells(panel()).empty


def test_delisting_position_is_sold_from_first_sellable_day(panel):
    delisting = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    delisting.loc[INDEX[2], "SZ000001"] = True
    data = _delist_panel(panel, "SZ000001", is_delisting=delisting)
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    liquidations = liquidation_cells(data)
    filtered, rejection, direction, executed, _, _ = apply_rules(
        data, targets, ExecutionSpec(), liquidations
    )
    assert filtered.loc[INDEX[2], "SZ000001"] == 0.0
    assert direction[2, 1] == -1
    assert executed["stock_code"].tolist() == ["SZ000001"]
    assert rejection.empty


def test_forced_exit_respects_suspension_and_limit_down(panel):
    """整理期强制退出不能绕过停牌或跌停。"""
    delisting = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    delisting.loc[INDEX[2], "SZ000001"] = True
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SZ000001"] = False
    can_sell.loc[INDEX[2], "SZ000001"] = False
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[2], "SZ000001"] = True
    data = _delist_panel(
        panel,
        can_sell_open_next=can_sell,
        is_suspended=suspended,
        is_delisting=delisting,
    )
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, direction, executed, _, state = apply_rules(
        data, targets, ExecutionSpec(), liquidation_cells(data)
    )
    assert np.isnan(filtered.loc[INDEX[2], "SZ000001"])
    assert direction[2, 1] == 0
    assert len(executed) == 1
    assert rejection.to_dict("records") == [
        {"trade_date": INDEX[2], "stock_code": "SZ000001", "reason": "suspended"}
    ]
    assert not state.held["SZ000001"]
    assert np.isnan(state.pending["SZ000001"])


def test_forced_exit_is_blocked_by_limit_down(panel):
    """整理期强制退出遇到跌停时保留持仓并等待下一交易日重试。"""
    delisting = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    delisting.loc[INDEX[2], "SZ000001"] = True
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SZ000001"] = False
    can_sell.loc[INDEX[2], "SZ000001"] = False
    data = _delist_panel(panel, can_sell_open_next=can_sell, is_delisting=delisting)
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, direction, _, _, state = apply_rules(
        data,
        targets,
        ExecutionSpec(),
        pd.DataFrame(columns=WRITEOFF_COLUMNS),
    )
    assert np.isnan(filtered.loc[INDEX[2], "SZ000001"])
    assert direction[2, 1] == 0
    assert rejection.to_dict("records") == [
        {
            "trade_date": INDEX[2],
            "stock_code": "SZ000001",
            "reason": "limit_down_or_not_sellable",
        }
    ]
    assert state.held["SZ000001"]


def test_writeoff_cells_follow_real_delist_date(make_data):
    """真实退市股：核销日不晚于退市日，参考价取最后有效收盘。"""
    codes = ["SZ002898", "SZ000004", "SH600193"]
    start, end = "2026-06-01", "2026-07-31"
    frame = quant.load_panel(
        start=start, end=end, fields=["open", "close"], adjust="raw", stocks=codes
    )
    open_wide = frame.pivot(index="trade_date", columns="stock_code", values="open")
    close_wide = frame.pivot(index="trade_date", columns="stock_code", values="close")
    open_wide.index = pd.DatetimeIndex(open_wide.index)
    close_wide.index = pd.DatetimeIndex(close_wide.index)
    basic = quant.load_stock_basic().set_index("stock_code")
    delist = pd.to_datetime(basic["delist_date"]).reindex(open_wide.columns)
    data = make_data(
        dates=open_wide.index,
        stocks=open_wide.columns,
        open=open_wide,
        close=close_wide,
        delist_date=delist,
    )
    cells = liquidation_cells(data)
    assert set(cells["stock_code"]) == set(codes)
    for row in cells.to_dict("records"):
        code = row["stock_code"]
        last_valid = close_wide[code].dropna().index[-1]
        assert row["trade_date"] <= row["delist_date"]
        assert row["trade_date"] >= last_valid
        assert row["reference_price"] == pytest.approx(float(close_wide.loc[last_valid, code]))

