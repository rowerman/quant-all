"""订单平移、规则过滤与费用矩阵。"""

import numpy as np
import pandas as pd
import pytest

import quant
from backtest.orders import (
    apply_rules,
    build_order_plan,
    execution_targets,
    fee_matrices,
    liquidation_cells,
    probe_matrices,
)
from backtest.specs import CostSpec, ExecutionSpec

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
    filtered, rejection, direction, _, stats = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[1], "SH600000"])
    assert filtered.loc[INDEX[2], "SH600000"] == 0.5
    assert rejection.to_dict("records") == [
        {"trade_date": INDEX[1], "stock_code": "SH600000", "reason": "suspended"}
    ]
    assert direction[1, 0] == 0 or direction[1, 0] == 1
    assert stats == {"retry_attempts": 1, "retried_filled_cells": 1, "pending_at_end_cells": 0}


def test_limit_up_blocks_buy_and_keeps_position(panel):
    can_buy = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_buy.loc[INDEX[0], "SH600000"] = False  # 信号日次一交易日开盘涨停
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
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
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[3], "SH600000"])
    assert rejection.iloc[0]["reason"] == "limit_down_or_not_sellable"
    # 区间最后一个交易日仍卖不掉，格子留在待重试状态里
    assert stats == {"retry_attempts": 0, "retried_filled_cells": 0, "pending_at_end_cells": 1}


def test_rule_switches_can_be_turned_off(panel):
    can_buy = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    execution = ExecutionSpec(reject_limit_up_buy=False)
    filtered, rejection, _, _, _ = apply_rules(data, targets, execution)
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
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
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
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    assert filtered.loc[INDEX[3], "SH600000"] == 0.0
    assert stats == {"retry_attempts": 1, "retried_filled_cells": 1, "pending_at_end_cells": 0}


def test_repeatedly_blocked_cell_writes_single_rejection(panel):
    """同一事件的多次重试失败不再重复写拒单，只计尝试次数。"""
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[1], "SH600000"] = True
    suspended.loc[INDEX[2], "SH600000"] = True
    data = panel(is_suspended=suspended)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
    assert len(rejection) == 1
    assert rejection.iloc[0]["trade_date"] == INDEX[1]
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5
    assert stats == {"retry_attempts": 2, "retried_filled_cells": 1, "pending_at_end_cells": 0}


def test_new_rebalance_target_replaces_pending_retry(panel):
    """待重试的买入被新一期目标覆盖：股票始终没买进，不会再补单。"""
    can_buy = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_buy.loc[INDEX[0], "SH600000"] = False  # INDEX[1] 执行日涨停买不到
    can_buy.loc[INDEX[1], "SH600000"] = False  # 次日的重试仍然涨停
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [0.0, np.nan]])
    filtered, rejection, _, _, stats = apply_rules(data, targets, ExecutionSpec())
    assert np.isnan(filtered.loc[INDEX[1], "SH600000"])
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[3] 的新目标是不持有，本来就空仓，因此没有订单也不留待重试
    assert np.isnan(filtered.loc[INDEX[3], "SH600000"])
    assert set(rejection["reason"]) == {"limit_up_or_not_buyable"}
    assert len(rejection) == 1
    assert stats == {"retry_attempts": 1, "retried_filled_cells": 0, "pending_at_end_cells": 0}


def test_rebalance_cell_is_skipped_when_a_side_is_blocked(panel):
    """方向未知的再平衡在买卖任一侧受限时不下单。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 的再平衡一侧受限
    data = panel(can_sell_open_next=can_sell)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [0.5, np.nan], [0.5, np.nan]])
    filtered, _, _, _, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[2] 的信号日两侧都正常，再平衡可以提交
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5


def test_flat_position_with_zero_target_is_not_submitted(panel):
    data = panel()
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, _, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.notna().sum().sum() == 2
    assert rejection.empty


def test_fee_matrix_charges_stamp_tax_on_sell_only(panel):
    costs = CostSpec(commission_rate=0.0003, minimum_commission=5.0, stamp_tax_rate=0.0005)
    side = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    value = np.zeros((len(INDEX), len(COLUMNS)), dtype="float32")
    side[1, 0], value[1, 0] = 1, 100000.0   # 大额买入：只收佣金
    side[2, 1], value[2, 1] = -1, 100000.0  # 大额卖出：佣金 + 印花税
    keep = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    keep.iloc[1, 0] = keep.iloc[2, 1] = True
    direction = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    fees, fixed = fee_matrices(INDEX, COLUMNS, keep, side, value, direction, costs)
    assert fees.iloc[1, 0] == pytest.approx(0.0003)
    assert fees.iloc[2, 1] == pytest.approx(0.0008)
    assert fixed.iloc[1, 0] == pytest.approx(0.0)
    assert fees.iloc[3, 0] == pytest.approx(0.0)


def test_minimum_commission_becomes_fixed_fee(panel):
    costs = CostSpec(commission_rate=0.0003, minimum_commission=5.0)
    side = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    value = np.zeros((len(INDEX), len(COLUMNS)), dtype="float32")
    side[1, 0], value[1, 0] = 1, 5000.0  # 佣金 1.5 元，不足 5 元下限
    keep = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    keep.iloc[1, 0] = True
    direction = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    fees, fixed = fee_matrices(INDEX, COLUMNS, keep, side, value, direction, costs)
    assert fixed.iloc[1, 0] == pytest.approx(3.5)
    assert fees.iloc[1, 0] * 5000.0 + fixed.iloc[1, 0] == pytest.approx(5.0)


def test_fee_matrix_uses_intent_direction_without_probe_record(panel):
    costs = CostSpec(commission_rate=0.0003, stamp_tax_rate=0.0005)
    side = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    value = np.zeros((len(INDEX), len(COLUMNS)), dtype="float32")
    direction = np.zeros((len(INDEX), len(COLUMNS)), dtype="int8")
    direction[1, 0] = -1  # 探针没记录，但意图是卖出
    keep = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    keep.iloc[1, 0] = True
    fees, fixed = fee_matrices(INDEX, COLUMNS, keep, side, value, direction, costs)
    assert fees.iloc[1, 0] == pytest.approx(0.0008)
    assert fixed.iloc[1, 0] == pytest.approx(0.0)


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
    """构造一只股票在窗口内退市的面板：最后一条行情在 INDEX[2]。"""
    prices = pd.DataFrame(10.0, index=INDEX, columns=COLUMNS)
    prices.loc[INDEX[3], stock] = np.nan
    delist = pd.Series(pd.NaT, index=COLUMNS)
    delist[stock] = INDEX[2]
    return panel(open=prices.astype("float32"), delist_date=delist, **overrides)


def test_liquidation_cells_pick_last_bar_before_window_end(panel):
    data = _delist_panel(panel, "SZ000001")
    cells = liquidation_cells(data)
    assert list(cells.columns) == ["trade_date", "stock_code", "price", "delist_date"]
    assert cells["stock_code"].tolist() == ["SZ000001"]
    assert cells["trade_date"].tolist() == [INDEX[2]]
    assert cells["price"].tolist() == [pytest.approx(10.0)]
    assert cells["delist_date"].tolist() == [INDEX[2]]


def test_liquidation_cells_skip_delist_on_window_end(panel):
    """退市日等于面板最后一天时不强卖，避免动到区间末的估值口径。"""
    delist = pd.Series(pd.NaT, index=COLUMNS)
    delist["SZ000001"] = INDEX[3]
    assert liquidation_cells(panel(delist_date=delist)).empty
    # 没有退市日序列时整体不生效
    assert liquidation_cells(panel()).empty


def test_delisted_position_is_sold_on_last_bar(panel):
    data = _delist_panel(panel, "SZ000001")
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    liquidations = liquidation_cells(data)
    filtered, rejection, direction, executed, _ = apply_rules(
        data, targets, ExecutionSpec(), liquidations
    )
    assert filtered.loc[INDEX[2], "SZ000001"] == 0.0
    assert direction[2, 1] == -1
    assert executed["stock_code"].tolist() == ["SZ000001"]
    assert rejection.empty


def test_forced_liquidation_bypasses_limit_down(panel):
    """退市强卖不受跌停/停牌规则约束，也不写拒单。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SZ000001"] = False  # INDEX[2] 执行日跌停卖不掉
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[2], "SZ000001"] = True
    data = _delist_panel(panel, can_sell_open_next=can_sell, is_suspended=suspended)
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _, executed, _ = apply_rules(
        data, targets, ExecutionSpec(), liquidation_cells(data)
    )
    assert filtered.loc[INDEX[2], "SZ000001"] == 0.0
    assert len(executed) == 1
    assert rejection.empty


def test_liquidation_cells_follow_real_delist_date(make_data):
    """真实退市股：强卖格落在它最后一条有价的行情日。

    退市整理期结束前就停牌的股票（如 SH600193 一直停到退市日）最后一条有价行情在退市日之前，
    强卖格取的是那一天而不是退市日，这也是唯一能真正卖出的时点。
    """
    codes = ["SZ002898", "SZ000004", "SH600193"]
    start, end = "2026-06-01", "2026-07-31"
    frame = quant.load_panel(
        start=start, end=end, fields=["open"], adjust="raw", stocks=codes
    )
    wide = frame.pivot(index="trade_date", columns="stock_code", values="open")
    wide.index = pd.DatetimeIndex(wide.index)
    basic = quant.load_stock_basic().set_index("stock_code")
    delist = pd.to_datetime(basic["delist_date"]).reindex(wide.columns)
    data = make_data(dates=wide.index, stocks=wide.columns, open=wide, delist_date=delist)
    cells = liquidation_cells(data)
    assert set(cells["stock_code"]) == set(codes)
    for row in cells.to_dict("records"):
        last_bar = wide[row["stock_code"]].dropna().index[-1]
        assert row["trade_date"] == last_bar
        assert pd.Timestamp(row["delist_date"]) >= last_bar
        assert row["price"] == pytest.approx(float(wide.loc[last_bar, row["stock_code"]]))

