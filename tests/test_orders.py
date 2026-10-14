"""订单平移、规则过滤与费用矩阵。"""

import numpy as np
import pandas as pd
import pytest

from backtest.orders import (
    apply_rules,
    build_order_plan,
    execution_targets,
    fee_matrices,
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


def test_suspended_execution_day_is_blocked(panel):
    suspended = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    suspended.loc[INDEX[1], "SH600000"] = True
    data = panel(is_suspended=suspended)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, direction = apply_rules(data, targets, ExecutionSpec())
    assert filtered.notna().sum().sum() == 0
    assert rejection.to_dict("records") == [
        {"trade_date": INDEX[1], "stock_code": "SH600000", "reason": "suspended"}
    ]
    assert direction[1, 0] == 0 or direction[1, 0] == 1


def test_limit_up_blocks_buy_and_keeps_position(panel):
    can_buy = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_buy.loc[INDEX[0], "SH600000"] = False  # 信号日次一交易日开盘涨停
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] != filtered.loc[INDEX[1], "SH600000"]  # NaN
    assert rejection.iloc[0]["reason"] == "limit_up_or_not_buyable"


def test_limit_down_blocks_sell_only(panel):
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[2], "SH600000"] = False  # INDEX[3] 执行日不能卖
    data = panel(can_sell_open_next=can_sell)
    targets = _targets(
        [[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [0.0, np.nan]]
    )
    filtered, rejection, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[3], "SH600000"])
    assert rejection.iloc[0]["reason"] == "limit_down_or_not_sellable"


def test_rule_switches_can_be_turned_off(panel):
    can_buy = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    data = panel(can_buy_open_next=can_buy)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    execution = ExecutionSpec(reject_limit_up_buy=False)
    filtered, rejection, _ = apply_rules(data, targets, execution)
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert rejection.empty


def test_blocked_sell_is_retried_on_the_next_rebalance(panel):
    """被拦下的卖出必须在下一个调仓日重试，否则仓位会永久留在组合里。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 执行日卖不掉
    data = panel(can_sell_open_next=can_sell)
    targets = _targets(
        [[np.nan, np.nan], [0.5, np.nan], [0.0, np.nan], [0.0, np.nan]]
    )
    filtered, rejection, _ = apply_rules(data, targets, ExecutionSpec())
    assert rejection.iloc[0]["reason"] == "limit_down_or_not_sellable"
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # 状态仍为持有，目标一直是 0，所以下一个调仓日会重新提交卖出
    assert filtered.loc[INDEX[3], "SH600000"] == 0.0


def test_rebalance_cell_is_skipped_when_a_side_is_blocked(panel):
    """方向未知的再平衡在买卖任一侧受限时不下单。"""
    can_sell = pd.DataFrame(True, index=INDEX, columns=COLUMNS)
    can_sell.loc[INDEX[1], "SH600000"] = False  # INDEX[2] 的再平衡一侧受限
    data = panel(can_sell_open_next=can_sell)
    targets = _targets([[np.nan, np.nan], [0.5, np.nan], [0.5, np.nan], [0.5, np.nan]])
    filtered, _, _ = apply_rules(data, targets, ExecutionSpec())
    assert filtered.loc[INDEX[1], "SH600000"] == 0.5
    assert np.isnan(filtered.loc[INDEX[2], "SH600000"])
    # INDEX[2] 的信号日两侧都正常，再平衡可以提交
    assert filtered.loc[INDEX[3], "SH600000"] == 0.5


def test_flat_position_with_zero_target_is_not_submitted(panel):
    data = panel()
    targets = _targets([[np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
    filtered, rejection, _ = apply_rules(data, targets, ExecutionSpec())
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
    plan = build_order_plan(data, targets, ExecutionSpec(), probe)
    assert plan.plan_stats["probe_orders"] == 1
    assert plan.plan_stats["planned_orders"] == 1
    assert plan.plan_stats["rejected_orders"] == 0
    # 本来就空仓、目标也是 0，不需要下单
    assert np.isnan(plan.target_percent.loc[INDEX[1], "SZ000001"])

