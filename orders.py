"""把目标权重转换成 VectorBT 的订单输入和规则拒单记录。

这里不维护现金账本：下单量由 VectorBT 按目标百分比换算，规则层只负责
「哪些格子不许下单」和「每笔该收多少费用」。

方向判断用一个**持仓状态机**，只看调仓目标和可交易性，不看现金：

- 目标是 0 且状态为持有 → 卖出；卖出被拦下时状态保持持有，下次调仓继续卖；
- 目标是正数且状态为空仓 → 买入；买入被拦下时状态保持空仓，下次调仓继续买；
- 目标是正数且状态为持有 → 与上一个目标比较：变大算买入、变小算卖出、
  完全一致算「再平衡」（方向未知，两侧任一被封就整格不下单）；

状态机保证被规则拦下的交易不会永久丢失：这一点很关键，否则一次停牌错过的
卖出会让这只股票一直留在组合里。

佣金、卖出印花税和最低佣金仍然按一遍**探针运行**（同目标、无费用）的真实
成交额校准；探针没有记录的格子用保守费率兜底，笔数计入报告。

T+1 在本执行口径下结构性满足：订单最早在信号日次一交易日开盘执行，
同一天不会对同一只股票既买又卖，因此不单独产生 T+1 拒单。

退市持仓单独处理：股票最后一条行情之后价格为空，卖出永远无法成交、仓位会永久留在
组合里，因此对区间内退市的股票在它最后一条有效行情日强制按开盘价卖出（见
``liquidation_cells``），不受停牌与涨跌停规则约束。

最终一遍的下单量不是目标百分比，而是由 :mod:`backtest.allocation` 按执行日开盘时刻的
真实组合价值算出的**目标股数**：这样下单量天生是交易单位整数倍，也不会因为现金不足
把排在后面的格子整笔裁掉。基准遍的费率矩阵由 :func:`baseline_inputs` 提供，它不依赖
分配结果，因此基准遍可以先于分配运行。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.allocation import allocate_shares, execution_price
from backtest.data import MarketData
from backtest.specs import CostSpec, ExecutionSpec

REJECTION_COLUMNS = ["trade_date", "stock_code", "reason"]
PROBE_COLUMNS = ["trade_date", "stock_code", "side", "value"]
LIQUIDATION_COLUMNS = ["trade_date", "stock_code", "price", "delist_date"]

# 目标权重与上一个目标视为一致时的比较容差
WEIGHT_EPS = 1e-12


@dataclass(frozen=True)
class OrderPlan:
    """VectorBT 订单矩阵、规则拒单与计划期统计。

    ``shares`` 是最终一遍实际提交的目标股数矩阵，``allocation`` 是它的逐格明细。
    """

    target_percent: pd.DataFrame
    shares: pd.DataFrame
    estimate_side: np.ndarray
    fees: pd.DataFrame
    fixed_fees: pd.DataFrame
    rejection: pd.DataFrame
    direction: np.ndarray
    liquidations: pd.DataFrame
    allocation: pd.DataFrame
    plan_stats: dict


def execution_targets(target_weights: pd.DataFrame) -> pd.DataFrame:
    """把信号日目标权重平移到下一交易日执行。

    参数:
        target_weights: 信号日目标权重，``NaN`` 表示当天不调仓。
    """
    return target_weights.shift(1)


def probe_matrices(
    probe: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """把探针成交明细铺成与面板同形的方向矩阵和成交额矩阵。

    参数:
        probe: 规范化成交明细，含 ``trade_date``、``stock_code``、``side``、
            ``value`` 四列；``side`` 为 1（买）或 -1（卖）。
        index: 面板日期索引。
        columns: 面板股票列。
    """
    side = np.zeros((len(index), len(columns)), dtype="int8")
    value = np.zeros((len(index), len(columns)), dtype="float32")
    if probe.empty:
        return side, value
    rows = index.get_indexer(pd.DatetimeIndex(probe["trade_date"]))
    cols = columns.get_indexer(probe["stock_code"])
    valid = (rows >= 0) & (cols >= 0)
    side[rows[valid], cols[valid]] = probe["side"].to_numpy()[valid]
    value[rows[valid], cols[valid]] = probe["value"].to_numpy()[valid]
    return side, value


def _shifted_flags(frame: pd.DataFrame) -> np.ndarray:
    """把「下一交易日」口径的标记移到执行日。

    第一行没有前一交易日，按「不可交易」处理。
    """
    values = frame.to_numpy(dtype=bool)
    shifted = np.zeros_like(values)
    shifted[1:] = values[:-1]
    return shifted


def _rejection_rows(mask: np.ndarray, index: pd.Index, columns: pd.Index, reason: str) -> pd.DataFrame:
    """把布尔掩码转成拒单明细。"""
    rows, cols = np.where(mask)
    if not len(rows):
        return pd.DataFrame(columns=REJECTION_COLUMNS)
    return pd.DataFrame({"trade_date": index[rows], "stock_code": columns[cols], "reason": reason})


def liquidation_cells(data: MarketData) -> pd.DataFrame:
    """列出退市持仓的强制清算格。

    对回测区间内退市的股票，取它在面板内最后一条有效开盘价所在日作为强卖格；
    最后一条行情日就是面板最后一天的股票不强卖（它的处置属于区间末的估值口径）。

    参数:
        data: 回测输入面板。
    """
    if data.delist_date is None:
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
    index = data.index
    if not len(index):
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
    delisted = pd.to_datetime(data.delist_date.reindex(data.columns))
    in_window = delisted.notna().to_numpy() & (delisted.to_numpy() <= index[-1].to_datetime64())
    prices = data.open.to_numpy(dtype="float64")
    has_price = ~np.isnan(prices)
    last_row = np.where(
        has_price.any(axis=0), len(index) - 1 - np.argmax(has_price[::-1], axis=0), -1
    )
    picked = np.flatnonzero(in_window & (last_row >= 0) & (last_row < len(index) - 1))
    if not len(picked):
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
    rows = last_row[picked]
    frame = pd.DataFrame(
        {
            "trade_date": index[rows],
            "stock_code": data.columns[picked],
            "price": prices[rows, picked],
            "delist_date": delisted.to_numpy()[picked],
        }
    )
    return frame.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _liquidation_positions(
    frame: pd.DataFrame | None, index: pd.Index, columns: pd.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把清算明细的日期与股票代码换算成面板下标。"""
    if frame is None or frame.empty:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=bool)
    rows = index.get_indexer(pd.DatetimeIndex(frame["trade_date"]))
    cols = columns.get_indexer(frame["stock_code"])
    return rows, cols, (rows >= 0) & (cols >= 0)


def _liquidation_matrix(
    frame: pd.DataFrame | None, index: pd.Index, columns: pd.Index
) -> np.ndarray:
    """把清算明细铺成与面板同形的布尔矩阵。"""
    mask = np.zeros((len(index), len(columns)), dtype=bool)
    if frame is None or frame.empty:
        return mask
    rows, cols, valid = _liquidation_positions(frame, index, columns)
    mask[rows[valid], cols[valid]] = True
    return mask


def _liquidation_rows(
    frame: pd.DataFrame | None, mask: np.ndarray, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """从清算明细里挑出真正提交的格子。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
    rows, cols, valid = _liquidation_positions(frame, index, columns)
    positions = np.flatnonzero(valid)
    positions = positions[mask[rows[valid], cols[valid]]]
    if not len(positions):
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
    return frame.iloc[positions].reset_index(drop=True)


def apply_rules(
    data: MarketData,
    targets: pd.DataFrame,
    execution: ExecutionSpec,
    liquidations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """按 A 股规则过滤订单，被拦住的格子置 ``NaN``（保持原仓位）。

    参数:
        data: 回测输入面板。
        targets: 已平移到执行日的目标权重。
        execution: 交易限制。
        liquidations: 退市强制清算格，来自 :func:`liquidation_cells`；这些格子绕过涨跌停与
            停牌规则，按最后一条有效行情的开盘价卖出。

    返回:
        ``(过滤后的目标权重, 拒单明细, 意图方向矩阵, 实际提交的清算明细)``；
        方向矩阵 1 买 / -1 卖 / 0 再平衡。
    """
    planned = np.array(targets.to_numpy(dtype="float64"), copy=True)
    force = _liquidation_matrix(liquidations, targets.index, targets.columns)
    # 清算格的目标一律写 0：状态机认为空仓时自然丢掉，认为持有时判为卖出
    planned[force] = 0.0
    buy_ok = _shifted_flags(data.can_buy_open_next)
    sell_ok = _shifted_flags(data.can_sell_open_next)
    suspended = data.is_suspended.to_numpy()
    keep = np.zeros(planned.shape, dtype=bool)
    direction = np.zeros(planned.shape, dtype="int8")
    blocked = {reason: np.zeros(planned.shape, dtype=bool) for reason in
               ("suspended", "limit_up_or_not_buyable", "limit_down_or_not_sellable")}
    held = np.zeros(planned.shape[1], dtype=bool)
    shadow = np.zeros(planned.shape[1], dtype="float64")
    for row in np.flatnonzero(~np.isnan(planned).all(axis=1)):
        desired = planned[row]
        has = ~np.isnan(desired)
        positive = has & (desired > 0)
        sell_intent = has & held & ((desired == 0) | (desired < shadow - WEIGHT_EPS))
        buy_intent = positive & (~held | (desired > shadow + WEIGHT_EPS))
        rebalance = positive & held & (np.abs(desired - shadow) <= WEIGHT_EPS)
        suspended_row = suspended[row]
        # 停牌当天一律不下单：价格为空，VectorBT 也不会成交，先按规则记下来
        stopped = (sell_intent | buy_intent | rebalance) & suspended_row & execution.reject_suspended
        stopped_buy = buy_intent & ~suspended_row & ~buy_ok[row] & execution.reject_limit_up_buy
        stopped_sell = sell_intent & ~suspended_row & ~sell_ok[row] & execution.reject_limit_down_sell
        # 方向未知的再平衡：买卖两侧任一被封就不下单，避免误触涨停买入/跌停卖出
        stopped_rebalance = rebalance & ~suspended_row & (~buy_ok[row] | ~sell_ok[row])
        keep_sell = sell_intent & ~stopped_sell & ~stopped
        keep_buy = buy_intent & ~stopped_buy & ~stopped
        keep_rebalance = rebalance & ~stopped & ~stopped_rebalance
        # 退市强卖只看状态机是否认为持仓，不受停牌与涨跌停约束
        forced_cell = force[row] & held
        keep[row] = keep_sell | keep_buy | keep_rebalance | forced_cell
        direction[row] = np.where(keep_buy, 1, np.where(keep_sell | forced_cell, -1, 0))
        blocked["suspended"][row] = stopped & ~forced_cell
        blocked["limit_up_or_not_buyable"][row] = stopped_buy & ~forced_cell
        blocked["limit_down_or_not_sellable"][row] = stopped_sell & ~forced_cell
        # 被拦下的格子状态不变，下一次调仓会重新尝试
        held[keep_sell | forced_cell] = False
        held[keep_buy | keep_rebalance] = True
        shadow[keep_sell | forced_cell] = 0.0
        shadow[keep_buy | keep_rebalance] = desired[keep_buy | keep_rebalance]

    filtered = pd.DataFrame(
        np.where(keep, planned, np.nan), index=targets.index, columns=targets.columns
    )
    frames = [
        _rejection_rows(blocked[reason], targets.index, targets.columns, reason)
        for reason in ("suspended", "limit_up_or_not_buyable", "limit_down_or_not_sellable")
    ]
    frames = [frame for frame in frames if len(frame)]
    rejection = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=REJECTION_COLUMNS)
    )
    if len(rejection):
        rejection = rejection.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    executed = _liquidation_rows(liquidations, keep & force, targets.index, targets.columns)
    return filtered, rejection, direction, executed


def fee_matrices(
    index: pd.Index,
    columns: pd.Index,
    keep: pd.DataFrame,
    side: np.ndarray,
    value: np.ndarray,
    direction: np.ndarray,
    costs: CostSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按成交估计生成逐格费率与最低佣金差额。

    VectorBT 的费率是「日期 × 股票」单元格口径，不区分买卖；这里用成交估计的
    方向把卖出印花税只加在卖出格子上，最低佣金用 ``fixed_fees`` 补差额。

    参数:
        index: 面板日期索引。
        columns: 面板股票列。
        keep: 最终会下单的布尔矩阵。
        side: 成交估计方向矩阵，1 买 / -1 卖 / 0 没有估计。
        value: 成交估计金额矩阵。
        direction: 意图方向矩阵，没有成交估计时用它判断买卖方向。
        costs: 交易成本参数。
    """
    # 没有成交估计的格子用意图方向判断印花税；成交额未知时只收佣金、不加最低佣金差额
    known = side != 0
    sell = np.where(known, side < 0, direction < 0)
    rate = np.where(sell, costs.commission_rate + costs.stamp_tax_rate, costs.commission_rate)
    kept = keep.to_numpy()
    fees = np.where(kept, rate, 0.0).astype("float32")
    deficiency = np.maximum(0.0, costs.minimum_commission - costs.commission_rate * value)
    fixed = np.where(known & kept, deficiency, 0.0).astype("float32")
    return (
        pd.DataFrame(fees, index=index, columns=columns),
        pd.DataFrame(fixed, index=index, columns=columns),
    )


def baseline_inputs(
    data: MarketData,
    targets: pd.DataFrame,
    execution: ExecutionSpec,
    probe: pd.DataFrame,
    costs: CostSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """给出基准遍需要的输入：规则过滤后的目标权重与逐格费用矩阵。

    基准遍只负责提供执行日的真实现金与持股数（分配预算的来源），费用按状态机
    意图方向估计即可，因此这一步不依赖分配结果，可以排在分配之前。

    参数:
        data: 回测输入面板。
        targets: 已平移到执行日的目标权重。
        execution: 交易限制。
        probe: 探针成交明细。
        costs: 交易成本参数。
    """
    costs = costs or CostSpec()
    side, value = probe_matrices(probe, targets.index, targets.columns)
    filtered, _, intent, _ = apply_rules(data, targets, execution, liquidation_cells(data))
    fees, fixed_fees = fee_matrices(
        targets.index, targets.columns, filtered.notna(), side, value, intent, costs
    )
    return filtered, fees, fixed_fees


def build_order_plan(
    data: MarketData,
    targets: pd.DataFrame,
    execution: ExecutionSpec,
    probe: pd.DataFrame,
    budget: pd.Series,
    baseline_shares: pd.DataFrame,
    costs: CostSpec | None = None,
) -> OrderPlan:
    """把目标权重、探针成交明细与执行预算变成最终订单计划。

    参数:
        data: 回测输入面板。
        targets: 已平移到执行日的目标权重。
        execution: 交易限制。
        probe: 探针成交明细，含 ``trade_date``、``stock_code``、``side``、``value``。
        budget: 逐日执行预算，来自 :func:`backtest.allocation.execution_budget`。
        baseline_shares: 基准遍逐日持股数，用于给出实际下单方向。
        costs: 交易成本参数。
    """
    costs = costs or CostSpec()
    side, value = probe_matrices(probe, targets.index, targets.columns)
    liquidations = liquidation_cells(data)
    filtered, rejection, direction, executed = apply_rules(data, targets, execution, liquidations)
    price = execution_price(data.open, data.close.ffill(axis=0))
    shares, allocation, allocated_direction, allocation_stats = allocate_shares(
        filtered, budget, price, baseline_shares, execution.lot_size, execution.cash_buffer
    )
    # 成交估计：分配出的股数差最贴近最终一遍的真实成交额（基准遍与最终遍的持仓
    # 只差一个取整范围）；没有估计的格子退回探针成交额，两者都没有就按意图方向计费。
    baseline_values = (
        baseline_shares.reindex(index=targets.index, columns=targets.columns)
        .to_numpy(dtype="float64")
    )
    prices = price.reindex(index=targets.index, columns=targets.columns).to_numpy(dtype="float64")
    delta = shares.to_numpy(dtype="float64") - np.nan_to_num(baseline_values, nan=0.0)
    estimated_side = np.where(np.isfinite(delta), np.sign(delta), 0.0).astype("int8")
    # 估算成交额时按买卖方向带上滑点，和 VectorBT 的实际成交价一致，最低佣金差额才算得准
    slippage = float(costs.slippage)
    traded_price = np.where(estimated_side > 0, prices * (1.0 + slippage), prices * (1.0 - slippage))
    estimated_value = np.where(np.isfinite(delta), np.abs(delta) * traded_price, np.nan)
    estimated_value = np.nan_to_num(estimated_value, nan=0.0)
    unknown = estimated_side == 0
    side_source = np.where(unknown, side, estimated_side).astype("int8")
    value_source = np.where(unknown, value, estimated_value).astype("float64")
    mixed = np.where(allocated_direction != 0, allocated_direction, direction).astype("int8")
    fees, fixed_fees = fee_matrices(
        targets.index, targets.columns, filtered.notna(), side_source, value_source, mixed, costs
    )
    plan_stats = {
        "probe_orders": int((side != 0).sum()),
        "probe_buys": int((side > 0).sum()),
        "probe_sells": int((side < 0).sum()),
        "planned_orders": int(filtered.notna().to_numpy().sum()),
        "rebalance_orders": int(((direction == 0) & filtered.notna().to_numpy()).sum()),
        "rejected_orders": int(len(rejection)),
        "forced_liquidations": int(len(executed)),
        **allocation_stats,
    }
    return OrderPlan(
        target_percent=filtered,
        shares=shares,
        estimate_side=side_source,
        fees=fees,
        fixed_fees=fixed_fees,
        rejection=rejection,
        direction=direction,
        liquidations=executed,
        allocation=allocation,
        plan_stats=plan_stats,
    )

