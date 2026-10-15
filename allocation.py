"""把目标权重换成「一手整数倍」的目标股数。

最终一遍不在 VectorBT 内部按目标百分比换算下单量，而是由这里按执行日开盘时刻的
真实组合价值算出每只的目标股数：下单量天生是交易单位的整数倍，也不会出现
「现金不够时排在后面的格子被整笔裁掉」。

预算来自**基准遍**（带规则、带费用）的逐日现金与持股数：执行日开盘前
「现金 + 各持仓按执行价估值」就是本次调仓可用的总预算。探针一遍是免费的，
累计费用会让它的净值偏高，因此不能用它当预算。

目标金额不足一手的格子用预算内的取整余量补到一手；余量不够时保持 0 股并记为
``unaffordable``，不剔除也不超配。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.errors import ExecutionError

# 逐格分配状态：正常分配 / 余量补到一手 / 一手都装不下 / 缺执行价不下单 / 目标 0
STATUS_ALLOCATED = "allocated"
STATUS_BUMPED = "bumped"
STATUS_UNAFFORDABLE = "unaffordable"
STATUS_NO_PRICE = "no_price"
STATUS_FLAT = "flat"

ALLOCATION_COLUMNS = [
    "trade_date",
    "stock_code",
    "target_weight",
    "price",
    "baseline_shares",
    "allocated_shares",
    "status",
]

# 分配统计的键名，引擎按它把「资金分配」阶段的进度事件单独报出来
ALLOCATION_STAT_KEYS = (
    "allocated_cells",
    "bumped_one_lot_cells",
    "unaffordable_lot_cells",
    "no_price_cells",
)


def execution_price(open_price: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """给出执行价宽表：用开盘价，停牌（开盘价为空）时沿用前收盘。

    参数:
        open_price: 执行日开盘价宽表。
        valuation: 逐日估值价（前收盘 ``ffill``）。
    """
    price = open_price.astype("float64")
    return price.where(price.notna(), valuation.reindex_like(price).astype("float64"))


def execution_budget(
    cash: pd.Series,
    assets: pd.DataFrame,
    price: pd.DataFrame,
    init_cash: float,
) -> pd.Series:
    """重建每个交易日开盘时刻的组合价值，作为本次调仓的预算。

    参数:
        cash: 基准遍逐日现金。
        assets: 基准遍逐日持股数宽表。
        price: 执行价宽表，来自 :func:`execution_price`。
        init_cash: 初始资金，用于首个交易日的预算。
    """
    if not cash.index.equals(assets.index):
        raise ExecutionError("基准遍的现金与持仓日期不一致，无法计算执行预算")
    index = pd.DatetimeIndex(cash.index)
    price = price.reindex(index=index, columns=assets.columns).astype("float64").fillna(0.0)
    previous_shares = assets.shift(1).fillna(0.0).to_numpy(dtype="float64")
    previous_cash = cash.shift(1).to_numpy(dtype="float64")
    if len(index):
        previous_shares[0] = 0.0
        previous_cash[0] = float(init_cash)
    values = (previous_shares * price.to_numpy(dtype="float64")).sum(axis=1)
    return pd.Series(previous_cash + values, index=index)


def allocate_shares(
    targets: pd.DataFrame,
    budget: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    lot_size: int,
    cash_buffer: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """按预算把目标权重换成目标股数，返回 ``(股数矩阵, 逐格明细, 方向矩阵, 统计)``。

    参数:
        targets: 规则过滤后的目标权重，``NaN`` 表示当天不下单、``0`` 表示清仓。
        budget: 逐日执行预算，来自 :func:`execution_budget`。
        price: 执行价宽表，来自 :func:`execution_price`。
        baseline_shares: 基准遍逐日持股数，用于给出实际下单方向。
        lot_size: 交易单位（股）。
        cash_buffer: 预留现金比例，取值 ``[0, 1)``。
    """
    if not 0.0 <= float(cash_buffer) < 1.0:
        raise ExecutionError(f"现金缓冲必须在 [0, 1) 之间：{cash_buffer}")
    index = pd.DatetimeIndex(targets.index)
    columns = targets.columns
    prices = price.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    base = baseline_shares.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    budgets = budget.reindex(index=index).to_numpy(dtype="float64")
    weights = targets.to_numpy(dtype="float64")
    shares = np.full(weights.shape, np.nan, dtype="float64")
    direction = np.zeros(weights.shape, dtype="int8")
    rows: list[dict] = []
    stats = {
        "allocated_cells": 0,
        "bumped_one_lot_cells": 0,
        "unaffordable_lot_cells": 0,
        "no_price_cells": 0,
    }
    for row in np.flatnonzero(~np.isnan(weights).all(axis=1)):
        cells = np.flatnonzero(~np.isnan(weights[row]))
        if not len(cells):
            continue
        held = np.nan_to_num(base[row, cells], nan=0.0)
        # 被规则拦下的持仓当天卖不掉，它的市值不能算进可动用现金，否则平价进货会买空
        stranded = float((np.nan_to_num(base[row], nan=0.0) * np.nan_to_num(prices[row], nan=0.0)).sum())
        available_cash = max(budgets[row] - stranded, 0.0)
        tradable_value = float((held * np.nan_to_num(prices[row, cells], nan=0.0)).sum())
        usable = (available_cash + tradable_value) * (1.0 - float(cash_buffer))
        if not np.isfinite(usable) or usable <= 0:
            usable = 0.0
        cell_price = prices[row, cells]
        tradable = np.isfinite(cell_price) & (cell_price > 0)
        cell_weight = weights[row, cells]
        positive = tradable & (cell_weight > 0)
        ideal = np.where(positive, usable * cell_weight, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ideal = np.where(tradable & (cell_price > 0), ideal / cell_price, 0.0)
        # 空仓向下取整到手；已有持仓按「差够一手才动」调整，避免预算微动带来的来回调仓
        opened = np.where(np.isfinite(ideal), np.floor(ideal / lot_size), 0.0) * lot_size
        kept_lots = np.trunc((ideal - held) / lot_size)
        cell_shares = np.where(held > 0, held + np.nan_to_num(kept_lots) * lot_size, opened)
        cell_shares = np.where(tradable, np.maximum(cell_shares, 0.0), 0.0)
        bumpable = positive & (cell_shares == 0)
        used = float((cell_shares * np.where(tradable, cell_price, 0.0)).sum())
        leftover = max(usable - used, 0.0)
        # 余量补一手：按一手金额从小到大，用预算内的取整余量把没建仓的格子补到一手
        bump = np.zeros(len(cells), dtype=bool)
        for cell in np.flatnonzero(bumpable)[np.argsort(
            lot_size * cell_price[bumpable], kind="stable"
        )]:
            lot_value = lot_size * cell_price[cell]
            if lot_value <= leftover:
                cell_shares[cell] = lot_size
                leftover -= lot_value
                bump[cell] = True
        shares[row, cells[tradable]] = cell_shares[tradable]
        direction[row, cells[tradable]] = np.sign(
            cell_shares[tradable] - held[tradable]
        ).astype("int8")
        stats["allocated_cells"] += int((cell_shares > 0).sum())
        stats["bumped_one_lot_cells"] += int(bump.sum())
        stats["unaffordable_lot_cells"] += int((bumpable & ~bump).sum())
        stats["no_price_cells"] += int((~tradable).sum())
        for position, cell in enumerate(cells):
            if not tradable[position]:
                status = STATUS_NO_PRICE
            elif cell_weight[position] <= 0:
                status = STATUS_FLAT
            elif bump[position]:
                status = STATUS_BUMPED
            elif cell_shares[position] > 0:
                status = STATUS_ALLOCATED
            else:
                status = STATUS_UNAFFORDABLE
            rows.append(
                {
                    "trade_date": index[row],
                    "stock_code": columns[cell],
                    "target_weight": float(cell_weight[position]),
                    "price": float(cell_price[position]),
                    "baseline_shares": float(held[position]),
                    "allocated_shares": float(cell_shares[position]),
                    "status": status,
                }
            )
    records = (
        pd.DataFrame(rows, columns=ALLOCATION_COLUMNS)
        if rows
        else pd.DataFrame(columns=ALLOCATION_COLUMNS)
    )
    return (
        pd.DataFrame(shares, index=index, columns=columns),
        records,
        direction,
        stats,
    )
