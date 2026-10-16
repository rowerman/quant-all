"""把目标权重换成「一手整数倍」的目标股数。

最终一遍不在 VectorBT 内部按目标百分比换算下单量，而是由这里按执行日开盘时刻的
真实组合价值算出每只的目标股数：下单量天生是交易单位的整数倍，也不会出现
「现金不够时排在后面的格子被整笔裁掉」。

预算来自**基准遍**（带规则、带费用）的逐日现金与持股数：执行日开盘前
「现金 + 各持仓按执行价估值」就是本次调仓可用的总预算。探针一遍是免费的，
累计费用会让它的净值偏高，因此不能用它当预算。

目标权重按这份总预算解释（再乘 ``1 - cash_buffer``）：重试日只有少数格子下单，
它们的权重仍然是「占整个组合的比例」，不能拿这些格子自身的市值当基准。

目标金额不足一手的格子用预算内的取整余量补到一手；余量不够时保持 0 股并记为
``unaffordable``，不剔除也不超配。

买单还受资金约束：本行买单总额不能超过「开盘现金 + 本行卖出所得」，超了就等比缩小，
被缩小的行记 ``cash_shrunk_cells``；原始买单产生实际缺口时记为 ``cash_limited``，经
:func:`cash_carry` 变成下一交易日的补回事件。已有补回再次缺现金只保留
``cash_shrunk_shares`` 并延续原事件，不重复创建补回事件。普通一手取整造成的股数差
不会进入现金补回队列。
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
    "cash_limited",
    "cash_shrunk_shares",
]

# 分配统计的键名，引擎按它把「资金分配」阶段的进度事件单独报出来
ALLOCATION_STAT_KEYS = (
    "allocated_cells",
    "bumped_one_lot_cells",
    "unaffordable_lot_cells",
    "no_price_cells",
    "cash_shrunk_cells",
    "cash_limited_cells",
    "cash_limited_value",
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


def execution_cash(cash: pd.Series, init_cash: float) -> pd.Series:
    """给出每个交易日开盘时刻可用的现金：上一交易日收盘现金，首日用初始资金。

    参数:
        cash: 基准遍逐日收盘现金。
        init_cash: 初始资金。
    """
    index = pd.DatetimeIndex(cash.index)
    values = cash.shift(1).to_numpy(dtype="float64", copy=True)
    if len(index):
        values[0] = float(init_cash)
    return pd.Series(values, index=index)


def allocate_shares(
    targets: pd.DataFrame,
    budget: pd.Series,
    open_cash: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    lot_size: int,
    cash_buffer: float,
    carry: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """按预算把目标权重换成目标股数，返回 ``(股数矩阵, 逐格明细, 方向矩阵, 统计)``。

    参数:
        targets: 规则过滤后的目标权重，``NaN`` 表示当天不下单、``0`` 表示清仓。
        budget: 逐日执行预算，来自 :func:`execution_budget`。
        open_cash: 逐日开盘可用现金，来自 :func:`execution_cash`。
        price: 执行价宽表，来自 :func:`execution_price`。
        baseline_shares: 基准遍逐日持股数，用于给出实际下单方向。
        lot_size: 交易单位（股）。
        cash_buffer: 预留现金比例，取值 ``[0, 1)``。
        carry: 本轮已有现金补回目标；重试再次缺现金时继续传播，不新建补回事件。
    """
    if not 0.0 <= float(cash_buffer) < 1.0:
        raise ExecutionError(f"现金缓冲必须在 [0, 1) 之间：{cash_buffer}")
    index = pd.DatetimeIndex(targets.index)
    columns = targets.columns
    prices = price.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    base = baseline_shares.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    budgets = budget.reindex(index=index).to_numpy(dtype="float64")
    cashes = open_cash.reindex(index=index).to_numpy(dtype="float64")
    # 上一交易日收盘持股 = 执行日开盘时刻的真实持股（首日为空仓）
    previous = np.zeros_like(np.nan_to_num(base, nan=0.0))
    previous[1:] = np.nan_to_num(base, nan=0.0)[:-1]
    weights = targets.to_numpy(dtype="float64")
    carried = (
        np.zeros(weights.shape, dtype=bool)
        if carry is None
        else carry.reindex_like(targets).notna().to_numpy(dtype=bool)
    )
    shares = np.full(weights.shape, np.nan, dtype="float64")
    direction = np.zeros(weights.shape, dtype="int8")
    rows: list[dict] = []
    stats = {
        "allocated_cells": 0,
        "bumped_one_lot_cells": 0,
        "unaffordable_lot_cells": 0,
        "no_price_cells": 0,
        "cash_shrunk_cells": 0,
        "cash_limited_cells": 0,
        "cash_limited_value": 0.0,
    }
    for row in np.flatnonzero(~np.isnan(weights).all(axis=1)):
        cells = np.flatnonzero(~np.isnan(weights[row]))
        if not len(cells):
            continue
        held = np.nan_to_num(base[row, cells], nan=0.0)
        # 目标权重按「执行日开盘时刻的组合价值」解释：重试日只有少数格子下单，
        # 若拿这些格子自己的市值当基准，补单量会被压成目标的一个零头。
        usable = budgets[row] * (1.0 - float(cash_buffer))
        if not np.isfinite(usable) or usable <= 0:
            usable = 0.0
        cash_open = max(float(cashes[row]), 0.0)
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
        # 现金约束：本行买单总额不能超过「开盘现金 + 本行卖出所得」，超了就等比缩小买单。
        # 口径用执行日开盘时刻的真实持股（上一交易日收盘），被规则拦下的持仓卖不掉，
        # 它的市值换不到现金，只能少买。
        tradable_price = np.where(tradable, cell_price, 0.0)
        previous_held = previous[row, cells]
        previous_value = previous_held * tradable_price
        target_value = cell_shares * tradable_price
        buy_need = float(np.sum(np.maximum(target_value - previous_value, 0.0)))
        sell_funds = float(np.sum(np.maximum(previous_value - target_value, 0.0)))
        funds = cash_open + sell_funds
        # 缩小前的目标股数，用来判断哪些买单被现金限制、差多少股
        intended = cell_shares.copy()
        buying = np.zeros(len(cells), dtype=bool)
        if buy_need > funds:
            scale = max(funds, 0.0) / buy_need
            buying = target_value > previous_value
            scaled = previous_value + (target_value - previous_value) * scale
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled_shares = np.floor(scaled / np.where(tradable, cell_price, 1.0) / lot_size) * lot_size
            cell_shares = np.where(buying, np.maximum(scaled_shares, 0.0), cell_shares)
            target_value = cell_shares * tradable_price
        bumpable = positive & (cell_shares == 0)
        used = float(target_value.sum())
        # 补一手的余量同样受资金约束：不超过「开盘现金 + 本行持仓市值」，也不超过可用预算
        capacity = cash_open + float(previous_value.sum())
        leftover = max(min(usable, capacity) - used, 0.0)
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
        # 被现金缩小的买单：补一手之后仍未达到缩小前的目标股数。股数总是交易单位的
        # 整数倍，因此用半个交易单位消除浮点误差；普通取整不会进入这里。
        shortfall = np.where(buying, intended - cell_shares, 0.0)
        shrunk = buying & (shortfall >= lot_size * 0.5)
        cash_limited = shrunk & ~carried[row, cells]
        shares[row, cells[tradable]] = cell_shares[tradable]
        direction[row, cells[tradable]] = np.sign(
            cell_shares[tradable] - held[tradable]
        ).astype("int8")
        stats["allocated_cells"] += int((cell_shares > 0).sum())
        stats["bumped_one_lot_cells"] += int(bump.sum())
        stats["unaffordable_lot_cells"] += int((bumpable & ~bump).sum())
        stats["no_price_cells"] += int((~tradable).sum())
        stats["cash_shrunk_cells"] += int(shrunk.sum())
        stats["cash_limited_cells"] += int(cash_limited.sum())
        stats["cash_limited_value"] += float(
            np.sum(np.where(cash_limited, shortfall * tradable_price, 0.0))
        )
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
                    "cash_limited": bool(cash_limited[position]),
                    "cash_shrunk_shares": float(shortfall[position]),
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


def cash_carry(
    allocation: pd.DataFrame | None,
    targets: pd.DataFrame,
    carry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """推进现金补回状态，并产生下一交易日的补回目标。

    ``allocation`` 中的现金缩水有两种来源：原始调仓和已有补回。原始调仓产生
    新的补回事件；已有补回如果尚未成交则继续向后传播，成交后停止传播。这样不会
    把同一个补回事件在每次重试时重复登记，避免固定点在长链上来回摆动。

    参数:
        allocation: 逐格分配明细，含 ``trade_date``、``stock_code``、``target_weight``、
            ``cash_limited``；None 表示还没有分配结果。
        targets: 执行日的原始目标权重矩阵（不含补回目标）。
        carry: 本轮传入的补回目标；用于判断 allocation 行是否属于已有补回事件。
    """
    next_carry = pd.DataFrame(
        np.nan, index=targets.index, columns=targets.columns, dtype="float64"
    )
    if not len(targets.index):
        return next_carry

    previous = None if carry is None else carry.reindex_like(targets).astype("float64")
    values = targets.to_numpy(dtype="float64")
    next_values = next_carry.to_numpy(dtype="float64")

    allocation_lookup: dict[tuple[pd.Timestamp, object], tuple[bool, float, float]] = {}
    if allocation is not None and not allocation.empty:
        for row in allocation.itertuples(index=False):
            cash_limited = bool(row.cash_limited)
            shrunk = getattr(row, "cash_shrunk_shares", np.nan)
            if pd.isna(shrunk):
                shrunk = 1.0 if cash_limited else 0.0
            else:
                shrunk = float(shrunk)
            allocation_lookup[(pd.Timestamp(row.trade_date), row.stock_code)] = (
                cash_limited,
                float(getattr(row, "allocated_shares", 0.0)),
                shrunk,
            )

    # 先延续本轮已经存在、但尚未真正补齐的补回目标。
    if previous is not None:
        previous_values = previous.to_numpy(dtype="float64")
        for row, column in zip(*np.where(~np.isnan(previous_values))):
            next_row = row + 1
            if next_row >= len(targets.index) or not np.isnan(values[next_row, column]):
                continue
            key = (pd.Timestamp(targets.index[row]), targets.columns[column])
            status = allocation_lookup.get(key)
            filled = status is not None and status[1] > 0 and status[2] <= 0
            if status is None or not filled:
                next_values[next_row, column] = previous_values[row, column]

    # 再把本轮新产生的现金缩水事件推到下一交易日；已有补回不重复起新事件。
    if allocation is not None and not allocation.empty:
        limited = allocation[allocation["cash_limited"].astype(bool)]
        for row in limited.itertuples(index=False):
            position = targets.index.get_loc(pd.Timestamp(row.trade_date))
            column = targets.columns.get_loc(row.stock_code)
            if previous is not None and not np.isnan(previous.iloc[position, column]):
                continue
            next_row = position + 1
            if next_row < len(targets.index) and np.isnan(values[next_row, column]):
                next_values[next_row, column] = float(row.target_weight)

    return pd.DataFrame(next_values, index=targets.index, columns=targets.columns)


def cash_retry_filled(carry: pd.DataFrame, allocation: pd.DataFrame | None) -> int:
    """数出补回目标真正补齐的格数。

    补齐指当天有分配明细、拿到了仓位，并且没有再被现金缩小；被规则拦下或者
    又被缩小的格子只是「尝试过」，要留在待重试状态里。

    参数:
        carry: 本轮使用的补回目标矩阵，来自 :func:`cash_carry`。
        allocation: 本轮最终一遍的逐格分配明细。
    """
    carried = ~np.isnan(carry.to_numpy(dtype="float64"))
    if allocation is None or allocation.empty or not carried.any():
        return 0
    rows = carry.index.get_indexer(pd.DatetimeIndex(allocation["trade_date"]))
    columns = carry.columns.get_indexer(allocation["stock_code"])
    shrunk = (
        allocation["cash_shrunk_shares"].to_numpy(dtype="float64")
        if "cash_shrunk_shares" in allocation
        else allocation["cash_limited"].astype(bool).to_numpy(dtype="float64")
    )
    filled = (
        (rows >= 0)
        & (columns >= 0)
        & (shrunk <= 0)
        & (allocation["allocated_shares"].to_numpy(dtype="float64") > 0)
    )
    return int(carried[rows[filled], columns[filled]].sum())


def extend_carry(carry: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """把补回目标延续到下一次真实调仓目标之前。

    现金没到账时每天都得试一次，但补回目标依赖分配结果、分配结果又依赖补回目标，
    一轮只能推后一个交易日。用「一直补到下次调仓」当第一轮的种子，可以让整条链
    一轮铺开，之后几轮再按分配结果收敛回真正需要补的日子。

    参数:
        carry: 由分配结果派生出的补回目标。
        targets: 执行日的原始目标权重矩阵（不含补回目标）。
    """
    values = targets.to_numpy(dtype="float64")
    extended = carry.to_numpy(dtype="float64").copy()
    for row in range(1, len(extended)):
        # 上一行还在补回、本行既没有真实目标也没有补回目标的格子：继续补
        moving = (
            ~np.isnan(extended[row - 1])
            & np.isnan(extended[row])
            & np.isnan(values[row])
        )
        extended[row] = np.where(moving, extended[row - 1], extended[row])
    return pd.DataFrame(extended, index=carry.index, columns=carry.columns)


def pending_carry(
    carry: pd.DataFrame | None,
    allocation: pd.DataFrame | None,
    targets: pd.DataFrame,
) -> pd.Series:
    """给出「最后一个交易日之后」仍要补回的目标权重。

    :func:`cash_carry` 只把补回目标推到下一交易日，而最后一天的下一日在面板之外，
    因此单独把「末日新产生的现金缩水」与「末日仍没成交的旧补回」取出来，供分块回测
    交接给下一块继续补回。

    参数:
        carry: 本轮的补回目标矩阵；None 表示没有补回。
        allocation: 逐格分配明细，含 ``trade_date``、``stock_code``、``cash_limited``、
            ``allocated_shares``、``cash_shrunk_shares``。
        targets: 执行日的原始目标权重矩阵。
    """
    out = pd.Series(np.nan, index=targets.columns, dtype="float64")
    if not len(targets.index):
        return out
    last = pd.Timestamp(targets.index[-1])
    status: dict[str, tuple[bool, float, float]] = {}
    weights: dict[str, float] = {}
    if allocation is not None and not allocation.empty:
        for row in allocation.itertuples(index=False):
            if pd.Timestamp(row.trade_date) != last:
                continue
            shrunk = getattr(row, "cash_shrunk_shares", np.nan)
            shrunk = float(shrunk) if not pd.isna(shrunk) else (1.0 if row.cash_limited else 0.0)
            status[row.stock_code] = (bool(row.cash_limited), float(row.allocated_shares), shrunk)
            weights[row.stock_code] = float(row.target_weight)
    previous = None
    if carry is not None and len(carry.index):
        last_carry = carry.iloc[-1].astype("float64")
        if last_carry.notna().any():
            previous = last_carry
    # 末日仍没成交的旧补回继续往后推
    if previous is not None:
        for column in previous.index[previous.notna()]:
            item = status.get(column)
            filled = item is not None and item[1] > 0 and item[2] <= 0
            if not filled:
                out[column] = float(previous[column])
    # 末日新产生的现金缩水推到下一交易日；已在旧补回里的不重复起事件
    for column, item in status.items():
        if not item[0]:
            continue
        if previous is not None and not pd.isna(previous.get(column, np.nan)):
            continue
        if column in weights:
            out[column] = weights[column]
    return out
