"""把目标权重换成符合板块申报规则的目标股数。

最终一遍不在 VectorBT 内部按目标百分比换算下单量，而是由这里按执行日开盘时刻的
真实组合价值算出每只的目标股数：下单量天生是交易单位的整数倍，也不会出现
「现金不够时排在后面的格子被整笔裁掉」。

预算来自**基准遍**（带规则、带费用）的逐日现金与持股数：执行日开盘前
「现金 + 各持仓按执行价估值」就是本次调仓可用的总预算。探针一遍是免费的，
累计费用会让它的净值偏高，因此不能用它当预算。

目标权重按这份总预算解释（再乘 ``1 - cash_buffer``）：重试日只有少数格子下单，
它们的权重仍然是「占整个组合的比例」，不能拿这些格子自身的市值当基准。

目标金额不足最小申报量的格子用预算内的取整余量补到最小申报量；余量不够时保持 0 股并记为
``unaffordable``，不剔除也不超配。

买单还受资金约束：本行买单总额不能超过「开盘现金 + 本行卖出所得」，超了就等比缩小，
被缩小的行记 ``cash_shrunk_cells``；原始买单产生实际缺口时记为 ``cash_limited``，经
:func:`cash_carry` 变成下一交易日的补回事件。已有补回再次缺现金只保留
``cash_shrunk_shares`` 并延续原事件，不重复创建补回事件。普通申报单位取整造成的股数差
不会进入现金补回队列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.errors import ExecutionError
from backtest.costs import CostSpec, exchange_for
from backtest.trading_rules import BoardRule, rule_for

# 逐格分配状态：正常分配 / 余量补到最小申报量 / 最小申报量装不下 / 缺执行价不下单 / 目标 0
STATUS_ALLOCATED = "allocated"
STATUS_BUMPED = "bumped"
STATUS_UNAFFORDABLE = "unaffordable"
STATUS_NO_PRICE = "no_price"
STATUS_FLAT = "flat"
STATUS_BELOW_MIN_ORDER = "below_min_order"

ALLOCATION_COLUMNS = [
    "trade_date",
    "stock_code",
    "target_weight",
    "price",
    "baseline_shares",
    "allocated_shares",
    "board",
    "min_buy_shares",
    "order_shares",
    "order_valid",
    "status",
    "cash_limited",
    "cash_shrunk_shares",
    "capacity_limit_value",
    "capacity_requested_value",
    "capacity_executed_value",
    "capacity_limited",
    "capacity_unavailable",
    "capacity_shortfall_shares",
]

# 分配统计的键名，引擎按它把「资金分配」阶段的进度事件单独报出来
ALLOCATION_STAT_KEYS = (
    "allocated_cells",
    "bumped_min_order_cells",
    "unaffordable_min_order_cells",
    "below_min_order_cells",
    "no_price_cells",
    "cash_shrunk_cells",
    "cash_limited_cells",
    "cash_limited_value",
    "liquidity_limited_cells",
    "liquidity_limited_value",
    "liquidity_unavailable_cells",
    "liquidity_shortfall_shares",
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


def opening_budget(
    opening_cash: pd.Series,
    opening_assets: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.Series:
    """按账本给出的开盘前现金与持股计算当日组合价值。

    参数:
        opening_cash: 当日开盘可用现金。
        opening_assets: 当日开盘前持股，已经包含除权送转。
        price: 执行价宽表。
    """
    if not opening_cash.index.equals(opening_assets.index):
        raise ExecutionError("开盘现金与持股日期不一致")
    values = opening_assets.reindex(
        index=opening_cash.index, columns=opening_assets.columns
    ).astype("float64")
    prices = price.reindex(
        index=opening_cash.index, columns=opening_assets.columns
    ).astype("float64").fillna(0.0)
    return opening_cash.astype("float64") + (values * prices).sum(axis=1)


def _rules_for(columns: pd.Index, boards: pd.Series) -> list[BoardRule]:
    """按股票列整理板块规则，缺板块或未知板块直接报错。"""
    if not isinstance(boards, pd.Series):
        raise ExecutionError("交易单位需要以股票代码为索引的板块序列")
    aligned = boards.reindex(columns)
    missing = aligned.isna() | (aligned.astype(str).str.strip() == "")
    if missing.any():
        sample = [str(item) for item in columns[missing.to_numpy()][:5]]
        raise ExecutionError(f"交易单位缺少股票板块：{sample}")
    return [rule_for(str(item)) for item in aligned]


def allocate_shares(
    targets: pd.DataFrame,
    budget: pd.Series,
    open_cash: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    boards: pd.Series,
    cash_buffer: float,
    carry: pd.DataFrame | None = None,
    costs: CostSpec | None = None,
    amount_cny: pd.DataFrame | None = None,
    max_participation_rate: float | None = None,
    opening_shares: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """按预算把目标权重换成目标股数，返回 ``(股数矩阵, 逐格明细, 方向矩阵, 统计)``。

    参数:
        targets: 规则过滤后的目标权重，``NaN`` 表示当天不下单、``0`` 表示清仓。
        budget: 逐日执行预算，来自 :func:`execution_budget`。
        open_cash: 逐日开盘可用现金，来自 :func:`execution_cash`。
        price: 执行价宽表，来自 :func:`execution_price`。
        baseline_shares: 基准遍逐日持股数，用于给出实际下单方向。
        boards: 以股票代码为索引的板块序列，用于解析交易单位。
        cash_buffer: 预留现金比例，取值 ``[0, 1)``。
        carry: 本轮已有现金补回目标；重试再次缺现金时继续传播，不新建补回事件。
        costs: 最终成交使用的佣金、印花税和滑点口径；None 表示零成本。
        amount_cny: 执行日未复权成交额；None 表示不启用成交额参与率。
        max_participation_rate: 单票成交额参与率；None 表示关闭。
    """
    if not 0.0 <= float(cash_buffer) < 1.0:
        raise ExecutionError(f"现金缓冲必须在 [0, 1) 之间：{cash_buffer}")
    if max_participation_rate is not None and amount_cny is None:
        raise ExecutionError("启用成交额参与率时缺少未复权成交额面板")
    costs = costs or CostSpec.zero()
    index = pd.DatetimeIndex(targets.index)
    columns = targets.columns
    rules = _rules_for(columns, boards)
    weights = targets.to_numpy(dtype="float64")
    prices = price.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    base = baseline_shares.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    budgets = budget.reindex(index=index).to_numpy(dtype="float64")
    cashes = open_cash.reindex(index=index).to_numpy(dtype="float64")
    amounts = (
        None
        if amount_cny is None
        else amount_cny.reindex(index=index, columns=columns).to_numpy(dtype="float64")
    )
    # 执行日开盘前持股已经包含当日公司行动；没有显式传入时退回旧的平移口径。
    if opening_shares is None:
        previous = np.zeros_like(np.nan_to_num(base, nan=0.0))
        previous[1:] = np.nan_to_num(base, nan=0.0)[:-1]
        pre_trade = np.nan_to_num(base, nan=0.0)
    else:
        pre_trade = opening_shares.reindex(index=index, columns=columns).fillna(0.0).to_numpy(
            dtype="float64"
        )
        previous = pre_trade
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
        "bumped_min_order_cells": 0,
        "unaffordable_min_order_cells": 0,
        "below_min_order_cells": 0,
        "no_price_cells": 0,
        "cash_shrunk_cells": 0,
        "cash_limited_cells": 0,
        "cash_limited_value": 0.0,
        "liquidity_limited_cells": 0,
        "liquidity_limited_value": 0.0,
        "liquidity_unavailable_cells": 0,
        "liquidity_shortfall_shares": 0.0,
    }
    for row in np.flatnonzero(~np.isnan(weights).all(axis=1)):
        cells = np.flatnonzero(~np.isnan(weights[row]))
        if not len(cells):
            continue
        held = np.nan_to_num(pre_trade[row, cells], nan=0.0)
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
        # 空仓向下取整到最小申报量；已有持仓只在差额跨过合法申报单位时才调整。
        cell_shares = np.zeros(len(cells), dtype="float64")
        below_min = np.zeros(len(cells), dtype=bool)
        for position in range(len(cells)):
            if not tradable[position] or cell_weight[position] <= 0:
                continue
            rule = rules[cells[position]]
            cell_shares[position] = rule.normalize_target(held[position], ideal[position])
            below_min[position] = (
                held[position] > 0
                and abs(cell_shares[position] - held[position]) < 0.5
                and abs(ideal[position] - held[position]) >= 0.5
            )
        # 现金约束：本行买单总额不能超过「开盘现金 + 本行卖出所得」，超了就等比缩小买单。
        # 口径用执行日开盘时刻的真实持股（上一交易日收盘），被规则拦下的持仓卖不掉，
        # 它的市值换不到现金，只能少买。
        tradable_price = np.where(tradable, cell_price, 0.0)
        cell_codes = np.array(columns[cells], dtype=object)
        cell_boards = np.array([rules[cell].board for cell in cells], dtype=object)
        cell_exchanges = np.array(
            [
                exchange_for(code, board)
                for code, board in zip(cell_codes, cell_boards)
            ],
            dtype=object,
        )
        previous_held = previous[row, cells]
        previous_value = previous_held * tradable_price
        target_value = cell_shares * tradable_price
        buy_delta = np.maximum(cell_shares - previous_held, 0.0)
        sell_delta = np.maximum(previous_held - cell_shares, 0.0)

        def charges_for(delta: np.ndarray, sell: bool = False):
            active = delta > 0.0
            directions = np.where(active, -1 if sell else 1, 0).astype("int8")
            return costs.charge_arrays(
                index[row],
                directions,
                np.abs(delta),
                tradable_price,
                cell_codes,
                cell_boards,
                cell_exchanges,
            )

        def buy_cost(delta: np.ndarray) -> np.ndarray:
            charge = charges_for(delta)
            return np.where(delta > 0.0, charge.fill_amount + charge.fees, 0.0)

        def sell_proceeds(delta: np.ndarray) -> np.ndarray:
            charge = charges_for(delta, sell=True)
            return np.where(delta > 0.0, charge.fill_amount - charge.fees, 0.0)

        buy_need = float(buy_cost(buy_delta).sum())
        sell_funds = float(sell_proceeds(sell_delta).sum())
        funds = cash_open + sell_funds
        # 缩小前的目标股数，用来判断哪些买单被现金限制、差多少股
        intended = cell_shares.copy()
        buying = np.zeros(len(cells), dtype=bool)
        if buy_need > funds + 1e-9:
            scale = max(funds, 0.0) / buy_need
            buying = intended > previous_held + 0.5
            for position in np.flatnonzero(buying):
                scaled_target = previous_held[position] + buy_delta[position] * scale
                cell_shares[position] = rules[cells[position]].normalize_target(
                    previous_held[position], scaled_target
                )
            # 最低佣金是非线性的，比例缩放后仍可能差几元；逐档回退到真实现金可成交。
            while float(buy_cost(np.maximum(cell_shares - previous_held, 0.0)).sum()) > funds + 1e-9:
                candidates = np.flatnonzero((cell_shares - previous_held) >= 0.5)
                if not len(candidates):
                    break
                previous_targets = np.array(
                    [
                        rules[cells[position]].previous_buy_target(
                            previous_held[position], cell_shares[position]
                        )
                        for position in candidates
                    ],
                    dtype="float64",
                )
                marginal_delta = np.zeros(len(cells), dtype="float64")
                marginal_delta[candidates] = cell_shares[candidates] - previous_targets
                marginal = buy_cost(marginal_delta)
                selected = candidates[int(np.argmax(marginal))]
                cell_shares[selected] = rules[cells[selected]].previous_buy_target(
                    previous_held[selected], cell_shares[selected]
                )
            target_value = cell_shares * tradable_price
        bumpable = positive & (previous_held <= 0.5) & (cell_shares == 0)
        used = float(target_value.sum())
        # 补最小申报量的余量同样受资金约束：不超过「开盘现金 + 本行持仓市值」，也不超过可用预算。
        capacity = cash_open + float(previous_value.sum())
        leftover = max(min(usable, capacity) - used, 0.0)
        cash_left = max(
            funds - float(buy_cost(np.maximum(cell_shares - previous_held, 0.0)).sum()),
            0.0,
        )
        # 余量补最小申报量：按最小订单金额从小到大，用预算内的取整余量把没建仓的格子补上。
        bump = np.zeros(len(cells), dtype=bool)
        minimum_values = np.array(
            [
                rules[cells[position]].min_buy_shares * cell_price[position]
                if tradable[position] else np.inf
                for position in range(len(cells))
            ],
            dtype="float64",
        )
        for cell in np.flatnonzero(bumpable)[np.argsort(
            minimum_values[bumpable], kind="stable"
        )]:
            minimum_shares = rules[cells[cell]].min_buy_shares
            minimum_value = minimum_shares * cell_price[cell]
            minimum_cash = float(
                buy_cost(np.eye(1, len(cells), cell).ravel() * minimum_shares).sum()
            )
            if minimum_value <= leftover and minimum_cash <= cash_left + 1e-9:
                cell_shares[cell] = minimum_shares
                leftover -= minimum_value
                cash_left -= minimum_cash
                bump[cell] = True
        # 被现金缩小的买单：补最小申报量之后仍未达到缩小前的目标股数。
        shortfall = np.where(buying, intended - cell_shares, 0.0)
        shrunk = buying & (shortfall >= 0.5)
        cash_limited = shrunk & ~carried[row, cells]

        # 成交额参与率限制订单的持仓变化，而不是目标持仓总额。这样已有仓位的
        # 部分卖出和补回买入都能在下一交易日继续推进；退市强平由规则层显式豁免。
        capacity_limit = np.full(len(cells), np.nan, dtype="float64")
        capacity_unavailable = np.zeros(len(cells), dtype=bool)
        if max_participation_rate is not None and amounts is not None:
            raw_amount = amounts[row, cells]
            valid_amount = np.isfinite(raw_amount) & (raw_amount > 0)
            capacity_limit = np.where(
                valid_amount,
                raw_amount * float(max_participation_rate),
                0.0,
            )
            capacity_unavailable = ~valid_amount
            before_capacity = cell_shares.copy()
            delta = before_capacity - previous_held
            active_delta = np.abs(delta) >= 0.5
            capacity_unavailable &= active_delta
            max_shares = np.floor(
                np.divide(
                    capacity_limit,
                    np.maximum(tradable_price, 1e-12),
                    out=np.zeros_like(capacity_limit),
                    where=tradable_price > 0,
                )
            )
            capped_targets = np.array(
                [
                    rules[cells[position]].cap_target(
                        previous_held[position], before_capacity[position], max_shares[position]
                    )
                    if not capacity_unavailable[position]
                    else previous_held[position]
                    for position in range(len(cells))
                ],
                dtype="float64",
            )
            cell_shares = np.where(tradable, capped_targets, 0.0)
            capped_delta = cell_shares - previous_held
            capacity_shortfall = np.where(
                active_delta,
                np.maximum(np.abs(delta) - np.abs(capped_delta), 0.0),
                0.0,
            )
            capacity_limited = (
                active_delta
                & (capacity_shortfall >= 0.5)
            )
            capacity_limited &= ~capacity_unavailable
            capacity_requested_value = np.abs(delta) * tradable_price
            capacity_executed_value = np.abs(capped_delta) * tradable_price
        else:
            capacity_shortfall = np.zeros(len(cells), dtype="float64")
            capacity_limited = np.zeros(len(cells), dtype=bool)
            capacity_requested_value = np.zeros(len(cells), dtype="float64")
            capacity_executed_value = np.zeros(len(cells), dtype="float64")

        shares[row, cells[tradable]] = cell_shares[tradable]
        order_shares = cell_shares - held
        order_valid = np.array(
            [
                rules[cells[position]].is_valid_order(held[position], order_shares[position])
                for position in range(len(cells))
            ],
            dtype=bool,
        )
        if not order_valid.all():
            invalid = np.flatnonzero(~order_valid)
            details = [
                (
                    str(columns[cells[position]]),
                    float(held[position]),
                    float(order_shares[position]),
                    rules[cells[position]].board,
                )
                for position in invalid[:5]
            ]
            raise ExecutionError(f"资金分配生成了非法板块订单：{details}")
        direction[row, cells[tradable]] = np.sign(
            cell_shares[tradable] - held[tradable]
        ).astype("int8")
        stats["allocated_cells"] += int((cell_shares > 0).sum())
        stats["bumped_min_order_cells"] += int(bump.sum())
        stats["unaffordable_min_order_cells"] += int((bumpable & ~bump).sum())
        stats["below_min_order_cells"] += int((below_min & ~bump).sum())
        stats["no_price_cells"] += int((~tradable).sum())
        stats["cash_shrunk_cells"] += int(shrunk.sum())
        stats["cash_limited_cells"] += int(cash_limited.sum())
        stats["cash_limited_value"] += float(
            np.sum(np.where(cash_limited, shortfall * tradable_price, 0.0))
        )
        stats["liquidity_limited_cells"] += int(capacity_limited.sum())
        stats["liquidity_limited_value"] += float(
            np.sum(np.where(capacity_limited, capacity_shortfall * tradable_price, 0.0))
        )
        stats["liquidity_unavailable_cells"] += int(capacity_unavailable.sum())
        stats["liquidity_shortfall_shares"] += float(capacity_shortfall.sum())
        for position, cell in enumerate(cells):
            if not tradable[position]:
                status = STATUS_NO_PRICE
            elif capacity_unavailable[position]:
                status = "liquidity_unavailable"
            elif capacity_limited[position]:
                status = "liquidity_limited"
            elif cell_weight[position] <= 0:
                status = STATUS_FLAT
            elif bump[position]:
                status = STATUS_BUMPED
            elif below_min[position] and abs(order_shares[position]) < 0.5:
                status = STATUS_BELOW_MIN_ORDER
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
                    "board": rules[cell].board,
                    "min_buy_shares": int(rules[cell].min_buy_shares),
                    "order_shares": float(order_shares[position]),
                    "order_valid": bool(order_valid[position]),
                    "status": status,
                    "cash_limited": bool(cash_limited[position]),
                    "cash_shrunk_shares": float(shortfall[position]),
                    "capacity_limit_value": float(capacity_limit[position]),
                    "capacity_requested_value": float(capacity_requested_value[position]),
                    "capacity_executed_value": float(capacity_executed_value[position]),
                    "capacity_limited": bool(capacity_limited[position]),
                    "capacity_unavailable": bool(capacity_unavailable[position]),
                    "capacity_shortfall_shares": float(capacity_shortfall[position]),
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
    """推进现金或流动性补回状态，并产生下一交易日的补回目标。

    ``allocation`` 中的现金缩水和成交额限制都可能产生补回事件。原始调仓产生
    新事件；已有补回如果尚未完成则继续向后传播，完成后停止传播。这样不会把同一
    个事件在每次重试时重复登记，避免固定点在长链上来回摆动。

    参数:
        allocation: 逐格分配明细，含 ``trade_date``、``stock_code``、``target_weight``、
        ``cash_limited``、``capacity_limited`` 或 ``capacity_unavailable``；None 表示还没有分配结果。
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

    allocation_lookup: dict[tuple[pd.Timestamp, object], tuple[float, float, bool, float]] = {}
    if allocation is not None and not allocation.empty:
        for row in allocation.itertuples(index=False):
            shrunk = getattr(row, "cash_shrunk_shares", np.nan)
            if pd.isna(shrunk):
                shrunk = 1.0 if bool(row.cash_limited) else 0.0
            else:
                shrunk = float(shrunk)
            capacity_shortfall = getattr(row, "capacity_shortfall_shares", 0.0)
            capacity_shortfall = 0.0 if pd.isna(capacity_shortfall) else float(capacity_shortfall)
            allocation_lookup[(pd.Timestamp(row.trade_date), row.stock_code)] = (
                shrunk,
                capacity_shortfall,
                bool(getattr(row, "capacity_unavailable", False)),
                float(getattr(row, "allocated_shares", 0.0)),
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
            filled = status is not None and status[0] <= 0 and status[1] <= 0 and not status[2]
            if status is None or not filled:
                next_values[next_row, column] = previous_values[row, column]

    # 再把本轮新产生的现金缩水事件推到下一交易日；已有补回不重复起新事件。
    if allocation is not None and not allocation.empty:
        capacity_limited = (
            allocation["capacity_limited"].astype(bool)
            if "capacity_limited" in allocation
            else pd.Series(False, index=allocation.index)
        )
        capacity_unavailable = (
            allocation["capacity_unavailable"].astype(bool)
            if "capacity_unavailable" in allocation
            else pd.Series(False, index=allocation.index)
        )
        limited = allocation.loc[
            allocation["cash_limited"].astype(bool) | capacity_limited | capacity_unavailable
        ]
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
    cash_shrunk = (
        allocation["cash_shrunk_shares"].to_numpy(dtype="float64")
        if "cash_shrunk_shares" in allocation
        else allocation["cash_limited"].astype(bool).to_numpy(dtype="float64")
    )
    capacity_shortfall = (
        allocation["capacity_shortfall_shares"].to_numpy(dtype="float64")
        if "capacity_shortfall_shares" in allocation
        else np.zeros(len(allocation), dtype="float64")
    )
    unavailable = (
        allocation["capacity_unavailable"].astype(bool).to_numpy()
        if "capacity_unavailable" in allocation
        else np.zeros(len(allocation), dtype=bool)
    )
    filled = (
        (rows >= 0)
        & (columns >= 0)
        & (cash_shrunk <= 0)
        & (capacity_shortfall <= 0)
        & ~unavailable
    )
    return int(carried[rows[filled], columns[filled]].sum())


def liquidity_retry_filled(carry: pd.DataFrame, allocation: pd.DataFrame | None) -> int:
    """数出由成交额限制产生、随后真正完成的补回格数。"""
    if allocation is None or allocation.empty or "capacity_shortfall_shares" not in allocation:
        return 0
    carried = ~np.isnan(carry.to_numpy(dtype="float64"))
    rows = carry.index.get_indexer(pd.DatetimeIndex(allocation["trade_date"]))
    columns = carry.columns.get_indexer(allocation["stock_code"])
    capacity_shortfall = allocation["capacity_shortfall_shares"].to_numpy(dtype="float64")
    unavailable = (
        allocation["capacity_unavailable"].astype(bool).to_numpy()
        if "capacity_unavailable" in allocation
        else np.zeros(len(allocation), dtype=bool)
    )
    filled = (
        (rows >= 0)
        & (columns >= 0)
        & (capacity_shortfall <= 0)
        & ~unavailable
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
    status: dict[str, tuple[bool, float, float, bool]] = {}
    weights: dict[str, float] = {}
    if allocation is not None and not allocation.empty:
        for row in allocation.itertuples(index=False):
            if pd.Timestamp(row.trade_date) != last:
                continue
            shrunk = getattr(row, "cash_shrunk_shares", np.nan)
            shrunk = float(shrunk) if not pd.isna(shrunk) else (1.0 if row.cash_limited else 0.0)
            capacity_shortfall = float(getattr(row, "capacity_shortfall_shares", 0.0) or 0.0)
            unavailable = bool(getattr(row, "capacity_unavailable", False))
            status[row.stock_code] = (
                bool(row.cash_limited),
                shrunk,
                capacity_shortfall,
                unavailable,
            )
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
            filled = item is not None and item[1] <= 0 and item[2] <= 0 and not item[3]
            if not filled:
                out[column] = float(previous[column])
    # 末日新产生的现金缩水推到下一交易日；已在旧补回里的不重复起事件
    for column, item in status.items():
        if not (item[0] or item[2] > 0 or item[3]):
            continue
        if previous is not None and not pd.isna(previous.get(column, np.nan)):
            continue
        if column in weights:
            out[column] = weights[column]
    return out
