"""严格 T+1 订单守卫。

当前日频策略在收盘生成信号、次日开盘执行，同一天不会对同一只股票既买又卖，
因此结构上已经满足 T+1。本模块把规则显式落到订单层：任何卖出都不能超过
上一交易日收盘持股，供未来接入同日双向或动态订单时复用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_sells_to_prior_holdings(
    orders: pd.DataFrame,
    prior_holdings: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    enforce: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """把超过前收盘持股的卖出截断。

    参数:
        orders: 有符号订单矩阵，正数买入、负数卖出、NaN 表示不下单。
        prior_holdings: 同一时点的开盘前持股，通常来自上一交易日收盘持股。
        prices: 可选成交价矩阵，用于估算被拦金额。
        enforce: 是否启用严格 T+1；关闭时原样返回。

    返回:
        ``(截断后的订单, 被拦掩码, 统计)``。统计含被拦格数、股数和金额。
    """
    values = orders.to_numpy(dtype="float64", copy=True)
    holdings = prior_holdings.reindex_like(orders).to_numpy(dtype="float64")
    holdings = np.nan_to_num(holdings, nan=0.0, posinf=0.0, neginf=0.0)
    blocked = np.zeros(values.shape, dtype=bool)
    blocked_shares = 0.0
    excess = np.zeros(values.shape, dtype="float64")
    if enforce:
        sell = np.isfinite(values) & (values < 0)
        blocked = sell & (np.abs(values) > holdings)
        requested = np.where(blocked, np.abs(values), 0.0)
        allowed = np.where(blocked, holdings, 0.0)
        excess = np.maximum(requested - allowed, 0.0)
        blocked_shares = float(excess.sum())
        values[blocked] = -allowed[blocked]
    price_values = (
        np.zeros(values.shape, dtype="float64")
        if prices is None
        else np.nan_to_num(
            prices.reindex_like(orders).to_numpy(dtype="float64"),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    )
    blocked_value = float((excess * price_values).sum())
    stats = {
        "t_plus_1_blocked_cells": int(blocked.sum()),
        "t_plus_1_blocked_shares": blocked_shares,
        "t_plus_1_blocked_value": blocked_value,
    }
    return (
        pd.DataFrame(values, index=orders.index, columns=orders.columns),
        blocked,
        stats,
    )
