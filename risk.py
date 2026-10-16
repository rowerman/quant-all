"""回测目标权重的单股上限与单边换手约束。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.errors import ExecutionError
from backtest.specs import RiskSpec


def cap_single_weights(
    weights: pd.DataFrame,
    max_single_weight: float | None,
) -> tuple[pd.DataFrame, dict]:
    """封顶单股目标权重，并把可分配余量按原权重比例重新分配。

    参数:
        weights: 信号日目标权重，NaN 表示非调仓日。
        max_single_weight: 单股权重上限，None 表示关闭。
    """
    if max_single_weight is None:
        return weights, {
            "single_cap_days": 0,
            "single_cap_cells": 0,
            "single_cap_excess_weight": 0.0,
            "single_cap_residual_cash": 0.0,
            "single_cap_max_weight": 0.0,
        }
    cap = float(max_single_weight)
    values = weights.to_numpy(dtype="float64", copy=True)
    stats = {
        "single_cap_days": 0,
        "single_cap_cells": 0,
        "single_cap_excess_weight": 0.0,
        "single_cap_residual_cash": 0.0,
        "single_cap_max_weight": 0.0,
    }
    for row in range(values.shape[0]):
        active = np.isfinite(values[row]) & (values[row] > 0)
        if not active.any():
            continue
        original = values[row, active].copy()
        capped = original.copy()
        capped_cells = np.zeros(len(original), dtype=bool)
        for _ in range(len(original) + 1):
            over = capped > cap + 1e-12
            if not over.any():
                break
            excess = float((capped[over] - cap).sum())
            capped[over] = cap
            capped_cells |= over
            candidates = ~capped_cells & (capped < cap - 1e-12)
            if not candidates.any():
                stats["single_cap_residual_cash"] += excess
                break
            base = capped[candidates]
            share = base / base.sum() if base.sum() > 0 else np.full(candidates.sum(), 1.0 / candidates.sum())
            capped[candidates] += excess * share
        values[row, np.flatnonzero(active)] = capped
        stats["single_cap_max_weight"] = max(stats["single_cap_max_weight"], float(capped.max()))
        if capped_cells.any():
            stats["single_cap_days"] += 1
            stats["single_cap_cells"] += int(capped_cells.sum())
            stats["single_cap_excess_weight"] += float(np.maximum(original - capped, 0.0).sum())
    return pd.DataFrame(values, index=weights.index, columns=weights.columns), stats


def constrain_one_way_turnover(
    targets: pd.DataFrame,
    budget: pd.Series,
    cash_open: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    max_one_way_turnover: float | None,
) -> tuple[pd.DataFrame, dict]:
    """把超出单边换手上限的目标沿当前组合方向整体缩放。

    参数:
        targets: 已平移到执行日的目标权重。
        budget: 执行日开盘时刻的组合价值。
        cash_open: 执行日开盘现金。
        price: 执行价宽表。
        baseline_shares: 上一遍组合逐日收盘持股数。
        max_one_way_turnover: 含现金的单边换手上限，None 表示关闭。
    """
    if max_one_way_turnover is None:
        return targets, {
            "turnover_limited_days": 0,
            "turnover_limited_cells": 0,
            "turnover_before_max": 0.0,
            "turnover_after_max": 0.0,
            "turnover_limit_error_max": 0.0,
        }
    limit = float(max_one_way_turnover)
    if not budget.index.equals(targets.index) or not cash_open.index.equals(targets.index):
        raise ExecutionError("风险约束的预算、现金与目标日期不一致")
    previous = baseline_shares.reindex_like(targets).shift(1).fillna(0.0)
    values = targets.to_numpy(dtype="float64", copy=True)
    budgets = budget.to_numpy(dtype="float64")
    cash_values = cash_open.to_numpy(dtype="float64")
    prices = price.reindex_like(targets).to_numpy(dtype="float64")
    previous_values = previous.to_numpy(dtype="float64")
    stats = {
        "turnover_limited_days": 0,
        "turnover_limited_cells": 0,
        "turnover_before_max": 0.0,
        "turnover_after_max": 0.0,
        "turnover_limit_error_max": 0.0,
    }
    for row in range(values.shape[0]):
        target = values[row]
        if not np.isfinite(target).any():
            continue
        total = budgets[row]
        if not np.isfinite(total) or total <= 0:
            continue
        tradable_value = np.where(np.isfinite(prices[row]) & (prices[row] > 0), previous_values[row] * prices[row], 0.0)
        current_stock = tradable_value / total
        current_cash = max(float(cash_values[row]), 0.0) / total
        target_stock = np.nan_to_num(target, nan=0.0)
        target_cash = max(0.0, 1.0 - float(target_stock.sum()))
        planned = 0.5 * (float(np.abs(target_stock - current_stock).sum()) + abs(target_cash - current_cash))
        stats["turnover_before_max"] = max(stats["turnover_before_max"], planned)
        if planned <= limit + 1e-12:
            stats["turnover_after_max"] = max(stats["turnover_after_max"], planned)
            continue
        scale = limit / planned if planned > 0 else 0.0
        constrained = current_stock + scale * (target_stock - current_stock)
        constrained = np.maximum(constrained, 0.0)
        values[row] = constrained
        after = 0.5 * (float(np.abs(constrained - current_stock).sum()) + abs((1.0 - float(constrained.sum())) - current_cash))
        stats["turnover_limited_days"] += 1
        stats["turnover_limited_cells"] += int(np.count_nonzero(np.abs(constrained - target_stock) > 1e-12))
        stats["turnover_after_max"] = max(stats["turnover_after_max"], after)
        stats["turnover_limit_error_max"] = max(stats["turnover_limit_error_max"], max(0.0, after - limit))
    return pd.DataFrame(values, index=targets.index, columns=targets.columns), stats


def apply_risk_constraints(
    targets: pd.DataFrame,
    budget: pd.Series,
    cash_open: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    risk: RiskSpec,
) -> tuple[pd.DataFrame, dict]:
    """依次应用单股上限和换手上限，返回实际执行目标。"""
    capped, cap_stats = cap_single_weights(targets, risk.max_single_weight)
    constrained, turnover_stats = constrain_one_way_turnover(
        capped, budget, cash_open, price, baseline_shares, risk.max_one_way_turnover
    )
    return constrained, {**cap_stats, **turnover_stats}
