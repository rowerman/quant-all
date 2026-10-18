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
    return cap_target_weights(weights, max_single_weight)


def cap_target_weights(
    weights: pd.DataFrame,
    max_single_weight: float | None,
    industries: pd.DataFrame | None = None,
    max_industry_weight: float | None = None,
    industry_labels: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """联合封顶单股和行业目标权重，并按原权重比例使用剩余容量。"""
    if max_industry_weight is not None:
        if industries is None:
            raise ExecutionError("启用行业上限时缺少申万一级行业面板")
        if not industries.index.equals(weights.index) or not industries.columns.equals(weights.columns):
            raise ExecutionError("行业面板与目标权重的日期或股票列不一致")
    if max_single_weight is None and max_industry_weight is None:
        return weights, {
            "single_cap_days": 0,
            "single_cap_cells": 0,
            "single_cap_excess_weight": 0.0,
            "single_cap_residual_cash": 0.0,
            "single_cap_max_weight": 0.0,
            "industry_cap_days": 0,
            "industry_cap_groups": 0,
            "industry_cap_cells": 0,
            "industry_cap_excess_weight": 0.0,
            "industry_cap_residual_cash": 0.0,
            "industry_cap_target_max_weight": 0.0,
            "industry_missing_cells": 0,
            "risk_residual_cash": 0.0,
        }
    single_cap = None if max_single_weight is None else float(max_single_weight)
    industry_cap = None if max_industry_weight is None else float(max_industry_weight)
    values = weights.to_numpy(dtype="float64", copy=True)
    industry_values = None if industries is None else industries.to_numpy(dtype="int16", copy=False)
    stats = {
        "single_cap_days": 0,
        "single_cap_cells": 0,
        "single_cap_excess_weight": 0.0,
        "single_cap_residual_cash": 0.0,
        "single_cap_max_weight": 0.0,
        "industry_cap_days": 0,
        "industry_cap_groups": 0,
        "industry_cap_cells": 0,
        "industry_cap_excess_weight": 0.0,
        "industry_cap_residual_cash": 0.0,
        "industry_cap_target_max_weight": 0.0,
        "industry_missing_cells": 0,
        "risk_residual_cash": 0.0,
    }
    for row in range(values.shape[0]):
        active = np.isfinite(values[row]) & (values[row] > 0)
        if not active.any():
            continue
        original = values[row, active].copy()
        capped = original.copy()
        groups = None if industry_values is None else industry_values[row, active]
        if industry_cap is not None and groups is not None and (groups <= 0).any():
            missing_columns = weights.columns[np.flatnonzero(active)[groups <= 0]].astype(str).tolist()
            stats["industry_missing_cells"] += len(missing_columns)
            shown = ", ".join(missing_columns[:10])
            suffix = "..." if len(missing_columns) > 10 else ""
            raise ExecutionError(
                f"申万一级行业缺失：{weights.index[row].date()} 的目标股票 {shown}{suffix}"
            )
        single_hits = np.zeros(len(original), dtype=bool)
        group_hits: set[int] = set()
        single_removed = 0.0
        industry_removed = 0.0
        target_total = float(original.sum())
        for _ in range(len(original) * 2 + 4):
            if single_cap is not None:
                over = capped > single_cap + 1e-12
                if over.any():
                    single_removed += float((capped[over] - single_cap).sum())
                    capped[over] = single_cap
                    single_hits |= over
            if industry_cap is not None and groups is not None:
                for group in np.unique(groups):
                    members = groups == group
                    total = float(capped[members].sum())
                    if total > industry_cap + 1e-12:
                        industry_removed += total - industry_cap
                        capped[members] *= industry_cap / total
                        group_hits.add(int(group))
            excess = target_total - float(capped.sum())
            if excess <= 1e-12:
                break
            single_headroom = (
                np.full(len(capped), np.inf)
                if single_cap is None else np.maximum(single_cap - capped, 0.0)
            )
            candidates = single_headroom > 1e-12
            group_headroom: dict[int, float] = {}
            if industry_cap is not None and groups is not None:
                for group in np.unique(groups):
                    members = groups == group
                    headroom = max(industry_cap - float(capped[members].sum()), 0.0)
                    group_headroom[int(group)] = headroom
                    if headroom <= 1e-12:
                        candidates[members] = False
            if not candidates.any():
                break
            base = original[candidates]
            proposal = excess * base / float(base.sum())
            scale = 1.0
            finite = np.isfinite(single_headroom[candidates]) & (proposal > 0)
            if finite.any():
                scale = min(scale, float(np.min(single_headroom[candidates][finite] / proposal[finite])))
            if industry_cap is not None and groups is not None:
                candidate_groups = groups[candidates]
                for group in np.unique(candidate_groups):
                    proposed = float(proposal[candidate_groups == group].sum())
                    if proposed > 0:
                        scale = min(scale, group_headroom[int(group)] / proposed)
            if scale <= 1e-12:
                break
            candidate_positions = np.flatnonzero(candidates)
            capped[candidate_positions] += proposal * min(scale, 1.0)
            if single_cap is not None:
                reached = capped[candidate_positions] >= single_cap - 1e-12
                single_hits[candidate_positions[reached]] = True
            if industry_cap is not None and groups is not None:
                for group in np.unique(groups[candidates]):
                    if float(capped[groups == group].sum()) >= industry_cap - 1e-12:
                        group_hits.add(int(group))
        values[row, np.flatnonzero(active)] = capped
        stats["single_cap_max_weight"] = max(stats["single_cap_max_weight"], float(capped.max()))
        residual = max(target_total - float(capped.sum()), 0.0)
        stats["risk_residual_cash"] += residual
        if single_cap is not None and single_hits.any():
            stats["single_cap_days"] += 1
            stats["single_cap_cells"] += int(single_hits.sum())
            stats["single_cap_excess_weight"] += single_removed
            stats["single_cap_residual_cash"] += residual
        if industry_cap is not None and groups is not None:
            group_weights = [float(capped[groups == group].sum()) for group in np.unique(groups)]
            stats["industry_cap_target_max_weight"] = max(
                stats["industry_cap_target_max_weight"], max(group_weights, default=0.0)
            )
            if group_hits:
                stats["industry_cap_days"] += 1
                stats["industry_cap_groups"] += len(group_hits)
                stats["industry_cap_cells"] += int(
                    np.isin(groups, np.fromiter(group_hits, dtype="int16")).sum()
                )
                stats["industry_cap_excess_weight"] += industry_removed
                stats["industry_cap_residual_cash"] += residual
    return pd.DataFrame(values, index=weights.index, columns=weights.columns), stats


def industry_weight_witnesses(
    targets: pd.DataFrame,
    industries: pd.DataFrame | None,
    max_industry_weight: float | None,
) -> dict:
    """统计换手缩放后的行业暴露及继承超限。"""
    stats = {
        "industry_post_turnover_max_weight": 0.0,
        "industry_inherited_over_cap_days": 0,
        "industry_inherited_over_cap_groups": 0,
        "industry_inherited_excess_weight": 0.0,
    }
    if max_industry_weight is None or industries is None:
        return stats
    cap = float(max_industry_weight)
    values = targets.to_numpy(dtype="float64", copy=False)
    groups = industries.to_numpy(dtype="int16", copy=False)
    for row in range(len(targets)):
        active = np.isfinite(values[row]) & (values[row] > 0)
        if not active.any():
            continue
        over_groups = 0
        for group in np.unique(groups[row, active]):
            if group <= 0:
                continue
            weight = float(values[row, active & (groups[row] == group)].sum())
            stats["industry_post_turnover_max_weight"] = max(
                stats["industry_post_turnover_max_weight"], weight
            )
            if weight > cap + 1e-12:
                over_groups += 1
                stats["industry_inherited_excess_weight"] += weight - cap
        if over_groups:
            stats["industry_inherited_over_cap_days"] += 1
            stats["industry_inherited_over_cap_groups"] += over_groups
    return stats


def constrain_one_way_turnover(
    targets: pd.DataFrame,
    budget: pd.Series,
    cash_open: pd.Series,
    price: pd.DataFrame,
    baseline_shares: pd.DataFrame,
    max_one_way_turnover: float | None,
    opening_shares: bool = False,
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
    previous = (
        baseline_shares.reindex_like(targets).fillna(0.0)
        if opening_shares
        else baseline_shares.reindex_like(targets).shift(1).fillna(0.0)
    )
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
    industries: pd.DataFrame | None = None,
    industry_labels: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """依次应用集中度上限和换手上限，返回实际执行目标。"""
    capped, cap_stats = cap_target_weights(
        targets,
        risk.max_single_weight,
        industries,
        risk.max_industry_weight,
        industry_labels,
    )
    constrained, turnover_stats = constrain_one_way_turnover(
        capped, budget, cash_open, price, baseline_shares, risk.max_one_way_turnover
    )
    industry_stats = industry_weight_witnesses(
        constrained, industries, risk.max_industry_weight
    )
    return constrained, {**cap_stats, **turnover_stats, **industry_stats}
