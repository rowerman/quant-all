"""基于 VectorBT 的回测执行器。

一次回测跑两遍 VectorBT：

1. **探针**：无规则、无费用，只为拿到每格的买卖方向和真实成交额；
2. **最终**：把规则拦住的格子置空，并按探针成交额写入佣金、卖出印花税和
   最低佣金差额，得到权威的现金、持仓、费用与净值。
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import ExecutionError
from backtest.logging_conf import get_logger
from backtest.metrics import summarize
from backtest.orders import (
    PROBE_COLUMNS,
    OrderPlan,
    build_order_plan,
    execution_targets,
    probe_matrices,
)
from backtest.signals import build_target_weights
from backtest.specs import BacktestParams, StrategySpec

logger = get_logger("backtest.engine")

Progress = Callable[[str, dict], None]


@dataclass(frozen=True)
class BacktestResult:
    """VectorBT 结果和项目标准化输出。"""

    portfolio: Any
    daily: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    holdings: pd.DataFrame
    target_weights: pd.DataFrame
    rejections: pd.DataFrame
    metrics: dict
    metadata: dict


def _notify(progress: Progress | None, stage: str, **payload) -> None:
    """向调用方报告阶段进度；没有回调时什么也不做。"""
    if progress is not None:
        progress(stage, payload)


def _run_portfolio(
    data: MarketData,
    valuation: pd.DataFrame,
    targets: pd.DataFrame,
    params: BacktestParams,
    strategy: StrategySpec,
    fees: pd.DataFrame | None,
    fixed_fees: pd.DataFrame | None,
):
    """调用一次 VectorBT 的 ``from_orders``。

    参数:
        data: 回测输入面板。
        valuation: 逐日估值价（停牌日沿用上一有效收盘价）。
        targets: 目标权重矩阵，``NaN`` 表示当天不下单。
        params: 区间与初始资金。
        strategy: 策略配置。
        fees: 逐格费率；None 表示免费。
        fixed_fees: 逐格固定费用；None 表示没有。
    """
    try:
        import vectorbt as vbt

        return vbt.Portfolio.from_orders(
            close=valuation,
            size=targets,
            size_type="targetpercent",
            price=data.open,
            fees=0.0 if fees is None else fees,
            fixed_fees=0.0 if fixed_fees is None else fixed_fees,
            slippage=strategy.costs.slippage,
            size_granularity=strategy.execution.lot_size,
            init_cash=float(params.init_cash),
            cash_sharing=True,
            call_seq="auto",
            # 100 股取整后必然与原目标有零头，allow_partial=False 会把订单整笔拒掉
            allow_partial=True,
            raise_reject=False,
            log=True,
            freq="1D",
        )
    except Exception as error:  # VectorBT 内部异常统一转成回测错误
        raise ExecutionError(f"VectorBT 执行失败：{error}") from error


def _order_records(portfolio, label: str) -> pd.DataFrame:
    """提取 VectorBT 成交明细。"""
    try:
        frame = portfolio.orders.records_readable.copy()
    except Exception as error:
        raise ExecutionError(f"无法提取 VectorBT 订单记录：{error}") from error
    if frame.empty:
        return pd.DataFrame(columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"])
    frame = frame.rename(columns={"Column": "stock_code", "Timestamp": "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    logger.info("%s：订单 %d 笔", label, len(frame))
    return frame


def _probe_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """把成交明细规范化成探针口径：方向加成交额。"""
    if orders.empty:
        return pd.DataFrame(columns=PROBE_COLUMNS)
    return pd.DataFrame(
        {
            "trade_date": orders["trade_date"],
            "stock_code": orders["stock_code"],
            "side": np.where(orders["Side"].astype(str).str.lower().eq("buy"), 1, -1).astype("int8"),
            "value": (orders["Size"].abs() * orders["Price"].astype(float)).astype("float64"),
        }
    )


def _daily_frame(portfolio, orders: pd.DataFrame, benchmark: pd.Series | None) -> pd.DataFrame:
    """标准化每日组合结果。"""
    value = portfolio.value()
    daily = pd.DataFrame({"portfolio_value": value, "cash": portfolio.cash()})
    daily.index = pd.DatetimeIndex(value.index)
    daily["holdings_value"] = daily["portfolio_value"] - daily["cash"]
    daily["return"] = daily["portfolio_value"].pct_change().fillna(0.0)
    daily["nav"] = daily["portfolio_value"] / float(daily["portfolio_value"].iloc[0])
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1.0
    if not orders.empty:
        stamp = pd.DatetimeIndex(orders["trade_date"])
        daily["cost"] = orders["Fees"].astype(float).groupby(stamp).sum().reindex(daily.index).fillna(0.0)
        traded = (orders["Size"].abs() * orders["Price"].astype(float)).groupby(stamp).sum()
        daily["turnover"] = (
            traded.reindex(daily.index).fillna(0.0)
            / daily["portfolio_value"].shift(1).replace(0, np.nan)
        )
    else:
        daily["cost"] = 0.0
        daily["turnover"] = 0.0
    if benchmark is not None:
        benchmark = benchmark.reindex(daily.index).astype(float)
        daily["benchmark_return"] = benchmark.pct_change().fillna(0.0)
        daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
        daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily


def _holdings_frame(portfolio) -> pd.DataFrame:
    """提取逐日逐股持仓市值与权重，只保留有仓位的格子。

    参数:
        portfolio: VectorBT 组合。
    """
    value = portfolio.value().astype("float64")
    try:
        asset = portfolio.asset_value(group_by=False)
    except Exception as error:
        raise ExecutionError(f"无法提取逐股持仓市值：{error}") from error
    asset = asset.reindex(pd.DatetimeIndex(value.index))
    # 逐格取非零持仓，避免把整块「日期 × 股票」矩阵堆成长表
    matrix = asset.to_numpy(dtype="float64")
    rows, cols = np.where(matrix > 0)
    if not len(rows):
        return pd.DataFrame(columns=["trade_date", "stock_code", "market_value", "weight"])
    frame = pd.DataFrame(
        {
            "trade_date": asset.index[rows],
            "stock_code": asset.columns[cols],
            "market_value": matrix[rows, cols],
        }
    )
    frame["weight"] = frame["market_value"] / frame["trade_date"].map(value)
    return frame.reset_index(drop=True)


def _fee_witnesses(
    orders: pd.DataFrame,
    plan: OrderPlan,
    probe_side: np.ndarray,
) -> dict:
    """核对最终成交的费用是否按逐格费率矩阵收准。

    参数:
        orders: 最终一遍的成交明细。
        plan: 订单计划，含费率矩阵。
        probe_side: 探针方向矩阵，用于统计「探针没记录、走了兜底费率」的笔数。
    """
    if orders.empty:
        return {"final_orders": 0, "fallback_fee_orders": 0, "fee_matrix_max_diff": 0.0}
    rows = plan.fees.index.get_indexer(pd.DatetimeIndex(orders["trade_date"]))
    cols = plan.fees.columns.get_indexer(orders["stock_code"])
    if (rows < 0).any() or (cols < 0).any():
        raise ExecutionError("成交明细落在目标面板之外，无法核对费用")
    value = orders["Size"].abs().to_numpy() * orders["Price"].astype(float).to_numpy()
    expected = (
        plan.fees.to_numpy()[rows, cols].astype("float64") * value
        + plan.fixed_fees.to_numpy()[rows, cols].astype("float64")
    )
    actual = orders["Fees"].astype(float).to_numpy()
    return {
        "final_orders": int(len(orders)),
        "final_buys": int(orders["Side"].astype(str).str.lower().eq("buy").sum()),
        "final_sells": int(orders["Side"].astype(str).str.lower().eq("sell").sum()),
        "fallback_fee_orders": int((probe_side[rows, cols] == 0).sum()),
        "fee_matrix_max_diff": float(np.max(np.abs(expected - actual))) if len(actual) else 0.0,
    }


def _weight_witnesses(plan: OrderPlan, holdings: pd.DataFrame) -> dict:
    """统计实际权重与目标权重的偏差。

    目标权重逐格下单后，当日收盘的实际权重应当贴近目标；偏差来自 100 股取整、
    现金不足被裁剪以及盘中与收盘的估值差。这里给出分位数作为验收口径。

    参数:
        plan: 订单计划，``target_percent`` 非空表示当时提交了订单。
        holdings: 逐日逐股持仓市值与权重。
    """
    targets = plan.target_percent.to_numpy(dtype="float64")
    rows, cols = np.where(~np.isnan(targets))
    if not len(rows):
        return {"target_cells": 0, "weight_deviation_median": None, "weight_deviation_p95": None,
                "weight_deviation_max": None, "weight_deviation_over_1pct": 0}
    realized = np.zeros_like(targets)
    if not holdings.empty:
        holding_rows = plan.target_percent.index.get_indexer(pd.DatetimeIndex(holdings["trade_date"]))
        holding_cols = plan.target_percent.columns.get_indexer(holdings["stock_code"])
        valid = (holding_rows >= 0) & (holding_cols >= 0)
        realized[holding_rows[valid], holding_cols[valid]] = holdings["weight"].to_numpy()[valid]
    deviation = np.abs(realized[rows, cols] - targets[rows, cols])
    return {
        "target_cells": int(len(rows)),
        "weight_deviation_median": float(np.median(deviation)),
        "weight_deviation_p95": float(np.quantile(deviation, 0.95)),
        "weight_deviation_max": float(deviation.max()),
        "weight_deviation_over_1pct": int((deviation > 0.01).sum()),
    }


def run_backtest(
    data: MarketData,
    strategy: StrategySpec,
    params: BacktestParams,
    progress: Progress | None = None,
) -> BacktestResult:
    """执行一次 VectorBT 回测。

    参数:
        data: 回测输入面板。
        strategy: 策略配置。
        params: 区间与初始资金。
        progress: 可选阶段回调，签名为 ``progress(stage, payload)``。
    """
    if not data.index.size:
        raise ExecutionError("回测数据为空")
    if strategy.universe != "all_a" and data.membership is None:
        raise ExecutionError(
            f"股票池 {strategy.universe} 需要成分矩阵，但输入面板没有；"
            "请用 load_market_data(universe=...) 取数"
        )
    data = data.crop(params.start, params.end)
    started = time.perf_counter()
    valuation = data.close.ffill(axis=0)
    target_weights, ranks = build_target_weights(
        data, strategy.signal, strategy.portfolio, strategy.rebalance
    )
    _notify(
        progress,
        "signals",
        rebalance_days=int(target_weights.notna().any(axis=1).sum()),
        tradable_cells=int(ranks.notna().sum().sum()),
    )

    targets = execution_targets(target_weights)
    probe = _run_portfolio(data, valuation, targets, params, strategy, None, None)
    probe_orders = _probe_frame(_order_records(probe, "探针运行"))
    _notify(progress, "probe", probe_orders=int(len(probe_orders)))

    plan = build_order_plan(data, targets, strategy.execution, probe_orders, strategy.costs)
    _notify(progress, "orders", **plan.plan_stats)

    portfolio = _run_portfolio(
        data, valuation, plan.target_percent, params, strategy, plan.fees, plan.fixed_fees
    )
    orders = _order_records(portfolio, "最终运行")
    side_matrix, _ = probe_matrices(probe_orders, targets.index, targets.columns)
    witnesses = _fee_witnesses(orders, plan, side_matrix)
    _notify(progress, "execute", **witnesses)

    daily = _daily_frame(portfolio, orders, data.benchmark)
    holdings = _holdings_frame(portfolio)
    calibration = {**plan.plan_stats, **witnesses, **_weight_witnesses(plan, holdings)}
    metadata = {
        "strategy": strategy.to_dict(),
        "strategy_hash": strategy.config_hash,
        "params": {"start": params.start, "end": params.end, "init_cash": params.init_cash},
        "universe": strategy.universe,
        "vectorbt_version": getattr(__import__("vectorbt"), "__version__", "unknown"),
        "panel": {"trading_days": int(len(data.index)), "stocks": int(len(data.columns))},
        "rank_rows": int(ranks.notna().sum().sum()),
        "calibration": calibration,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger.info(
        "回测完成：区间=%s~%s 订单=%d 拒单=%d 耗时=%.1fs",
        params.start,
        params.end,
        len(orders),
        len(plan.rejection),
        metadata["elapsed_seconds"],
    )
    return BacktestResult(
        portfolio=portfolio,
        daily=daily,
        orders=orders,
        positions=portfolio.positions.records_readable.copy(),
        holdings=holdings,
        target_weights=plan.target_percent,
        rejections=plan.rejection,
        metrics=summarize(daily),
        metadata=metadata,
    )



