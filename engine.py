"""基于 VectorBT 的回测执行器。

一次回测跑三遍 VectorBT：

1. **探针**：无规则、无费用，只为拿到每格的买卖方向和真实成交额；
2. **基准**：规则照过滤、费用照收，按 ``1 - cash_buffer`` 的目标百分比下单，
   它的逐日现金与持股数用来算「执行日开盘时刻的组合价值」，也就是资金分配的预算；
3. **最终**：按分配出的目标股数（交易单位整数倍）下单，得到权威的现金、持仓、
   费用与净值。

分配预算必须来自带费用的基准遍：探针一遍是免费的，累计费用拖累会让它的净值
系统性偏高，直接拿来当预算会超买。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.allocation import ALLOCATION_STAT_KEYS, execution_budget, execution_price
from backtest.config import MAX_ALLOCATION_ITERATIONS
from backtest.data import PRICE_ADJUST, MarketData
from backtest.errors import ExecutionError
from backtest.logging_conf import get_logger
from backtest.metrics import summarize
from backtest.orders import (
    PROBE_COLUMNS,
    OrderPlan,
    baseline_inputs,
    build_order_plan,
    execution_targets,
)
from backtest.signals import build_target_weights
from backtest.specs import BacktestParams, StrategySpec

logger = get_logger("backtest.engine")

Progress = Callable[[str, dict], None]


@dataclass(frozen=True)
class BacktestResult:
    """VectorBT 结果和项目标准化输出。

    ``allocations`` 是逐格的资金分配明细（目标股数、补一手与未建仓的记录）。
    """

    portfolio: Any
    daily: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    holdings: pd.DataFrame
    target_weights: pd.DataFrame
    rejections: pd.DataFrame
    metrics: dict
    metadata: dict
    liquidations: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocations: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    size_type: str = "targetpercent",
):
    """调用一次 VectorBT 的 ``from_orders``。

    参数:
        data: 回测输入面板。
        valuation: 逐日估值价（停牌日沿用上一有效收盘价）。
        targets: 下单矩阵，``NaN`` 表示当天不下单；``size_type`` 为
            ``targetpercent`` 时是目标权重，为 ``targetamount`` 时是目标股数。
        params: 区间与初始资金。
        strategy: 策略配置。
        fees: 逐格费率；None 表示免费。
        fixed_fees: 逐格固定费用；None 表示没有。
        size_type: VectorBT 的下单量口径。
    """
    try:
        import vectorbt as vbt

        return vbt.Portfolio.from_orders(
            close=valuation,
            size=targets,
            size_type=size_type,
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


def _state_frames(portfolio) -> tuple[pd.Series, pd.DataFrame]:
    """提取逐日现金与逐股持股数，用于重建执行预算。

    参数:
        portfolio: 基准遍的 VectorBT 组合。
    """
    try:
        cash = portfolio.cash()
        assets = portfolio.assets()
    except Exception as error:
        raise ExecutionError(f"无法提取基准遍的现金与持仓：{error}") from error
    cash = pd.Series(cash.to_numpy(dtype="float64"), index=pd.DatetimeIndex(cash.index))
    assets = assets.copy()
    assets.index = pd.DatetimeIndex(assets.index)
    return cash, assets.astype("float64")


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
) -> dict:
    """核对最终成交的费用是否按逐格费率矩阵收准。

    参数:
        orders: 最终一遍的成交明细。
        plan: 订单计划，含费率矩阵与成交估计方向。
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
        "fallback_fee_orders": int((plan.estimate_side[rows, cols] == 0).sum()),
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


def _liquidation_records(plan: OrderPlan, orders: pd.DataFrame) -> pd.DataFrame:
    """把强制清算格与实际成交明细对齐，给出落库用的明细。

    参数:
        plan: 订单计划，``liquidations`` 是实际提交的清算格。
        orders: 最终一遍的成交明细。
    """
    columns = ["trade_date", "stock_code", "delist_date", "size", "price", "value"]
    frame = plan.liquidations.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.rename(columns={"price": "reference_price"})
    if orders.empty:
        frame["size"] = np.nan
        frame["price"] = np.nan
    else:
        filled = orders[["trade_date", "stock_code", "Size", "Price"]].rename(
            columns={"Size": "size", "Price": "price"}
        )
        frame = frame.merge(filled, on=["trade_date", "stock_code"], how="left")
    missing = frame["price"].isna()
    if missing.any():
        logger.warning("强制清算有 %d 格没有对应成交，按参考价记账", int(missing.sum()))
        frame.loc[missing, "price"] = frame.loc[missing, "reference_price"]
        frame["size"] = frame["size"].fillna(0.0)
    frame["value"] = frame["size"].abs() * frame["price"].astype("float64")
    return frame[columns]


def _same_shares(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """比较两次分配的目标股数是否完全一致（``NaN`` 视为同值）。"""
    if first.shape != second.shape:
        return False
    if not first.index.equals(second.index) or not first.columns.equals(second.columns):
        return False
    left = first.to_numpy(dtype="float64")
    right = second.to_numpy(dtype="float64")
    left_nan, right_nan = np.isnan(left), np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        return False
    return bool(np.array_equal(left[~left_nan], right[~right_nan]))


def _clipped_cells(plan: OrderPlan, portfolio) -> dict:
    """核对最终一遍的实际持股是否等于分配出的目标股数。

    目标股数是交易单位整数倍、预算又留了现金缓冲，正常情况应当逐格相等；
    出现差额说明基准遍与最终遍的状态差超出了缓冲，被 VectorBT 按现金裁剪了。

    参数:
        plan: 订单计划，``shares`` 是分配出的目标股数。
        portfolio: 最终一遍的 VectorBT 组合。
    """
    targets = plan.shares.to_numpy(dtype="float64")
    rows, cols = np.where(~np.isnan(targets))
    if not len(rows):
        return {"cash_clipped_cells": 0, "allocation_gap_max_shares": 0.0}
    try:
        realized = portfolio.assets().reindex(
            index=plan.shares.index, columns=plan.shares.columns
        )
    except Exception as error:
        raise ExecutionError(f"无法提取最终一遍的持股数：{error}") from error
    gap = np.abs(realized.to_numpy(dtype="float64")[rows, cols] - targets[rows, cols])
    clipped = int((gap > 1e-9).sum())
    if clipped:
        logger.warning("最终一遍有 %d 个格子被现金裁剪，最大差 %.0f 股", clipped, float(gap.max()))
    return {"cash_clipped_cells": clipped, "allocation_gap_max_shares": float(gap.max())}


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
    del probe  # 探针只提供成交明细，矩阵尽早释放
    _notify(progress, "probe", probe_orders=int(len(probe_orders)))

    # 基准遍：规则照过滤、费用照收，只是按留了缓冲的目标百分比下单，用来提供分配预算
    budget_scale = 1.0 - float(strategy.execution.cash_buffer)
    baseline_targets, baseline_fees, baseline_fixed = baseline_inputs(
        data, targets, strategy.execution, probe_orders, strategy.costs
    )
    baseline = _run_portfolio(
        data, valuation, baseline_targets * budget_scale, params, strategy,
        baseline_fees, baseline_fixed,
    )
    state_cash, state_shares = _state_frames(baseline)
    del baseline
    _notify(
        progress,
        "baseline",
        cash_buffer=float(strategy.execution.cash_buffer),
        rebalance_days=int((~np.isnan(baseline_targets.to_numpy())).any(axis=1).sum()),
    )

    # 分配要迭代到自洽：目标股数依赖「执行日开盘时刻的真实持仓与现金」，
    # 而这份状态只有跑完一遍才拿得到。分配结果与上一遍一致时，上一遍的成交与净值
    # 就是最终结果（同样的目标股数与费率矩阵必然跑出同样的组合）。
    price = execution_price(data.open, valuation)
    previous: pd.DataFrame | None = None
    plan: OrderPlan | None = None
    portfolio = None
    converged = False
    iteration = 0
    for iteration in range(1, MAX_ALLOCATION_ITERATIONS + 1):
        budget = execution_budget(state_cash, state_shares, price, params.init_cash)
        candidate = build_order_plan(
            data, targets, strategy.execution, probe_orders, budget, state_shares, strategy.costs
        )
        if previous is not None and _same_shares(previous, candidate.shares):
            converged = True
            break
        if iteration == 1:
            _notify(
                progress,
                "orders",
                **{key: value for key, value in candidate.plan_stats.items()
                   if key not in ALLOCATION_STAT_KEYS},
            )
        portfolio = _run_portfolio(
            data, valuation, candidate.shares, params, strategy, candidate.fees, candidate.fixed_fees,
            size_type="targetamount",
        )
        state_cash, state_shares = _state_frames(portfolio)
        plan = candidate
        previous = candidate.shares
        _notify(
            progress,
            "allocation",
            iteration=iteration,
            **{key: candidate.plan_stats[key] for key in ALLOCATION_STAT_KEYS},
        )
    else:
        logger.warning("资金分配 %d 次迭代仍未稳定，使用最后一次结果", MAX_ALLOCATION_ITERATIONS)
    if plan is None or portfolio is None:
        raise ExecutionError("资金分配没有产出可执行的目标股数")

    orders = _order_records(portfolio, "最终运行")
    witnesses = {**_fee_witnesses(orders, plan), **_clipped_cells(plan, portfolio)}
    _notify(progress, "execute", **witnesses)

    daily = _daily_frame(portfolio, orders, data.benchmark)
    holdings = _holdings_frame(portfolio)
    liquidations = _liquidation_records(plan, orders)
    calibration = {
        "cash_buffer": float(strategy.execution.cash_buffer),
        "allocation_iterations": int(iteration),
        "allocation_converged": bool(converged),
        **plan.plan_stats,
        **witnesses,
        **_weight_witnesses(plan, holdings),
    }
    metadata = {
        "strategy": strategy.to_dict(),
        "strategy_hash": strategy.config_hash,
        "params": {"start": params.start, "end": params.end, "init_cash": params.init_cash},
        "universe": strategy.universe,
        "benchmark": data.benchmark_info,
        "price_adjust": PRICE_ADJUST,
        "execution": {
            "sizing": "target_amount",
            "cash_buffer": float(strategy.execution.cash_buffer),
            "lot_size": strategy.execution.lot_size,
            "allocation_iterations": int(iteration),
            "allocation_converged": bool(converged),
        },
        "versions": data.versions,
        "vectorbt_version": getattr(__import__("vectorbt"), "__version__", "unknown"),
        "panel": {"trading_days": int(len(data.index)), "stocks": int(len(data.columns))},
        "rank_rows": int(ranks.notna().sum().sum()),
        "calibration": calibration,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger.info(
        "回测完成：区间=%s~%s 订单=%d 拒单=%d 强制清算=%d 耗时=%.1fs",
        params.start,
        params.end,
        len(orders),
        len(plan.rejection),
        len(liquidations),
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
        liquidations=liquidations,
        allocations=plan.allocation,
    )



