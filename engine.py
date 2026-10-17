"""基于 VectorBT 的回测执行器。

一次回测跑三遍 VectorBT：

1. **探针**：无规则、无费用，只为拿到每格的买卖方向和真实成交额；
2. **基准**：规则照过滤、费用照收，按 ``1 - cash_buffer`` 的目标百分比下单，
   它的逐日现金与持股数用来算「执行日开盘时刻的组合价值」，也就是资金分配的预算；
3. **最终**：按分配出的目标股数（交易单位整数倍）下单，得到权威的现金、持仓、
   费用与净值。

分配预算必须来自带费用的基准遍：探针一遍是免费的，累计费用拖累会让它的净值
系统性偏高，直接拿来当预算会超买。

被现金缩小的买单会变成下一交易日的补回目标，因此「分配 → 最终」还要再迭代一轮：
用上一轮最终遍的真实状态重新分配，直到「本轮用的补回目标」与「由分配结果派生出的
补回目标」一致为止，两层共用 ``MAX_ALLOCATION_ITERATIONS``。首轮用「一直补到下次
调仓」当种子把整条链铺开，避免链条长时一轮只推后一个交易日。没有这类格子的运行
一轮就收敛，耗时与原来相同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.allocation import (
    ALLOCATION_STAT_KEYS,
    cash_carry,
    cash_retry_filled,
    execution_budget,
    execution_cash,
    execution_price,
    extend_carry,
    pending_carry,
)
from backtest.config import MAX_ALLOCATION_ITERATIONS
from backtest.data import PRICE_ADJUST, MarketData
from backtest.errors import ExecutionError
from backtest.logging_conf import get_logger
from backtest.metrics import summarize
from backtest.orders import (
    PROBE_COLUMNS,
    OrderPlan,
    RuleState,
    baseline_inputs,
    build_order_plan,
    execution_targets,
)
from backtest.risk import cap_single_weights, constrain_one_way_turnover
from backtest.signals import build_target_weights
from backtest.specs import BacktestParams, StrategySpec

logger = get_logger("backtest.engine")

Progress = Callable[[str, dict], None]


@dataclass(frozen=True)
class ChunkState:
    """分块回测的跨块交接状态。

    ``cash`` 与 ``shares`` 是上一块最后一个交易日收盘的真实现金与持股，
    ``mark_prices`` 是同一时点的估值价（停牌股沿用前收盘），
    ``rule_state`` 是规则状态机，``carry`` 是下一交易日仍要补回的目标权重。
    """

    cash: float
    shares: pd.Series
    mark_prices: pd.Series
    last_prices: pd.Series
    rule_state: RuleState
    carry: pd.Series


@dataclass(frozen=True)
class SeedPlan:
    """跨块播种的执行口径。

    ``shares`` 是上一块收盘持股，``percent`` 是预热行的目标百分比（基准遍与探针遍用），
    ``fixed_fees`` 是预热行的固定费用：数值等于播种成交金额的相反数，使播种那笔交易的
    现金变化精确为 0（成交金额与固定费用在浮点里正好相加为 0），组合因此从上一块收盘
    的现金与持股起步，结果与不分块逐位一致。
    """

    shares: pd.Series
    percent: pd.Series
    fixed_fees: pd.Series


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
    final_state: "ChunkState | None" = None


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
    init_cash: float | None = None,
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
        init_cash: 初始资金；None 表示用回测参数里的初始资金。分块回测从第二块起
            用「交接现金 + 交接持仓按播种价估值」当初始资金。
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
            init_cash=float(params.init_cash if init_cash is None else init_cash),
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


def derive_daily(
    portfolio_value: pd.Series,
    cash: pd.Series,
    orders: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """由组合价值、现金与成交明细推导标准化日结果。

    分块回测拼接后也走这里，保证单块与分块的派生列口径完全一致。

    参数:
        portfolio_value: 逐日组合价值。
        cash: 逐日现金。
        orders: 成交明细，含 ``trade_date``、``Size``、``Price``、``Fees``。
        benchmark: 基准指数收盘价；None 表示不算超额。
    """
    daily = pd.DataFrame(
        {
            "portfolio_value": pd.Series(portfolio_value).astype("float64"),
            "cash": pd.Series(cash).astype("float64"),
        }
    )
    daily.index = pd.DatetimeIndex(daily.index)
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


def _daily_frame(portfolio, orders: pd.DataFrame, benchmark: pd.Series | None) -> pd.DataFrame:
    """标准化每日组合结果。"""
    return derive_daily(portfolio.value(), portfolio.cash(), orders, benchmark)


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


def weight_witnesses(target_percent: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    """统计实际权重与目标权重的偏差。

    目标权重逐格下单后，当日收盘的实际权重应当贴近目标；偏差来自 100 股取整、
    现金不足被裁剪以及盘中与收盘的估值差。这里给出分位数作为验收口径。

    参数:
        target_percent: 实际提交的目标权重矩阵，非空表示当时提交了订单。
        holdings: 逐日逐股持仓市值与权重。
    """
    targets = target_percent.to_numpy(dtype="float64")
    rows, cols = np.where(~np.isnan(targets))
    if not len(rows):
        return {"target_cells": 0, "weight_deviation_median": None, "weight_deviation_p95": None,
                "weight_deviation_max": None, "weight_deviation_over_1pct": 0}
    realized = np.zeros_like(targets)
    if not holdings.empty:
        holding_rows = target_percent.index.get_indexer(pd.DatetimeIndex(holdings["trade_date"]))
        holding_cols = target_percent.columns.get_indexer(holdings["stock_code"])
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


def _weight_witnesses(plan: OrderPlan, holdings: pd.DataFrame) -> dict:
    """按单块订单计划统计权重偏差。"""
    return weight_witnesses(plan.target_percent, holdings)


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


def _same_matrix(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """比较两个同形矩阵是否完全一致（``NaN`` 视为同值）。"""
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


def _matrix_signature(frame: pd.DataFrame) -> tuple[tuple[int, ...], bytes]:
    """生成固定点迭代的稳定签名，NaN 位置和有限值都参与比较。"""
    values = frame.to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(values)
    normalized = np.where(finite, values, 0.0)
    return tuple(finite.ravel().astype("uint8")), normalized.tobytes()


def _same_masked_matrix(
    first: pd.DataFrame,
    second: pd.DataFrame,
    mask: np.ndarray,
) -> bool:
    """比较指定格子的矩阵值，补回重试日之外的原始目标必须稳定。"""
    if first.shape != second.shape or first.shape != mask.shape:
        return False
    left = first.to_numpy(dtype="float64")[mask]
    right = second.to_numpy(dtype="float64")[mask]
    left_nan, right_nan = np.isnan(left), np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        return False
    return bool(np.array_equal(left[~left_nan], right[~right_nan]))


def _cash_event_signature(allocation: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """提取新的现金补回事件，忽略重试日重新计算的股数。"""
    if allocation.empty or "cash_limited" not in allocation:
        return ()
    limited = allocation.loc[allocation["cash_limited"].astype(bool), ["trade_date", "stock_code"]]
    return tuple(
        sorted((str(pd.Timestamp(row.trade_date)), str(row.stock_code))
               for row in limited.itertuples(index=False))
    )


def _cash_pending_on_last_day(allocation: pd.DataFrame, last_day: pd.Timestamp) -> int:
    """统计最后交易日仍有现金缩水缺口的补回格。"""
    if allocation.empty:
        return 0
    rows = allocation.loc[pd.DatetimeIndex(allocation["trade_date"]) == last_day]
    if rows.empty:
        return 0
    if "cash_shrunk_shares" in rows:
        return int((rows["cash_shrunk_shares"].to_numpy(dtype="float64") > 0).sum())
    return int(rows["cash_limited"].astype(bool).sum())


def seed_inputs(
    initial_state: "ChunkState",
    data: MarketData,
    valuation: pd.DataFrame,
    slippage: float,
) -> SeedPlan:
    """给出跨块播种的执行口径。

    播种价取预热行执行价（开盘价，停牌时沿用前收盘），初始资金就是交接现金。预热行的
    买单带一笔等额负固定费用，`成交金额 + 固定费用` 在浮点里正好相加为 0，于是这一笔
    的现金变化精确为 0：组合从上一块收盘的现金与持股起步，后续每一天与不分块时逐位一致。
    最终遍按目标股数下单，基准遍与探针遍按目标百分比下单，两种口径都精确落到交接持股。

    参数:
        initial_state: 上一块的收尾状态。
        data: 本块输入面板（首行为预热行）。
        valuation: 逐日估值价。
        slippage: 滑点比例。
    """
    price_row = execution_price(data.open, valuation).iloc[0].astype("float64")
    shares = initial_state.shares.reindex(data.columns).fillna(0.0).astype("float64")
    held = shares > 0
    if held.any():
        missing = price_row.index[held.to_numpy() & ~(price_row > 0).to_numpy()]
        if len(missing):
            raise ExecutionError(f"分块播种缺少执行价：{list(missing[:5])}")
    # 百分比略微放大一点，保证 VectorBT 再把百分比换算成股数并向下取整到 100 股时
    # 正好落在交接持股上（浮点误差可能让结果差一个最小变动单位）
    percent = (shares * price_row / float(initial_state.cash) * (1.0 + 1e-12)).astype("float64")
    cost = shares * (price_row * (1.0 + float(slippage)))
    return SeedPlan(
        shares=shares,
        percent=percent,
        fixed_fees=(-cost).where(held, 0.0).astype("float64"),
    )


def _seed_target_row(size: pd.Series) -> pd.Series:
    """把播种结果写成下单矩阵的一行：没有持仓的格子留空，不提交 0 股订单。"""
    return size.where(size > 0, np.nan)


def _with_seed_row(matrix: pd.DataFrame, percent: pd.Series) -> pd.DataFrame:
    """把播种行的目标百分比写进下单矩阵。"""
    if not len(matrix.index):
        return matrix
    frame = matrix.copy()
    frame.iloc[0] = (
        _seed_target_row(percent).reindex(matrix.columns).to_numpy(dtype="float64")
    )
    return frame


def _drop_warmup_orders(orders: pd.DataFrame, index: pd.Index, warmup_rows: int) -> pd.DataFrame:
    """丢掉预热行的成交并把订单号重新从 0 编起（预热行的播种成交不属于运行结果）。"""
    if not warmup_rows or orders.empty:
        return orders
    kept = orders.loc[pd.DatetimeIndex(orders["trade_date"]) >= index[warmup_rows]].copy()
    kept["Order Id"] = np.arange(len(kept), dtype="int64")
    return kept.reset_index(drop=True)


def _with_seed_fixed_fees(
    fixed_fees: pd.DataFrame | None,
    index: pd.Index,
    columns: pd.Index,
    plan: SeedPlan,
) -> pd.DataFrame:
    """把预热行的固定费用写进费用矩阵；没有矩阵时按面板形状新建一个。

    写进去之前统一按 ``float64`` 存放：费率矩阵原本是 ``float32``，升成 ``float64`` 只是
    精确提升，不会改变其它格子的数值，但能让「成交金额 + 固定费用」正好相加为 0。
    """
    frame = (
        pd.DataFrame(0.0, index=index, columns=columns)
        if fixed_fees is None
        else fixed_fees
    ).astype("float64")
    values = plan.fixed_fees.reindex(columns).fillna(0.0).to_numpy(dtype="float64")
    frame.iloc[0] = values
    return frame


def _check_seed_cash(state_cash: pd.Series, initial_state: "ChunkState", label: str) -> None:
    """核对播种后的开盘现金与上一块收盘现金一致，防止浮点偏差悄悄改变执行口径。"""
    if not len(state_cash.index):
        return
    seeded = float(state_cash.iloc[0])
    if seeded != float(initial_state.cash):
        raise ExecutionError(
            f"{label} 播种后的现金 {seeded!r} 与上一块收盘现金 {float(initial_state.cash)!r} 不一致"
        )


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
    *,
    initial_state: "ChunkState | None" = None,
    warmup_rows: int = 0,
    rebalance_days: pd.Series | None = None,
    liquidations: pd.DataFrame | None = None,
    chunk_boundary: bool = False,
) -> BacktestResult:
    """执行一次 VectorBT 回测。

    参数:
        data: 回测输入面板。
        strategy: 策略配置。
        params: 区间与初始资金。
        progress: 可选阶段回调，签名为 ``progress(stage, payload)``。
        initial_state: 分块回测从上一块继承的组合与规则状态；None 表示从初始资金空仓起步。
        warmup_rows: 预热行数。分块回测的块首行只提供价格与平移口径，不下单、不进结果。
        rebalance_days: 整段区间的调仓日掩码；None 表示按本面板现算。
        liquidations: 整段区间的退市强制清算格；None 表示按本面板现算。
        chunk_boundary: 本块之后还有下一块。块末仍有现金补回待办时记为跨块携带，
            不算未收敛。
    """
    if not data.index.size:
        raise ExecutionError("回测数据为空")
    if strategy.universe != "all_a" and data.membership is None:
        raise ExecutionError(
            f"股票池 {strategy.universe} 需要成分矩阵，但输入面板没有；"
            "请用 load_market_data(universe=...) 取数"
        )
    data = data.crop(params.start, params.end)
    if warmup_rows and warmup_rows >= len(data.index):
        raise ExecutionError(f"预热行数 {warmup_rows} 不小于面板交易日数 {len(data.index)}")
    started = time.perf_counter()
    valuation = data.close.ffill(axis=0)
    target_weights, ranks = build_target_weights(
        data, strategy.signal, strategy.portfolio, strategy.rebalance, rebalance_days
    )
    target_weights, single_cap_stats = cap_single_weights(
        target_weights, strategy.risk.max_single_weight
    )
    _notify(
        progress,
        "signals",
        rebalance_days=int(target_weights.notna().any(axis=1).sum()),
        tradable_cells=int(ranks.notna().sum().sum()),
        **single_cap_stats,
    )

    targets = execution_targets(target_weights)
    seed_plan = None
    init_cash = float(params.init_cash)
    if initial_state is not None:
        seed_plan = seed_inputs(
            initial_state, data, valuation, float(strategy.costs.slippage)
        )
        # 播种的现金变化为 0，组合直接从上一块收盘的现金起步
        init_cash = float(initial_state.cash)
    probe_size = targets if seed_plan is None else _with_seed_row(targets, seed_plan.percent)
    probe = _run_portfolio(
        data,
        valuation,
        probe_size,
        params,
        strategy,
        None,
        None if seed_plan is None else _with_seed_fixed_fees(None, targets.index, targets.columns, seed_plan),
        init_cash=init_cash,
    )
    probe_orders = _probe_frame(_order_records(probe, "探针运行"))
    if warmup_rows:
        stamps = pd.DatetimeIndex(probe_orders["trade_date"])
        probe_orders = probe_orders.loc[stamps >= data.index[warmup_rows]].reset_index(drop=True)
    del probe  # 探针只提供成交明细，矩阵尽早释放
    _notify(progress, "probe", probe_orders=int(len(probe_orders)))

    # 基准遍：规则照过滤、费用照收，只是按留了缓冲的目标百分比下单，用来提供分配预算
    budget_scale = 1.0 - float(strategy.execution.cash_buffer)
    baseline_targets, baseline_fees, baseline_fixed = baseline_inputs(
        data,
        targets,
        strategy.execution,
        probe_orders,
        strategy.costs,
        liquidations,
        None if initial_state is None else initial_state.rule_state,
        warmup_rows,
    )
    baseline_size = baseline_targets * budget_scale
    baseline_fixed_fees = baseline_fixed
    if seed_plan is not None:
        baseline_size = _with_seed_row(baseline_size, seed_plan.percent)
        baseline_fixed_fees = _with_seed_fixed_fees(
            baseline_fixed, targets.index, targets.columns, seed_plan
        )
    baseline = _run_portfolio(
        data, valuation, baseline_size, params, strategy,
        baseline_fees, baseline_fixed_fees, init_cash=init_cash,
    )
    state_cash, state_shares = _state_frames(baseline)
    if seed_plan is not None:
        _check_seed_cash(state_cash, initial_state, "基准遍")
    del baseline
    _notify(
        progress,
        "baseline",
        cash_buffer=float(strategy.execution.cash_buffer),
        rebalance_days=int((~np.isnan(baseline_targets.to_numpy())).any(axis=1).sum()),
    )

    # 分配要迭代到自洽：目标股数依赖「执行日开盘时刻的真实持仓与现金」，
    # 而这份状态只有跑完一遍才拿得到。目标股数与补回目标都不再变化时，
    # 上一遍的成交与净值就是最终结果（同样的目标股数与费率矩阵必然跑出同样的组合）。
    #
    # 被现金缩小的买单是下一交易日的补回目标，而补回目标又依赖分配结果，
    # 因此两层一起迭代：首轮用「一直补到下次调仓」当种子把整条链铺开，
    # 之后按分配结果收敛回真正需要补的日子。
    price = execution_price(data.open, valuation)
    carry = cash_carry(None, targets)
    if initial_state is not None and len(carry.index):
        # 上一块交接过来的补回目标写在预热行上，之后按当天的真实状态继续推进
        carry.iloc[0] = (
            initial_state.carry.reindex(targets.columns).astype("float64").to_numpy(dtype="float64")
        )
    plan: OrderPlan | None = None
    portfolio = None
    previous_shares: pd.DataFrame | None = None
    previous_carry: pd.DataFrame | None = None
    previous_effective_targets: pd.DataFrame | None = None
    previous_clipped_cells: int | None = None
    previous_no_pending = False
    carry_iterations = 0
    carry_converged = False
    iteration = 0
    converged = False
    seen_states: dict[
        tuple[
            tuple[tuple[int, ...], bytes],
            tuple[tuple[int, ...], bytes],
            tuple[tuple[int, ...], bytes],
        ],
        int,
    ] = {}
    last_plan_carry: pd.DataFrame | None = None
    previous_cash_events: tuple[tuple[str, str], ...] | None = None
    original_target_mask = ~np.isnan(targets.to_numpy(dtype="float64"))
    effective_targets = targets
    risk_stats = dict(single_cap_stats)
    for iteration in range(1, MAX_ALLOCATION_ITERATIONS + 1):
        budget = execution_budget(state_cash, state_shares, price, init_cash)
        cash_open = execution_cash(state_cash, init_cash)
        effective_targets, turnover_stats = constrain_one_way_turnover(
            targets,
            budget,
            cash_open,
            price,
            state_shares,
            strategy.risk.max_one_way_turnover,
        )
        risk_stats = {**single_cap_stats, **turnover_stats}
        candidate = build_order_plan(
            data, effective_targets, strategy.execution, probe_orders, budget, cash_open, state_shares,
            strategy.costs, carry,
            initial_state=None if initial_state is None else initial_state.rule_state,
            warmup_rows=warmup_rows,
            liquidations=liquidations,
            seed_shares=None if seed_plan is None else seed_plan.shares,
            seed_fixed_fees=None if seed_plan is None else seed_plan.fixed_fees,
        )
        if (
            previous_shares is not None
            and _same_matrix(previous_shares, candidate.shares)
            and _same_matrix(previous_carry, carry)
            and _same_matrix(previous_effective_targets, effective_targets)
            and previous_clipped_cells == 0
            and previous_no_pending
            # 有下一调仓段时，稳定的补回目标可以交给下一段继续；最后一段没有后续交易日，
            # 必须确认区间末不存在仍被现金缩小的缺口。
            and (
                chunk_boundary
                or _cash_pending_on_last_day(candidate.allocation, effective_targets.index[-1]) == 0
            )
        ):
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
            data, valuation, candidate.shares, params, strategy, candidate.fees,
            candidate.fixed_fees, size_type="targetamount", init_cash=init_cash,
        )
        state_cash, state_shares = _state_frames(portfolio)
        if seed_plan is not None:
            _check_seed_cash(state_cash, initial_state, "最终遍")
        plan = candidate
        last_plan_carry = carry
        previous_shares = candidate.shares
        previous_carry = carry
        previous_effective_targets = effective_targets
        next_carry = cash_carry(plan.allocation, effective_targets, carry)
        if iteration == 1:
            next_carry = extend_carry(next_carry, effective_targets)
        if not _same_matrix(next_carry, carry):
            carry_iterations += 1
        carry = next_carry
        cash_events = _cash_event_signature(candidate.allocation)
        allocation_witnesses = _clipped_cells(candidate, portfolio)
        no_pending = _cash_pending_on_last_day(candidate.allocation, effective_targets.index[-1]) == 0
        state_key = (
            _matrix_signature(candidate.shares),
            _matrix_signature(carry),
            _matrix_signature(effective_targets),
        )
        prior_iteration = seen_states.get(state_key)
        if prior_iteration is not None and prior_iteration < iteration - 1:
            logger.warning(
                "资金分配检测到重复状态：第 %d 轮重复第 %d 轮，停止继续重算",
                iteration,
                prior_iteration,
            )
            break
        seen_states[state_key] = iteration
        if (
            previous_cash_events is not None
            and previous_cash_events == cash_events
            and previous_shares is not None
            and _same_masked_matrix(previous_shares, candidate.shares, original_target_mask)
            # 分块运行时额外要求目标股数与补回目标本身稳定：只比「原始目标格」会让结果
            # 依赖迭代停在哪一轮，从而依赖回测区间的终点，分块与不分块就对不上了
            and (
                not chunk_boundary
                or (
                    _same_matrix(previous_shares, candidate.shares)
                    and _same_matrix(previous_carry, carry)
                    and _same_matrix(previous_effective_targets, effective_targets)
                )
            )
            and allocation_witnesses["cash_clipped_cells"] == 0
            and (no_pending or chunk_boundary)
        ):
            converged = True
            logger.info("现金补回事件在第 %d 轮稳定，停止继续重算", iteration)
        _notify(
            progress,
            "allocation",
            iteration=iteration,
            carry_iterations=carry_iterations,
            **{key: candidate.plan_stats[key] for key in ALLOCATION_STAT_KEYS},
        )
        previous_cash_events = cash_events
        previous_clipped_cells = int(allocation_witnesses["cash_clipped_cells"])
        previous_no_pending = bool(no_pending)
        if converged:
            break
    else:
        logger.warning("资金分配 %d 次迭代仍未稳定，使用最后一次结果", MAX_ALLOCATION_ITERATIONS)
    if plan is None or portfolio is None:
        raise ExecutionError("资金分配没有产出可执行的目标股数")
    carry_converged = _same_matrix(
        carry,
        cash_carry(plan.allocation, effective_targets, last_plan_carry),
    )

    orders = _order_records(portfolio, "最终运行")
    orders = _drop_warmup_orders(orders, data.index, warmup_rows)
    witnesses = {**_fee_witnesses(orders, plan), **_clipped_cells(plan, portfolio)}
    _notify(progress, "execute", **witnesses)

    daily = _daily_frame(portfolio, orders, data.benchmark)
    holdings = _holdings_frame(portfolio)
    liquidation_records = _liquidation_records(plan, orders)
    # 区间末仍被现金限制、来不及补回的格子（末日的补回目标没有下一交易日可落）
    last_allocations = plan.allocation.loc[
        pd.DatetimeIndex(plan.allocation["trade_date"]) == effective_targets.index[-1]
    ]
    if "cash_shrunk_shares" in last_allocations:
        pending_at_end = int(
            (last_allocations["cash_shrunk_shares"].to_numpy(dtype="float64") > 0).sum()
        )
    else:
        pending_at_end = int(last_allocations["cash_limited"].astype(bool).sum())
    if plan.rule_state is None:
        raise ExecutionError("订单计划缺少规则状态，无法交接给下一块")
    calibration = {
        "cash_buffer": float(strategy.execution.cash_buffer),
        "allocation_iterations": int(iteration),
        "allocation_converged": bool(converged),
        "carry_iterations": int(carry_iterations),
        "carry_converged": bool(carry_converged),
        "cash_retry_filled_cells": cash_retry_filled(carry, plan.allocation),
        "cash_pending_at_end_cells": pending_at_end,
        "cash_carry_over_cells": int(pending_at_end if chunk_boundary else 0),
        **risk_stats,
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
        "cash_retry_policy": "all_cash_shortfall_event",
        "risk": strategy.risk.to_dict(),
        "execution": {
            "sizing": "target_amount",
            "cash_buffer": float(strategy.execution.cash_buffer),
            "lot_size": strategy.execution.lot_size,
            "allocation_iterations": int(iteration),
            "allocation_converged": bool(converged),
            "carry_iterations": int(carry_iterations),
            "carry_converged": bool(carry_converged),
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
        len(liquidation_records),
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
        liquidations=liquidation_records,
        allocations=plan.allocation,
        final_state=ChunkState(
            cash=float(state_cash.iloc[-1]),
            shares=state_shares.iloc[-1].astype("float64"),
            mark_prices=valuation.iloc[-1].astype("float64"),
            last_prices=execution_price(data.open, valuation).iloc[-1].astype("float64"),
            rule_state=plan.rule_state,
            carry=pending_carry(carry, plan.allocation, effective_targets),
        ),
    )



