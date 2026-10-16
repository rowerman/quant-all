"""分块回测：把超过单块上限的区间按时间切块，逐块执行并交接状态，最后拼成一次运行结果。

分块只解决内存：每块面板控制在 ``CHUNK_CELL_BUDGET`` 以内，块与块之间交接真实组合状态
（现金、持股、规则状态机、现金补回目标），因此结果与「不分块跑同一区间」逐位一致。

每块首行是**预热行**：它是上一块的最后一个交易日，只用来把交易标记与信号平移到本块首日，
不下单、不进结果。播种也在这一行完成：按上一块收盘的持股与估值价买回来，初始资金为
「交接现金 + 播种市值 × (1 + 滑点)」，成交后现金正好回到交接现金。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Callable

import numpy as np
import pandas as pd

import quant
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
from backtest.config import (
    CHUNK_CELL_BUDGET,
    MAX_ALLOCATION_ITERATIONS,
    MAX_CHUNKS,
    MAX_PANEL_CELLS,
    MAX_TRADE_DAYS,
)
from backtest.data import (
    MarketData,
    PRICE_ADJUST,
    load_benchmark,
    load_liquidation_candidates,
    load_market_data,
)
from backtest.engine import (
    BacktestResult,
    ChunkState,
    derive_daily,
    seed_inputs,
    weight_witnesses,
)
import backtest.engine as engine
from backtest.errors import ResourceLimitError, RunCancelled
from backtest.logging_conf import get_logger
from backtest.metrics import summarize
from backtest.orders import (
    RuleState,
    baseline_inputs_with_state,
    build_order_plan,
    execution_targets,
)
from backtest.risk import cap_single_weights, constrain_one_way_turnover
from backtest.signals import build_target_weights, rebalance_mask
from backtest.specs import BacktestParams, StrategySpec
from backtest.universe import resolve_universe, universe_stocks

logger = get_logger("backtest.chunked")

Progress = Callable[[str, dict], None]
Cancelled = Callable[[], bool]

# 校准字段的聚合口径：笔数/格数类求和，极值类取最大，区间末遗留取最后一块
SUM_KEYS = (
    "probe_orders",
    "probe_buys",
    "probe_sells",
    "planned_orders",
    "rebalance_orders",
    "rejected_orders",
    "forced_liquidations",
    "retry_attempts",
    "retried_filled_cells",
    "cash_retry_attempts",
    "cash_retry_filled_cells",
    "allocated_cells",
    "bumped_one_lot_cells",
    "unaffordable_lot_cells",
    "no_price_cells",
    "cash_shrunk_cells",
    "cash_limited_cells",
    "cash_limited_value",
    "final_orders",
    "final_buys",
    "final_sells",
    "fallback_fee_orders",
    "cash_clipped_cells",
    "weight_deviation_over_1pct",
    "single_cap_days",
    "single_cap_cells",
    "single_cap_excess_weight",
    "single_cap_residual_cash",
    "turnover_limited_days",
    "turnover_limited_cells",
    "cash_carry_over_cells",
)
MAX_KEYS = (
    "fee_matrix_max_diff",
    "allocation_gap_max_shares",
    "single_cap_max_weight",
    "turnover_before_max",
    "turnover_after_max",
    "turnover_limit_error_max",
)
LAST_KEYS = ("pending_at_end_cells", "cash_pending_at_end_cells")


def _notify(progress: Progress | None, stage: str, **payload) -> None:
    """向调用方报告阶段进度；没有回调时什么也不做。"""
    if progress is not None:
        progress(stage, payload)


@dataclass(frozen=True)
class ChunkWindow:
    """一个分块窗口：本块的交易日区间，以及可选的前一块最后交易日（预热日）。"""

    start: str
    end: str
    warmup: str | None = None


def _trade_day_index(start: str, end: str, db_path=None) -> pd.DatetimeIndex:
    """读取区间内的交易日索引（升序）。"""
    calendar = quant.load_calendar(start=start, end=end, db_path=db_path)
    days = pd.DatetimeIndex(sorted(pd.to_datetime(calendar["trade_date"])))
    if not len(days):
        raise ResourceLimitError(f"回测区间 {start} ~ {end} 内没有交易日")
    return days


def _alive_counts(days: pd.DatetimeIndex, universe: str, start: str, end: str, db_path=None) -> np.ndarray:
    """给出每个交易日「区间内可能出现」的股票数上界。

    全 A 池按股票基础表的上市/退市日逐日累计；指数池用整段区间的成分并集（保守但足够小）。

    参数:
        days: 交易日索引。
        universe: 股票池标识。
        start: 起始交易日。
        end: 截止交易日。
        db_path: 行情库路径。
    """
    spec = resolve_universe(universe)
    if spec.is_index:
        total = len(universe_stocks(spec, start, end, db_path=db_path) or ())
        return np.full(len(days), total, dtype="int64")
    basic = quant.load_stock_basic(db_path=db_path)
    listed = np.sort(pd.to_datetime(basic["list_date"]).to_numpy())
    delisted = np.sort(pd.to_datetime(basic["delist_date"]).dropna().to_numpy())
    stamps = days.to_numpy()
    return np.searchsorted(listed, stamps, side="right") - np.searchsorted(
        delisted, stamps, side="left"
    )


def plan_chunks(
    start: str,
    end: str,
    universe: str = "all_a",
    *,
    db_path=None,
    cell_budget: int = CHUNK_CELL_BUDGET,
    max_chunks: int = MAX_CHUNKS,
    max_days: int = MAX_TRADE_DAYS,
    max_cells: int = MAX_PANEL_CELLS,
) -> list[ChunkWindow]:
    """按单元格预算把区间贪心切成若干块。

    参数:
        start: 起始交易日。
        end: 截止交易日。
        universe: 股票池标识。
        db_path: 行情库路径。
        cell_budget: 单块目标规模（交易日 × 股票数）。
        max_chunks: 块数上限。
        max_days: 单块交易日上限。
        max_cells: 单块单元格硬上限。

    返回:
        分块窗口列表，覆盖区间且互不重叠；除第一块外都带预热日。
    """
    days = _trade_day_index(start, end, db_path)
    counts = _alive_counts(days, universe, start, end, db_path)
    windows: list[ChunkWindow] = []
    begin = 0
    for position in range(1, len(days)):
        span = position - begin + 1
        cells = span * int(counts[begin : position + 1].max())
        if cells <= cell_budget:
            continue
        windows.append(_window(days, begin, position - 1))
        begin = position
    windows.append(_window(days, begin, len(days) - 1))
    if len(windows) > max_chunks:
        raise ResourceLimitError(
            f"回测区间需要切成 {len(windows)} 块，超过上限 {max_chunks} 块；请缩短区间"
        )
    for window in windows:
        chunk_days = int(days.get_loc(pd.Timestamp(window.end))) - int(
            days.get_loc(pd.Timestamp(window.start))
        ) + 1
        if chunk_days > max_days:
            raise ResourceLimitError(f"分块 {window.start} ~ {window.end} 含 {chunk_days} 个交易日，超过上限 {max_days}")
    return windows


def _window(days: pd.DatetimeIndex, first: int, last: int) -> ChunkWindow:
    """按交易日下标构造一个窗口，除第一块外带上预热日。"""
    warmup = None if first == 0 else str(days[first - 1].date())
    return ChunkWindow(
        start=str(days[first].date()),
        end=str(days[last].date()),
        warmup=warmup,
    )


def _warm_up(data: MarketData, state: ChunkState) -> MarketData:
    """把预热行的价格补成上一块收盘时的估值价。

    预热行属于上一块：停牌股在这一行没有开收盘价，按原样留着会让本块起点的估值链
    断在空值上（``close.ffill`` 从空值开始），播种也算不出成交价。

    参数:
        data: 本块输入面板，首行为预热行。
        state: 上一块的收尾状态。
    """
    marks = state.mark_prices.reindex(data.columns).astype("float64").to_numpy()
    known = np.isfinite(marks)
    open_panel = data.open.copy()
    close_panel = data.close.copy()
    open_row = open_panel.iloc[0].astype("float64").to_numpy()
    close_row = close_panel.iloc[0].astype("float64").to_numpy()
    open_row[known] = marks[known]
    close_row[known] = marks[known]
    open_panel.iloc[0] = open_row
    close_panel.iloc[0] = close_row
    return replace(data, open=open_panel, close=close_panel)


@dataclass(frozen=True)
class _PreparedChunk:
    """分块中不随固定点轮次变化的信号结果。"""

    window: ChunkWindow
    target_weights: pd.DataFrame
    targets: pd.DataFrame
    rank_rows: int
    single_cap_stats: dict


class _StateStore:
    """把一轮的逐日现金与非零持股暂存到 Parquet，避免跨块常驻内存。"""

    def __init__(self) -> None:
        self._temp = TemporaryDirectory(prefix="backtest-chunk-state-")
        self.root = Path(self._temp.name)

    def write(self, position: int, cash: pd.Series, shares: pd.DataFrame) -> None:
        cash_frame = pd.DataFrame(
            {"trade_date": pd.DatetimeIndex(cash.index), "cash": cash.to_numpy(dtype="float64")}
        )
        cash_frame.to_parquet(self.root / f"cash-{position}.parquet", index=False)
        values = shares.to_numpy(dtype="float64")
        rows, columns = np.where(values != 0.0)
        share_frame = pd.DataFrame(
            {
                "trade_date": shares.index[rows],
                "stock_code": shares.columns[columns],
                "shares": values[rows, columns],
            }
        )
        share_frame.to_parquet(self.root / f"shares-{position}.parquet", index=False)

    def read(
        self,
        position: int,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> tuple[pd.Series, pd.DataFrame]:
        cash_frame = pd.read_parquet(self.root / f"cash-{position}.parquet")
        cash = pd.Series(
            cash_frame["cash"].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(cash_frame["trade_date"]),
        ).reindex(index)
        shares = pd.DataFrame(0.0, index=index, columns=columns, dtype="float64")
        share_frame = pd.read_parquet(self.root / f"shares-{position}.parquet")
        if not share_frame.empty:
            rows = index.get_indexer(pd.DatetimeIndex(share_frame["trade_date"]))
            cols = columns.get_indexer(share_frame["stock_code"])
            valid = (rows >= 0) & (cols >= 0)
            shares.to_numpy()[rows[valid], cols[valid]] = share_frame.loc[
                valid, "shares"
            ].to_numpy(dtype="float64")
        return cash, shares

    def cleanup(self) -> None:
        self._temp.cleanup()


def _empty_rule_state(columns: pd.Index) -> RuleState:
    """构造空仓规则状态。"""
    return RuleState(
        held=pd.Series(False, index=columns),
        shadow=pd.Series(0.0, index=columns, dtype="float64"),
        pending=pd.Series(np.nan, index=columns, dtype="float64"),
    )


def _empty_carry(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """构造与本块面板同形的空补回矩阵。"""
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype="float64")


def _physical_state(
    cash: pd.Series,
    shares: pd.DataFrame,
    data: MarketData,
    valuation: pd.DataFrame,
    rule_state: RuleState | None = None,
) -> ChunkState:
    """从一次组合执行的收尾状态构造下一块播种状态。"""
    columns = shares.columns
    return ChunkState(
        cash=float(cash.iloc[-1]),
        shares=shares.iloc[-1].astype("float64"),
        mark_prices=valuation.iloc[-1].astype("float64"),
        last_prices=execution_price(data.open, valuation).iloc[-1].astype("float64"),
        rule_state=rule_state or _empty_rule_state(columns),
        carry=pd.Series(np.nan, index=columns, dtype="float64"),
    )


def _seeded_inputs(
    matrix: pd.DataFrame,
    fixed_fees: pd.DataFrame | None,
    state: ChunkState | None,
    data: MarketData,
    valuation: pd.DataFrame,
    slippage: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, object | None]:
    """把跨块播种行写入下单与费用矩阵。"""
    if state is None:
        return matrix, fixed_fees, np.nan, None
    plan = seed_inputs(state, data, valuation, slippage)
    seeded = engine._with_seed_row(matrix, plan.percent)
    seeded_fees = engine._with_seed_fixed_fees(
        fixed_fees, matrix.index, matrix.columns, plan
    )
    return seeded, seeded_fees, float(state.cash), plan


def _logical_frame(frame: pd.DataFrame, window: ChunkWindow) -> pd.DataFrame:
    """去掉重复的预热行，得到本块真正属于整段区间的行。"""
    if window.warmup is None:
        return frame
    return frame.loc[frame.index >= pd.Timestamp(window.start)]


def _concat_logical(
    frames: list[pd.DataFrame], prepared: list[_PreparedChunk]
) -> pd.DataFrame:
    """按交易日拼接各块矩阵并去掉预热行。"""
    return pd.concat(
        [_logical_frame(frame, item.window) for frame, item in zip(frames, prepared, strict=True)]
    ).sort_index()


def _split_logical(
    frame: pd.DataFrame, prepared: list[_PreparedChunk]
) -> list[pd.DataFrame]:
    """把整段矩阵切回各块形状，预热行保持为空。"""
    parts: list[pd.DataFrame] = []
    for item in prepared:
        part = _empty_carry(item.targets.index, item.targets.columns)
        logical_index = part.index if item.window.warmup is None else part.index[1:]
        part.loc[logical_index] = frame.reindex(
            index=logical_index, columns=part.columns
        ).to_numpy(dtype="float64")
        parts.append(part)
    return parts


def _frames_equal(first: list[pd.DataFrame], second: list[pd.DataFrame]) -> bool:
    """逐块精确比较矩阵列表。"""
    return len(first) == len(second) and all(
        engine._same_matrix(left, right)
        for left, right in zip(first, second, strict=True)
    )


def _carry_events(allocations: list[pd.DataFrame]) -> tuple[tuple[str, str], ...]:
    """提取整段本轮新产生的现金补回事件。"""
    events: list[tuple[str, str]] = []
    for allocation in allocations:
        events.extend(engine._cash_event_signature(allocation))
    return tuple(sorted(events))


def _check_cancelled(should_cancel: Cancelled | None) -> None:
    """在分块安全边界检查取消。"""
    if should_cancel is not None and should_cancel():
        raise RunCancelled("回测已取消")


def _report_chunk(
    progress: Progress | None,
    phase: str,
    position: int,
    count: int,
    window: ChunkWindow,
    iteration: int | None = None,
) -> None:
    """报告当前整段阶段与块位置。"""
    payload = {
        "phase": phase,
        "chunk_index": position,
        "chunk_count": count,
        "start": window.start,
        "end": window.end,
        "warmup": window.warmup,
    }
    if iteration is not None:
        payload["iteration"] = iteration
    _notify(progress, "chunk", **payload)


def _prepare_chunks(
    windows: list[ChunkWindow],
    strategy: StrategySpec,
    days: pd.DatetimeIndex,
    loader: Callable[[ChunkWindow, ChunkState | None], MarketData],
    progress: Progress | None,
    should_cancel: Cancelled | None,
) -> list[_PreparedChunk]:
    """逐块计算一次信号目标，后续固定点轮次复用。"""
    mask = rebalance_mask(days, strategy.rebalance.frequency)
    prepared: list[_PreparedChunk] = []
    total_rank_rows = 0
    total_rebalance_days = 0
    for position, window in enumerate(windows, start=1):
        _check_cancelled(should_cancel)
        _report_chunk(progress, "prepare", position, len(windows), window)
        data = loader(window, None)
        target_weights, ranks = build_target_weights(
            data,
            strategy.signal,
            strategy.portfolio,
            strategy.rebalance,
            mask,
        )
        target_weights, single_cap_stats = cap_single_weights(
            target_weights, strategy.risk.max_single_weight
        )
        prepared.append(
            _PreparedChunk(
                window=window,
                target_weights=target_weights,
                targets=execution_targets(target_weights),
                rank_rows=int(ranks.notna().sum().sum()),
                single_cap_stats=single_cap_stats,
            )
        )
        total_rank_rows += int(ranks.notna().sum().sum())
        total_rebalance_days += int(target_weights.notna().any(axis=1).sum())
        del data, ranks
        gc.collect()
    _notify(
        progress,
        "signals",
        rebalance_days=total_rebalance_days,
        tradable_cells=total_rank_rows,
    )
    return prepared


def _run_probe_pass(
    prepared: list[_PreparedChunk],
    strategy: StrategySpec,
    params: BacktestParams,
    loader: Callable[[ChunkWindow, ChunkState | None], MarketData],
    progress: Progress | None,
    should_cancel: Cancelled | None,
) -> list[pd.DataFrame]:
    """按整段顺序执行探针遍，块间只交接探针组合状态。"""
    state: ChunkState | None = None
    probes: list[pd.DataFrame] = []
    for position, item in enumerate(prepared, start=1):
        _check_cancelled(should_cancel)
        _report_chunk(progress, "probe", position, len(prepared), item.window)
        data = loader(item.window, state)
        valuation = data.close.ffill(axis=0)
        fixed = None
        size = item.targets
        init_cash = float(params.init_cash)
        if state is not None:
            size, fixed, init_cash, _ = _seeded_inputs(
                size,
                fixed,
                state,
                data,
                valuation,
                float(strategy.costs.slippage),
            )
        portfolio = engine._run_portfolio(
            data,
            valuation,
            size,
            params,
            strategy,
            None,
            fixed,
            init_cash=init_cash,
        )
        orders = engine._probe_frame(engine._order_records(portfolio, "探针运行"))
        warmup_rows = 0 if item.window.warmup is None else 1
        if warmup_rows:
            orders = orders.loc[
                pd.DatetimeIndex(orders["trade_date"]) >= data.index[warmup_rows]
            ].reset_index(drop=True)
        cash, shares = engine._state_frames(portfolio)
        if state is not None:
            engine._check_seed_cash(cash, state, "探针遍")
        state = _physical_state(cash, shares, data, valuation)
        probes.append(orders)
        del data, portfolio, cash, shares
        gc.collect()
    _notify(progress, "probe", probe_orders=sum(len(frame) for frame in probes))
    return probes


def _run_baseline_pass(
    prepared: list[_PreparedChunk],
    probes: list[pd.DataFrame],
    strategy: StrategySpec,
    params: BacktestParams,
    loader: Callable[[ChunkWindow, ChunkState | None], MarketData],
    liquidations: pd.DataFrame | None,
    progress: Progress | None,
    should_cancel: Cancelled | None,
) -> _StateStore:
    """按整段顺序执行基准遍，返回第一轮分配所需的逐日状态。"""
    store = _StateStore()
    state: ChunkState | None = None
    rule_state: RuleState | None = None
    try:
        for position, (item, probe) in enumerate(zip(prepared, probes, strict=True), start=1):
            _check_cancelled(should_cancel)
            _report_chunk(progress, "baseline", position, len(prepared), item.window)
            data = loader(item.window, state)
            valuation = data.close.ffill(axis=0)
            warmup_rows = 0 if item.window.warmup is None else 1
            targets, fees, fixed, rule_state = baseline_inputs_with_state(
                data,
                item.targets,
                strategy.execution,
                probe,
                strategy.costs,
                liquidations,
                rule_state,
                warmup_rows,
            )
            size = targets * (1.0 - float(strategy.execution.cash_buffer))
            init_cash = float(params.init_cash)
            if state is not None:
                size, fixed, init_cash, _ = _seeded_inputs(
                    size,
                    fixed,
                    state,
                    data,
                    valuation,
                    float(strategy.costs.slippage),
                )
            portfolio = engine._run_portfolio(
                data,
                valuation,
                size,
                params,
                strategy,
                fees,
                fixed,
                init_cash=init_cash,
            )
            cash, shares = engine._state_frames(portfolio)
            if state is not None:
                engine._check_seed_cash(cash, state, "基准遍")
            store.write(position, cash, shares)
            state = _physical_state(cash, shares, data, valuation, rule_state)
            del data, portfolio, cash, shares
            gc.collect()
    except BaseException:
        store.cleanup()
        raise
    _notify(
        progress,
        "baseline",
        cash_buffer=float(strategy.execution.cash_buffer),
        rebalance_days=sum(int(item.targets.notna().any(axis=1).sum()) for item in prepared),
    )
    return store


def _result_from_iteration(
    data: MarketData,
    strategy: StrategySpec,
    params: BacktestParams,
    portfolio,
    plan,
    orders: pd.DataFrame,
    holdings: pd.DataFrame,
    liquidations: pd.DataFrame,
    allocation_witnesses: dict,
    risk_stats: dict,
    carry: pd.DataFrame,
    effective_targets: pd.DataFrame,
    rank_rows: int,
    iteration: int,
) -> BacktestResult:
    """把一块当前固定点轮次的执行结果整理成标准结果。"""
    fee_witnesses = engine._fee_witnesses(orders, plan)
    witnesses = {**fee_witnesses, **allocation_witnesses}
    pending = engine._cash_pending_on_last_day(plan.allocation, effective_targets.index[-1])
    calibration = {
        "cash_buffer": float(strategy.execution.cash_buffer),
        "allocation_iterations": int(iteration),
        "allocation_converged": False,
        "carry_iterations": 0,
        "carry_converged": False,
        "cash_retry_filled_cells": cash_retry_filled(carry, plan.allocation),
        "cash_pending_at_end_cells": int(pending),
        "cash_carry_over_cells": 0,
        **risk_stats,
        **plan.plan_stats,
        **witnesses,
        **engine._weight_witnesses(plan, holdings),
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
        },
        "versions": data.versions,
        "vectorbt_version": getattr(__import__("vectorbt"), "__version__", "unknown"),
        "panel": {"trading_days": int(len(data.index)), "stocks": int(len(data.columns))},
        "rank_rows": int(rank_rows),
        "calibration": calibration,
        "elapsed_seconds": 0.0,
    }
    return BacktestResult(
        portfolio=portfolio,
        daily=engine._daily_frame(portfolio, orders, data.benchmark),
        orders=orders,
        positions=portfolio.positions.records_readable.copy(),
        holdings=holdings,
        target_weights=plan.target_percent,
        rejections=plan.rejection,
        metrics={},
        metadata=metadata,
        liquidations=liquidations,
        allocations=plan.allocation,
        final_state=None,
    )


def run_windows(
    windows: list[ChunkWindow],
    strategy: StrategySpec,
    params: BacktestParams,
    *,
    days: pd.DatetimeIndex,
    loader: Callable[[ChunkWindow, ChunkState | None], MarketData],
    benchmark: pd.Series | None = None,
    benchmark_info: dict | None = None,
    liquidations: pd.DataFrame | None = None,
    progress: Progress | None = None,
    should_cancel: Cancelled | None = None,
) -> BacktestResult:
    """按给定窗口做整段固定点流式回测，并返回拼接结果。"""
    started = time.perf_counter()
    logger.info("整段流式回测：区间=%s~%s 共 %d 块", params.start, params.end, len(windows))
    prepared = _prepare_chunks(windows, strategy, days, loader, progress, should_cancel)
    probes = _run_probe_pass(prepared, strategy, params, loader, progress, should_cancel)
    previous_store = _run_baseline_pass(
        prepared,
        probes,
        strategy,
        params,
        loader,
        liquidations,
        progress,
        should_cancel,
    )
    carry_frames = [_empty_carry(item.targets.index, item.targets.columns) for item in prepared]
    previous_shares: list[pd.DataFrame] | None = None
    previous_effective: list[pd.DataFrame] | None = None
    previous_carry: list[pd.DataFrame] | None = None
    previous_cash_events: tuple[tuple[str, str], ...] | None = None
    carry_iterations = 0
    converged = False
    carry_converged = False
    final_results: list[tuple[ChunkWindow, BacktestResult]] = []
    final_opens: list[pd.DataFrame] = []
    final_marks: list[pd.DataFrame] = []
    iteration = 0
    try:
        for iteration in range(1, MAX_ALLOCATION_ITERATIONS + 1):
            current_store = _StateStore()
            state: ChunkState | None = None
            rule_state: RuleState | None = None
            current_results: list[tuple[ChunkWindow, BacktestResult]] = []
            open_parts: list[pd.DataFrame] = []
            mark_parts: list[pd.DataFrame] = []
            share_frames: list[pd.DataFrame] = []
            effective_frames: list[pd.DataFrame] = []
            raw_next_carry: list[pd.DataFrame] = []
            allocations: list[pd.DataFrame] = []
            boundary_carry: list[pd.Series] = []
            clipped_cells = 0
            allocation_totals = {key: 0.0 for key in ALLOCATION_STAT_KEYS}
            try:
                for position, (item, probe, carry) in enumerate(
                    zip(prepared, probes, carry_frames, strict=True), start=1
                ):
                    _check_cancelled(should_cancel)
                    _report_chunk(
                        progress,
                        "allocation",
                        position,
                        len(prepared),
                        item.window,
                        iteration,
                    )
                    data = loader(item.window, state)
                    valuation = data.close.ffill(axis=0)
                    price = execution_price(data.open, valuation)
                    prior_cash, prior_shares = previous_store.read(
                        position, data.index, data.columns
                    )
                    prior_init_cash = float(
                        params.init_cash if item.window.warmup is None else prior_cash.iloc[0]
                    )
                    budget = execution_budget(prior_cash, prior_shares, price, prior_init_cash)
                    cash_open = execution_cash(prior_cash, prior_init_cash)
                    effective_targets, turnover_stats = constrain_one_way_turnover(
                        item.targets,
                        budget,
                        cash_open,
                        price,
                        prior_shares,
                        strategy.risk.max_one_way_turnover,
                    )
                    risk_stats = {**item.single_cap_stats, **turnover_stats}
                    seed_plan = None
                    init_cash = float(params.init_cash)
                    if state is not None:
                        seed_plan = seed_inputs(
                            state, data, valuation, float(strategy.costs.slippage)
                        )
                        init_cash = float(state.cash)
                    warmup_rows = 0 if item.window.warmup is None else 1
                    plan = build_order_plan(
                        data,
                        effective_targets,
                        strategy.execution,
                        probe,
                        budget,
                        cash_open,
                        prior_shares,
                        strategy.costs,
                        carry,
                        rule_state,
                        warmup_rows,
                        liquidations,
                        None if seed_plan is None else seed_plan.shares,
                        None if seed_plan is None else seed_plan.fixed_fees,
                    )
                    portfolio = engine._run_portfolio(
                        data,
                        valuation,
                        plan.shares,
                        params,
                        strategy,
                        plan.fees,
                        plan.fixed_fees,
                        size_type="targetamount",
                        init_cash=init_cash,
                    )
                    cash, shares = engine._state_frames(portfolio)
                    if state is not None:
                        engine._check_seed_cash(cash, state, "最终遍")
                    current_store.write(position, cash, shares)
                    allocation_witnesses = engine._clipped_cells(plan, portfolio)
                    clipped_cells += int(allocation_witnesses["cash_clipped_cells"])
                    orders = engine._drop_warmup_orders(
                        engine._order_records(portfolio, "最终运行"), data.index, warmup_rows
                    )
                    holdings = engine._holdings_frame(portfolio)
                    liquidation_records = engine._liquidation_records(plan, orders)
                    result = _result_from_iteration(
                        data,
                        strategy,
                        params,
                        portfolio,
                        plan,
                        orders,
                        holdings,
                        liquidation_records,
                        allocation_witnesses,
                        risk_stats,
                        carry,
                        effective_targets,
                        item.rank_rows,
                        iteration,
                    )
                    if len(orders):
                        order_days = pd.DatetimeIndex(orders["trade_date"])
                        rows = data.index.get_indexer(order_days)
                        cols = data.columns.get_indexer(orders["stock_code"])
                        open_parts.append(
                            pd.DataFrame(
                                {
                                    "trade_date": order_days.to_numpy(),
                                    "stock_code": orders["stock_code"].to_numpy(),
                                    "open": data.open.to_numpy()[rows, cols],
                                }
                            )
                        )
                        mark_parts.append(valuation.loc[order_days.unique()])
                    mark_parts.append(valuation.iloc[[-1]])
                    state = _physical_state(cash, shares, data, valuation, plan.rule_state)
                    rule_state = plan.rule_state
                    share_frames.append(plan.shares)
                    effective_frames.append(effective_targets)
                    allocations.append(plan.allocation)
                    raw_next_carry.append(cash_carry(plan.allocation, effective_targets, carry))
                    boundary_carry.append(
                        pending_carry(carry, plan.allocation, effective_targets)
                    )
                    for key in ALLOCATION_STAT_KEYS:
                        allocation_totals[key] += float(plan.plan_stats.get(key, 0) or 0)
                    current_results.append((item.window, replace(result, portfolio=None)))
                    del data, portfolio, cash, shares, prior_cash, prior_shares
                    gc.collect()
            except BaseException:
                current_store.cleanup()
                raise

            for position in range(1, len(raw_next_carry)):
                pending = boundary_carry[position - 1]
                if pending.notna().any():
                    row = raw_next_carry[position].index.get_loc(
                        pd.Timestamp(prepared[position].window.start)
                    )
                    columns = pending.index[pending.notna()]
                    raw_next_carry[position].loc[
                        raw_next_carry[position].index[row], columns
                    ] = pending.loc[columns].to_numpy(dtype="float64")
            next_carry = raw_next_carry
            if iteration == 1:
                global_carry = _concat_logical(next_carry, prepared)
                global_targets = _concat_logical(effective_frames, prepared)
                next_carry = _split_logical(
                    extend_carry(global_carry, global_targets), prepared
                )
            if not _frames_equal(next_carry, carry_frames):
                carry_iterations += 1
            stable = (
                previous_shares is not None
                and previous_effective is not None
                and previous_carry is not None
                and _frames_equal(previous_shares, share_frames)
                and _frames_equal(previous_effective, effective_frames)
                and _frames_equal(previous_carry, carry_frames)
            )
            cash_events = _carry_events(allocations)
            last_pending = engine._cash_pending_on_last_day(
                allocations[-1], effective_frames[-1].index[-1]
            )
            converged = stable or (
                previous_cash_events is not None
                and previous_cash_events == cash_events
                and clipped_cells == 0
                and last_pending == 0
            )
            carry_converged = _frames_equal(next_carry, raw_next_carry)
            _notify(
                progress,
                "allocation",
                iteration=iteration,
                carry_iterations=carry_iterations,
                **allocation_totals,
            )
            previous_store.cleanup()
            previous_store = current_store
            previous_shares = share_frames
            previous_effective = effective_frames
            previous_carry = carry_frames
            carry_frames = next_carry
            previous_cash_events = cash_events
            final_results = current_results
            final_opens = open_parts
            final_marks = mark_parts
            if converged:
                logger.info("整段资金分配在第 %d 轮稳定", iteration)
                break
        else:
            logger.warning("整段资金分配 %d 次迭代仍未稳定", MAX_ALLOCATION_ITERATIONS)

        elapsed = time.perf_counter() - started
        result = _stitch(
            final_results,
            final_opens,
            final_marks,
            days,
            benchmark,
            benchmark_info,
            strategy,
            params,
            elapsed,
        )
        metadata = dict(result.metadata)
        calibration = dict(metadata["calibration"])
        calibration.update(
            {
                "allocation_iterations": int(iteration),
                "allocation_converged": bool(converged),
                "carry_iterations": int(carry_iterations),
                "carry_converged": bool(carry_converged),
                "cash_carry_over_cells": 0,
            }
        )
        metadata["calibration"] = calibration
        metadata["execution"] = {
            **metadata.get("execution", {}),
            "mode": "streamed_fixed_point",
            "chunk_count": len(windows),
            "allocation_iterations": int(iteration),
            "allocation_converged": bool(converged),
        }
        return replace(result, metadata=metadata)
    finally:
        previous_store.cleanup()


def _load_window(
    window: ChunkWindow,
    state: ChunkState | None,
    strategy: StrategySpec,
    params: BacktestParams,
    db_path=None,
    factor_dir=None,
    stock_codes: tuple[str, ...] | None = None,
) -> MarketData:
    """按窗口取数：带上预热行，并把预热行的价格补成上一块收盘时的估值价。"""
    data = load_market_data(
        window.warmup or window.start,
        window.end,
        list(strategy.signal.factors),
        universe=strategy.universe,
        db_path=db_path,
        factor_dir=factor_dir,
        benchmark_code=strategy.benchmark.index_code,
        membership_end=params.end,
        stock_codes=stock_codes,
    )
    if window.warmup is None or state is None:
        return data
    return _warm_up(data, state)


def _global_stock_codes(
    universe: str,
    start: str,
    end: str,
    db_path=None,
) -> tuple[str, ...]:
    """取得整段区间股票并集，保证所有块使用同一列集合。"""
    spec = resolve_universe(universe)
    stocks = universe_stocks(spec, start, end, db_path=db_path)
    if stocks:
        return tuple(sorted(stocks))
    basic = quant.load_stock_basic(db_path=db_path)
    listed = pd.to_datetime(basic["list_date"])
    delisted = pd.to_datetime(basic["delist_date"])
    alive = (listed <= pd.Timestamp(end)) & (
        delisted.isna() | (delisted >= pd.Timestamp(start))
    )
    return tuple(sorted(basic.loc[alive, "stock_code"].astype(str)))


def run_chunked(
    strategy: StrategySpec,
    params: BacktestParams,
    *,
    cell_budget: int = CHUNK_CELL_BUDGET,
    progress: Progress | None = None,
    should_cancel: Cancelled | None = None,
    db_path=None,
    factor_dir=None,
    windows: list[ChunkWindow] | None = None,
) -> BacktestResult:
    """分块执行一次回测，返回拼接后的整段结果。

    参数:
        strategy: 策略配置。
        params: 整段区间与初始资金。
        cell_budget: 单块目标规模；未显式给窗口时用来切块。
        progress: 可选阶段回调，块序号写在 ``chunk_index`` / ``chunk_count`` 里。
        should_cancel: 可选取消检查，在每块开始前检查一次。
        db_path: 行情库路径。
        factor_dir: 因子数据根目录。
        windows: 直接指定分块窗口；None 表示按预算自动切块。
    """
    windows = windows or plan_chunks(
        params.start, params.end, strategy.universe, db_path=db_path, cell_budget=cell_budget
    )
    days = _trade_day_index(params.start, params.end, db_path)
    stock_codes = _global_stock_codes(
        strategy.universe, params.start, params.end, db_path
    )
    liquidations = load_liquidation_candidates(params.start, params.end, db_path=db_path)
    benchmark, benchmark_info = load_benchmark(
        strategy.benchmark.index_code, params.start, params.end, days, db_path
    )
    return run_windows(
        windows,
        strategy,
        params,
        days=days,
        loader=lambda window, state: _load_window(
            window, state, strategy, params, db_path, factor_dir, stock_codes
        ),
        benchmark=benchmark,
        benchmark_info=benchmark_info,
        liquidations=liquidations,
        progress=progress,
        should_cancel=should_cancel,
    )


POSITION_COLUMNS = (
    "Position Id",
    "Column",
    "Size",
    "Entry Timestamp",
    "Avg Entry Price",
    "Entry Fees",
    "Exit Timestamp",
    "Avg Exit Price",
    "Exit Fees",
    "PnL",
    "Return",
    "Direction",
    "Status",
)


def _positions_frame(rows: list[dict]) -> pd.DataFrame:
    """把拼接后的持仓记录整理成结果表。"""
    if not rows:
        return pd.DataFrame(columns=list(POSITION_COLUMNS))
    frame = pd.DataFrame(rows)
    frame["Position Id"] = np.arange(len(frame), dtype="int64")
    return frame[list(POSITION_COLUMNS)].reset_index(drop=True)


def _trade_stats(life: dict) -> tuple[float, float]:
    """按 VectorBT 的口径算出持仓的盈亏与收益率。"""
    entry_value = life["size"] * life["entry_price"]
    exit_value = life["size"] * life["exit_price"]
    pnl = (exit_value - entry_value) - life["entry_fees"] - life["exit_fees"]
    return pnl, pnl / entry_value


def _mark_exit_price(life: dict, price: float) -> float:
    """开仓状态下按最新价重算出场价（VectorBT 在每笔成交与区间末都会做这一步）。"""
    open_size = abs(life["position"])
    previous = life["exit_price"]
    if previous is None or not np.isfinite(previous):
        return float(price)
    exit_size_sum = life["size"] - open_size
    return (exit_size_sum * previous + open_size * float(price)) / life["size"]


def _replay_positions(
    orders: pd.DataFrame,
    opens: pd.DataFrame,
    marks: pd.DataFrame,
    days: pd.DatetimeIndex,
    columns: pd.Index,
    init_cash: float,
    slippage: float,
) -> pd.DataFrame:
    """用整段区间的成交单重放一遍 VectorBT，得到跨块连续的持仓表。

    分块运行时每块开头会把上一块收盘持股当成一次买入播种，直接拼各块的持仓表会把
    跨块持仓拆成多段。这里把整段区间的成交单（与不分块时逐笔一致）交给 VectorBT
    重跑一遍记账：成交价按记录的成交价、滑点设为 0、费用按记录的逐笔费用放进固定费用，
    因此每一笔的股数、成交价与费用都和正式运行相同，持仓表也与不分块一致。

    参数:
        orders: 整段区间的成交明细，含 ``trade_date``、``stock_code``、``Size``、``Price``、
            ``Fees``、``Side``。
        opens: 每笔成交对应的原始开盘价，含 ``trade_date``、``stock_code``、``open``。
        marks: 成交日与区间末的估值价（收盘价前向填充），用来重估持仓。
        days: 整段区间的交易日索引。
        columns: 股票列。
        init_cash: 初始资金。
        slippage: 滑点比例。
    """
    import vectorbt as vbt

    price = pd.DataFrame(np.nan, index=days, columns=columns, dtype="float32")
    close = marks.reindex(index=days, columns=columns).astype("float32")
    size = pd.DataFrame(np.nan, index=days, columns=columns, dtype="float64")
    fixed_fees = pd.DataFrame(0.0, index=days, columns=columns, dtype="float64")
    if not orders.empty:
        rows = days.get_indexer(pd.DatetimeIndex(orders["trade_date"]))
        cols = columns.get_indexer(orders["stock_code"])
        kept = (rows >= 0) & (cols >= 0)
        rows, cols = rows[kept], cols[kept]
        moved = orders.loc[kept]
        signed = np.where(moved["Side"].to_numpy() == "Buy", 1.0, -1.0)
        opening = (
            opens.set_index(["trade_date", "stock_code"])["open"]
            .reindex(
                pd.MultiIndex.from_arrays(
                    [moved["trade_date"].to_numpy(), moved["stock_code"].to_numpy()]
                )
            )
            .to_numpy(dtype="float64")
        )
        price.to_numpy()[rows, cols] = opening
        size.to_numpy()[rows, cols] = orders["Size"].to_numpy(dtype="float64")[kept] * signed
        fixed_fees.to_numpy()[rows, cols] = orders["Fees"].to_numpy(dtype="float64")[kept]
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="amount",
        price=price,
        fees=0.0,
        fixed_fees=fixed_fees,
        slippage=float(slippage),
        val_price=price,
        size_granularity=100,
        init_cash=float(init_cash),
        cash_sharing=True,
        call_seq="auto",
        allow_partial=True,
        log=True,
        freq="1D",
    )
    return portfolio.positions.records_readable.copy()


def _aggregate_calibration(items: list[dict]) -> dict:
    """按口径把各块的校准字段聚合成整段运行的校准。

    笔数与格数求和、极值取最大、区间末遗留取最后一块、收敛标记取「全部为真」；
    迭代次数取块内最大值，逐块明细记在运行元数据的 ``chunks`` 里。

    参数:
        items: 各块的校准字典。
    """
    merged: dict = {}
    for key in SUM_KEYS:
        merged[key] = sum(float(item.get(key, 0) or 0) for item in items)
    for key in MAX_KEYS:
        merged[key] = max(float(item.get(key, 0) or 0) for item in items)
    for key in LAST_KEYS:
        merged[key] = int(items[-1].get(key, 0) or 0)
    merged["cash_buffer"] = items[0].get("cash_buffer")
    merged["allocation_iterations"] = max(int(item.get("allocation_iterations", 0) or 0) for item in items)
    merged["carry_iterations"] = max(int(item.get("carry_iterations", 0) or 0) for item in items)
    merged["allocation_converged"] = all(bool(item.get("allocation_converged")) for item in items)
    merged["carry_converged"] = all(bool(item.get("carry_converged")) for item in items)
    return merged


def _stitch(
    results: list[tuple[ChunkWindow, BacktestResult]],
    open_parts: list[pd.DataFrame],
    mark_parts: list[pd.DataFrame],
    days: pd.DatetimeIndex,
    benchmark: pd.Series | None,
    benchmark_info: dict | None,
    strategy: StrategySpec,
    params: BacktestParams,
    elapsed: float,
) -> BacktestResult:
    """把各块结果拼成一次运行的标准化输出。

    参数:
        results: 各块的窗口与结果。
        open_parts: 各块成交对应的开盘价明细，用来重放持仓表。
        mark_parts: 各块成交日与区间末的估值价，用来重放持仓表。
        days: 整段区间的交易日索引。
        benchmark: 整段区间的基准收盘价。
        benchmark_info: 基准口径说明。
        strategy: 策略配置。
        params: 整段区间与初始资金。
        elapsed: 分块执行总耗时。
    """
    orders_parts: list[pd.DataFrame] = []
    holding_parts: list[pd.DataFrame] = []
    rejection_parts: list[pd.DataFrame] = []
    liquidation_parts: list[pd.DataFrame] = []
    allocation_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    value_parts: list[pd.Series] = []
    cash_parts: list[pd.Series] = []
    order_offset = 0
    for window, result in results:
        warmup = None if window.warmup is None else pd.Timestamp(window.warmup)
        start = pd.Timestamp(window.start)
        orders = result.orders.copy()
        if warmup is not None:
            seeded = pd.DatetimeIndex(orders["trade_date"]) <= warmup
            orders = orders.loc[~seeded].copy()
            orders["Order Id"] = orders["Order Id"].astype("int64") - int(seeded.sum()) + order_offset
        order_offset += len(orders)
        orders_parts.append(orders.reset_index(drop=True))
        for frame, bucket in (
            (result.holdings, holding_parts),
            (result.rejections, rejection_parts),
            (result.liquidations, liquidation_parts),
            (result.allocations, allocation_parts),
        ):
            if frame.empty:
                continue
            kept = frame.loc[pd.DatetimeIndex(frame["trade_date"]) >= start]
            bucket.append(kept.reset_index(drop=True))
        target = result.target_weights
        daily = result.daily
        if warmup is not None:
            target = target.loc[target.index >= start]
            daily = daily.loc[daily.index >= start]
        target_parts.append(target)
        value_parts.append(daily["portfolio_value"])
        cash_parts.append(daily["cash"])

    orders = pd.concat(orders_parts, ignore_index=True) if orders_parts else pd.DataFrame()
    holdings = (
        pd.concat(holding_parts, ignore_index=True)
        if holding_parts
        else pd.DataFrame(columns=["trade_date", "stock_code", "market_value", "weight"])
    )
    rejections = (
        pd.concat(rejection_parts, ignore_index=True) if rejection_parts else pd.DataFrame(columns=["trade_date", "stock_code", "reason"])
    )
    liquidations = (
        pd.concat(liquidation_parts, ignore_index=True)
        if liquidation_parts
        else pd.DataFrame(columns=["trade_date", "stock_code", "delist_date", "size", "price", "value"])
    )
    allocations = (
        pd.concat(allocation_parts, ignore_index=True) if allocation_parts else pd.DataFrame()
    )
    target_weights = pd.concat(target_parts)
    target_weights = target_weights.reindex(index=days).sort_index()
    target_weights = target_weights.reindex(columns=sorted(target_weights.columns))
    native_target_index = results[0][1].target_weights.index
    target_weights.index = target_weights.index.astype(native_target_index.dtype)
    target_weights.index.name = native_target_index.name
    value = pd.concat(value_parts).sort_index()
    cash = pd.concat(cash_parts).sort_index()
    opens = pd.concat(open_parts, ignore_index=True) if open_parts else pd.DataFrame(
        columns=["trade_date", "stock_code", "open"]
    )
    marks = pd.concat(mark_parts).groupby(level=0).last() if mark_parts else pd.DataFrame()
    positions = _replay_positions(
        orders,
        opens,
        marks,
        days,
        target_weights.columns,
        params.init_cash,
        float(strategy.costs.slippage),
    )
    # VectorBT 的可读持仓表沿用输入索引的时间精度；整段交易日索引在拼接时可能被
    # pandas 升成 ns。按块内原生结果恢复 dtype，保证分块与单块结果逐位、逐类型一致。
    native_positions = next((item.positions for _, item in results if not item.positions.empty), None)
    if native_positions is not None:
        for column in ("Entry Timestamp", "Exit Timestamp"):
            if column in positions and column in native_positions:
                positions[column] = positions[column].astype(native_positions[column].dtype)
    daily = derive_daily(value, cash, orders, benchmark)
    calibration = _aggregate_calibration([item.metadata["calibration"] for _, item in results])
    calibration.update(weight_witnesses(target_weights, holdings))
    first = results[0][1]
    metadata = {
        "strategy": first.metadata.get("strategy"),
        "strategy_hash": first.metadata.get("strategy_hash"),
        "params": {"start": params.start, "end": params.end, "init_cash": params.init_cash},
        "universe": strategy.universe,
        "benchmark": benchmark_info,
        "price_adjust": first.metadata.get("price_adjust"),
        "cash_retry_policy": first.metadata.get("cash_retry_policy"),
        "risk": first.metadata.get("risk"),
        "execution": {**first.metadata.get("execution", {}), "chunk_count": len(results)},
        "versions": first.metadata.get("versions"),
        "vectorbt_version": first.metadata.get("vectorbt_version"),
        "panel": {"trading_days": int(len(days)), "stocks": int(len(target_weights.columns))},
        "rank_rows": int(sum(item.metadata.get("rank_rows", 0) or 0 for _, item in results)),
        "chunks": [
            {
                "index": index,
                "start": window.start,
                "end": window.end,
                "warmup": window.warmup,
                "trading_days": int(len(item.daily) - (0 if window.warmup is None else 1)),
                "stocks": int(len(item.target_weights.columns)),
                "allocation_iterations": int(item.metadata["calibration"]["allocation_iterations"]),
                "allocation_converged": bool(item.metadata["calibration"]["allocation_converged"]),
                "cash_carry_over_cells": int(
                    item.metadata["calibration"].get("cash_carry_over_cells", 0) or 0
                ),
            }
            for index, (window, item) in enumerate(results, start=1)
        ],
        "calibration": calibration,
        "elapsed_seconds": round(elapsed, 3),
    }
    logger.info(
        "分块回测完成：区间=%s~%s 块数=%d 订单=%d 耗时=%.1fs",
        params.start,
        params.end,
        len(results),
        len(orders),
        metadata["elapsed_seconds"],
    )
    return BacktestResult(
        portfolio=None,
        daily=daily,
        orders=orders,
        positions=positions,
        holdings=holdings,
        target_weights=target_weights,
        rejections=rejections,
        metrics=summarize(daily),
        metadata=metadata,
        liquidations=liquidations,
        allocations=allocations,
        final_state=None,
    )
