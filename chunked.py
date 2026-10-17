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
import time
from typing import Callable

import numpy as np
import pandas as pd

import quant
from backtest.config import (
    CHUNK_CELL_BUDGET,
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
    weight_witnesses,
)
import backtest.engine as engine
from backtest.errors import ResourceLimitError, RunCancelled
from backtest.logging_conf import get_logger
from backtest.metrics import summarize
from backtest.orders import (
    execution_targets,
    liquidation_cells,
)
from backtest.risk import cap_target_weights
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
    "liquidity_retry_filled_cells",
    "allocated_cells",
    "bumped_one_lot_cells",
    "unaffordable_lot_cells",
    "no_price_cells",
    "cash_shrunk_cells",
    "cash_limited_cells",
    "cash_limited_value",
    "liquidity_limited_cells",
    "liquidity_limited_value",
    "liquidity_unavailable_cells",
    "liquidity_shortfall_shares",
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
    "industry_cap_days",
    "industry_cap_groups",
    "industry_cap_cells",
    "industry_cap_excess_weight",
    "industry_cap_residual_cash",
    "industry_missing_cells",
    "risk_residual_cash",
    "industry_inherited_over_cap_days",
    "industry_inherited_over_cap_groups",
    "industry_inherited_excess_weight",
    "turnover_limited_days",
    "turnover_limited_cells",
    "cash_carry_over_cells",
)
MAX_KEYS = (
    "fee_matrix_max_diff",
    "allocation_gap_max_shares",
    "single_cap_max_weight",
    "industry_cap_target_max_weight",
    "industry_post_turnover_max_weight",
    "turnover_before_max",
    "turnover_after_max",
    "turnover_limit_error_max",
)
LAST_KEYS = (
    "pending_at_end_cells",
    "cash_pending_at_end_cells",
    "liquidity_pending_at_end_cells",
)


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
    risk_stats: dict


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


def _segment_windows(
    days: pd.DatetimeIndex,
    prepared: list[_PreparedChunk],
) -> list[ChunkWindow]:
    """按真实调仓执行日划分因果求解段。

    第一段从回测起点开始并包含第一次调仓；之后每个执行日开启新段。后一段带上
    前一交易日作为预热行，使信号平移、持仓播种和规则状态都能连续交接。
    """
    targets = _concat_logical([item.targets for item in prepared], prepared).reindex(days)
    execution_days = pd.DatetimeIndex(
        targets.index[targets.notna().any(axis=1).to_numpy()]
    )
    starts = [pd.Timestamp(days[0])]
    starts.extend(pd.Timestamp(day) for day in execution_days[1:])
    starts = list(dict.fromkeys(starts))
    positions = {pd.Timestamp(day): position for position, day in enumerate(days)}
    windows: list[ChunkWindow] = []
    for index, start in enumerate(starts):
        start_position = positions[start]
        if index + 1 < len(starts):
            end = pd.Timestamp(days[positions[starts[index + 1]] - 1])
        else:
            end = pd.Timestamp(days[-1])
        warmup = None if start_position == 0 else pd.Timestamp(days[start_position - 1])
        windows.append(
            ChunkWindow(
                start=str(start.date()),
                end=str(end.date()),
                warmup=None if warmup is None else str(warmup.date()),
            )
        )
    return windows


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
        target_weights, risk_stats = cap_target_weights(
            target_weights,
            strategy.risk.max_single_weight,
            data.industry_l1,
            strategy.risk.max_industry_weight,
            data.industry_labels,
        )
        prepared.append(
            _PreparedChunk(
                window=window,
                target_weights=target_weights,
                targets=execution_targets(target_weights),
                rank_rows=int(ranks.notna().sum().sum()),
                risk_stats=risk_stats,
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
    """按给定资源窗口准备信号，再按调仓执行段因果求解并拼接结果。"""
    started = time.perf_counter()
    logger.info("因果分段回测：区间=%s~%s 共 %d 个资源块", params.start, params.end, len(windows))
    prepared = _prepare_chunks(windows, strategy, days, loader, progress, should_cancel)
    segments = _segment_windows(days, prepared)
    mask = rebalance_mask(days, strategy.rebalance.frequency)
    results: list[tuple[ChunkWindow, BacktestResult]] = []
    open_parts: list[pd.DataFrame] = []
    mark_parts: list[pd.DataFrame] = []
    state: ChunkState | None = None
    for position, window in enumerate(segments, start=1):
        _check_cancelled(should_cancel)
        _report_chunk(progress, "segment", position, len(segments), window)
        data = loader(window, state)
        valuation = data.close.ffill(axis=0)

        def segment_progress(stage: str, payload: dict) -> None:
            _notify(
                progress,
                stage,
                **payload,
                segment_index=position,
                segment_count=len(segments),
            )

        result = engine.run_backtest(
            data,
            strategy,
            params,
            progress=segment_progress,
            initial_state=state,
            warmup_rows=0 if window.warmup is None else 1,
            rebalance_days=mask,
            liquidations=liquidations,
            chunk_boundary=position < len(segments),
        )
        if result.final_state is None:
            raise ResourceLimitError("调仓段执行没有返回可交接状态")
        if len(result.orders):
            order_days = pd.DatetimeIndex(result.orders["trade_date"])
            rows = data.index.get_indexer(order_days)
            cols = data.columns.get_indexer(result.orders["stock_code"])
            open_parts.append(
                pd.DataFrame(
                    {
                        "trade_date": order_days.to_numpy(),
                        "stock_code": result.orders["stock_code"].to_numpy(),
                        "open": data.open.to_numpy()[rows, cols],
                    }
                )
            )
            mark_parts.append(valuation.loc[order_days.unique()])
        mark_parts.append(valuation.iloc[[-1]])
        state = result.final_state
        results.append((window, replace(result, portfolio=None)))
        del data, result, valuation
        gc.collect()

    return _stitch(
        results,
        open_parts,
        mark_parts,
        days,
        benchmark,
        benchmark_info,
        strategy,
        params,
        time.perf_counter() - started,
        resource_windows=windows,
    )


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
        industry_level="L1" if strategy.risk.max_industry_weight is not None else None,
    )
    if window.warmup is None or state is None:
        return data
    return _warm_up(data, state)


def run_loaded(
    data: MarketData,
    strategy: StrategySpec,
    params: BacktestParams,
    *,
    progress: Progress | None = None,
    should_cancel: Cancelled | None = None,
) -> BacktestResult:
    """对已经加载到内存的面板执行因果分段回测。

    小面板与数据库分块面板共用 :func:`run_windows`，因此资源切块不会改变求解边界。
    """
    window = ChunkWindow(params.start, params.end)

    def loader(part: ChunkWindow, state: ChunkState | None) -> MarketData:
        chunk = data.crop(part.warmup or part.start, part.end)
        return chunk if state is None else _warm_up(chunk, state)

    return run_windows(
        [window],
        strategy,
        params,
        days=pd.DatetimeIndex(data.index),
        loader=loader,
        benchmark=data.benchmark,
        benchmark_info=data.benchmark_info,
        liquidations=liquidation_cells(data),
        progress=progress,
        should_cancel=should_cancel,
    )


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
    iterations = [int(item.get("allocation_iterations", 0) or 0) for item in items]
    merged["allocation_segments"] = len(items)
    merged["allocation_iterations_total"] = sum(iterations)
    merged["allocation_iterations_max"] = max(iterations, default=0)
    merged["nonconverged_segments"] = sum(
        not bool(item.get("allocation_converged")) for item in items
    )
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
    resource_windows: list[ChunkWindow] | None = None,
) -> BacktestResult:
    """把各调仓段结果拼成一次运行的标准化输出。

    参数:
        results: 各调仓执行段的窗口与结果。
        open_parts: 各块成交对应的开盘价明细，用来重放持仓表。
        mark_parts: 各块成交日与区间末的估值价，用来重放持仓表。
        days: 整段区间的交易日索引。
        benchmark: 整段区间的基准收盘价。
        benchmark_info: 基准口径说明。
        strategy: 策略配置。
        params: 整段区间与初始资金。
        elapsed: 分块执行总耗时。
        resource_windows: 原始资源窗口；只用于记录取数切块，不参与求解。
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
        "execution": {
            **first.metadata.get("execution", {}),
            "mode": "causal_segmented",
            "chunk_count": len(resource_windows or results),
            "segment_count": len(results),
            "allocation_iterations": calibration["allocation_iterations_max"],
            "allocation_converged": calibration["allocation_converged"],
        },
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
                "trading_days": int(
                    ((days >= pd.Timestamp(window.start)) & (days <= pd.Timestamp(window.end))).sum()
                ),
                "stocks": int(len(target_weights.columns)),
            }
            for index, window in enumerate(resource_windows or [window for window, _ in results], start=1)
        ],
        "segments": [
            {
                "index": index,
                "start": window.start,
                "end": window.end,
                "warmup": window.warmup,
                "allocation_iterations": int(item.metadata["calibration"]["allocation_iterations"]),
                "allocation_converged": bool(item.metadata["calibration"]["allocation_converged"]),
                "cash_clipped_cells": int(
                    item.metadata["calibration"].get("cash_clipped_cells", 0) or 0
                ),
            }
            for index, (window, item) in enumerate(results, start=1)
        ],
        "calibration": calibration,
        "elapsed_seconds": round(elapsed, 3),
    }
    logger.info(
        "因果分段回测完成：区间=%s~%s 资源块=%d 调仓段=%d 订单=%d 耗时=%.1fs",
        params.start,
        params.end,
        len(resource_windows or results),
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
