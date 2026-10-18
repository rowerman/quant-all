"""分块回测的等价性测试：分块结果必须与不分块逐位一致。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import quant
from backtest.chunked import ChunkWindow, _warm_up, plan_chunks, run_loaded, run_windows
from backtest.config import MAX_CHUNKS
from backtest.data import MarketData, load_market_data
from backtest.errors import ResourceLimitError, RunCancelled
from backtest.orders import liquidation_cells
from backtest.providers import register_provider
from backtest.specs import (
    BacktestParams,
    BenchmarkSpec,
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    RiskSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.snapshot import data_snapshot_from_market_data

DATES = pd.date_range("2024-01-29", periods=60, freq="B")
STOCKS = ["SH600000", "SH600001", "SH600002", "SH600003", "SZ000001", "SZ000002"]
# 退市股的最后一条行情日：正好落在第一块的最后一个交易日，用来盯住块边界的清算归属
DELIST_ROW = 23
DELIST_STOCK = "SH600003"
BENCHMARK = pd.Series(np.linspace(1000.0, 1100.0, len(DATES)), index=DATES)
BENCHMARK_INFO = {
    "code": "H00300.CSI",
    "name": "300收益",
    "return_type": "total_return",
    "start": str(DATES[0].date()),
    "end": str(DATES[-1].date()),
}


class WeeklyTopTwoProvider:
    """每周一按 MOM_20 选前两只等权，用于验证提供者分块一致性。"""

    name = "test_chunk_weekly_top2"
    version = "1"
    description = "测试用周频 Top-2 提供者"
    required_factors = ("MOM_20",)

    def build(self, data, eligible, params):
        score = data.factors["MOM_20"].astype("float64")
        weights = pd.DataFrame(
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        for day in data.index[data.index.dayofweek == 0]:
            chosen = (
                score.loc[day]
                .where(eligible.loc[day])
                .dropna()
                .sort_values(ascending=False)
                .head(2)
                .index
            )
            weights.loc[day] = 0.0
            if len(chosen):
                weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


register_provider(WeeklyTopTwoProvider())


def _panel(make_data, *, block_buy_row=None, suspended_row=None) -> MarketData:
    """构造 60 个交易日、6 只股票的面板：周频调仓、一只中途退市、可选涨停与停牌。"""
    steps = np.arange(len(DATES), dtype="float64")
    prices = {
        "SH600000": 12.0 + 0.05 * steps,
        "SH600001": 35.0 + 0.10 * steps,
        "SH600002": 8.0 - 0.02 * steps,
        "SH600003": 210.0 + 0.30 * steps,
        "SZ000001": 16.0 + 0.03 * steps,
        "SZ000002": 5.0 + 0.01 * steps,
    }
    scores = {
        "SH600000": 6.0,
        "SH600001": 5.0,
        "SH600002": 4.0,
        "SH600003": 3.0,
        "SZ000001": 2.0,
        "SZ000002": 1.0,
    }
    open_price = pd.DataFrame({code: value for code, value in prices.items()}, index=DATES)
    close_price = open_price * 1.002
    score = pd.DataFrame({code: [value] * len(DATES) for code, value in scores.items()}, index=DATES)
    can_buy = pd.DataFrame(True, index=DATES, columns=STOCKS)
    can_sell = pd.DataFrame(True, index=DATES, columns=STOCKS)
    suspended = pd.DataFrame(False, index=DATES, columns=STOCKS)
    # 退市股：最后一条行情日之后没有价格，因子也取不到
    after = DATES[DELIST_ROW + 1 :]
    open_price.loc[after, DELIST_STOCK] = np.nan
    close_price.loc[after, DELIST_STOCK] = np.nan
    score.loc[after, DELIST_STOCK] = np.nan
    delist = pd.Series(pd.NaT, index=STOCKS)
    delist[DELIST_STOCK] = DATES[DELIST_ROW + 2]
    if block_buy_row is not None:
        can_buy.loc[DATES[block_buy_row], "SH600001"] = False
    if suspended_row is not None:
        suspended.loc[DATES[suspended_row], "SH600002"] = True
    return make_data(
        dates=DATES,
        stocks=STOCKS,
        open=open_price.astype("float32"),
        close=close_price.astype("float32"),
        can_buy_open_next=can_buy,
        can_sell_open_next=can_sell,
        is_suspended=suspended,
        factors={"MOM_20": score.astype("float32")},
        delist_date=delist,
        benchmark=BENCHMARK,
        benchmark_info=BENCHMARK_INFO,
    )


def _strategy(**overrides) -> StrategySpec:
    """默认周频调仓、4 只等权的测试策略。"""
    fields = {
        "signal": SignalSpec(factors=("MOM_20",), weights=(1.0,), standardize=False),
        "portfolio": PortfolioSpec(top_n=4),
        "rebalance": RebalanceSpec(frequency="weekly"),
    }
    fields.update(overrides)
    return StrategySpec(**fields)


def _provider_strategy(**overrides) -> StrategySpec:
    """默认周一的 Top-2 提供者策略。"""
    fields = {
        "signal": SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec("test_chunk_weekly_top2"),
        ),
        "portfolio": PortfolioSpec(selection="provider"),
        "rebalance": RebalanceSpec(frequency="custom"),
    }
    fields.update(overrides)
    return StrategySpec(**fields)


def _windows(cuts: list[int]) -> list[ChunkWindow]:
    """按交易日下标切块，除第一块外都带上预热日。"""
    bounds = [0, *cuts, len(DATES)]
    windows = []
    for first, last in zip(bounds[:-1], bounds[1:]):
        windows.append(
            ChunkWindow(
                start=str(DATES[first].date()),
                end=str(DATES[last - 1].date()),
                warmup=None if first == 0 else str(DATES[first - 1].date()),
            )
        )
    return windows


def _loader(data: MarketData):
    """返回分块取数回调：按窗口切面板，并把预热行的价格补成上一块收盘估值价。"""

    def load(window: ChunkWindow, state):
        chunk = data.crop(window.warmup or window.start, window.end)
        return chunk if state is None else _warm_up(chunk, state)

    return load


def _run_chunked(data: MarketData, strategy: StrategySpec, params: BacktestParams, cuts: list[int]):
    """按给定切点跑一次分块回测。"""
    data = replace(
        data,
        data_snapshot=data.data_snapshot or data_snapshot_from_market_data(data),
    )
    return run_windows(
        _windows(cuts),
        strategy,
        params,
        days=pd.DatetimeIndex(data.index),
        loader=_loader(data),
        benchmark=data.benchmark,
        benchmark_info=data.benchmark_info,
        liquidations=liquidation_cells(data),
        data_snapshot=data.data_snapshot,
    )


def _run_single(data: MarketData, strategy: StrategySpec, params: BacktestParams):
    """用一个资源窗口运行，作为资源分块等价性的基准。"""
    data = replace(
        data,
        data_snapshot=data.data_snapshot or data_snapshot_from_market_data(data),
    )
    return run_loaded(data, strategy, params)


def _assert_same(single, chunked) -> None:
    """逐位比较单块与分块的全部结果表。"""
    pd.testing.assert_frame_equal(single.daily, chunked.daily, check_exact=True)
    pd.testing.assert_frame_equal(single.orders, chunked.orders, check_exact=True)
    pd.testing.assert_frame_equal(single.holdings, chunked.holdings, check_exact=True)
    pd.testing.assert_frame_equal(single.rejections, chunked.rejections, check_exact=True)
    pd.testing.assert_frame_equal(single.liquidations, chunked.liquidations, check_exact=True)
    pd.testing.assert_frame_equal(single.allocations, chunked.allocations, check_exact=True)
    pd.testing.assert_frame_equal(single.positions, chunked.positions, check_exact=True)
    pd.testing.assert_frame_equal(
        single.target_weights, chunked.target_weights, check_exact=True, check_like=True
    )
    assert single.metrics == chunked.metrics
    assert single.metadata["calibration"] == chunked.metadata["calibration"]
    assert single.metadata["snapshot_id"] == chunked.metadata["snapshot_id"]
    assert (
        single.metadata["snapshot"]["components"]
        == chunked.metadata["snapshot"]["components"]
    )
    assert (
        single.metadata["snapshot"]["scope"]
        == chunked.metadata["snapshot"]["scope"]
    )


def test_chunked_matches_single_block(make_data):
    """普通切块：订单、日结果、持仓、目标权重、分配明细与指标逐位一致。"""
    data = _panel(make_data)
    strategy = _strategy()
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 300_000.0)

    single = _run_single(data, strategy, params)
    chunked = _run_chunked(data, strategy, params, [24, 40])

    assert chunked.metadata["execution"]["chunk_count"] == 3
    assert len(chunked.metadata["chunks"]) == 3
    assert chunked.metadata["chunks"][1]["warmup"] == str(DATES[23].date())
    _assert_same(single, chunked)


def test_chunked_matches_single_block_with_provider(make_data):
    """提供者策略跨资源块时目标权重和全部结果表逐位一致。"""
    data = _panel(make_data)
    strategy = _provider_strategy()
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 300_000.0)

    single = _run_single(data, strategy, params)
    chunked = _run_chunked(data, strategy, params, [24, 40])

    assert chunked.metadata["signal"]["kind"] == "provider"
    assert chunked.metadata["signal"]["name"] == "test_chunk_weekly_top2"
    _assert_same(single, chunked)


def test_chunked_matches_single_block_with_industry_cap(make_data):
    """行业面板跨资源块裁剪后，单块与分块结果仍逐位一致。"""
    data = _panel(make_data)
    industry = pd.DataFrame(
        [[1, 1, 1, 2, 2, 3]] * len(DATES),
        index=DATES,
        columns=STOCKS,
        dtype="int16",
    )
    data = __import__("dataclasses").replace(
        data, industry_l1=industry, industry_labels={1: "A", 2: "B", 3: "C"}
    )
    strategy = _strategy(risk=RiskSpec(max_industry_weight=0.5))
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 300_000.0)
    single = _run_single(data, strategy, params)
    chunked = _run_chunked(data, strategy, params, [24, 40])
    assert single.metadata["calibration"]["industry_cap_days"] > 0
    _assert_same(single, chunked)


def test_chunked_matches_single_block_across_blocked_and_delisted_cells(make_data):
    """涨停被拦、停牌与块边界退市清算同时出现时分块仍逐位一致。"""
    data = _panel(make_data, block_buy_row=23, suspended_row=25)
    strategy = _strategy()
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 300_000.0)

    single = _run_single(data, strategy, params)
    assert len(single.rejections) > 0
    assert len(single.liquidations) == 1
    chunked = _run_chunked(data, strategy, params, [23, 24, 41])
    _assert_same(single, chunked)


def test_chunked_matches_single_block_for_single_cut(make_data):
    """只切一条边界（含退市日）时也逐位一致。"""
    data = _panel(make_data)
    strategy = _strategy()
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 1_000_000.0)

    single = _run_single(data, strategy, params)
    chunked = _run_chunked(data, strategy, params, [24])

    assert chunked.metadata["execution"]["chunk_count"] == 2
    _assert_same(single, chunked)


def test_causal_segments_are_not_limited_by_global_iteration_count(make_data):
    """超过 24 个调仓段时逐段收敛，不能再依赖整段迭代向后传播。"""
    data = _panel(make_data)
    # 本用例只验证长调仓链；恢复测试面板里专门用于退市边界测试的股票。
    steps = np.arange(len(DATES), dtype="float64")
    data.open.loc[:, DELIST_STOCK] = (210.0 + 0.30 * steps).astype("float32")
    data.close.loc[:, DELIST_STOCK] = (data.open[DELIST_STOCK] * 1.002).astype("float32")
    data.factors["MOM_20"].loc[:, DELIST_STOCK] = 3.0
    data.delist_date.loc[DELIST_STOCK] = pd.NaT
    strategy = _strategy(rebalance=RebalanceSpec(frequency="daily"))
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 300_000.0)

    result = _run_single(data, strategy, params)
    calibration = result.metadata["calibration"]

    assert calibration["allocation_segments"] > 24
    assert calibration["allocation_iterations_total"] >= calibration["allocation_segments"]
    assert calibration["allocation_iterations_max"] <= 24
    nonconverged = [
        (item["start"], item["end"], item["allocation_iterations"], item["cash_clipped_cells"])
        for item in result.metadata["segments"]
        if not item["allocation_converged"]
    ]
    assert calibration["nonconverged_segments"] == 0, nonconverged
    assert calibration["allocation_converged"] is True
    clipped = [
        (item["start"], item["end"], item["cash_clipped_cells"])
        for item in result.metadata["segments"]
        if item["cash_clipped_cells"]
    ]
    assert calibration["cash_clipped_cells"] == 0, clipped


def test_causal_segments_honor_cancel_between_segments(make_data):
    """因果求解要在调仓段边界响应取消。"""
    data = _panel(make_data)
    strategy = _strategy()
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 1_000_000.0)
    calls = 0

    def cancel_during_allocation() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 6

    with pytest.raises(RunCancelled):
        run_windows(
            _windows([24, 40]),
            strategy,
            params,
            days=DATES,
            loader=_loader(data),
            benchmark=BENCHMARK,
            benchmark_info=BENCHMARK_INFO,
            liquidations=liquidation_cells(data),
            should_cancel=cancel_during_allocation,
        )
    assert calls >= 6


def test_plan_chunks_splits_by_cell_budget():
    """分块计划要覆盖整段区间、互不重叠，并带上上一块最后一天的预热日。"""
    start, end = "2016-01-01", "2026-10-14"
    windows = plan_chunks(start, end, "all_a", cell_budget=4_000_000, max_chunks=MAX_CHUNKS)

    assert len(windows) >= 2
    assert windows[0].warmup is None
    for previous, current in zip(windows[:-1], windows[1:]):
        assert current.warmup == previous.end
        assert current.start > previous.end
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(quant.load_calendar(start=start, end=end)["trade_date"]))
    )
    covered: list[pd.Timestamp] = []
    for window in windows:
        covered.extend(calendar[(calendar >= window.start) & (calendar <= window.end)])
    assert list(covered) == list(calendar)


def test_plan_chunks_rejects_too_many_blocks():
    """块数超过上限要直接报错，而不是把机器占满。"""
    with pytest.raises(ResourceLimitError, match="超过上限"):
        plan_chunks("2016-01-01", "2026-10-14", "all_a", cell_budget=20_000, max_chunks=3)


def test_chunked_matches_single_block_on_real_data():
    """真实数据（沪深300 池）：任意切块后全部标准结果与单块逐位一致。"""
    from backtest.chunked import run_chunked

    strategy = StrategySpec(
        name="真实数据分块对照",
        universe="index:000300.SH",
        signal=SignalSpec(factors=("MOM_20",), weights=(1.0,)),
        portfolio=PortfolioSpec(top_n=20),
        rebalance=RebalanceSpec(frequency="weekly"),
        benchmark=BenchmarkSpec(index_code="H00300.CSI"),
    )
    params = BacktestParams("2025-01-01", "2026-10-14", 1_000_000.0)
    data = load_market_data(
        params.start, params.end, ["MOM_20"], universe=strategy.universe,
        benchmark_code="H00300.CSI",
    )
    single = run_loaded(data, strategy, params)
    chunked = run_chunked(
        strategy,
        params,
        windows=[
            ChunkWindow("2025-01-01", "2025-09-30"),
            ChunkWindow("2025-10-01", "2026-03-31", warmup="2025-09-30"),
            ChunkWindow("2026-04-01", "2026-10-14", warmup="2026-03-31"),
        ],
    )

    assert chunked.metadata["execution"]["chunk_count"] == 3
    _assert_same(single, chunked)
