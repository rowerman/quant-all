"""执行层持仓状态与状态提供者的黄金用例。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.chunked import ChunkWindow, _warm_up, run_loaded, run_windows
from backtest.engine import BacktestResult, run_backtest
from backtest.errors import DataAlignmentError, ExecutionError, SpecError
from backtest.execution_state import (
    POSITION_STATE_COLUMNS,
    ProviderState,
)
from backtest.ledger import LedgerPortfolio, LedgerSnapshot
from backtest.orders import liquidation_cells
from backtest.pipeline import _stateful_runner
from backtest.providers import register_provider
from backtest.specs import (
    BacktestParams,
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.strategies.momentum_trailing_stop import MomentumTrailingStopProvider
from backtest.snapshot import data_snapshot_from_market_data
from backtest.trading_rules import BOARD_MAIN

BOARDS = pd.Series({"SH600000": BOARD_MAIN, "SZ000001": BOARD_MAIN})


def _state_rows(dates: pd.DatetimeIndex, code: str, prices: list[float], **overrides):
    """构造一只股票的逐日状态行。"""
    rows = []
    for day, price in zip(dates, prices):
        row = {
            "trade_date": day,
            "stock_code": code,
            "shares": 100.0,
            "avg_cost": 10.0,
            "price": price,
            "market_value": 100.0 * price,
            "weight": 1.0,
            "unrealized_return": price / 10.0 - 1.0,
            "high_water_price": max(prices[: len(rows) + 1]),
            "holding_since": dates[0],
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _provider_state(dates, rows, values=None) -> ProviderState:
    """构造仅用于测试的状态对象。"""
    values = [1000.0] * len(dates) if values is None else values
    return ProviderState(
        cash=pd.Series(0.0, index=dates),
        portfolio_value=pd.Series(values, index=dates),
        positions=pd.DataFrame(rows, columns=POSITION_STATE_COLUMNS),
    )


def _result(targets: pd.DataFrame, state: ProviderState | None) -> BacktestResult:
    """构造固定点测试需要的最小结果。"""
    return BacktestResult(
        portfolio=None,
        daily=pd.DataFrame(index=targets.index),
        orders=pd.DataFrame(),
        positions=pd.DataFrame(),
        holdings=pd.DataFrame(),
        target_weights=targets,
        rejections=pd.DataFrame(),
        metrics={},
        metadata={"execution": {}},
        provider_state=state,
    )


def test_provider_state_at_weights_slice_and_merge():
    """状态快照按日期读取、切片并合并。"""
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    rows = _state_rows(dates[:2], "SH600000", [10.0, 9.0])
    state = _provider_state(dates, rows)

    snapshot = state.at(dates[1])
    assert snapshot.cash == 0.0
    assert snapshot.weights().loc["SH600000"] == pytest.approx(1.0)
    assert snapshot.positions.loc["SH600000", "unrealized_return"] == pytest.approx(-0.1)
    assert state.at(dates[2]).positions.empty

    sliced = state.slice(dates[1:])
    assert list(sliced.dates) == list(dates[1:])
    assert sliced.positions["trade_date"].nunique() == 1
    with pytest.raises(DataAlignmentError):
        state.at(pd.Timestamp("2024-02-01"))
    with pytest.raises(DataAlignmentError):
        state.slice(pd.DatetimeIndex([pd.Timestamp("2024-02-01")]))

    merged = ProviderState.merge([state.slice(dates[:2]), state.slice(dates[1:])])
    assert merged is not None
    assert list(merged.dates) == list(dates)
    assert len(merged.positions) == 2


def test_ledger_state_keeps_total_return_after_split():
    """送转后平均成本和高水位同步调整，浮盈不出现伪跳变。"""
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = pd.DataFrame([10.0, 12.0, 6.0, 6.6], index=dates, columns=["SH600000"])
    targets = pd.DataFrame(
        [100.0, 100.0, 200.0, 200.0], index=dates, columns=["SH600000"]
    )
    action = pd.DataFrame(
        [
            {
                "stock_code": "SH600000",
                "ex_date": dates[2],
                "record_date": dates[1],
                "pay_date": dates[2],
                "share_multiplier": 2.0,
                "cash_div_per_share": 0.0,
                "action_type": "stock",
                "source": "test",
                "source_event_key": "split-1",
            }
        ]
    )
    portfolio = LedgerPortfolio.from_orders(
        close,
        targets,
        size_type="targetamount",
        price=close,
        init_cash=10_000.0,
        corporate_actions=action,
        boards=BOARDS,
    )
    state = portfolio.provider_state()
    split_day = state.at(dates[2]).positions.loc["SH600000"]
    assert split_day["shares"] == pytest.approx(200.0)
    assert split_day["avg_cost"] == pytest.approx(5.0)
    assert split_day["price"] == pytest.approx(6.0)
    assert split_day["unrealized_return"] == pytest.approx(0.2)
    assert split_day["high_water_price"] == pytest.approx(6.0)
    next_day = state.at(dates[3]).positions.loc["SH600000"]
    assert next_day["high_water_price"] == pytest.approx(6.6)
    assert portfolio.positions.records_readable["PnL"].iloc[0] == pytest.approx(320.0)


def test_chunk_snapshot_carries_high_water():
    """跨块交接保留建仓以来最高收盘价。"""
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame([10.0, 12.0, 11.5], index=dates, columns=["SH600000"])
    targets = pd.DataFrame([100.0, 100.0, 100.0], index=dates, columns=["SH600000"])
    first = LedgerPortfolio.from_orders(
        close.iloc[:2],
        targets.iloc[:2],
        size_type="targetamount",
        price=close.iloc[:2],
        init_cash=10_000.0,
        boards=BOARDS,
    )
    snapshot = first.snapshot()
    assert snapshot.high_water is not None
    second = LedgerPortfolio.from_orders(
        close.iloc[1:],
        targets.iloc[1:],
        size_type="targetamount",
        price=close.iloc[1:],
        init_cash=float(snapshot.cash),
        initial_state=LedgerSnapshot(
            cash=snapshot.cash,
            shares=snapshot.shares,
            lots=snapshot.lots,
            receivables=snapshot.receivables,
            last_prices=snapshot.last_prices,
            high_water=snapshot.high_water,
        ),
        boards=BOARDS,
        warmup_rows=1,
    )
    state = second.provider_state()
    assert state.at(dates[2]).positions.loc["SH600000", "high_water_price"] == pytest.approx(
        12.0
    )


def test_stateful_provider_must_accept_state():
    """声明读取状态却没有 state 参数时注册失败。"""

    class BadProvider:
        name = "test_bad_state_provider"
        version = "1"
        description = "缺少 state 参数"
        required_factors = ()
        uses_position_state = True

        def build(self, data, eligible, params):
            return data.close * 0

    with pytest.raises(SpecError, match="state"):
        register_provider(BadProvider())


def test_state_feedback_converges_and_fails_on_oscillation():
    """目标重复即收敛，振荡超过上限直接报错。"""
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    columns = ["SH600000"]
    first = pd.DataFrame([[1.0], [np.nan]], index=dates, columns=columns)
    second = pd.DataFrame([[0.0], [np.nan]], index=dates, columns=columns)
    state = _provider_state(dates, _state_rows(dates, "SH600000", [10.0, 10.0]))

    calls: list[int] = []
    events: list[dict] = []

    def stable(report, provider_state):
        calls.append(1)
        return _result(first, state)

    result = _stateful_runner(stable, lambda stage, payload: events.append(payload), None)
    assert len(calls) == 2
    assert result.metadata["execution"]["provider_state_converged"] is True
    assert result.metadata["execution"]["provider_state_passes"] == 2
    assert events[-1]["converged"] is True

    sequence = [first, second]
    calls = 0

    def oscillate(report, provider_state):
        nonlocal calls
        result = _result(sequence[calls % 2], state)
        calls += 1
        return result

    with pytest.raises(ExecutionError, match="未收敛"):
        _stateful_runner(oscillate, lambda stage, payload: None, None)


def test_momentum_trailing_stop_uses_state_at_close():
    """止损行出现在信号日，当日权重把触发股票清零。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    columns = pd.Index(["SH600000", "SZ000001"], name="stock_code")
    score = pd.DataFrame(
        {"SH600000": 2.0, "SZ000001": 1.0}, index=dates, columns=columns
    )
    eligible = pd.DataFrame(True, index=dates, columns=columns)
    data = type(
        "Data",
        (),
        {
            "index": dates,
            "columns": columns,
            "factors": {"MOM_20": score},
        },
    )()
    rows = _state_rows(dates, "SH600000", [10.0, 10.0, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9])
    state = _provider_state(dates, rows)
    weights = MomentumTrailingStopProvider().build(
        data,
        eligible,
        {"top_n": 1, "stop_loss": 0.10, "trailing_stop": 0.5},
        state=state,
    )
    assert weights.loc[dates[2], "SH600000"] == pytest.approx(0.0)
    assert weights.loc[dates[2], "SZ000001"] == pytest.approx(0.0)
    assert weights.loc[dates[5], "SZ000001"] == pytest.approx(1.0)


def test_trailing_stop_uses_high_water():
    """移动止损按建仓以来最高收盘价触发。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    columns = pd.Index(["SH600000"], name="stock_code")
    score = pd.DataFrame(1.0, index=dates, columns=columns)
    eligible = pd.DataFrame(True, index=dates, columns=columns)
    data = type(
        "Data",
        (),
        {"index": dates, "columns": columns, "factors": {"MOM_20": score}},
    )()
    rows = _state_rows(dates, "SH600000", [10.0, 12.0, 11.0, 10.1, 10.1, 10.1, 10.1, 10.1])
    for row in rows:
        row["avg_cost"] = 9.0
        row["unrealized_return"] = row["price"] / 9.0 - 1.0
    state = _provider_state(dates, rows)
    weights = MomentumTrailingStopProvider().build(
        data,
        eligible,
        {"top_n": 1, "stop_loss": 0.5, "trailing_stop": 0.15},
        state=state,
    )
    assert weights.loc[dates[3], "SH600000"] == pytest.approx(0.0)


def test_stateful_provider_sells_next_open(make_data):
    """止损在信号日收盘触发，成交落在次一交易日开盘。"""
    dates = pd.date_range("2024-01-01", periods=14, freq="B")
    stocks = ["SH600000", "SZ000001", "SZ000002"]
    prices = pd.DataFrame(10.0, index=dates, columns=stocks)
    prices.loc[dates[8] :, "SH600000"] = 9.2
    # 价格下跌后动量转负，止损卖出后不会在下一次调仓立刻买回。
    factors = pd.DataFrame(
        {
            "SH600000": [3.0 if index < 10 else -3.0 for index in range(len(dates))],
            "SZ000001": 2.0,
            "SZ000002": 1.0,
        },
        index=dates,
        columns=stocks,
    )
    data = make_data(
        dates=dates,
        stocks=stocks,
        open=prices.astype("float32"),
        close=prices.astype("float32"),
        factors={"MOM_20": factors.astype("float32")},
    )
    strategy = StrategySpec(
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec(
                "momentum_trailing_stop",
                params={"top_n": 1, "stop_loss": 0.05, "trailing_stop": 0.5},
            ),
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    params = BacktestParams(str(dates[0].date()), str(dates[-1].date()), 100_000.0)
    first = run_backtest(data, strategy, params)
    second = run_backtest(data, strategy, params, provider_state=first.provider_state)
    assert not first.orders.empty and first.orders["Side"].eq("Buy").any()
    sells = second.orders.loc[second.orders["stock_code"].eq("SH600000")]
    sells = sells.loc[sells["Side"].eq("Sell")]
    assert not sells.empty
    trigger = sells["trade_date"].min()
    assert trigger == dates[9]


def _windows(dates: pd.DatetimeIndex, cuts: list[int]) -> list[ChunkWindow]:
    """按交易日下标构造资源窗口。"""
    bounds = [0, *cuts, len(dates)]
    return [
        ChunkWindow(
            start=str(dates[first].date()),
            end=str(dates[last - 1].date()),
            warmup=None if first == 0 else str(dates[first - 1].date()),
        )
        for first, last in zip(bounds[:-1], bounds[1:])
    ]


def _fixed_point(run):
    """按目标权重重复执行到自洽。"""
    state = None
    previous = None
    for _ in range(4):
        result = run(state)
        if previous is not None and previous.equals(result.target_weights):
            return result
        previous = result.target_weights
        state = result.provider_state
    raise AssertionError("状态提供者没有在 4 轮内收敛")


def test_chunked_matches_single_block_with_state_provider(make_data):
    """状态提供者在单块与多块下逐位一致。"""
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    stocks = ["SH600000", "SZ000001", "SZ000002"]
    prices = pd.DataFrame(10.0, index=dates, columns=stocks)
    decline = np.power(0.98, np.arange(31))
    prices.loc[dates[9] :, "SH600000"] = decline
    factors = pd.DataFrame(
        {
            "SH600000": [3.0 if index < 10 else -3.0 for index in range(len(dates))],
            "SZ000001": 2.0,
            "SZ000002": 1.0,
        },
        index=dates,
        columns=stocks,
    )
    data = make_data(
        dates=dates,
        stocks=stocks,
        open=prices.astype("float32"),
        close=prices.astype("float32"),
        factors={"MOM_20": factors.astype("float32")},
    )
    data = replace(data, data_snapshot=data_snapshot_from_market_data(data))
    strategy = StrategySpec(
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec(
                "momentum_trailing_stop",
                params={"top_n": 1, "stop_loss": 0.05, "trailing_stop": 0.5},
            ),
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    params = BacktestParams(str(dates[0].date()), str(dates[-1].date()), 300_000.0)

    windows = _windows(dates, [13, 27])

    def load(window: ChunkWindow, state):
        chunk = data.crop(window.warmup or window.start, window.end)
        return chunk if state is None else _warm_up(chunk, state)

    single = _fixed_point(
        lambda state: run_loaded(data, strategy, params, provider_state=state)
    )
    chunked = _fixed_point(
        lambda state: run_windows(
            windows,
            strategy,
            params,
            days=dates,
            loader=load,
            benchmark=data.benchmark,
            benchmark_info=data.benchmark_info,
            liquidations=liquidation_cells(data),
            data_snapshot=data.data_snapshot,
            provider_state=state,
        )
    )
    for name in (
        "daily",
        "orders",
        "holdings",
        "positions",
        "rejections",
        "liquidations",
        "allocations",
        "industry_exposure",
    ):
        pd.testing.assert_frame_equal(
            getattr(single, name), getattr(chunked, name), check_exact=True
        )
    pd.testing.assert_frame_equal(
        single.target_weights,
        chunked.target_weights,
        check_exact=True,
        check_like=True,
    )
    assert single.metrics == chunked.metrics
    assert single.metadata["snapshot_id"] == chunked.metadata["snapshot_id"]
    snapshot = single.metadata["snapshot"]
    assert snapshot["schema_version"] == "backtest_input_snapshot_v5"
    rules = snapshot["components"]["rules"]
    assert rules["provider_state_model"] == "execution_position_state_v1"
    assert rules["provider"]["uses_position_state"] is True
