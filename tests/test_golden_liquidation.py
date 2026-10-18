"""退市整理期退出与零价值核销的手工账本黄金用例。"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.errors import ExecutionError
from backtest.orders import WRITEOFF_COLUMNS
from backtest.specs import (
    BacktestParams,
    CostSpec,
    ExecutionSpec,
    PortfolioSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)

DATES = pd.date_range("2026-01-05", periods=30, freq="B")
STOCK = "SZ000001"
BUY_DATE = pd.Timestamp("2026-02-02")
DELIST_DATE = pd.Timestamp("2026-02-09")
LAST_DATE = DATES[-1]
INIT_CASH = 100_000.0


def _panel(
    make_data,
    *,
    delist_date=DELIST_DATE,
    delisting_from=DELIST_DATE,
    suspended=None,
    amount=1e9,
    delist_amount=None,
):
    index = pd.DatetimeIndex(DATES)
    columns = pd.Index([STOCK], name="stock_code")

    def wide(value):
        return pd.DataFrame(value, index=index, columns=columns)

    open_price = wide(10.0)
    close_price = wide(10.0)
    after = index > pd.Timestamp(delist_date)
    open_price.loc[after, STOCK] = np.nan
    close_price.loc[after, STOCK] = np.nan
    is_suspended = wide(False)
    if suspended is not None:
        is_suspended.loc[suspended, STOCK] = True
    is_delisting = wide(False)
    is_delisting.loc[index >= pd.Timestamp(delisting_from), STOCK] = True
    amount_frame = wide(amount)
    if delist_amount is not None:
        amount_frame.loc[pd.Timestamp(delist_date), STOCK] = delist_amount
    return make_data(
        dates=index,
        stocks=[STOCK],
        open=open_price.astype("float32"),
        close=close_price.astype("float32"),
        is_suspended=is_suspended,
        is_delisting=is_delisting,
        amount_cny=amount_frame.astype("float32"),
        delist_date=pd.Series([pd.Timestamp(delist_date)], index=columns),
    )


def _strategy(*, participation=None):
    return StrategySpec(
        signal=SignalSpec(factors=("MOM_20",), weights=(1.0,), standardize=False),
        portfolio=PortfolioSpec(top_n=1),
        rebalance=RebalanceSpec(frequency="monthly"),
        execution=ExecutionSpec(
            cash_buffer=0.0,
            max_participation_rate=participation,
        ),
        costs=CostSpec.zero(),
    )


def _params():
    return BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), INIT_CASH)


def test_suspended_delisting_writeoff_matches_manual_ledger(make_data):
    """退市日停牌卖不掉时，10,000 股按零价值核销，现金不变。"""
    data = _panel(make_data, suspended=[DELIST_DATE])
    result = run_backtest(data, _strategy(), _params())

    assert result.orders[["trade_date", "stock_code", "Side", "Size"]].to_dict("records") == [
        {
            "trade_date": BUY_DATE,
            "stock_code": STOCK,
            "Side": "Buy",
            "Size": 10_000.0,
        }
    ]
    assert result.liquidations.to_dict("records") == [
        {
            "trade_date": DELIST_DATE,
            "stock_code": STOCK,
            "delist_date": DELIST_DATE,
            "size": 10_000.0,
            "price": 0.0,
            "value": 0.0,
            "reference_price": 10.0,
            "reason": "zero_writeoff",
        }
    ]
    settled = result.daily.loc[DELIST_DATE]
    assert settled["cash"] == pytest.approx(0.0)
    assert settled["holdings_value"] == pytest.approx(0.0)
    assert settled["portfolio_value"] == pytest.approx(0.0)
    assert settled["writeoff_size"] == pytest.approx(10_000.0)
    assert settled["writeoff_value"] == pytest.approx(100_000.0)

    closed = result.positions.iloc[0]
    assert closed["Status"] == "Closed"
    assert closed["Exit Timestamp"] == DELIST_DATE
    assert closed["Avg Exit Price"] == pytest.approx(0.0)
    assert closed["PnL"] == pytest.approx(-100_000.0)
    assert result.final_state is not None
    assert result.final_state.shares[STOCK] == 0.0
    assert result.final_state.cash == pytest.approx(0.0)
    assert result.rejections.to_dict("records") == [
        {"trade_date": DELIST_DATE, "stock_code": STOCK, "reason": "suspended"}
    ]

    calibration = result.metadata["calibration"]
    assert calibration["forced_liquidations"] == 1
    assert calibration["delisting_writeoff_shares"] == pytest.approx(10_000.0)
    assert calibration["delisting_writeoff_value"] == pytest.approx(100_000.0)
    assert calibration["delisting_residual_shares_at_end"] == pytest.approx(0.0)
    assert calibration["forced_exit_orders"] == 0


def test_first_sellable_day_exits_before_delist(make_data):
    """整理期首日停牌、下一交易日可卖时，按开盘价正常卖出，不产生核销。"""
    first_delisting = pd.Timestamp("2026-02-06")
    data = _panel(
        make_data,
        delisting_from=first_delisting,
        suspended=[first_delisting],
    )
    result = run_backtest(data, _strategy(), _params())

    sells = result.orders.loc[result.orders["Side"].astype(str).str.lower() == "sell"]
    assert len(sells) == 1
    sell = sells.iloc[0]
    assert sell["trade_date"] == DELIST_DATE
    assert sell["Size"] == -10_000.0
    assert sell["Price"] == pytest.approx(10.0)
    assert bool(sell["forced_exit"])
    assert result.liquidations.empty
    assert result.daily.loc[DELIST_DATE, "cash"] == pytest.approx(INIT_CASH)
    assert result.metadata["calibration"]["forced_exit_orders"] == 1
    assert result.metadata["calibration"]["forced_liquidations"] == 0


def test_window_end_delisting_is_written_off_without_residual(make_data):
    """退市日等于区间末且不可卖时，区间末核销且不留下持仓。"""
    data = _panel(
        make_data,
        delist_date=LAST_DATE,
        delisting_from=LAST_DATE,
        suspended=[LAST_DATE],
    )
    result = run_backtest(data, _strategy(), _params())

    assert result.liquidations["trade_date"].tolist() == [LAST_DATE]
    assert result.liquidations["size"].tolist() == [10_000.0]
    assert result.final_state is not None
    assert result.final_state.shares[STOCK] == 0.0
    assert result.metadata["calibration"]["delisting_residual_shares_at_end"] == 0.0


def test_capacity_shortfall_is_sold_then_written_off(make_data):
    """成交额容量只允许卖出 100 股，剩余 9,900 股在退市日核销。"""
    data = _panel(make_data, delist_amount=10_000.0)
    result = run_backtest(data, _strategy(participation=0.1), _params())

    sells = result.orders.loc[result.orders["Side"].astype(str).str.lower() == "sell"]
    assert len(sells) == 1
    assert sells.iloc[0]["Size"] == -100.0
    assert bool(sells.iloc[0]["forced_exit"])
    liquidation = result.liquidations.iloc[0]
    assert liquidation["size"] == pytest.approx(9_900.0)
    assert liquidation["value"] == pytest.approx(0.0)
    assert result.daily.loc[DELIST_DATE, "cash"] == pytest.approx(1_000.0)
    calibration = result.metadata["calibration"]
    assert calibration["delisting_capacity_shortfall_shares"] == pytest.approx(9_900.0)
    assert calibration["delisting_residual_shares_at_end"] == 0.0


def test_delisting_stock_is_never_bought(make_data):
    """整理期股票即使因子有值也不能进入组合。"""
    data = _panel(make_data, delisting_from=DATES[0])
    result = run_backtest(data, _strategy(), _params())
    assert result.orders.empty
    assert result.metadata["calibration"]["delisting_buy_orders"] == 0


def test_removing_writeoff_candidates_is_rejected_by_residual_check(make_data):
    """移除核销候选时，区间末残留退市持仓必须直接报错，不能静默保留。"""
    with pytest.raises(ExecutionError, match="退市持仓未清零"):
        run_backtest(
            _panel(make_data, suspended=[DELIST_DATE]),
            _strategy(),
            _params(),
            liquidations=pd.DataFrame(columns=WRITEOFF_COLUMNS),
        )
