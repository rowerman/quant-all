"""退市强制清算的手工账本黄金用例。"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.orders import LIQUIDATION_COLUMNS
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
STOCKS = ["SH600000", "SZ000001"]
DELIST_STOCK = "SZ000001"
BUY_DATE = pd.Timestamp("2026-02-02")
BEFORE_DELIST = pd.Timestamp("2026-02-06")
DELIST_DATE = pd.Timestamp("2026-02-09")
INIT_CASH = 100_000.0


def _delisting_panel(make_data):
    """构造一只股票在非调仓日退市、且当日停牌不可卖的双股票面板。"""
    index = pd.DatetimeIndex(DATES)
    columns = pd.Index(STOCKS, name="stock_code")
    open_price = pd.DataFrame(10.0, index=index, columns=columns)
    close_price = pd.DataFrame(10.5, index=index, columns=columns)
    after_delist = index > DELIST_DATE
    open_price.loc[after_delist, DELIST_STOCK] = np.nan
    close_price.loc[after_delist, DELIST_STOCK] = np.nan

    suspended = pd.DataFrame(False, index=index, columns=columns)
    suspended.loc[DELIST_DATE, DELIST_STOCK] = True
    can_sell = pd.DataFrame(True, index=index, columns=columns)
    can_sell.loc[BEFORE_DELIST, DELIST_STOCK] = False
    score = pd.DataFrame(1.0, index=index, columns=columns)
    delist = pd.Series(pd.NaT, index=columns)
    delist[DELIST_STOCK] = DELIST_DATE
    return make_data(
        dates=index,
        stocks=STOCKS,
        open=open_price.astype("float32"),
        close=close_price.astype("float32"),
        can_sell_open_next=can_sell,
        is_suspended=suspended,
        factors={"MOM_20": score.astype("float32")},
        delist_date=delist,
        amount_cny=pd.DataFrame(1e9, index=index, columns=columns).astype("float32"),
    )


def _strategy():
    """月度等权两只股票，关闭现金缓冲以便手工复算加减仓金额。"""
    return StrategySpec(
        signal=SignalSpec(factors=("MOM_20",), weights=(1.0,), standardize=False),
        portfolio=PortfolioSpec(top_n=2),
        rebalance=RebalanceSpec(frequency="monthly"),
        execution=ExecutionSpec(cash_buffer=0.0),
        costs=CostSpec(),
    )


def _params():
    """返回与面板一致的运行区间和初始资金。"""
    return BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), INIT_CASH)


def test_delisting_liquidation_matches_manual_ledger(make_data):
    """非调仓日退市仍应在最后有价日强卖，现金与费用逐项对上手算账本。"""
    result = run_backtest(_delisting_panel(make_data), _strategy(), _params())

    orders = result.orders.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    buys = orders.loc[orders["trade_date"] == BUY_DATE]
    assert buys["stock_code"].tolist() == STOCKS
    assert buys["Size"].tolist() == [4900.0, 4900.0]
    assert buys["Price"].tolist() == pytest.approx([10.005, 10.005])
    assert buys["commission"].tolist() == pytest.approx([14.71, 14.71])
    assert buys["stamp_tax"].tolist() == pytest.approx([0.0, 0.0])
    assert buys["transfer_fee"].tolist() == pytest.approx([0.49, 0.49])
    assert buys["Fees"].tolist() == pytest.approx([15.20, 15.20])
    assert buys["slippage_cost"].tolist() == pytest.approx([24.50, 24.50])

    sells = orders.loc[orders["Side"].astype(str).str.lower() == "sell"]
    assert len(sells) == 1
    sell = sells.iloc[0]
    assert sell["trade_date"] == DELIST_DATE
    assert sell["stock_code"] == DELIST_STOCK
    assert sell["Size"] == -4900.0
    assert sell["Price"] == pytest.approx(9.995)
    assert sell["commission"] == pytest.approx(14.69)
    assert sell["stamp_tax"] == pytest.approx(24.49)
    assert sell["transfer_fee"] == pytest.approx(0.49)
    assert sell["Fees"] == pytest.approx(39.67)
    assert sell["slippage_cost"] == pytest.approx(24.50)
    assert len(orders) == 3

    assert list(result.liquidations.columns) == [
        "trade_date", "stock_code", "delist_date", "size", "price", "value",
    ]
    liquidation = result.liquidations.iloc[0]
    assert liquidation["trade_date"] == DELIST_DATE
    assert liquidation["stock_code"] == DELIST_STOCK
    assert liquidation["delist_date"] == DELIST_DATE
    assert liquidation["size"] == pytest.approx(4900.0)
    assert liquidation["price"] == pytest.approx(9.995)
    assert liquidation["value"] == pytest.approx(48_975.50)
    assert result.rejections.empty

    before = result.daily.loc[BEFORE_DELIST]
    assert before["cash"] == pytest.approx(1_920.60)
    assert before["holdings_value"] == pytest.approx(102_900.0)
    assert before["portfolio_value"] == pytest.approx(104_820.60)

    settled = result.daily.loc[DELIST_DATE]
    assert settled["cash"] == pytest.approx(50_856.43)
    assert settled["holdings_value"] == pytest.approx(51_450.0)
    assert settled["portfolio_value"] == pytest.approx(102_306.43)
    assert result.daily.loc[result.daily.index > DELIST_DATE, "portfolio_value"].tolist() == pytest.approx(
        [102_306.43] * int((result.daily.index > DELIST_DATE).sum())
    )

    assert DELIST_STOCK not in set(
        result.holdings.loc[result.holdings["trade_date"] >= DELIST_DATE, "stock_code"]
    )
    last_holdings = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert last_holdings["stock_code"].tolist() == ["SH600000"]
    assert last_holdings["market_value"].tolist() == pytest.approx([51_450.0])

    closed = result.positions.loc[result.positions["Column"] == DELIST_STOCK]
    assert len(closed) == 1
    assert closed.iloc[0]["Status"] == "Closed"
    assert closed.iloc[0]["Size"] == pytest.approx(4900.0)
    assert closed.iloc[0]["Exit Timestamp"] == DELIST_DATE
    assert closed.iloc[0]["Exit Fees"] == pytest.approx(39.67)
    assert closed.iloc[0]["PnL"] == pytest.approx(-103.87)

    assert result.metadata["calibration"]["forced_liquidations"] == 1
    assert result.metadata["calibration"]["final_orders"] == 3
    assert result.metadata["calibration"]["final_sells"] == 1
    assert result.metadata["calibration"]["cash_clipped_cells"] == 0
    assert result.final_state is not None
    assert result.final_state.shares[DELIST_STOCK] == 0.0
    assert result.final_state.cash == pytest.approx(50_856.43)


def test_without_liquidation_candidates_leaves_delisted_position(make_data):
    """负向对照：移除强制清算候选后，同一面板会留下退市持仓。"""
    result = run_backtest(
        _delisting_panel(make_data),
        _strategy(),
        _params(),
        liquidations=pd.DataFrame(columns=LIQUIDATION_COLUMNS),
    )

    assert result.liquidations.empty
    assert result.metadata["calibration"]["forced_liquidations"] == 0
    last_holdings = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert DELIST_STOCK in set(last_holdings["stock_code"])
    assert result.final_state is not None
    assert result.final_state.shares[DELIST_STOCK] == pytest.approx(4900.0)
