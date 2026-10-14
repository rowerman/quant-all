"""VectorBT 执行器的集成测试：取整、规则拒单、费用与持仓口径。"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.errors import DataAlignmentError, ExecutionError
from backtest.specs import (
    BacktestParams,
    CostSpec,
    ExecutionSpec,
    PortfolioSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)

DATES = pd.date_range("2024-01-29", periods=6, freq="B")
STOCKS = ["SH600000", "SZ000001", "SZ000002"]


def _frames():
    """构造 6 个交易日、3 只股票、价格恒定的输入面板。"""
    score = pd.DataFrame(
        {
            "SH600000": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "SZ000001": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "SZ000002": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=DATES,
    )
    price = pd.DataFrame(10.0, index=DATES, columns=STOCKS)
    return score, price


def _strategy(**overrides):
    """默认日频调仓、两只股票等权的测试策略。"""
    fields = {
        "signal": SignalSpec(factors=("MOM_20",), weights=(1.0,), standardize=False),
        "portfolio": PortfolioSpec(top_n=2),
        "rebalance": RebalanceSpec(frequency="daily"),
    }
    fields.update(overrides)
    return StrategySpec(**fields)


def test_engine_produces_lot_sized_orders_and_exact_fees(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    stages: list[str] = []
    result = run_backtest(data, _strategy(), params, progress=lambda stage, _: stages.append(stage))

    assert stages == ["signals", "probe", "orders", "execute"]
    assert not result.orders.empty
    assert (result.orders["Size"] % 100 == 0).all()
    assert result.metadata["calibration"]["fee_matrix_max_diff"] == pytest.approx(0.0)
    assert result.metadata["calibration"]["fallback_fee_orders"] == 0
    # 下单当日的实际权重应当贴近目标权重（只受 100 股取整与现金裁剪影响）
    assert result.metadata["calibration"]["weight_deviation_p95"] < 0.01
    assert result.metadata["calibration"]["target_cells"] > 0
    assert result.daily.index.equals(DATES)
    assert result.daily["portfolio_value"].iloc[0] == pytest.approx(100_000.0)
    assert {"cost", "turnover", "drawdown", "nav"} <= set(result.daily.columns)
    assert result.metrics["trading_days"] == len(DATES)


def test_engine_holdings_match_target_weights(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    held = result.holdings.groupby("trade_date")["stock_code"].nunique()
    assert held.max() == 2
    weights = result.holdings.groupby("trade_date")["weight"].sum()
    # 满仓 + 100 股取整 + 费用，实际权重贴近目标但不会超过 1
    assert weights.max() <= 1.0 + 1e-9
    assert weights.max() > 0.95
    last = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert set(last["stock_code"]) == {"SH600000", "SZ000001"}


def test_engine_records_limit_up_rejection(make_data):
    score, price = _frames()
    can_buy = pd.DataFrame(True, index=DATES, columns=STOCKS)
    can_buy.loc[DATES[0], "SH600000"] = False  # 次一交易日开盘涨停
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), can_buy_open_next=can_buy,
        factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    assert "limit_up_or_not_buyable" in set(result.rejections["reason"])
    assert result.metadata["calibration"]["rejected_orders"] == len(result.rejections)


def test_engine_reports_missing_membership(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    with pytest.raises(ExecutionError, match="成分矩阵"):
        run_backtest(data, _strategy(universe="index:000300.SH"), params)


def test_engine_rejects_range_without_trading_days(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    with pytest.raises(DataAlignmentError, match="没有交易日"):
        run_backtest(
            data, _strategy(), BacktestParams("2024-03-01", "2024-03-05", 100_000.0)
        )


def test_minimum_commission_is_charged_per_order(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    costs = CostSpec(commission_rate=0.0003, minimum_commission=5.0, stamp_tax_rate=0.0005)
    strategy = _strategy(costs=costs)
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 1_000_000.0)
    result = run_backtest(data, strategy, params)
    fees = result.orders["Fees"].astype(float)
    value = result.orders["Size"].abs() * result.orders["Price"].astype(float)
    sells = result.orders["Side"].astype(str).str.lower().eq("sell")
    expected = np.maximum(value * costs.commission_rate, costs.minimum_commission) + np.where(
        sells, value * costs.stamp_tax_rate, 0.0
    )
    assert np.allclose(fees.to_numpy(), expected, atol=1e-6)


def test_execution_switches_off_keeps_orders(make_data):
    score, price = _frames()
    suspended = pd.DataFrame(False, index=DATES, columns=STOCKS)
    suspended.loc[DATES[1], "SH600000"] = True
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), is_suspended=suspended,
        factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(execution=ExecutionSpec(reject_suspended=False)), params)
    assert result.rejections.empty

