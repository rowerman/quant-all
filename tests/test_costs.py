"""历史成本模型与成本归因的独立黄金用例。"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.costs import CostSpec, order_cost_frame, round_cent


@pytest.mark.parametrize(
    ("day", "buy", "sell"),
    [
        ("2001-11-15", 0.004, 0.004),
        ("2001-11-16", 0.002, 0.002),
        ("2005-01-21", 0.002, 0.002),
        ("2005-01-24", 0.001, 0.001),
        ("2007-05-29", 0.001, 0.001),
        ("2007-05-30", 0.003, 0.003),
        ("2008-04-23", 0.003, 0.003),
        ("2008-04-24", 0.001, 0.001),
        ("2008-09-18", 0.001, 0.001),
        ("2008-09-19", 0.0, 0.001),
        ("2023-08-25", 0.0, 0.001),
        ("2023-08-28", 0.0, 0.0005),
    ],
)
def test_stamp_tax_schedule_boundaries(day, buy, sell):
    assert CostSpec().stamp_rates(day) == pytest.approx((buy, sell))


@pytest.mark.parametrize(
    ("day", "exchange", "rate", "basis", "minimum"),
    [
        ("2012-05-31", "SH", 0.0005, "par_value", 1.0),
        ("2012-06-01", "SH", 0.000375, "par_value", 1.0),
        ("2012-08-31", "SH", 0.000375, "par_value", 1.0),
        ("2012-09-01", "SH", 0.0003, "par_value", 1.0),
        ("2015-07-31", "SH", 0.0003, "par_value", 1.0),
        ("2015-08-03", "SH", 0.00002, "amount", 0.0),
        ("2012-05-31", "SZ", 0.0000255, "amount", 0.0),
        ("2015-08-03", "SZ", 0.00002, "amount", 0.0),
        ("2021-11-16", "BJ", 0.000025, "amount", 0.0),
        ("2022-04-28", "BJ", 0.000025, "amount", 0.0),
        ("2022-04-29", "BJ", 0.00001, "amount", 0.0),
    ],
)
def test_transfer_fee_schedule_boundaries(day, exchange, rate, basis, minimum):
    assert CostSpec().transfer_terms(day, exchange) == pytest.approx(
        (rate, basis, minimum)
    )


def test_charge_hand_calculates_all_components():
    costs = CostSpec()
    buy = costs.charge("2024-01-02", "Buy", 1000, 10.0, "SH600000", "主板")
    assert buy.fill_price == pytest.approx(10.005)
    assert buy.fill_amount == pytest.approx(10005.0)
    assert buy.commission == pytest.approx(5.0)
    assert buy.stamp_tax == 0.0
    assert buy.transfer_fee == pytest.approx(0.10)
    assert buy.fees == pytest.approx(5.10)
    assert buy.slippage_cost == pytest.approx(5.0)
    assert buy.total_cost == pytest.approx(10.10)

    sell = costs.charge("2024-01-02", "Sell", 1000, 10.0, "SH600000", "主板")
    assert sell.fill_price == pytest.approx(9.995)
    assert sell.fill_amount == pytest.approx(9995.0)
    assert sell.commission == pytest.approx(5.0)
    assert sell.stamp_tax == pytest.approx(5.0)
    assert sell.transfer_fee == pytest.approx(0.10)
    assert sell.fees == pytest.approx(10.10)
    assert sell.total_cost == pytest.approx(15.10)


@pytest.mark.parametrize(
    ("day", "stamp"),
    [
        ("2016-01-04", 10.00),
        ("2023-08-25", 10.00),
        ("2023-08-28", 5.00),
        ("2026-01-05", 5.00),
    ],
)
def test_sell_stamp_tax_hand_calculation(day, stamp):
    charge = CostSpec().charge(day, "Sell", 1000, 10.0, "SH600000", "主板")
    assert charge.stamp_tax == pytest.approx(stamp)


def test_shanghai_par_value_fee_has_one_yuan_minimum():
    charge = CostSpec().charge("2012-08-31", "Sell", 1000, 10.0, "SH600000", "主板")
    assert charge.transfer_fee == pytest.approx(1.0)
    shenzhen = CostSpec().charge("2012-08-31", "Sell", 1000, 10.0, "SZ000001", "主板")
    assert shenzhen.transfer_fee == pytest.approx(0.25)


def test_legacy_fixed_stamp_tax_is_sell_only():
    costs = CostSpec(stamp_tax_rate=0.0005)
    buy = costs.charge("2016-01-04", "Buy", 1000, 10.0, "SH600000", "主板")
    sell = costs.charge("2016-01-04", "Sell", 1000, 10.0, "SH600000", "主板")
    assert buy.stamp_tax == 0.0
    assert sell.stamp_tax == pytest.approx(5.0)


def test_cost_can_be_fully_disabled():
    costs = CostSpec.zero()
    charge = costs.charge("2024-01-02", "Sell", 1000, 10.0, "SH600000", "主板")
    assert charge.fees == 0.0
    assert charge.slippage_cost == 0.0
    assert charge.total_cost == 0.0


def test_fee_rounding_is_half_up():
    assert round_cent(1.005) == pytest.approx(1.01)
    assert round_cent(2.675) == pytest.approx(2.68)
    assert round_cent(0.004) == pytest.approx(0.00)


@pytest.mark.parametrize(
    ("entry", "sell", "rate"),
    [
        ("2004-01-01", "2004-02-01", 0.20),
        ("2004-01-01", "2005-06-12", 0.20),
        ("2006-01-01", "2006-02-01", 0.10),
        ("2014-01-01", "2014-01-15", 0.20),
        ("2014-01-01", "2014-06-01", 0.10),
        ("2014-01-01", "2015-09-07", 0.05),
        ("2016-01-01", "2016-01-15", 0.20),
        ("2016-01-01", "2016-06-01", 0.10),
        ("2016-01-01", "2018-01-01", 0.0),
    ],
)
def test_dividend_tax_uses_historical_policy(entry, sell, rate):
    assert CostSpec().dividend_tax_rate(entry, sell) == pytest.approx(rate)
    assert CostSpec(dividend_tax_mode="disabled").dividend_tax_rate(entry, sell) == 0.0


def test_order_cost_frame_aggregates_components_and_falls_back_to_fees():
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    orders = pd.DataFrame(
        {
            "trade_date": [index[0], index[0], index[1]],
            "Fees": [5.0, 6.0, 7.0],
            "commission": [5.0, 5.0, 7.0],
            "stamp_tax": [0.0, 1.0, 0.0],
            "transfer_fee": [0.0, 0.0, 0.0],
            "slippage_cost": [0.5, 0.5, 0.7],
        }
    )
    frame = order_cost_frame(orders, index)
    assert frame.loc[index[0], "commission"] == pytest.approx(10.0)
    assert frame.loc[index[0], "stamp_tax"] == pytest.approx(1.0)
    assert frame.loc[index[1], "slippage_cost"] == pytest.approx(0.7)

    legacy = orders[["trade_date", "Fees"]]
    fallback = order_cost_frame(legacy, index)
    assert fallback.loc[index[0], "commission"] == pytest.approx(11.0)
    assert fallback.loc[index[0], "stamp_tax"] == 0.0
