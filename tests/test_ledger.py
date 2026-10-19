"""原始价执行账本的公司行动、分红和红利税黄金用例。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.ledger import LedgerPortfolio, LedgerSnapshot, _tax_rate
from backtest.specs import CostSpec
from backtest.trading_rules import BOARD_BSE, BOARD_MAIN, BOARD_STAR

BOARDS = pd.Series({"A": BOARD_MAIN})


def _prices(values, dates, columns=("A",)):
    """构造账本使用的价格宽表。"""
    return pd.DataFrame(values, index=dates, columns=columns, dtype="float64")


def _action(dates, **overrides):
    """构造一条公司行动记录。"""
    row = {
        "stock_code": "A",
        "ex_date": dates[1],
        "record_date": dates[0],
        "pay_date": dates[1],
        "share_multiplier": 1.0,
        "cash_div_per_share": 0.0,
        "action_type": "cash",
        "source": "test",
        "source_event_key": "event-1",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_split_keeps_value_and_moves_to_true_share_count():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = _prices([10.0, 5.0, 5.0], dates)
    open_price = _prices([10.0, 5.0, 5.0], dates)
    targets = _prices([100.0, 200.0, 200.0], dates)
    actions = _action(dates, share_multiplier=2.0, action_type="stock")

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=open_price,
        init_cash=1_000.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.assets().iloc[-1, 0] == 200
    assert portfolio.value().tolist() == pytest.approx([1_000.0, 1_000.0, 1_000.0])
    assert portfolio.orders.records_readable["Size"].tolist() == [100.0]


def test_cash_dividend_is_receivable_until_pay_date():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = _prices([10.0, 9.5, 9.5, 9.5], dates)
    open_price = close.copy()
    targets = _prices([100.0, 100.0, 100.0, 0.0], dates)
    actions = _action(dates, cash_div_per_share=0.5, pay_date=dates[2])

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=open_price,
        init_cash=1_000.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.receivables().tolist() == pytest.approx([0.0, 50.0, 0.0, 0.0])
    assert portfolio.cash().tolist() == pytest.approx([0.0, 0.0, 50.0, 990.0])
    assert portfolio.value().tolist() == pytest.approx([1_000.0, 1_000.0, 1_000.0, 990.0])
    assert portfolio.dividend_income().sum() == pytest.approx(50.0)


def test_fractional_share_is_cashed_out():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 10.0], dates)
    targets = _prices([100.0, 100.0], dates)
    # 100 × 1.005 = 100.5，向下取整后 0.5 股按 10 元折现。
    actions = _action(dates, share_multiplier=1.005, action_type="stock")

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_000.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.assets().iloc[-1, 0] == 100
    assert portfolio.cash().iloc[-1] == pytest.approx(5.0)
    assert portfolio.fractional_cash().sum() == pytest.approx(5.0)


def test_rights_subscription_uses_available_cash_and_adds_lot():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 8.0], dates)
    targets = _prices([100.0, 130.0], dates)
    actions = _action(
        dates,
        rights_ratio=0.3,
        rights_price=5.0,
        action_type="rights",
    )

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=2_000.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.assets().iloc[-1, 0] == 130
    assert portfolio.cash().iloc[-1] == pytest.approx(850.0)
    assert portfolio.rights_subscription_cash().sum() == pytest.approx(150.0)
    assert portfolio.rights_subscribed_shares().sum() == pytest.approx(30.0)
    row = portfolio.corporate_action_records().iloc[-1]
    assert row["kind"] == "rights_subscription"
    assert row["rights_entitled_shares"] == pytest.approx(30.0)
    assert row["rights_subscribed_shares"] == pytest.approx(30.0)
    assert row["rights_price"] == pytest.approx(5.0)


def test_rights_subscription_stops_at_cash_and_records_shortfall():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 8.0], dates)
    targets = _prices([100.0, 110.0], dates)
    actions = _action(
        dates,
        rights_ratio=0.3,
        rights_price=5.0,
        action_type="rights",
    )

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_050.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.assets().iloc[-1, 0] == 110
    assert portfolio.cash().iloc[-1] == pytest.approx(0.0)
    assert portfolio.rights_subscribed_shares().sum() == pytest.approx(10.0)
    assert portfolio.rights_unsubscribed_shares().sum() == pytest.approx(20.0)


def test_rights_and_share_adjustment_use_same_pre_event_holding():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 5.0], dates)
    targets = _prices([100.0, 180.0], dates)
    actions = _action(
        dates,
        share_multiplier=1.5,
        rights_ratio=0.3,
        rights_price=5.0,
        action_type="rights_mixed",
    )

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=2_000.0, corporate_actions=actions, boards=BOARDS,
    )

    assert portfolio.assets().iloc[-1, 0] == 180
    assert portfolio.rights_subscribed_shares().sum() == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("entry", "sell", "rate"),
    [
        ("2024-01-01", "2024-01-31", 0.20),
        ("2024-01-01", "2024-06-30", 0.10),
        ("2024-01-01", "2025-01-02", 0.0),
    ],
)
def test_dividend_tax_rate_by_holding_period(entry, sell, rate):
    assert _tax_rate(pd.Timestamp(entry), pd.Timestamp(sell)) == pytest.approx(rate)


def test_fifo_dividend_tax_is_deducted_when_lots_are_sold():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = _prices([10.0, 10.0, 9.5, 9.5], dates)
    # 第一天买 100，第二天登记分红 50，第三天卖出；持有不足一个月，税率 20%。
    targets = _prices([100.0, 100.0, 0.0, 0.0], dates)
    actions = _action(dates, cash_div_per_share=0.5, pay_date=dates[2])
    costs = CostSpec.zero()

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_000.0, corporate_actions=actions, costs=costs, boards=BOARDS,
    )

    assert portfolio.dividend_tax().sum() == pytest.approx(10.0)
    assert portfolio.corporate_action_records()["kind"].tolist().count("dividend_tax") == 1
    assert portfolio.cash().iloc[-1] == pytest.approx(990.0)


def test_disabled_dividend_tax_keeps_gross_cash():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = _prices([10.0, 10.0, 9.5, 9.5], dates)
    targets = _prices([100.0, 100.0, 0.0, 0.0], dates)
    actions = _action(dates, cash_div_per_share=0.5, pay_date=dates[2])
    costs = CostSpec.zero()

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_000.0, corporate_actions=actions, costs=costs,
        dividend_tax_mode="disabled", boards=BOARDS,
    )

    assert portfolio.dividend_tax().sum() == 0.0
    assert portfolio.cash().iloc[-1] == pytest.approx(1_000.0)


@pytest.mark.parametrize(
    ("dates", "stamp"),
    [
        (("2016-01-04", "2016-01-05"), 10.00),
        (("2023-08-24", "2023-08-25"), 10.00),
        (("2023-08-28", "2023-08-29"), 5.00),
        (("2026-01-05", "2026-01-06"), 5.00),
    ],
)
def test_ledger_charges_historical_stamp_tax(dates, stamp):
    index = pd.DatetimeIndex(dates)
    close = _prices([10.0, 10.0], index, columns=("SH600000",))
    targets = _prices([1000.0, 0.0], index, columns=("SH600000",))
    portfolio = LedgerPortfolio.from_orders(
        close,
        targets,
        size_type="targetamount",
        price=close,
        init_cash=20_000.0,
        boards=pd.Series({"SH600000": BOARD_MAIN}),
        costs=CostSpec(),
    )
    orders = portfolio.orders.records_readable
    sell = orders.loc[orders["Side"] == "Sell"].iloc[0]
    assert sell["stamp_tax"] == pytest.approx(stamp)
    assert sell["transfer_fee"] > 0
    assert sell["Fees"] == pytest.approx(
        sell["commission"] + sell["stamp_tax"] + sell["transfer_fee"]
    )
    assert sell["total_cost"] == pytest.approx(
        sell["Fees"] + sell["slippage_cost"]
    )


def test_snapshot_carries_lots_and_receivables_across_chunks():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 9.5], dates)
    targets = _prices([100.0, 100.0], dates)
    actions = _action(dates, cash_div_per_share=0.5, pay_date=dates[1])
    first = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_000.0, corporate_actions=actions, boards=BOARDS,
    )
    snapshot = first.snapshot()
    second = LedgerPortfolio.from_orders(
        close.iloc[-1:], targets.iloc[-1:], size_type="targetamount",
        price=close.iloc[-1:], init_cash=snapshot.cash,
        initial_state=snapshot, corporate_actions=actions, boards=BOARDS,
    )
    assert snapshot.shares["A"] == 100
    assert snapshot.receivables.empty
    assert second.assets().iloc[-1, 0] == 100


def test_bse_buy_can_use_single_share_increment():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 10.0], dates)
    targets = _prices([157.0, 157.0], dates)

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=2_000.0, boards=pd.Series({"A": BOARD_BSE}),
    )

    assert portfolio.orders.records_readable["Size"].tolist() == [157.0]
    assert portfolio.orders.records_readable["Board"].tolist() == [BOARD_BSE]
    assert portfolio.orders.records_readable["Order Valid"].tolist() == [True]


def test_star_rejects_invalid_small_buy():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 10.0], dates)
    targets = _prices([100.0, 100.0], dates)

    with pytest.raises(Exception, match="非法板块订单"):
        LedgerPortfolio.from_orders(
            close, targets, size_type="targetamount", price=close,
            init_cash=2_000.0, boards=pd.Series({"A": BOARD_STAR}),
        )


def test_star_can_sell_odd_lot_only_when_clearing_position():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    close = _prices([10.0, 10.0], dates)
    targets = _prices([201.0, 0.0], dates)
    state = LedgerSnapshot(
        cash=1_000.0,
        shares=pd.Series({"A": 201.0}),
        lots=pd.DataFrame(),
        receivables=pd.DataFrame(),
        last_prices=pd.Series({"A": 10.0}),
    )

    portfolio = LedgerPortfolio.from_orders(
        close, targets, size_type="targetamount", price=close,
        init_cash=1_000.0, initial_state=state,
        boards=pd.Series({"A": BOARD_STAR}), costs=CostSpec.zero(),
    )

    assert portfolio.orders.records_readable["Size"].tolist() == [-201.0]
    assert portfolio.assets().iloc[-1, 0] == 0
