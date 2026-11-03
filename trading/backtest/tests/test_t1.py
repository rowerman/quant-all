"""严格 T+1 订单守卫。"""

import pandas as pd

from backtest.t1 import cap_sells_to_prior_holdings


def test_t1_caps_sell_at_prior_holdings():
    orders = pd.DataFrame([[-150.0, -50.0]], columns=["A", "B"])
    holdings = pd.DataFrame([[100.0, 100.0]], columns=["A", "B"])
    prices = pd.DataFrame([[10.0, 20.0]], columns=["A", "B"])
    capped, blocked, stats = cap_sells_to_prior_holdings(orders, holdings, prices)
    assert capped.iloc[0, 0] == -100.0
    assert capped.iloc[0, 1] == -50.0
    assert blocked.tolist() == [[True, False]]
    assert stats == {
        "t_plus_1_blocked_cells": 1,
        "t_plus_1_blocked_shares": 50.0,
        "t_plus_1_blocked_value": 500.0,
    }


def test_t1_can_be_disabled():
    orders = pd.DataFrame([[-150.0]], columns=["A"])
    holdings = pd.DataFrame([[100.0]], columns=["A"])
    capped, blocked, stats = cap_sells_to_prior_holdings(
        orders, holdings, enforce=False
    )
    assert capped.iloc[0, 0] == -150.0
    assert not blocked.any()
    assert stats["t_plus_1_blocked_cells"] == 0


def test_t1_ignores_buys_and_missing_orders():
    orders = pd.DataFrame([[100.0, float("nan")]], columns=["A", "B"])
    holdings = pd.DataFrame([[50.0, 0.0]], columns=["A", "B"])
    capped, blocked, _ = cap_sells_to_prior_holdings(orders, holdings)
    assert capped.iloc[0, 0] == 100.0
    assert pd.isna(capped.iloc[0, 1])
    assert not blocked.any()
