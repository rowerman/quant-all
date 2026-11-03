"""A 股板块交易单位与目标持仓归一化。"""

import pytest

from backtest.errors import ExecutionError
from backtest.trading_rules import rule_for


def test_main_board_uses_100_share_buy_grid_and_allows_odd_lot_sell():
    rule = rule_for("主板")

    assert rule.normalize_target(0, 150) == 100
    assert rule.normalize_target(50, 150) == 150
    assert rule.normalize_target(150, 50) == 50
    assert rule.normalize_target(150, 120) == 150
    assert rule.is_valid_sell(150, 150)
    assert not rule.is_valid_sell(50, 150)


def test_star_board_uses_200_share_minimum_and_single_share_increment():
    rule = rule_for("科创板")

    assert rule.normalize_target(0, 150) == 0
    assert rule.normalize_target(0, 250) == 250
    assert rule.normalize_target(200, 399) == 200
    assert rule.normalize_target(200, 400) == 400
    assert rule.normalize_target(401, 250) == 401
    assert rule.normalize_target(401, 200) == 200
    assert rule.normalize_target(201, 0) == 0
    assert rule.is_valid_buy(200)
    assert rule.is_valid_buy(201)
    assert not rule.is_valid_buy(199)


def test_bse_uses_100_share_minimum_and_single_share_increment():
    rule = rule_for("北交所")

    assert rule.normalize_target(0, 99) == 0
    assert rule.normalize_target(0, 157) == 157
    assert rule.normalize_target(157, 200) == 157
    assert rule.normalize_target(157, 300) == 300
    assert rule.normalize_target(157, 180) == 157
    assert rule.normalize_target(157, 280) == 280
    assert rule.normalize_target(157, 150) == 157
    assert rule.normalize_target(157, 50) == 50
    assert rule.normalize_target(50, 0) == 0


def test_capacity_cap_keeps_orders_legal():
    star = rule_for("科创板")
    main = rule_for("主板")

    assert star.cap_target(0, 500, 150) == 0
    assert star.cap_target(0, 500, 250) == 250
    assert star.cap_target(500, 0, 150) == 500
    assert main.cap_target(0, 500, 250) == 200
    assert main.cap_target(500, 0, 250) == 300


def test_unknown_board_is_an_error():
    with pytest.raises(ExecutionError, match="未知板块"):
        rule_for("三板")
