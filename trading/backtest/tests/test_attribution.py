"""成本、换手和行业暴露归因口径。"""

import pandas as pd
import pytest

from backtest.attribution import (
    build_report,
    cost_attribution,
    industry_exposure_frame,
    summarize_industry_exposure,
    turnover_attribution,
)


def _daily():
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_010_000.0, 999_000.0],
            "cash": [100_000.0, 10_000.0, 50_000.0],
            "nav": [1.0, 1.01, 1.0],
            "return": [0.0, 0.01, -0.010891],
            "drawdown": [0.0, 0.0, -0.010891],
            "cost": [0.0, 10.0, 10.0],
            "commission": [0.0, 5.0, 5.0],
            "stamp_tax": [0.0, 5.0, 5.0],
            "transfer_fee": [0.0, 0.0, 0.0],
            "dividend_tax": [0.0, 0.0, 0.0],
            "slippage_cost": [0.0, 2.0, 2.0],
            "impact_cost": [0.0, 3.0, 3.0],
            "total_cost": [0.0, 15.0, 15.0],
            "turnover": [1.0, 2.0, 0.0],
            "turnover_one_way": [1.0, 1.0, 0.0],
            "buy_value": [1_000_000.0, 1_010_000.0, 0.0],
            "sell_value": [0.0, 1_010_000.0, 0.0],
        },
        index=index,
    )


def _orders():
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "stock_code": ["SH600000", "SH600000", "SZ000001", "SH600000"],
            "Side": ["Buy", "Buy", "Sell", "Sell"],
            "Size": [50_000.0, 50_000.0, 100_000.0, 10_000.0],
            "Price": [10.0, 10.0, 10.1, 10.0],
            "commission": [5.0, 5.0, 5.0, 5.0],
            "stamp_tax": [0.0, 0.0, 5.0, 5.0],
            "transfer_fee": [0.0, 0.0, 0.0, 0.0],
            "slippage_cost": [1.0, 1.0, 1.0, 2.0],
            "impact_cost": [1.5, 1.5, 1.5, 3.0],
            "forced_exit": [False, False, False, True],
        }
    )


def test_turnover_attribution_separates_one_and_two_way():
    result = turnover_attribution(_daily(), _orders())
    assert result["buy_value"] == pytest.approx(1_000_000.0)
    assert result["sell_value"] == pytest.approx(1_110_000.0)
    assert result["forced_exit_sell_value"] == pytest.approx(100_000.0)
    day2_sell = 100_000.0 / 1_000_000.0
    assert result["two_way_annual"] == pytest.approx((1.0 + 1.01 + day2_sell) * 252 / 3)
    assert result["one_way_annual"] == pytest.approx((1.01 + day2_sell) * 252 / 3)
    assert result["one_way_annual"] <= result["two_way_annual"]


def test_cost_attribution_sums_components_and_sides():
    result = cost_attribution(_daily(), _orders(), risk_free_rate=0.02)
    total = result["total"]
    assert total["commission"] == pytest.approx(10.0)
    assert total["stamp_tax"] == pytest.approx(10.0)
    assert total["slippage_cost"] == pytest.approx(4.0)
    assert total["impact_cost"] == pytest.approx(6.0)
    assert total["cash_cost"] == pytest.approx(20.0)
    assert total["total_cost"] == pytest.approx(30.0)
    assert result["by_side"]["buy"]["total_cost"] == pytest.approx(15.0)
    assert result["by_side"]["sell"]["total_cost"] == pytest.approx(27.5)
    assert result["bps_of_nav"]["total_cost"] == pytest.approx(30.0 / 1_003_000.0 * 10_000)


def test_industry_exposure_includes_cash_and_unclassified():
    daily = _daily()
    industry = pd.DataFrame(
        [[1, 1, 0], [1, 1, 0], [2, 1, 0]],
        index=daily.index,
        columns=["SH600000", "SZ000001", "SZ000002"],
    )
    holdings = pd.DataFrame(
        {
            "trade_date": [daily.index[0], daily.index[0], daily.index[1]],
            "stock_code": ["SH600000", "SZ000001", "SZ000002"],
            "market_value": [500_000.0, 400_000.0, 100_000.0],
            "weight": [0.5, 0.4, 0.1],
        }
    )
    exposure = industry_exposure_frame(
        daily, holdings, industry, {1: "SW2014 801010 农林牧渔", 2: "SW2014 801020 采掘"}
    )
    first = exposure.loc[exposure["trade_date"].eq(daily.index[0])]
    assert set(first["industry"]) == {"SW2014 801010 农林牧渔", "现金"}
    assert first.loc[first["is_cash"], "weight"].iloc[0] == pytest.approx(0.1)
    second = exposure.loc[exposure["trade_date"].eq(daily.index[1])]
    assert "未分类" in set(second["industry"])
    summary = summarize_industry_exposure(exposure, daily)
    assert summary["available"] is True
    assert summary["cash_avg_weight"] == pytest.approx(
        (0.1 + (1.0 - 100_000.0 / 1_010_000.0) + 1.0) / 3
    )
    assert summary["unclassified_avg_weight"] > 0


def test_build_report_handles_provider_cash_and_nan_rows():
    daily = _daily()
    # 目标权重可以有现金、NaN 行或全 0 行；归因只看实际订单和持仓。
    report = build_report(
        daily,
        orders=_orders(),
        holdings=pd.DataFrame(),
        risk_free_rate=0.02,
        impact_model="sqrt",
    )
    assert report.metrics["risk_free_rate"] == pytest.approx(0.02)
    assert report.metrics["turnover_one_way_annual"] <= report.metrics["turnover_two_way_annual"]
    assert report.metrics["cost_attribution"]["impact_model"] == "sqrt"
    assert report.metrics["industry_exposure"]["available"] is False

