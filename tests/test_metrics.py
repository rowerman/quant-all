"""绩效指标口径。"""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import summarize


def _daily(values, benchmark=None):
    """按净值序列构造标准化日结果。"""
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    nav = pd.Series(values, index=index, dtype=float)
    daily = pd.DataFrame(
        {
            "portfolio_value": nav * 1_000_000.0,
            "cash": 0.0,
            "nav": nav,
            "return": nav.pct_change().fillna(0.0),
            "drawdown": nav / nav.cummax() - 1.0,
            "cost": 100.0,
            "turnover": 0.01,
        }
    )
    if benchmark is not None:
        bench = pd.Series(benchmark, index=index, dtype=float)
        daily["benchmark_nav"] = bench
        daily["benchmark_return"] = bench.pct_change().fillna(0.0)
        daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily


def test_summary_matches_manual_calculation():
    daily = _daily([1.0, 1.02, 1.01, 1.03])
    result = summarize(daily)
    assert result["trading_days"] == 4
    assert result["total_return"] == pytest.approx(0.03)
    assert result["max_drawdown"] == pytest.approx(1.01 / 1.02 - 1.0)
    returns = daily["return"]
    assert result["annual_volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
    assert result["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert result["total_cost"] == pytest.approx(400.0)
    assert result["turnover_annual"] == pytest.approx(0.01 * 4 * 252 / 4)
    assert result["months"] >= 1


def test_summary_reports_benchmark_excess():
    daily = _daily([1.0, 1.02, 1.01, 1.03], benchmark=[1.0, 1.01, 1.005, 1.01])
    result = summarize(daily)
    assert result["benchmark_total_return"] == pytest.approx(0.01)
    assert result["excess_annual_return"] == pytest.approx(
        result["annual_return"] - result["benchmark_annual_return"]
    )
    assert result["information_ratio"] is not None


def test_summary_without_benchmark_has_no_excess_keys():
    result = summarize(_daily([1.0, 0.99]))
    assert "excess_annual_return" not in result
    assert result["annual_return"] < 0


def test_summary_separates_cash_and_economic_cost():
    daily = _daily([1.0, 1.01])
    daily["total_cost"] = daily["cost"] + 25.0
    result = summarize(daily)
    assert result["cash_cost"] == pytest.approx(200.0)
    assert result["total_cost"] == pytest.approx(250.0)


def test_summary_falls_back_when_cost_components_are_null():
    daily = _daily([1.0, 1.01])
    daily["total_cost"] = np.nan
    daily["commission"] = np.nan
    result = summarize(daily)
    assert result["total_cost"] == pytest.approx(200.0)
    assert "total_commission" not in result


def test_summary_of_empty_frame_is_empty():
    assert summarize(pd.DataFrame()) == {}

