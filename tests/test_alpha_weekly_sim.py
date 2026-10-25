"""快速周频研究模拟器的单元测试。"""

import pandas as pd
import pytest

from alpha.portfolio import HoldingPolicy
from scripts.build_alpha.weekly_sim import (
    _metrics,
    _select_top,
    simulate_score_frame,
    yearly_summary,
)


def test_select_top_respects_eligibility_and_order():
    """只从可交易股票中按分数降序选 Top-N。"""

    score = pd.Series([1.0, 3.0, 2.0], index=["A", "B", "C"])
    eligible = pd.Series([True, False, True], index=["A", "B", "C"])
    assert _select_top(score, eligible, 2).tolist() == ["C", "A"]


def test_metrics_reports_total_drawdown_and_cost():
    """周频指标与手工净值和成本一致。"""

    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    daily = pd.DataFrame(
        {
            "net_return": [0.1, -0.2, 0.05],
            "nav": [1.1, 0.88, 0.924],
            "cost": [10.0, 20.0, 5.0],
            "turnover": [1.0, 1.0, 0.5],
            "turnover_one_way": [0.5, 0.5, 0.25],
        },
        index=dates,
    )
    benchmark = pd.Series([0.0, 0.0, 0.0], index=dates)
    metrics = _metrics(daily, benchmark)
    assert metrics["total_return"] == pytest.approx(-0.076)
    assert metrics["max_drawdown"] == pytest.approx(-0.2)
    assert metrics["total_cost"] == pytest.approx(35.0)


def test_yearly_summary_compounds_weekly_returns():
    """逐年收益按当年周收益复利计算。"""

    daily = pd.DataFrame(
        {
            "net_return": [0.1, -0.1, 0.2],
            "nav": [1.1, 0.99, 1.188],
            "cost": [1.0, 1.0, 1.0],
            "turnover": [1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-12"]),
    )
    result = yearly_summary(daily)
    by_year = {row["year"]: row for row in result}
    assert by_year[2023]["total_return"] == pytest.approx(0.1)
    assert by_year[2024]["total_return"] == pytest.approx(0.08)


def test_simulator_passes_only_positive_previous_weights_to_buffer():
    """研究模拟器不能把零权重股票误当成上期持仓。"""

    dates = pd.DatetimeIndex(
        ["2023-01-06", "2023-01-13", "2023-01-20"]
    )
    stocks = ["A", "B", "C"]
    scores = pd.DataFrame(
        [[3.0, 2.0, 1.0], [2.1, 1.0, 3.0], [2.0, 1.0, 3.0]],
        index=dates,
        columns=stocks,
    )
    labels = pd.DataFrame(
        [[0.01, 0.0, -0.01]] * len(dates),
        index=dates,
        columns=stocks,
    )
    eligible = pd.DataFrame(True, index=dates, columns=stocks)
    amount = pd.DataFrame(1e12, index=dates, columns=stocks)
    impact = pd.DataFrame(0.01, index=dates, columns=stocks)
    benchmark = pd.Series(0.0, index=dates)
    board = pd.Series("主板", index=stocks)
    _, daily, orders = simulate_score_frame(
        scores,
        labels,
        eligible,
        start="2023-01-06",
        end="2023-01-20",
        top_n=1,
        auxiliary=(amount, impact, benchmark, board),
        holding_policy=HoldingPolicy(kind="rank_buffer", buffer_rank=2),
    )

    assert daily.loc[dates[1], "turnover"] == pytest.approx(0.0)
    assert orders.loc[orders["trade_date"] == dates[1]].empty
