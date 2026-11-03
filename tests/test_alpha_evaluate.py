"""模型指标、分箱和 P2 决策门测试。"""

import numpy as np
import pandas as pd
import pytest

from alpha.evaluate import (
    evaluate_score_frame,
    p2_decision,
    p3_decision_gate,
    select_p3_candidate,
)
from scripts.build_alpha.evaluate import _seed_stability


def _score_frames():
    """构造每天严格单调的分数和标签。"""

    dates = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    stocks = [f"S{index}" for index in range(8)]
    scores = pd.DataFrame(
        np.tile(np.arange(8, dtype="float64"), (len(dates), 1)),
        index=dates,
        columns=stocks,
    )
    labels = scores / 100.0
    eligible = pd.DataFrame(True, index=dates, columns=stocks)
    return scores, labels, eligible


def test_evaluate_score_frame_matches_perfect_rank_ic():
    """完美排序得到 RankIC=1、正单调性和正首尾差。"""

    scores, labels, eligible = _score_frames()
    result = evaluate_score_frame(scores, labels, eligible, quantiles=4)
    assert result["rank_ic_mean"] == pytest.approx(1.0)
    assert result["icir"] is None or result["icir"] > 0
    assert result["monotonicity"] == pytest.approx(1.0)
    assert result["top_minus_bottom_mean"] > 0
    assert result["score_autocorrelation_1w"] == pytest.approx(1.0)


def test_ineligible_rows_do_not_affect_metrics():
    """池外样本即使分数和标签极端也不参与评估。"""

    scores, labels, eligible = _score_frames()
    changed_scores = scores.copy()
    changed_labels = labels.copy()
    changed_scores.loc[:, "S0"] = -1000.0
    changed_labels.loc[:, "S0"] = 1000.0
    eligible.loc[:, "S0"] = False
    left = evaluate_score_frame(scores, labels, eligible, quantiles=4)
    right = evaluate_score_frame(changed_scores, changed_labels, eligible, quantiles=4)
    assert right["rank_ic_mean"] == pytest.approx(left["rank_ic_mean"])


def test_p2_decision_pass_and_clear_failure():
    """决策门能区分通过和明显失败。"""

    baseline = {
        "B3_ALL_ZSCORE": {
            "rank_ic_mean": 0.06,
            "icir": 0.50,
            "annual_return": 0.12,
            "max_drawdown": -0.40,
            "turnover_two_way_annual": 40.0,
        },
        "B4_ALL_RANK": {
            "rank_ic_mean": 0.05,
            "icir": 0.40,
            "annual_return": 0.10,
            "max_drawdown": -0.35,
            "turnover_two_way_annual": 50.0,
        },
    }
    passed = p2_decision(
        {
            "rank_ic_mean": 0.07,
            "icir": 0.60,
            "annual_return": 0.15,
            "max_drawdown": -0.30,
            "turnover_two_way_annual": 30.0,
        },
        baseline,
    )
    failed = p2_decision(
        {
            "rank_ic_mean": 0.03,
            "icir": 0.20,
            "annual_return": 0.05,
            "max_drawdown": -0.45,
            "turnover_two_way_annual": 45.0,
        },
        baseline,
    )
    assert passed["status"] == "pass"
    assert passed["enter_p3"]
    assert failed["status"] == "fail"
    assert not failed["enter_p3"]


def test_seed_stability_reports_every_fixed_seed():
    """稳定性报告保留全部种子，并单独给出集成结果。"""

    comparison = {
        "MODEL": {"rank_ic_mean": 0.09, "icir": 0.8, "annual_return": 0.12},
        **{
            f"SEED_{seed}": {
                "rank_ic_mean": 0.08 + seed * 0.001,
                "icir": 0.7,
                "annual_return": 0.10,
                "turnover_two_way_annual": 80.0,
                "max_drawdown": -0.30,
            }
            for seed in range(5)
        },
    }
    report = _seed_stability(
        {
            "model_type": "lightgbm_ensemble",
            "seeds": [0, 1, 2, 3, 4],
        },
        comparison,
    )
    assert [row["seed"] for row in report["members"]] == [0, 1, 2, 3, 4]
    assert report["ensemble"]["rank_ic_mean"] == 0.09
    assert report["summary"]["rank_ic_mean"]["mean"] == pytest.approx(0.082)


def test_p3_decision_gate_requires_improvements_turnover_and_yearly_stability():
    """P3 决策门同时检查改善项、换手和逐年稳定性。"""

    ridge = {
        "rank_ic_mean": 0.09,
        "icir": 0.80,
        "annual_return": 0.14,
        "max_drawdown": -0.32,
        "turnover_two_way_annual": 90.0,
    }
    candidate = {
        "rank_ic_mean": 0.10,
        "icir": 0.85,
        "annual_return": 0.17,
        "max_drawdown": -0.27,
        "turnover_two_way_annual": 100.0,
    }
    yearly = [
        {
            "year": year,
            "rank_ic_mean": 0.05,
            "excess_total_return": 0.02,
        }
        for year in range(2018, 2024)
    ]
    passed = p3_decision_gate(
        candidate,
        ridge,
        yearly,
        expected_years=range(2018, 2024),
    )
    too_high_turnover = p3_decision_gate(
        {**candidate, "turnover_two_way_annual": 181.0},
        ridge,
        yearly,
        expected_years=range(2018, 2024),
    )
    unstable = p3_decision_gate(
        candidate,
        ridge,
        [{**row, "excess_total_return": -0.01} for row in yearly],
        expected_years=range(2018, 2024),
    )

    assert passed["passed"]
    assert passed["improvement_count"] == 4
    assert not too_high_turnover["turnover_ok"]
    assert not unstable["yearly_stable"]
    assert unstable["negative_excess_years"] == list(range(2018, 2024))


def test_select_p3_candidate_uses_fixed_metric_order():
    """通过候选按收益、Sharpe、回撤和换手依次选择。"""

    def _candidate(annual, sharpe, drawdown, turnover, priority=0):
        return {
            "metrics": {
                "annual_return": annual,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "turnover_two_way_annual": turnover,
            },
            "gate": {"passed": True},
            "priority": priority,
        }

    selected = select_p3_candidate(
        {
            "A": _candidate(0.18, 0.7, -0.30, 80.0),
            "B": _candidate(0.20, 0.6, -0.35, 70.0),
            "C": _candidate(0.20, 0.8, -0.40, 90.0),
            "FAIL": {
                **_candidate(0.30, 0.9, -0.10, 50.0),
                "gate": {"passed": False},
            },
        }
    )
    assert selected["selected"] == "C"
    assert selected["ranked"] == ["C", "B", "A"]

    fallback = select_p3_candidate({"A": {**_candidate(0.1, 0.1, -0.3, 50.0), "gate": {"passed": False}}})
    assert fallback["selected"] is None
