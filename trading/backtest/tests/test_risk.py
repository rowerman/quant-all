"""风险约束的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from backtest.risk import (
    apply_risk_constraints,
    cap_single_weights,
    cap_target_weights,
    constrain_one_way_turnover,
    industry_weight_witnesses,
)
from backtest.errors import ExecutionError
from backtest.specs import RiskSpec


DATES = pd.date_range("2024-01-29", periods=3, freq="B")
STOCKS = ["A", "B", "C"]


def test_single_cap_redistributes_excess_inside_selected_pool():
    weights = pd.DataFrame([[0.6, 0.3, 0.1]], index=DATES[:1], columns=STOCKS)
    capped, stats = cap_single_weights(weights, 0.4)
    assert np.allclose(capped.iloc[0].to_numpy(), [0.4, 0.4, 0.2])
    assert stats["single_cap_cells"] == 2
    assert stats["single_cap_residual_cash"] == pytest.approx(0.0)


def test_single_cap_keeps_unallocatable_weight_as_cash():
    weights = pd.DataFrame([[0.7, 0.3]], index=DATES[:1], columns=["A", "B"])
    capped, stats = cap_single_weights(weights, 0.4)
    assert np.allclose(capped.iloc[0].to_numpy(), [0.4, 0.4])
    assert stats["single_cap_residual_cash"] == pytest.approx(0.2)


def test_single_cap_preserves_non_rebalance_nan_rows():
    weights = pd.DataFrame([[np.nan, np.nan], [0.8, 0.2]], index=DATES[:2], columns=["A", "B"])
    capped, _ = cap_single_weights(weights, 0.6)
    assert capped.iloc[0].isna().all()
    assert np.allclose(capped.iloc[1].to_numpy(), [0.6, 0.4])


def test_one_way_turnover_scales_from_current_portfolio_and_cash():
    targets = pd.DataFrame(
        [[np.nan, np.nan], [1.0, 0.0], [0.0, 1.0]], index=DATES, columns=["A", "B"]
    )
    budget = pd.Series([1000.0, 1000.0, 1000.0], index=DATES)
    cash = pd.Series([1000.0, 1000.0, 0.0], index=DATES)
    price = pd.DataFrame(10.0, index=DATES, columns=["A", "B"])
    baseline = pd.DataFrame([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0]], index=DATES, columns=["A", "B"])
    constrained, stats = constrain_one_way_turnover(
        targets, budget, cash, price, baseline, 0.25
    )
    assert np.allclose(constrained.iloc[1].to_numpy(), [0.25, 0.0])
    assert np.allclose(constrained.iloc[2].to_numpy(), [0.75, 0.25])
    assert stats["turnover_limited_days"] == 2
    assert stats["turnover_after_max"] == pytest.approx(0.25)
    assert stats["turnover_limit_error_max"] == pytest.approx(0.0)


def test_one_way_turnover_zero_freezes_target_at_current_position():
    targets = pd.DataFrame([[1.0, 0.0]], index=DATES[:1], columns=["A", "B"])
    budget = pd.Series([1000.0], index=DATES[:1])
    cash = pd.Series([0.0], index=DATES[:1])
    price = pd.DataFrame([[10.0, 10.0]], index=DATES[:1], columns=["A", "B"])
    baseline = pd.DataFrame([[100.0, 0.0]], index=DATES[:1], columns=["A", "B"])
    constrained, stats = constrain_one_way_turnover(
        targets, budget, cash, price, baseline, 0.0
    )
    assert np.allclose(constrained.iloc[0].to_numpy(), [0.0, 0.0])
    assert stats["turnover_limited_days"] == 1


def test_apply_risk_constraints_composes_both_limits():
    targets = pd.DataFrame([[1.0, 0.0]], index=DATES[:1], columns=["A", "B"])
    budget = pd.Series([1000.0], index=DATES[:1])
    cash = pd.Series([1000.0], index=DATES[:1])
    price = pd.DataFrame([[10.0, 10.0]], index=DATES[:1], columns=["A", "B"])
    baseline = pd.DataFrame([[0.0, 0.0]], index=DATES[:1], columns=["A", "B"])
    constrained, stats = apply_risk_constraints(
        targets, budget, cash, price, baseline,
        RiskSpec(max_single_weight=0.6, max_one_way_turnover=1.0),
    )
    assert np.allclose(constrained.iloc[0].to_numpy(), [0.6, 0.0])
    assert stats["single_cap_max_weight"] == pytest.approx(0.6)


def test_industry_cap_redistributes_across_industries_with_single_headroom():
    weights = pd.DataFrame([[0.5, 0.2, 0.2, 0.1]], index=DATES[:1], columns=list("ABCD"))
    industries = pd.DataFrame([[1, 1, 2, 3]], index=weights.index, columns=weights.columns)
    capped, stats = cap_target_weights(weights, 0.4, industries, 0.5)
    row = capped.iloc[0]
    assert row["A"] <= 0.4 + 1e-12
    assert row[["A", "B"]].sum() <= 0.5 + 1e-12
    assert row.sum() == pytest.approx(1.0)
    assert stats["industry_cap_target_max_weight"] <= 0.5 + 1e-12
    assert stats["industry_cap_groups"] >= 1


def test_industry_cap_keeps_unallocatable_weight_as_cash():
    weights = pd.DataFrame([[0.8, 0.2]], index=DATES[:1], columns=["A", "B"])
    industries = pd.DataFrame([[1, 1]], index=weights.index, columns=weights.columns)
    capped, stats = cap_target_weights(weights, None, industries, 0.5)
    assert capped.iloc[0].sum() == pytest.approx(0.5)
    assert stats["industry_cap_residual_cash"] == pytest.approx(0.5)


def test_industry_cap_rejects_missing_target_industry():
    weights = pd.DataFrame([[0.5, 0.5]], index=DATES[:1], columns=["A", "B"])
    industries = pd.DataFrame([[1, 0]], index=weights.index, columns=weights.columns)
    with pytest.raises(ExecutionError, match="行业缺失"):
        cap_target_weights(weights, None, industries, 0.5)


def test_industry_witness_reports_turnover_inherited_excess():
    targets = pd.DataFrame([[0.4, 0.3, 0.3]], index=DATES[:1], columns=STOCKS)
    industries = pd.DataFrame([[1, 1, 2]], index=targets.index, columns=targets.columns)
    stats = industry_weight_witnesses(targets, industries, 0.5)
    assert stats["industry_post_turnover_max_weight"] == pytest.approx(0.7)
    assert stats["industry_inherited_over_cap_days"] == 1
