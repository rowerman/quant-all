"""组合级风险门槛测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha.errors import AlphaEvaluationError
from alpha.risk import RiskGatePolicy, evaluate_risk_gate


def _inputs(size: float = 1.0, liquidity: float = 0.8):
    dates = pd.date_range("2023-01-06", periods=13, freq="W-FRI")
    styles = pd.DataFrame(
        [
            {
                "trade_date": day,
                "style": style,
                "exposure": (
                    size if style == "SIZE" else liquidity
                ),
            }
            for day in dates
            for style in ("SIZE", "LIQUIDITY")
        ]
    )
    industries = pd.DataFrame(
        [
            {
                "trade_date": day,
                "industry": industry,
                "active_weight": weight,
            }
            for day in dates
            for industry, weight in (("I1", 0.10), ("I2", -0.05))
        ]
    )
    return styles, industries


def test_risk_gate_uses_rolling_absolute_exposure():
    """滚动市值、流动性和行业暴露都在阈值内时通过。"""

    styles, industries = _inputs()
    result = evaluate_risk_gate(
        styles,
        industries,
        RiskGatePolicy(),
    )
    assert result.summary["passed"] is True
    assert result.summary["mature_windows"] == 6
    assert result.summary["max_size_abs_exposure"] == pytest.approx(1.0)
    assert result.summary["max_liquidity_abs_exposure"] == pytest.approx(0.8)
    assert result.summary["max_industry_active_weight"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    "style,industry,field",
    [
        ("size", 0.10, "size_ok"),
        ("liquidity", 0.10, "liquidity_ok"),
        ("size", 0.40, "industry_ok"),
    ],
)
def test_risk_gate_fails_closed_on_each_dimension(style, industry, field):
    """任一维度越限都使候选风险门槛失败。"""

    styles, industries = _inputs(
        size=2.1 if style == "size" else 1.0,
        liquidity=2.1 if style == "liquidity" else 0.8,
    )
    if industry > 0.35:
        industries.loc[:, "active_weight"] = industry
    result = evaluate_risk_gate(
        styles,
        industries,
        RiskGatePolicy(),
    )
    assert result.summary["passed"] is False
    assert result.summary["checks"][field] is False


def test_risk_gate_requires_enough_observations_and_columns():
    """观测不足或缺少必要风格时拒绝给出通过结论。"""

    styles, industries = _inputs()
    with pytest.raises(AlphaEvaluationError, match="滚动观测"):
        evaluate_risk_gate(
            styles.head(6),
            industries,
            RiskGatePolicy(),
        )
    with pytest.raises(AlphaEvaluationError, match="必要风格"):
        evaluate_risk_gate(
            styles.loc[styles["style"].ne("LIQUIDITY")],
            industries,
            RiskGatePolicy(),
        )
