"""候选组合的市值、流动性和行业暴露门槛。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from alpha.errors import AlphaEvaluationError


@dataclass(frozen=True)
class RiskGatePolicy:
    """组合级风险门槛的固定滚动口径。"""

    rolling_weeks: int = 13
    min_periods: int = 8
    max_size_abs_exposure: float = 2.0
    max_liquidity_abs_exposure: float = 2.0
    max_industry_active_weight: float = 0.35

    def __post_init__(self) -> None:
        if type(self.rolling_weeks) is not int or self.rolling_weeks <= 0:
            raise AlphaEvaluationError("风险门槛 rolling_weeks 必须是正整数")
        if (
            type(self.min_periods) is not int
            or not 1 <= self.min_periods <= self.rolling_weeks
        ):
            raise AlphaEvaluationError(
                "风险门槛 min_periods 必须在 1 和 rolling_weeks 之间"
            )
        for name in (
            "max_size_abs_exposure",
            "max_liquidity_abs_exposure",
            "max_industry_active_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise AlphaEvaluationError(f"风险门槛 {name} 必须是正有限数")

    def to_dict(self) -> dict:
        """返回稳定字典。"""

        return asdict(self)


@dataclass(frozen=True)
class RiskGateResult:
    """一次候选组合风险检查的时间序列和汇总结论。"""

    style_timeseries: pd.DataFrame
    industry_timeseries: pd.DataFrame
    summary: dict


def evaluate_risk_gate(
    style_series: pd.DataFrame,
    industry_series: pd.DataFrame,
    policy: RiskGatePolicy,
) -> RiskGateResult:
    """按滚动绝对暴露判断候选是否通过组合级风险门槛。"""

    style_required = {"trade_date", "style", "exposure"}
    style_missing = sorted(style_required - set(style_series.columns))
    if style_missing:
        raise AlphaEvaluationError(f"风险风格序列缺少列：{style_missing}")
    industry_required = {"trade_date", "industry", "active_weight"}
    industry_missing = sorted(industry_required - set(industry_series.columns))
    if industry_missing:
        raise AlphaEvaluationError(f"风险行业序列缺少列：{industry_missing}")

    styles = style_series.copy()
    styles["trade_date"] = pd.to_datetime(styles["trade_date"])
    styles["exposure"] = pd.to_numeric(styles["exposure"], errors="coerce")
    if not np.isfinite(styles["exposure"].to_numpy(dtype="float64")).all():
        raise AlphaEvaluationError("风险风格暴露必须全部为有限数值")
    style_pivot = styles.pivot(
        index="trade_date",
        columns="style",
        values="exposure",
    ).sort_index()
    missing_styles = {"SIZE", "LIQUIDITY"} - set(style_pivot.columns)
    if missing_styles:
        raise AlphaEvaluationError(
            f"风险风格序列缺少必要风格：{sorted(missing_styles)}"
        )
    rolling_style = style_pivot.abs().rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).mean()
    mature = rolling_style.notna().any(axis=1)
    if not mature.any():
        raise AlphaEvaluationError("风险门槛没有形成足够的滚动观测")

    industries = industry_series.copy()
    industries["trade_date"] = pd.to_datetime(industries["trade_date"])
    industries["active_weight"] = pd.to_numeric(
        industries["active_weight"], errors="coerce"
    ).fillna(0.0)
    dates = pd.DatetimeIndex(style_pivot.index)
    industry_pivot = industries.pivot_table(
        index="trade_date",
        columns="industry",
        values="active_weight",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=dates, fill_value=0.0).sort_index()
    rolling_industry = industry_pivot.abs().rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).mean()

    size = rolling_style["SIZE"].dropna()
    liquidity = rolling_style["LIQUIDITY"].dropna()
    industry = rolling_industry.stack(future_stack=True).dropna()
    size_peak = float(size.max())
    liquidity_peak = float(liquidity.max())
    industry_peak = float(industry.max())
    checks = {
        "size_ok": size_peak <= policy.max_size_abs_exposure,
        "liquidity_ok": (
            liquidity_peak <= policy.max_liquidity_abs_exposure
        ),
        "industry_ok": (
            industry_peak <= policy.max_industry_active_weight
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": all(checks.values()),
        "checks": checks,
        "rolling_weeks": policy.rolling_weeks,
        "min_periods": policy.min_periods,
        "mature_windows": int(mature.sum()),
        "max_size_abs_exposure": size_peak,
        "max_liquidity_abs_exposure": liquidity_peak,
        "max_industry_active_weight": industry_peak,
        "thresholds": {
            "max_size_abs_exposure": policy.max_size_abs_exposure,
            "max_liquidity_abs_exposure": (
                policy.max_liquidity_abs_exposure
            ),
            "max_industry_active_weight": (
                policy.max_industry_active_weight
            ),
        },
    }
    return RiskGateResult(
        style_timeseries=rolling_style.reset_index(),
        industry_timeseries=rolling_industry.reset_index(),
        summary=summary,
    )


__all__ = ["RiskGatePolicy", "RiskGateResult", "evaluate_risk_gate"]
