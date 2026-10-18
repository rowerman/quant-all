"""回测绩效指标：只从标准化日结果推导，前端不重算。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


def _number(value) -> float | None:
    """把指标值收敛成 JSON 可用的有限数；无效值返回 None。"""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """做除法，分母为 0 或无效时返回 None。"""
    if numerator is None or denominator in (None, 0):
        return None
    return _number(numerator / denominator)


def _annualize(total_return: float, days: int) -> float | None:
    """把区间累计收益折算成年化收益。"""
    if days <= 0 or total_return <= -1:
        return None
    return (1.0 + total_return) ** (TRADING_DAYS / days) - 1.0


def summarize(daily: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """汇总净值、风险、成本与基准超额。

    参数:
        daily: 标准化日结果，需要 ``portfolio_value``、``return``、``nav``、
            ``drawdown``、``cost``、``turnover`` 列；有基准时还要
            ``benchmark_return``、``benchmark_nav``、``excess_return``。
        risk_free_rate: 年化无风险利率，默认 2%。
    """
    if daily.empty:
        return {}
    risk_free_rate = float(risk_free_rate)
    if not math.isfinite(risk_free_rate) or not -1.0 < risk_free_rate <= 1.0:
        return {}
    returns = daily["return"].astype(float)
    days = int(len(daily))
    nav = daily["nav"].astype(float)
    total_return = float(nav.iloc[-1]) - 1.0
    annual_return = _annualize(total_return, days)
    volatility = float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS) if days > 1 else None
    max_drawdown = float(daily["drawdown"].astype(float).min())
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess_over_rf = returns - rf_daily
    downside = np.minimum(excess_over_rf.to_numpy(dtype="float64"), 0.0)
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside)))) * math.sqrt(TRADING_DAYS)
        if days > 0
        else None
    )
    portfolio_value = daily["portfolio_value"].astype(float)
    average_value = float(portfolio_value.mean()) if len(portfolio_value) else None
    cash_cost = float(daily["cost"].astype(float).sum())
    total_cost = float(
        daily["total_cost"].fillna(daily["cost"]).astype(float).sum()
        if "total_cost" in daily.columns and daily["total_cost"].notna().any()
        else cash_cost
    )
    turnover_two_way = daily["turnover"].astype(float) if "turnover" in daily else pd.Series(0.0, index=daily.index)
    turnover_one_way = (
        daily["turnover_one_way"].astype(float)
        if "turnover_one_way" in daily.columns
        else pd.Series(np.nan, index=daily.index)
    )
    result = {
        "start": str(pd.Timestamp(daily.index[0]).date()),
        "end": str(pd.Timestamp(daily.index[-1]).date()),
        "trading_days": days,
        "risk_free_rate": _number(risk_free_rate),
        "average_portfolio_value": _number(average_value),
        "total_return": _number(total_return),
        "annual_return": _number(annual_return),
        "annual_volatility": _number(volatility),
        "annual_excess_return_over_rf": _number(
            None if annual_return is None else annual_return - risk_free_rate
        ),
        "sharpe": _ratio(
            _number(excess_over_rf.mean()) * math.sqrt(TRADING_DAYS),
            _number(excess_over_rf.std(ddof=1)),
        )
        if days > 1
        else None,
        "sortino": _ratio(
            _number(excess_over_rf.mean()) * math.sqrt(TRADING_DAYS),
            _number(downside_deviation),
        )
        if days > 1 and downside_deviation
        else None,
        "max_drawdown": _number(max_drawdown),
        "calmar": _ratio(annual_return, abs(max_drawdown)) if max_drawdown else None,
        "daily_win_rate": _number((returns > 0).mean()),
        "turnover_annual": _number(turnover_two_way.sum() * TRADING_DAYS / days),
        "turnover_two_way_annual": _number(
            turnover_two_way.sum() * TRADING_DAYS / days
        ),
        "turnover_one_way_annual": _number(
            None
            if turnover_one_way.isna().all()
            else turnover_one_way.sum(skipna=True) * TRADING_DAYS / days
        ),
        "cash_cost": _number(cash_cost),
        "total_cost": _number(total_cost),
        "total_cost_bps_of_nav": _number(
            None
            if not average_value
            else total_cost / average_value * 10_000.0
        ),
    }
    for name in (
        "commission",
        "stamp_tax",
        "transfer_fee",
        "dividend_tax",
        "slippage_cost",
        "impact_cost",
    ):
        if name in daily.columns and daily[name].notna().any():
            result[f"total_{name}"] = _number(daily[name].astype(float).sum())
    monthly = (1.0 + returns).groupby(daily.index.to_period("M")).prod() - 1.0
    result["monthly_win_rate"] = _number((monthly > 0).mean()) if len(monthly) else None
    result["months"] = int(len(monthly))
    if "benchmark_return" in daily.columns:
        benchmark_return = float(daily["benchmark_nav"].astype(float).iloc[-1]) - 1.0
        benchmark_annual = _annualize(benchmark_return, days)
        excess = daily["excess_return"].astype(float)
        tracking_error = float(excess.std(ddof=1)) * math.sqrt(TRADING_DAYS) if days > 1 else None
        benchmark_returns = daily["benchmark_return"].astype(float)
        beta = None
        alpha_annual = None
        if days > 1:
            covariance = float(np.cov(returns, benchmark_returns, ddof=1)[0, 1])
            variance = float(np.var(benchmark_returns, ddof=1))
            beta = _ratio(covariance, variance)
            if beta is not None:
                alpha_annual = _number(
                    (
                        float(excess_over_rf.mean())
                        - beta * float((benchmark_returns - rf_daily).mean())
                    )
                    * TRADING_DAYS
                )
        result.update(
            {
                "benchmark_total_return": _number(benchmark_return),
                "benchmark_annual_return": _number(benchmark_annual),
                "excess_annual_return": _number(
                    None if annual_return is None or benchmark_annual is None
                    else annual_return - benchmark_annual
                ),
                "tracking_error": _number(tracking_error),
                "information_ratio": _ratio(
                    _number(excess.mean()) * math.sqrt(TRADING_DAYS),
                    _number(excess.std(ddof=1)),
                )
                if days > 1
                else None,
                "beta": beta,
                "alpha_annual": alpha_annual,
                "excess_win_rate": _number((excess > 0).mean()),
            }
        )
    return result

