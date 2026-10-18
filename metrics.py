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


def summarize(daily: pd.DataFrame) -> dict:
    """汇总净值、风险、成本与基准超额。

    参数:
        daily: 标准化日结果，需要 ``portfolio_value``、``return``、``nav``、
            ``drawdown``、``cost``、``turnover`` 列；有基准时还要
            ``benchmark_return``、``benchmark_nav``、``excess_return``。
    """
    if daily.empty:
        return {}
    returns = daily["return"].astype(float)
    days = int(len(daily))
    nav = daily["nav"].astype(float)
    total_return = float(nav.iloc[-1]) - 1.0
    annual_return = _annualize(total_return, days)
    volatility = float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS) if days > 1 else None
    max_drawdown = float(daily["drawdown"].astype(float).min())
    result = {
        "start": str(pd.Timestamp(daily.index[0]).date()),
        "end": str(pd.Timestamp(daily.index[-1]).date()),
        "trading_days": days,
        "total_return": _number(total_return),
        "annual_return": _number(annual_return),
        "annual_volatility": _number(volatility),
        "sharpe": _ratio(_number(returns.mean()) * math.sqrt(TRADING_DAYS), _number(returns.std(ddof=1)))
        if days > 1
        else None,
        "max_drawdown": _number(max_drawdown),
        "calmar": _ratio(annual_return, abs(max_drawdown)) if max_drawdown else None,
        "daily_win_rate": _number((returns > 0).mean()),
        "turnover_annual": _number(daily["turnover"].astype(float).sum() * TRADING_DAYS / days),
        "cash_cost": _number(daily["cost"].astype(float).sum()),
        "total_cost": _number(
            daily["total_cost"].fillna(daily["cost"]).astype(float).sum()
            if "total_cost" in daily.columns and daily["total_cost"].notna().any()
            else daily["cost"].astype(float).sum()
        ),
    }
    for name in ("commission", "stamp_tax", "transfer_fee", "dividend_tax", "slippage_cost"):
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
                "excess_win_rate": _number((excess > 0).mean()),
            }
        )
    return result

