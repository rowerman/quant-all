"""评估用的小工具：序列统计、净值累乘与最大回撤。

被 IC 家族与 binplot 家族共用，避免同一个口径在两处各写一遍。
"""

from __future__ import annotations

import numpy as np

# 年化换算用的交易日数
TRADING_DAYS = 252


def series_stats(values) -> dict:
    """把一条收益或相关序列压成均值、标准差、标准误、IR、t 值与胜率。

    参数:
        values: 逐日序列，允许含 NaN。
    """
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "std": None, "stderr": None, "ir": None, "t": None,
                "win_rate": None, "days": 0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else None
    stderr = (std / float(np.sqrt(array.size))) if std else None
    ir = (mean / std) if std else None
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ir": ir,
        "t": (mean / stderr) if stderr else None,
        "win_rate": float((array > 0).mean()),
        "days": int(array.size),
    }


def compound(returns) -> np.ndarray:
    """把逐日收益累乘成净值曲线。

    参数:
        returns: 逐日收益序列，缺失值按 0 处理。
    """
    array = np.asarray(returns, dtype="float64")
    array = np.where(np.isfinite(array), array, 0.0)
    return np.cumprod(1.0 + array)


def max_drawdown(equity) -> float:
    """最大回撤，返回负数（例如 -0.12 表示最深回撤 12%）。

    参数:
        equity: 净值曲线。
    """
    array = np.asarray(equity, dtype="float64")
    if array.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(np.concatenate(([1.0], array)))[1:]
    drawdown = array / peak - 1.0
    return float(drawdown.min())


def annualised_sharpe(returns, periods: int = TRADING_DAYS) -> float | None:
    """年化夏普（不扣费用、不做无风险利率调整），样本不足或波动为 0 时返回空。

    参数:
        returns: 逐日收益序列。
        periods: 一年的交易日数。
    """
    array = np.asarray(returns, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size < 2:
        return None
    std = float(array.std(ddof=1))
    if std <= 0:
        return None
    return float(array.mean()) / std * float(np.sqrt(periods))
