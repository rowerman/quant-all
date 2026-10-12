"""风格拆分：暴露系数与增量 IC。

两个指标各占一个 stage（`exposure` / `incremental_ic`），共用同一份回归结果
（`EvalContext.exposures()` 带缓存），所以同一档行业开关只算一次。

口径要点：

- 因子的值先做截面 3 倍 MAD 去极值 + z-score，再对 7 个同样标准化过的风格回归，
  系数含义统一为「因子每 1 个标准差对应的风格暴露」，不同风格之间可直接比大小；
- 每天一个截面回归（Fama-MacBeth），报系数的时序均值与它的 t 值，另报显著占比；
- 增量 IC = 残差的逐日 RankIC，raw 在残差的同一有效配对与日期上重算；
- 行业层默认关闭，原因写在 payload 的 `industry.note` 里，前端直接展示。
"""

from __future__ import annotations

import numpy as np

from calculate.daily import MIN_CROSS_SECTION, daily_corr
from calculate.exposure import (
    INSUFFICIENT_DAYS,
    MIN_REGRESSION_ROWS,
    SIGNIFICANCE_T,
    describe_styles,
    residual_rank_ic,
)
from calculate.metrics.base import Metric, MetricParam, register_metric
from calculate.stats import series_stats
from calculate.styles import INDUSTRY_OFF, STYLE_KEYS, industry_choices

# 行业开关：默认关闭，可选值由已注册的行业数据源给出（V11-4 只有静态行业）
INDUSTRY_PARAM = MetricParam(
    key="industry", label="行业层", kind="enum",
    default=INDUSTRY_OFF, choices=industry_choices(),
)


def _industry_key(options) -> str:
    """从指标参数里取行业开关取值。

    参数:
        options: 指标参数。
    """
    return (options or {}).get("industry") or INDUSTRY_OFF


def _fm_stats(coefficients: np.ndarray, t_values: np.ndarray) -> dict:
    """把一条日度系数序列压成均值、Fama-MacBeth t 值与显著占比。

    参数:
        coefficients: 日度暴露系数，允许含 NaN。
        t_values: 与之对应的日度 t 值。
    """
    values = np.asarray(coefficients, dtype="float64")
    stats = series_stats(values)
    finite_coefficient = values[np.isfinite(values)]
    finite_t = np.asarray(t_values, dtype="float64")
    finite_t = finite_t[np.isfinite(finite_t)]
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "t": stats["t"],
        "positive_share": (
            float((finite_coefficient > 0).mean()) if finite_coefficient.size else None
        ),
        "significant_share": (
            float((np.abs(finite_t) > SIGNIFICANCE_T).mean()) if finite_t.size else None
        ),
        "days": stats["days"],
    }


def _insufficient_payload(context, industry_key: str, days: int) -> dict:
    """有效回归日不足时的载荷。

    参数:
        context: 评估上下文。
        industry_key: 行业开关取值。
        days: 有效回归日数。
    """
    return {
        "styles": describe_styles(),
        "method": "逐日截面 OLS（Fama-MacBeth）：标准化因子 ~ 1 + 7 个标准化风格",
        "standardize": "因子与风格都按交易日做 3 倍 MAD 去极值 + z-score；风格缺失置 0（中性）",
        "sample": "池内且因子非空（不看标签），使暴露与标签周期解耦",
        "coefficient_note": "系数 = 风格增加 1 单位时标准化因子的变化；VALUE 为两项 z 的均值",
        "days": days,
        "skipped_days": None,
        "mean_rows": None,
        "mean_r2": None,
        "stats": [],
        "series": {"dates": [], "coefficient": {}, "r2": []},
        "significance_t": SIGNIFICANCE_T,
        "insufficient": True,
        "min_days": INSUFFICIENT_DAYS,
        "min_rows": MIN_REGRESSION_ROWS,
        "hint": (f"有效回归日只有 {days} 天，少于 {INSUFFICIENT_DAYS} 天，结论不可用；"
                 "请拉长评估区间"),
        "industry": {"chosen": industry_key, "enabled": industry_key != INDUSTRY_OFF},
    }


def compute_exposure(context, options) -> dict:
    """算风格暴露：逐日截面回归的系数、t 值与逐日序列。

    参数:
        context: 评估上下文。
        options: 指标参数（``industry``）。
    """
    industry_key = _industry_key(options)
    result = context.exposures(industry_key)
    days = len(result.dates)
    if days < INSUFFICIENT_DAYS:
        payload = _insufficient_payload(context, industry_key, days)
        payload["industry"] = result.industry
        payload["skipped_days"] = result.skipped_days
        return payload

    stats = []
    for index, key in enumerate(STYLE_KEYS):
        stats.append({
            "key": key,
            **_fm_stats(result.coefficients[:, index], result.t_stats[:, index]),
        })
    return {
        "styles": describe_styles(),
        "method": "逐日截面 OLS（Fama-MacBeth）：标准化因子 ~ 1 + 7 个标准化风格",
        "standardize": "因子与风格都按交易日做 3 倍 MAD 去极值 + z-score；风格缺失置 0（中性）",
        "sample": "池内且因子非空（不看标签），使暴露与标签周期解耦",
        "coefficient_note": "系数 = 风格增加 1 单位时标准化因子的变化；VALUE 为两项 z 的均值，"
                            "缺失填补后风格列不保证单位方差；t 值用当日残差标准误",
        "days": days,
        "skipped_days": result.skipped_days,
        "mean_rows": float(result.rows.mean()) if result.rows.size else None,
        "mean_r2": float(np.nanmean(result.r2)) if result.r2.size else None,
        "stats": stats,
        "series": {
            "dates": [day.strftime("%Y-%m-%d") for day in result.dates],
            "coefficient": {
                key: result.coefficients[:, index]
                for index, key in enumerate(STYLE_KEYS)
            },
            "r2": result.r2,
        },
        "significance_t": SIGNIFICANCE_T,
        "min_days": INSUFFICIENT_DAYS,
        "min_rows": MIN_REGRESSION_ROWS,
        "insufficient": False,
        "industry": result.industry,
        "note": (
            "暴露说的是「这个因子在哪个风格上有系统性下注」，"
            "回归是逐日截面做的，系数序列的均值与 IC 一样会受样本区间影响"
        ),
    }


def compute_incremental_ic(context, options) -> dict:
    """算增量 IC：剔除风格后残差的 RankIC，与原始 RankIC 对照。

    参数:
        context: 评估上下文。
        options: 指标参数（``industry``）。
    """
    industry_key = _industry_key(options)
    result = context.exposures(industry_key)
    series = residual_rank_ic(result)
    raw = daily_corr(result.residual, "factor", "label", rank=True)["corr"]
    # 常数残差日没有 RankIC，raw 同时排除这些日子，保证均值对照的日期相同。
    series = series.loc[np.isfinite(series["corr"])]
    residual_stats = series_stats(series["corr"].to_numpy(dtype="float64"))
    raw_stats = series_stats(
        raw.reindex(series.index).to_numpy(dtype="float64")
    )
    days = residual_stats["days"]
    insufficient = days < INSUFFICIENT_DAYS
    payload = {
        "horizon": context.params.horizon,
        "price": context.params.price,
        "label": context.params.label_column,
        "min_days": INSUFFICIENT_DAYS,
        "min_cross_section": MIN_CROSS_SECTION,
        "days": days,
        "insufficient": insufficient,
        "industry": result.industry,
        "basis": "残差 = 因子对 7 个风格（可含行业）逐日回归后的残差；两列 IC 的样本完全相同",
    }
    if insufficient:
        payload["hint"] = (f"有效交易日只有 {days} 天，少于 {INSUFFICIENT_DAYS} 天，"
                           "结论不可用；请拉长评估区间")
        payload["raw"] = {"mean": None, "t": None, "ir": None, "win_rate": None,
                          "days": days}
        payload["residual"] = dict(payload["raw"])
        return payload

    payload["raw"] = {key: raw_stats[key] for key in ("mean", "ir", "t", "win_rate", "days")}
    payload["residual"] = {
        key: residual_stats[key] for key in ("mean", "ir", "t", "win_rate", "days")
    }
    payload["delta"] = {
        "mean": (
            None if raw_stats["mean"] is None or residual_stats["mean"] is None
            else raw_stats["mean"] - residual_stats["mean"]
        ),
        "ir": (
            None if raw_stats["ir"] is None or residual_stats["ir"] is None
            else raw_stats["ir"] - residual_stats["ir"]
        ),
    }
    payload["survivor_share"] = (
        None if not raw_stats["mean"] else residual_stats["mean"] / raw_stats["mean"]
    )
    payload["series"] = {
        "dates": [day.strftime("%Y-%m-%d") for day in series.index],
        "raw": raw.reindex(series.index).to_numpy(dtype="float64"),
        "residual": series["corr"].to_numpy(dtype="float64"),
        "n": series["n"].to_numpy(),
    }
    payload["note"] = (
        "残差 IC 保留的是「风格解释不掉的那部分信号」；"
        "原始 IC 与残差 IC 使用同一批有效配对和日期；差值仅用于对照，"
        "不是可加的风格贡献，比例也不是风格解释率"
    )
    return payload


register_metric(
    Metric(key="exposure", label="风格暴露", stage="exposure",
           compute=compute_exposure, params=(INDUSTRY_PARAM,))
)
register_metric(
    Metric(key="incremental_ic", label="剔除风格后的增量 IC", stage="incremental_ic",
           compute=compute_incremental_ic, params=(INDUSTRY_PARAM,))
)


__all__ = ["compute_exposure", "compute_incremental_ic", "INDUSTRY_PARAM"]
