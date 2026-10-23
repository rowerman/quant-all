"""Top-N 研究组合的风格与行业暴露。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.artifacts import validate_dev_range
from alpha.errors import AlphaEvaluationError
from calculate.normalize import winsorize_by_day, zscore_by_day
from calculate.stats import series_stats
from calculate.styles import RAW_COLUMNS, STYLE_KEYS, STYLE_FACTORS


@dataclass(frozen=True)
class PortfolioExposureResult:
    """组合暴露的汇总、逐日序列和整体说明。"""

    style_summary: pd.DataFrame
    style_series: pd.DataFrame
    industry_summary: pd.DataFrame
    industry_series: pd.DataFrame
    summary: dict


def _select_holdings(
    score: pd.Series,
    eligible: pd.Series,
    label: pd.Series,
    top_n: int,
) -> pd.Series:
    """按周频模拟器口径选择 Top-N 等权目标持仓。"""

    chosen = (
        score.where(eligible)
        .dropna()
        .sort_values(ascending=False, kind="mergesort")
        .head(top_n)
        .index
    )
    chosen = label.reindex(chosen).dropna().index
    if not len(chosen):
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(chosen), index=chosen, dtype="float64")


def _standardized_styles(universe: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """把当日股票池的七类风格标准化。"""

    missing = sorted(set(RAW_COLUMNS) - set(universe.columns))
    if missing:
        raise AlphaEvaluationError(f"风格表缺少列：{missing}")
    dates = np.full(len(universe), day.to_datetime64(), dtype="datetime64[ns]")
    standardized = {
        column: zscore_by_day(
            dates,
            winsorize_by_day(
                dates,
                pd.to_numeric(universe[column], errors="coerce").to_numpy(
                    dtype="float64"
                ),
            ),
        )
        for column in RAW_COLUMNS
    }
    size = standardized["SIZE"]
    bp = standardized["BP"]
    ep = standardized["EP"]
    value = np.full(len(universe), np.nan, dtype="float64")
    has_bp = np.isfinite(bp)
    has_ep = np.isfinite(ep)
    both = has_bp & has_ep
    value[both] = (bp[both] + ep[both]) / 2.0
    value[has_bp & ~has_ep] = bp[has_bp & ~has_ep]
    value[~has_bp & has_ep] = ep[~has_bp & has_ep]
    return pd.DataFrame(
        {
            "SIZE": size,
            "VALUE": value,
            "MOMENTUM": standardized["MOMENTUM"],
            "SHORT_REV": standardized["SHORT_REV"],
            "VOL": standardized["VOL"],
            "LIQUIDITY": standardized["LIQUIDITY"],
            "BETA_60": standardized["BETA_60"],
        },
        index=universe.index,
    )


def compute_portfolio_exposure(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    *,
    top_n: int = 30,
    styles: pd.DataFrame | None = None,
    allow_holdout: bool = False,
    holdout_window: object | None = None,
) -> PortfolioExposureResult:
    """计算 Top-N 等权目标持仓的七类风格和行业暴露。"""

    required = {"trade_date", "stock_code", "score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise AlphaEvaluationError(f"预测表缺少列：{missing}")
    if type(top_n) is not int or top_n <= 0:
        raise AlphaEvaluationError("top_n 必须是正整数")
    work = predictions.loc[:, ["trade_date", "stock_code", "score"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    start = str(work["trade_date"].min().date())
    end = str(work["trade_date"].max().date())
    validate_dev_range(
        start,
        end,
        allow_holdout=allow_holdout,
        holdout_window=holdout_window,
    )
    scores = work.pivot(
        index="trade_date", columns="stock_code", values="score"
    )
    labels = data.pivot(
        index="trade_date", columns="stock_code", values="label_raw"
    )
    eligible = data.pivot(
        index="trade_date", columns="stock_code", values="eligible"
    ).astype("boolean").fillna(False).astype(bool)
    scores = scores.reindex(index=sorted(scores.index))
    labels = labels.reindex(index=scores.index, columns=scores.columns)
    eligible = eligible.reindex(index=scores.index, columns=scores.columns)
    industry = data.pivot(
        index="trade_date", columns="stock_code", values="industry_l1"
    ).reindex(index=scores.index, columns=scores.columns)

    if styles is None:
        from calculate.params import EvalParams
        from calculate.styles import build_styles

        styles = build_styles(
            EvalParams(
                factor="MODEL_SCORE",
                start=start,
                end=end,
                horizon=5,
                price="open",
            )
        )
    style_required = {"trade_date", "stock_code", *RAW_COLUMNS}
    style_missing = sorted(style_required - set(styles.columns))
    if style_missing:
        raise AlphaEvaluationError(f"风格表缺少列：{style_missing}")
    style_frame = styles.copy()
    style_frame["trade_date"] = pd.to_datetime(style_frame["trade_date"])
    style_frame["stock_code"] = style_frame["stock_code"].astype(str)
    if style_frame.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaEvaluationError("风格表存在重复日期和股票")

    style_rows: list[dict] = []
    industry_rows: list[dict] = []
    holding_counts: list[int] = []
    hhi_values: list[float] = []
    unknown_weights: list[float] = []
    for day in scores.index:
        score = scores.loc[day]
        holdings = _select_holdings(
            score,
            eligible.loc[day],
            labels.loc[day],
            top_n,
        )
        if holdings.empty:
            continue
        holding_counts.append(int(len(holdings)))
        hhi_values.append(float((holdings**2).sum()))
        universe_codes = eligible.columns[eligible.loc[day].fillna(False)]
        universe_styles = (
            style_frame.loc[style_frame["trade_date"].eq(day)]
            .set_index("stock_code")
            .reindex(universe_codes)
        )
        standardized = _standardized_styles(universe_styles, day)
        selected = standardized.reindex(holdings.index)
        coverage = selected.notna().mul(holdings, axis=0).sum() / holdings.sum()
        neutral = selected.fillna(0.0)
        for key in STYLE_KEYS:
            values = neutral[key].to_numpy(dtype="float64")
            exposure = float(holdings.to_numpy(dtype="float64") @ values)
            style_rows.append(
                {
                    "trade_date": day,
                    "style": key,
                    "exposure": exposure,
                    "coverage": float(coverage[key]),
                    "holdings": int(len(holdings)),
                }
            )

        industry_day = industry.loc[day].reindex(holdings.index).fillna("未知")
        weights = holdings.groupby(industry_day).sum().sort_values(ascending=False)
        universe_industry = (
            pd.Series("未知", index=universe_codes)
            if not len(universe_codes)
            else industry.loc[day].reindex(universe_codes).fillna("未知")
        )
        universe_weights = universe_industry.value_counts(normalize=True)
        unknown_weights.append(float(weights.get("未知", 0.0)))
        for name, weight in weights.items():
            industry_rows.append(
                {
                    "trade_date": day,
                    "industry": str(name),
                    "weight": float(weight),
                    "universe_weight": float(universe_weights.get(name, 0.0)),
                    "active_weight": float(weight - universe_weights.get(name, 0.0)),
                }
            )
    if not style_rows:
        raise AlphaEvaluationError("组合暴露没有可用持仓")

    style_series = pd.DataFrame(style_rows).sort_values(
        ["trade_date", "style"], kind="mergesort"
    )
    industry_series = pd.DataFrame(industry_rows).sort_values(
        ["trade_date", "weight", "industry"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    labels = {spec.key: spec.label for spec in STYLE_FACTORS}
    summary_rows = []
    for key, group in style_series.groupby("style", sort=False):
        values = group["exposure"].to_numpy(dtype="float64")
        stats = series_stats(values)
        summary_rows.append(
            {
                "style": key,
                "label": labels[key],
                "days": stats["days"],
                "mean": stats["mean"],
                "std": stats["std"],
                "t": stats["t"],
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "positive_share": float((values > 0).mean()),
                "mean_coverage": float(group["coverage"].mean()),
            }
        )
    style_summary = pd.DataFrame(summary_rows)

    industry_summary_rows = []
    for name, group in industry_series.groupby("industry", sort=True):
        industry_summary_rows.append(
            {
                "industry": name,
                "days_present": int(len(group)),
                "mean_weight": float(group["weight"].mean()),
                "max_weight": float(group["weight"].max()),
                "end_weight": float(group.sort_values("trade_date").iloc[-1]["weight"]),
                "mean_universe_weight": float(group["universe_weight"].mean()),
                "mean_active_weight": float(group["active_weight"].mean()),
            }
        )
    industry_summary = pd.DataFrame(industry_summary_rows).sort_values(
        ["mean_weight", "industry"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = {
        "start": start,
        "end": end,
        "days": int(len(holding_counts)),
        "top_n": top_n,
        "holdings_min": int(min(holding_counts)),
        "holdings_max": int(max(holding_counts)),
        "holdings_mean": float(np.mean(holding_counts)),
        "hhi_mean": float(np.mean(hhi_values)),
        "hhi_max": float(np.max(hhi_values)),
        "unknown_industry_mean_weight": float(np.mean(unknown_weights)),
        "style_min_coverage": float(style_summary["mean_coverage"].min()),
    }
    return PortfolioExposureResult(
        style_summary=style_summary,
        style_series=style_series,
        industry_summary=industry_summary,
        industry_series=industry_series,
        summary=summary,
    )


__all__ = ["PortfolioExposureResult", "compute_portfolio_exposure"]
