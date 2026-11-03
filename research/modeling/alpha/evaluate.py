"""模型分数评估：RankIC、分箱、自相关与 P2/P3 决策门。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from calculate.binning import assign_bins, bin_daily_returns
from calculate.daily import daily_ic
from calculate.stats import series_stats
from calculate.binning import row_correlation
from alpha.errors import AlphaEvaluationError


def rank_ic_series(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    label_column: str = "label",
) -> pd.Series:
    """计算逐日截面 RankIC。"""

    required = {"trade_date", score_column, label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlphaEvaluationError(f"RankIC 输入缺少列：{missing}")
    panel = frame.loc[:, ["trade_date", score_column, label_column]].copy()
    panel = panel.rename(columns={score_column: "factor", label_column: "label"})
    panel["in_universe"] = True
    return daily_ic(panel, winsorize=False, pool_only=True)["rank_ic"].dropna()


def rank_ic_summary(series: pd.Series) -> dict:
    """把 RankIC 序列汇总为均值和 ICIR。"""

    stats = series_stats(series.to_numpy(dtype="float64"))
    return {
        "rank_ic_mean": stats["mean"],
        "rank_ic_std": stats["std"],
        "icir": stats["ir"],
        "rank_ic_t": stats["t"],
        "rank_ic_win_rate": stats["win_rate"],
        "days": stats["days"],
    }


def yearly_rank_ic(series: pd.Series) -> list[dict]:
    """按自然年汇总 RankIC。"""

    frame = series.dropna().to_frame("rank_ic")
    if frame.empty:
        return []
    frame["year"] = pd.DatetimeIndex(frame.index).year
    rows: list[dict] = []
    for year, group in frame.groupby("year", sort=True):
        rows.append({"year": int(year), **rank_ic_summary(group["rank_ic"])})
    return rows


def score_panel(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """把宽分数、标签和股票池对齐成长表。"""

    index = scores.index.intersection(labels.index).intersection(eligible.index)
    columns = scores.columns.intersection(labels.columns).intersection(eligible.columns)
    if len(index) == 0 or len(columns) == 0:
        raise AlphaEvaluationError("分数、标签和股票池没有共同日期或股票")
    scores = scores.reindex(index=index, columns=columns)
    labels = labels.reindex(index=index, columns=columns)
    eligible = eligible.reindex(index=index, columns=columns).fillna(False).astype(bool)
    valid = eligible & scores.notna() & labels.notna()
    if not valid.any().any():
        raise AlphaEvaluationError("股票池内没有可评估的分数和标签")
    score_long = scores.where(valid).stack(future_stack=True).rename("factor")
    label_long = labels.where(valid).stack(future_stack=True).rename("label")
    panel = pd.concat([score_long, label_long], axis=1).reset_index()
    panel.columns = ["trade_date", "stock_code", "factor", "label"]
    panel["in_universe"] = True
    panel["total_mv"] = 1.0
    panel = panel.loc[
        np.isfinite(panel["factor"]) & np.isfinite(panel["label"])
    ].reset_index(drop=True)
    return panel


def _monotonicity(means: np.ndarray) -> float | None:
    """计算分位序号与平均收益的 Spearman 相关。"""

    valid = np.isfinite(means)
    if int(valid.sum()) < 2:
        return None
    left = pd.Series(np.arange(1, means.size + 1)[valid]).rank()
    right = pd.Series(means[valid]).rank()
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_score_frame(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    quantiles: int = 10,
) -> dict:
    """评估一组宽分数，返回模型层指标。"""

    panel = score_panel(scores, labels, eligible)
    series = daily_ic(panel, winsorize=False, pool_only=True)["rank_ic"].dropna()
    summary = rank_ic_summary(series)
    binned = assign_bins(panel, quantiles, label_columns=("label",))
    daily_returns = bin_daily_returns(
        binned, quantiles, label_column="label"
    ).reindex(columns=range(1, quantiles + 1))
    bins: list[dict] = []
    means = []
    for bin_id in range(1, quantiles + 1):
        values = (
            daily_returns[bin_id].to_numpy(dtype="float64")
            if bin_id in daily_returns.columns
            else np.asarray([], dtype="float64")
        )
        stats = series_stats(values)
        means.append(stats["mean"] if stats["mean"] is not None else np.nan)
        bins.append(
            {
                "bin": bin_id,
                "mean": stats["mean"],
                "t": stats["t"],
                "win_rate": stats["win_rate"],
                "days": stats["days"],
            }
        )
    ranked = scores.where(eligible.reindex_like(scores).fillna(False)).rank(
        axis=1, pct=True
    )
    autocorrelation = (
        float(
            row_correlation(
                ranked.iloc[1:],
                ranked.iloc[:-1],
                rank=True,
            ).mean()
        )
        if len(ranked) > 1
        else None
    )
    return {
        **summary,
        "yearly": yearly_rank_ic(series),
        "bins": bins,
        "monotonicity": _monotonicity(np.asarray(means, dtype="float64")),
        "top_minus_bottom_mean": (
            float(means[-1] - means[0])
            if np.isfinite(means[-1]) and np.isfinite(means[0])
            else None
        ),
        "score_autocorrelation_1w": autocorrelation,
    }


def p2_decision(
    model_metrics: dict,
    baselines: dict[str, dict],
) -> dict:
    """按 P2-4 规则给出通过、有条件完成或明显失败。"""

    candidates = {
        name: metrics
        for name, metrics in baselines.items()
        if name in ("B3_ALL_ZSCORE", "B4_ALL_RANK")
    }
    if not candidates:
        raise AlphaEvaluationError("P2 决策缺少 B3/B4 基线")
    reference_name, reference = max(
        candidates.items(),
        key=lambda item: (
            item[1].get("rank_ic_mean")
            if item[1].get("rank_ic_mean") is not None
            else -np.inf
        ),
    )
    model_rank = model_metrics.get("rank_ic_mean")
    base_rank = reference.get("rank_ic_mean")
    model_annual = model_metrics.get("annual_return")
    base_annual = reference.get("annual_return")
    model_icir = model_metrics.get("icir")
    base_icir = reference.get("icir")
    if None in (model_rank, base_rank, model_annual, base_annual):
        raise AlphaEvaluationError("P2 决策所需核心指标缺失")
    base_drawdown = abs(float(reference.get("max_drawdown") or 0.0))
    model_drawdown = abs(float(model_metrics.get("max_drawdown") or 0.0))
    base_turnover = float(reference.get("turnover_two_way_annual") or 0.0)
    model_turnover = float(model_metrics.get("turnover_two_way_annual") or 0.0)
    drawdown_improvement = (
        (base_drawdown - model_drawdown) / base_drawdown
        if base_drawdown > 0
        else 0.0
    )
    turnover_improvement = (
        (base_turnover - model_turnover) / base_turnover
        if base_turnover > 0
        else 0.0
    )
    reasons: list[str] = []
    if float(model_rank) >= float(base_rank):
        reasons.append("RankIC 不低于等权基线")
    if model_icir is not None and base_icir is not None and float(model_icir) >= float(base_icir):
        reasons.append("ICIR 不低于等权基线")
    if float(model_annual) >= float(base_annual):
        reasons.append("Top-N 年化净收益不低于等权基线")
    if drawdown_improvement >= 0.10 and turnover_improvement >= 0.10:
        reasons.append("最大回撤和换手均改善至少 10%")
    clear_failure = (
        float(model_rank) < 0.80 * float(base_rank)
        and float(model_annual) < float(base_annual) - 0.03
        and drawdown_improvement < 0.10
        and turnover_improvement < 0.10
    )
    status = "pass" if reasons else ("fail" if clear_failure else "conditional")
    return {
        "status": status,
        "reference": reference_name,
        "reasons": reasons,
        "rank_ic_ratio": float(model_rank) / float(base_rank) if float(base_rank) != 0 else None,
        "net_return_gap": float(model_annual) - float(base_annual),
        "drawdown_improvement": drawdown_improvement,
        "turnover_improvement": turnover_improvement,
        "enter_p3": status == "pass",
    }


def p3_decision_gate(
    candidate: dict,
    ridge: dict,
    yearly: Sequence[dict],
    *,
    expected_years: Sequence[int] | None = None,
    required_improvements: int = 2,
    turnover_multiple_limit: float = 2.0,
) -> dict:
    """按固定规则判断候选是否通过 P3，并给出逐项原因。"""

    metric_names = (
        "rank_ic_mean",
        "icir",
        "annual_return",
        "max_drawdown",
    )
    try:
        candidate_values = {
            name: float(candidate[name]) for name in metric_names
        }
        ridge_values = {name: float(ridge[name]) for name in metric_names}
        candidate_turnover = float(candidate["turnover_two_way_annual"])
        ridge_turnover = float(ridge["turnover_two_way_annual"])
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaEvaluationError("P3 决策所需核心指标缺失或非数值") from error
    if ridge_turnover <= 0:
        raise AlphaEvaluationError("Ridge 换手必须为正")

    improvements = [
        name
        for name in metric_names
        if candidate_values[name] > ridge_values[name]
    ]
    turnover_multiple = candidate_turnover / ridge_turnover
    turnover_ok = turnover_multiple <= float(turnover_multiple_limit)

    expected = (
        {int(year) for year in expected_years}
        if expected_years is not None
        else {int(row["year"]) for row in yearly}
    )
    by_year = {int(row["year"]): row for row in yearly}
    missing_years = sorted(expected - set(by_year))
    negative_rank_ic: list[int] = []
    negative_excess: list[int] = []
    for year in sorted(expected & set(by_year)):
        row = by_year[year]
        try:
            rank_ic = float(row["rank_ic_mean"])
            excess = float(row["excess_total_return"])
        except (KeyError, TypeError, ValueError):
            negative_rank_ic.append(year)
            negative_excess.append(year)
            continue
        if not np.isfinite(rank_ic) or rank_ic <= 0:
            negative_rank_ic.append(year)
        if not np.isfinite(excess) or excess <= 0:
            negative_excess.append(year)
    yearly_stable = (
        not missing_years and not negative_rank_ic and not negative_excess
    )
    passed = (
        len(improvements) >= int(required_improvements)
        and turnover_ok
        and yearly_stable
    )
    reasons: list[str] = []
    reasons.append(
        f"优于 Ridge 的核心指标 {len(improvements)} 项：{improvements}"
    )
    reasons.append(
        f"换手为 Ridge 的 {turnover_multiple:.3f} 倍，"
        f"上限 {float(turnover_multiple_limit):.3f}"
    )
    if missing_years:
        reasons.append(f"缺少年度：{missing_years}")
    if negative_rank_ic:
        reasons.append(f"RankIC 非正的年度：{negative_rank_ic}")
    if negative_excess:
        reasons.append(f"净超额非正的年度：{negative_excess}")
    if yearly_stable:
        reasons.append("分年 RankIC 和净超额均为正")
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "improvements": improvements,
        "improvement_count": len(improvements),
        "required_improvements": int(required_improvements),
        "turnover_multiple": turnover_multiple,
        "turnover_multiple_limit": float(turnover_multiple_limit),
        "turnover_ok": turnover_ok,
        "yearly_stable": yearly_stable,
        "missing_years": missing_years,
        "negative_rank_ic_years": negative_rank_ic,
        "negative_excess_years": negative_excess,
        "reasons": reasons,
    }


def select_p3_candidate(candidates: dict[str, dict]) -> dict:
    """从通过决策门的候选中按固定优先级选出一个。"""

    passed = {
        name: item
        for name, item in candidates.items()
        if bool((item.get("gate") or {}).get("passed"))
    }
    if not passed:
        return {
            "selected": None,
            "ranked": [],
            "reason": "没有候选通过 P3 决策门，保留 Ridge",
        }

    def _sort_key(item: tuple[str, dict]) -> tuple:
        name, payload = item
        metrics = payload["metrics"]
        annual = float(metrics["annual_return"])
        sharpe = float(metrics.get("sharpe") or -np.inf)
        drawdown = float(metrics.get("max_drawdown") or -np.inf)
        turnover = float(metrics["turnover_two_way_annual"])
        priority = int(payload.get("priority", 99))
        return (-annual, -sharpe, -drawdown, turnover, priority, name)

    ranked = sorted(passed.items(), key=_sort_key)
    selected = ranked[0][0]
    return {
        "selected": selected,
        "ranked": [name for name, _ in ranked],
        "reason": f"按固定优先级选择 {selected}",
    }
