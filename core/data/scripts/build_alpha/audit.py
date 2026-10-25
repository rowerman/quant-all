"""特征覆盖审计与基线 IC 的纯函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate.stats import series_stats

MAD_SCALE = 1.4826


def _missing_concentration(values: pd.Series, by: pd.Series) -> tuple[float, float]:
    """返回缺失样本在单个分组和前十分组中的最大占比。"""

    counts = by.value_counts(sort=False)
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0.0
    return float(counts.max() / total), float(counts.nlargest(10).sum() / total)


def audit_year(
    frame: pd.DataFrame,
    factor_names: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    """统计一年内每个因子在可交易池中的覆盖、缺失和分布。"""

    eligible = frame["eligible"].astype(bool)
    total = int(eligible.sum())
    rows: list[dict] = []
    for name in factor_names:
        values = pd.to_numeric(frame[name], errors="coerce")
        valid = eligible & values.notna() & np.isfinite(values)
        missing = eligible & ~valid
        sample = values.loc[valid].astype("float64")
        if sample.empty:
            median = np.nan
            mad_share = np.nan
            quantiles = [np.nan] * 5
        else:
            median = float(sample.median())
            mad = float((sample - median).abs().median())
            threshold = 3.0 * MAD_SCALE * mad
            mad_share = (
                0.0
                if not np.isfinite(threshold) or threshold <= 0
                else float(((sample - median).abs() > threshold).mean())
            )
            quantiles = sample.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).tolist()
        stock_share, stock_top10 = _missing_concentration(
            values.loc[missing], frame.loc[missing, "stock_code"]
        )
        date_share, date_top10 = _missing_concentration(
            values.loc[missing], frame.loc[missing, "trade_date"]
        )
        rows.append(
            {
                "factor": name,
                "year": int(year),
                "eligible_rows": total,
                "non_null_rows": int(valid.sum()),
                "coverage": float(valid.sum() / total) if total else np.nan,
                "missing_rows": int(missing.sum()),
                "missing_max_stock_share": stock_share,
                "missing_top10_stocks_share": stock_top10,
                "missing_max_date_share": date_share,
                "missing_top10_dates_share": date_top10,
                "nonfinite_rows": int(
                    (eligible & values.notna() & ~np.isfinite(values)).sum()
                ),
                "median": median,
                "p001": quantiles[0],
                "p01": quantiles[1],
                "p50": quantiles[2],
                "p99": quantiles[3],
                "p999": quantiles[4],
                "outlier_3mad_share": mad_share,
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(
    frame: pd.DataFrame,
    factor_names: tuple[str, ...],
    dates: pd.DatetimeIndex,
    *,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """计算多个截面 Spearman 相关的日期平均矩阵。"""

    total = pd.DataFrame(0.0, index=list(factor_names), columns=list(factor_names))
    count = pd.DataFrame(0.0, index=list(factor_names), columns=list(factor_names))
    selected = frame.loc[
        frame["trade_date"].isin(dates) & frame["eligible"].astype(bool),
        ["trade_date", "stock_code", *factor_names],
    ]
    for _, group in selected.groupby("trade_date", sort=True):
        values = group.set_index("stock_code")[list(factor_names)].astype("float64")
        values = values.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if len(values) < min_stocks:
            continue
        corr = values.corr(method="spearman", min_periods=min_stocks)
        valid = corr.notna()
        total = total.add(corr.fillna(0.0), fill_value=0.0)
        count = count.add(valid.astype("float64"), fill_value=0.0)
    return total.div(count.replace(0.0, np.nan))


def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """把相关矩阵摊成超过阈值的有序因子对。"""

    rows: list[dict] = []
    names = list(corr.index)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append(
                    {
                        "factor_a": left,
                        "factor_b": right,
                        "mean_spearman": float(value),
                        "abs_spearman": abs(float(value)),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["factor_a", "factor_b", "mean_spearman", "abs_spearman"]
        )
    return pd.DataFrame(rows).sort_values(
        "abs_spearman", ascending=False
    ).reset_index(drop=True)


def correlation_clusters(corr: pd.DataFrame, threshold: float = 0.9) -> dict[str, str]:
    """用并查集把高相关因子合并成簇，返回因子到簇代表的映射。"""

    names = [str(name) for name in corr.index]
    parent = {name: name for name in names}

    def find(name: str) -> str:
        """查找并压缩并查集路径。"""

        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        """合并两个因子所在的集合。"""

        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            keep, drop = sorted((left_root, right_root))
            parent[drop] = keep

    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                union(left, right)
    return {name: find(name) for name in names}


def feature_summary(
    coverage: pd.DataFrame,
    corr: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """汇总覆盖、方向和相关簇，给出 v1 特征清单。"""

    stats = (
        coverage.groupby("factor", sort=False)["coverage"]
        .agg(["min", "mean", "count"])
        .rename(columns={"min": "min_coverage", "mean": "avg_coverage", "count": "years"})
    )
    recent = coverage.loc[coverage["year"] >= 2021].groupby("factor")["coverage"].mean()
    stats["recent_coverage"] = recent
    clusters = correlation_clusters(corr)
    stats["cluster"] = [clusters.get(name, name) for name in stats.index]
    stats["sparse"] = stats["min_coverage"] < 0.7
    stats["very_sparse"] = stats["recent_coverage"] < 0.5
    stats["selected"] = ~stats["very_sparse"]
    summary = stats.reset_index().rename(columns={"index": "factor"})
    summary = summary.set_index("factor").reindex(list(factor_names)).reset_index()
    selected = summary.loc[summary["selected"], "factor"].tolist()
    excluded = summary.loc[~summary["selected"], "factor"].tolist()
    payload = {
        "rule": {
            "sparse_coverage": 0.7,
            "very_sparse_recent_coverage": 0.5,
            "correlation_report_threshold": 0.8,
            "correlation_cluster_threshold": 0.9,
        },
        "selected_factors": selected,
        "excluded_factors": excluded,
        "context_features": ["log_mv", "turnover_rate", "listed_days"],
        "note": "高相关组只报告，不在 v1 自动删除；留给 P3 消融。",
    }
    return summary, payload


def summarize_ic(series: pd.Series) -> dict:
    """把逐日 RankIC 压成均值、ICIR、t 值和胜率。"""

    return series_stats(series.to_numpy(dtype="float64"))


def yearly_ic(series: pd.Series) -> pd.DataFrame:
    """按自然年汇总逐日 RankIC。"""

    frame = series.dropna().to_frame("rank_ic")
    frame["year"] = pd.DatetimeIndex(frame.index).year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        stats = summarize_ic(group["rank_ic"])
        rows.append({"year": int(year), **stats})
    return pd.DataFrame(rows)
