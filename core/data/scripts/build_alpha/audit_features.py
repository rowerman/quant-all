"""P0 特征审计：覆盖、缺失、极端值、方向、相关簇与基线 IC。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import factors
from alpha.calendar import load_calendar, weekly_signal_dates
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.errors import AlphaConfigError
from alpha.logging_conf import get_logger
from calculate.history import load_daily_ic
from calculate.daily import daily_ic
from calculate.params import EvalParams
from scripts.build_alpha.audit import (
    audit_year,
    correlation_matrix,
    correlation_pairs,
    feature_summary,
    summarize_ic,
    yearly_ic,
)
from scripts.build_alpha.baselines import BASELINE_IDS, scores_for_baseline, write_json
from scripts.build_alpha.panel import (
    data_versions,
    implemented_factor_names,
    load_year_panel,
    monthly_dates,
)

logger = get_logger("scripts.build_alpha.audit_features")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="审计 alpha 特征并计算基线 RankIC")
    parser.add_argument("--start", default=DEFAULT_CONTRACT.dev_start)
    parser.add_argument("--end", default=DEFAULT_CONTRACT.dev_end)
    return parser


def _guard(start: str, end: str) -> None:
    """拒绝最终测试集。"""

    if start < DEFAULT_CONTRACT.dev_start or end > DEFAULT_CONTRACT.dev_end:
        raise AlphaConfigError(
            f"审计区间必须在 {DEFAULT_CONTRACT.dev_start} ~ {DEFAULT_CONTRACT.dev_end}"
        )


def _ic_panel(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    score: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """把宽分数摊成长表并合上标签与股票池标记。"""

    score = score.reindex(index=dates)
    long = score.stack(future_stack=True).rename("factor").reset_index()
    long.columns = ["trade_date", "stock_code", "factor"]
    reference = frame[["stock_code", "trade_date", label, "eligible"]].copy()
    reference = reference.loc[reference["trade_date"].isin(dates)]
    long = long.merge(reference, on=["stock_code", "trade_date"], how="left")
    long["in_universe"] = long["eligible"].eq(True)
    return long.rename(columns={label: "label"})


def _factor_directions(
    start: str,
    end: str,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """逐因子计算全开发期周频 RankIC 并与注册方向对照。"""

    rows = []
    for name in factor_names:
        params = EvalParams(
            factor=name,
            start=start,
            end=end,
            universe=DEFAULT_CONTRACT.universe,
            horizon=DEFAULT_CONTRACT.horizon,
            price=DEFAULT_CONTRACT.price,
            winsorize=False,
        )
        series = load_daily_ic(params)["rank_ic"].dropna()
        stats = summarize_ic(series)
        years = yearly_ic(series)
        direction = int(factors.resolve(name).direction)
        rows.append(
            {
                "factor": name,
                "registry_direction": direction,
                "rank_ic_mean": stats["mean"],
                "rank_ic_ir": stats["ir"],
                "rank_ic_t": stats["t"],
                "rank_ic_win_rate": stats["win_rate"],
                "days": stats["days"],
                "positive_years": int((years["mean"] > 0).sum()) if len(years) else 0,
                "total_years": int(len(years)),
                "direction_match": bool(
                    stats["mean"] is not None
                    and np.sign(stats["mean"]) == np.sign(direction)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_audit(start: str, end: str) -> dict:
    """执行审计并写出全部产物。"""

    _guard(start, end)
    factor_names = implemented_factor_names()
    years = sorted(
        pd.DatetimeIndex(pd.date_range(start, end, freq="YS")).year.tolist()
    )
    coverage: list[pd.DataFrame] = []
    corr_total = pd.DataFrame(
        0.0, index=list(factor_names), columns=list(factor_names)
    )
    corr_count = pd.DataFrame(
        0.0, index=list(factor_names), columns=list(factor_names)
    )
    ic_series = {name: [] for name in BASELINE_IDS}
    full_weekly = weekly_signal_dates(load_calendar(start, end))
    for year in years:
        logger.info("读取并审计 %s 年", year)
        panel = load_year_panel(year, factor_names)
        coverage.append(audit_year(panel.frame, factor_names, year))
        dates = monthly_dates(panel.calendar)
        corr = correlation_matrix(panel.frame, factor_names, dates)
        corr_total = corr_total.add(corr.fillna(0.0), fill_value=0.0)
        corr_count = corr_count.add(corr.notna().astype("float64"), fill_value=0.0)
        signal_dates = full_weekly[
            (full_weekly >= pd.Timestamp(f"{year}-01-01"))
            & (full_weekly <= pd.Timestamp(f"{year}-12-31"))
        ]
        for baseline_id in BASELINE_IDS:
            score = scores_for_baseline(panel.frame, baseline_id, factor_names)
            panel_long = _ic_panel(
                panel.frame, signal_dates, score, DEFAULT_CONTRACT.label
            )
            series = daily_ic(panel_long, winsorize=False, pool_only=True)["rank_ic"]
            ic_series[baseline_id].append(series)
    coverage_frame = pd.concat(coverage, ignore_index=True)
    correlation = corr_total.div(corr_count.replace(0.0, np.nan))
    summary, selection = feature_summary(coverage_frame, correlation, factor_names)
    directions = _factor_directions(start, end, factor_names)
    summary = summary.merge(directions, on="factor", how="left")
    conflicts = summary.loc[~summary["direction_match"], "factor"].tolist()
    selection["direction_conflicts"] = conflicts
    root = MODEL_DATA_DIR / "audit"
    root.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(root / "feature_coverage.csv", index=False)
    summary.to_csv(root / "feature_summary.csv", index=False)
    directions.to_csv(root / "feature_direction.csv", index=False)
    correlation_pairs(correlation).to_csv(
        root / "feature_correlation.csv", index=False
    )
    write_json(root / "feature_selection.json", selection)
    ic_root = root / "baseline_ic"
    for baseline_id, chunks in ic_series.items():
        series = pd.concat(chunks).sort_index()
        series = series[~series.index.duplicated(keep="last")].dropna()
        folder = ic_root / baseline_id
        folder.mkdir(parents=True, exist_ok=True)
        series.to_frame("rank_ic").to_parquet(folder / "ic_daily.parquet")
        stats = summarize_ic(series)
        payload = {
            "baseline_id": baseline_id,
            "rank_ic_mean": stats["mean"],
            "rank_ic_std": stats["std"],
            "rank_ic_ir": stats["ir"],
            "rank_ic_t": stats["t"],
            "rank_ic_win_rate": stats["win_rate"],
            "days": stats["days"],
            "yearly": yearly_ic(series).to_dict(orient="records"),
        }
        write_json(folder / "summary.json", payload)
        logger.info(
            "%s RankIC 均值 %.4f，ICIR %s，有效日期 %d",
            baseline_id,
            stats["mean"] if stats["mean"] is not None else float("nan"),
            f"{stats['ir']:.3f}" if stats["ir"] is not None else "无定义",
            stats["days"],
        )
    code = Path(__file__).read_bytes()
    manifest = {
        "start": start,
        "end": end,
        "factors": list(factor_names),
        "factor_count": len(factor_names),
        "data_versions": data_versions(),
        "code_sha256": hashlib.sha256(code).hexdigest(),
        "selected_factors": selection["selected_factors"],
        "excluded_factors": selection["excluded_factors"],
        "direction_conflicts": selection["direction_conflicts"],
    }
    write_json(root / "audit_manifest.json", manifest)
    return {
        "coverage_rows": len(coverage_frame),
        "selected_factors": len(selection["selected_factors"]),
        "excluded_factors": selection["excluded_factors"],
        "ic": {
            name: summarize_ic(pd.concat(chunks).sort_index())["mean"]
            for name, chunks in ic_series.items()
        },
    }


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = run_audit(args.start, args.end)
    print(result)


if __name__ == "__main__":
    main()
