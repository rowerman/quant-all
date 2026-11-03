"""Alpha 模型评估与同口径基线对比。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.ensemble import combine_rank_predictions
from alpha.errors import AlphaArtifactError
from alpha.evaluate import evaluate_score_frame, p2_decision
from alpha.logging_conf import get_logger
from alpha.predict import load_prediction_parts, load_predictions
from scripts.build_alpha.baselines import BASELINE_IDS, scores_for_baseline
from scripts.build_alpha.weekly_sim import simulate_score_frame, yearly_summary

logger = get_logger("scripts.build_alpha.evaluate")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="评估 Alpha 模型并对比 P0 基线")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--ridge-model-id",
        help="可选 Ridge 模型 ID；提供时输出固定 50/50 rank 融合对照",
    )
    parser.add_argument("--force", action="store_true", help="重新计算已评估模型")
    return parser


def _wide_frames(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把数据集标签与股票池转成宽表。"""

    labels = data.pivot(index="trade_date", columns="stock_code", values="label_raw")
    eligible = data.pivot(
        index="trade_date", columns="stock_code", values="eligible"
    ).astype(bool)
    return labels, eligible


def _wide_scores(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """把标准预测表转换为模型层评估使用的宽表。"""

    return predictions.pivot(
        index="trade_date", columns="stock_code", values="score"
    ).reindex(index=dates)


def _member_scores(
    folder: Path,
    manifest: dict,
    dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """读取集成模型的逐种子预测。"""

    result: dict[str, pd.DataFrame] = {}
    members = manifest.get("member_predictions") or {}
    for seed in manifest.get("seeds", ()):
        key = str(seed)
        metadata = members.get(key)
        if not metadata:
            raise AlphaArtifactError(f"集成模型缺少 seed={seed} 预测元数据")
        predictions = load_prediction_parts(
            folder / "member_predictions" / f"seed={seed}",
            metadata["files"],
            str(metadata["sha256"]),
            label=f"seed={seed} 预测",
        )
        result[f"SEED_{seed}"] = _wide_scores(predictions, dates)
    return result


def _component_predictions(
    model_id: str,
    expected_manifest: dict,
) -> tuple[Path, dict, pd.DataFrame]:
    """读取参与融合的另一个模型并校验研究口径。"""

    folder, manifest, predictions = load_predictions(model_id)
    if manifest.get("dataset_id") != expected_manifest.get("dataset_id"):
        raise AlphaArtifactError("融合模型的数据集 ID 不一致")
    if manifest.get("target") != expected_manifest.get("target"):
        raise AlphaArtifactError("融合模型的目标不一致")
    left_folds = [
        (str(fold["fold_id"]), str(fold["test_start"]), str(fold["test_end"]))
        for fold in expected_manifest["folds"]
    ]
    right_folds = [
        (str(fold["fold_id"]), str(fold["test_start"]), str(fold["test_end"]))
        for fold in manifest["folds"]
    ]
    if left_folds != right_folds:
        raise AlphaArtifactError("融合模型的折或测试区间不一致")
    return folder, manifest, predictions


def _single_factor_scores(
    data: pd.DataFrame,
    manifest: dict,
    dates: pd.DatetimeIndex,
    stocks: pd.Index,
) -> pd.DataFrame:
    """按每折验证集选出的单因子构造测试分数。"""

    scores = pd.DataFrame(
        np.nan,
        index=dates,
        columns=stocks,
        dtype="float64",
    )
    for item in manifest["best_single_factors"]:
        selected_dates = dates[
            (dates >= pd.Timestamp(item["test_start"]))
            & (dates <= pd.Timestamp(item["test_end"]))
        ]
        if selected_dates.empty:
            continue
        subset = data.loc[
            pd.to_datetime(data["trade_date"]).isin(selected_dates),
            ["trade_date", "stock_code", item["factor"]],
        ]
        wide = subset.pivot(
            index="trade_date", columns="stock_code", values=item["factor"]
        ).astype("float64")
        scores.loc[selected_dates, wide.columns] = wide * float(item["sign"])
    return scores


def _portfolio_with_yearly(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    start: str,
    end: str,
    top_n: int,
) -> tuple[dict, list[dict]]:
    """运行 Top-N 模拟并补充逐年基准和超额。"""

    metrics, daily, _ = simulate_score_frame(
        scores,
        labels,
        eligible,
        start=start,
        end=end,
        top_n=top_n,
    )
    rows = yearly_summary(daily)
    for row in rows:
        group = daily.loc[pd.DatetimeIndex(daily.index).year == row["year"]]
        benchmark = float(
            (1.0 + group["benchmark_return"].astype(float)).prod() - 1.0
        )
        row["benchmark_total_return"] = benchmark
        row["excess_total_return"] = row["total_return"] - benchmark
    return metrics, rows


def _combined_row(model_level: dict, portfolio: dict) -> dict:
    """合并模型层与组合层指标。"""

    return {
        "rank_ic_mean": model_level.get("rank_ic_mean"),
        "icir": model_level.get("icir"),
        "rank_ic_t": model_level.get("rank_ic_t"),
        "rank_ic_win_rate": model_level.get("rank_ic_win_rate"),
        "monotonicity": model_level.get("monotonicity"),
        "score_autocorrelation_1w": model_level.get("score_autocorrelation_1w"),
        "annual_return": portfolio.get("annual_return"),
        "excess_annual_return": portfolio.get("excess_annual_return"),
        "turnover_two_way_annual": portfolio.get("turnover_two_way_annual"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "total_cost": portfolio.get("total_cost"),
    }


def _seed_stability(manifest: dict, comparison: dict[str, dict]) -> dict | None:
    """汇总固定种子成员的样本外表现，不挑选种子。"""

    if manifest.get("model_type") != "lightgbm_ensemble":
        return None
    metric_names = (
        "rank_ic_mean",
        "icir",
        "annual_return",
        "turnover_two_way_annual",
        "max_drawdown",
    )
    rows: list[dict] = []
    for seed in manifest["seeds"]:
        name = f"SEED_{seed}"
        if name not in comparison:
            raise AlphaArtifactError(f"稳定性报告缺少 {name}")
        rows.append(
            {
                "seed": int(seed),
                **{metric: comparison[name].get(metric) for metric in metric_names},
            }
        )
    summary: dict[str, dict | None] = {}
    for metric in metric_names:
        values = [
            float(row[metric])
            for row in rows
            if row[metric] is not None and np.isfinite(float(row[metric]))
        ]
        if not values:
            summary[metric] = None
            continue
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "seeds": [int(seed) for seed in manifest["seeds"]],
        "policy": "all fixed seeds; equal rank weights; no seed selection",
        "members": rows,
        "summary": summary,
        "ensemble": {
            metric: comparison["MODEL"].get(metric) for metric in metric_names
        },
    }


def _upsert_experiment(manifest: dict, metrics: dict) -> None:
    """把评估结果写回实验索引。"""

    path = MODEL_DATA_DIR / "experiments.csv"
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    model_id = manifest["model_id"]
    if not existing.empty:
        existing = existing.loc[existing["experiment_id"].astype(str) != model_id]
    comparison = metrics["comparison"]["MODEL"]
    is_gbdt = manifest.get("model_type") in (
        "lightgbm",
        "lightgbm_ensemble",
    )
    is_ensemble = manifest.get("model_type") == "lightgbm_ensemble"
    row = {
        "experiment_id": model_id,
        "created_at": manifest["created_at"],
        "batch": "P3" if is_gbdt else "P2",
        "name": model_id,
        "kind": "model",
        "stage": "development",
        "start": manifest["prediction_start"],
        "end": manifest["prediction_end"],
        "config_hash": manifest["model_id"],
        "feature_set": f"all_implemented_{manifest['feature_count']}",
        "data_version": manifest["data_versions"].get("data_version"),
        "factor_version": manifest["data_versions"].get("factor_version"),
        "snapshot_id": None,
        "metrics_json": json.dumps(
            {
                "rank_ic_mean": comparison.get("rank_ic_mean"),
                "icir": comparison.get("icir"),
                "annual_return": comparison.get("annual_return"),
                "decision": metrics["decision"]["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "notes": (
            "P3 LightGBM five-seed rank ensemble; development years only"
            if is_ensemble
            else (
                "P3 LightGBM; development years only"
                if is_gbdt
                else "P2 linear model; development years only"
            )
        ),
    }
    frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, index=False)


def _decision(manifest: dict, comparison: dict[str, dict]) -> dict:
    """按模型阶段选择决策口径；P3 正式决策留给 P3-5。"""

    if manifest.get("model_type") in ("lightgbm", "lightgbm_ensemble"):
        stored = manifest.get("p3_decision")
        if stored:
            status = str(stored.get("status", "pending_p3"))
            return {
                "status": status,
                "reference": stored.get("reference_model_id"),
                "reasons": list(stored.get("reasons", [])),
                "enter_p3": status == "pass_p3",
                "analysis_id": stored.get("analysis_id"),
                "selected_candidate": stored.get("selected_candidate"),
            }
        return {
            "status": "pending_p3",
            "reference": None,
            "reasons": ["P3-5 前只记录与基线的同口径指标"],
            "enter_p3": None,
        }
    return p2_decision(comparison["MODEL"], comparison)


def evaluate_model(
    model_id: str,
    *,
    ridge_model_id: str | None = None,
    force: bool = False,
) -> dict:
    """评估模型并与基线、种子成员和可选 Ridge 融合做同区间比较。"""

    folder, manifest, predictions = load_predictions(model_id)
    metrics_path = folder / "metrics.json"
    if metrics_path.is_file() and not force:
        current = json.loads(metrics_path.read_text(encoding="utf-8"))
        if current.get("evaluated") and (
            not ridge_model_id
            or current.get("ridge_model_id") == ridge_model_id
        ):
            return current
    data, dataset_manifest = load_dataset(manifest["dataset_id"])
    labels, eligible = _wide_frames(data)
    test_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(predictions["trade_date"]).unique())
    )
    start = str(test_dates.min().date())
    end = str(test_dates.max().date())
    factor_names = tuple(str(name) for name in dataset_manifest["features"])
    scores: dict[str, pd.DataFrame] = {
        "MODEL": _wide_scores(predictions, test_dates)
    }
    if manifest.get("model_type") == "lightgbm_ensemble":
        scores.update(_member_scores(folder, manifest, test_dates))
    if ridge_model_id:
        _, _, ridge_predictions = _component_predictions(
            ridge_model_id,
            manifest,
        )
        scores["RIDGE"] = _wide_scores(ridge_predictions, test_dates)
        blended = combine_rank_predictions(
            [predictions, ridge_predictions],
            weights=(0.5, 0.5),
            names=("MODEL", "RIDGE"),
        )
        scores["LIGHTGBM_RIDGE_RANK_BLEND"] = _wide_scores(
            blended, test_dates
        )
    for baseline_id in BASELINE_IDS:
        scores[baseline_id] = scores_for_baseline(
            data, baseline_id, factor_names
        ).reindex(index=test_dates)
    scores["S1_FOLD_BEST_SINGLE"] = _single_factor_scores(
        data, manifest, test_dates, labels.columns
    )
    comparison: dict[str, dict] = {}
    model_level: dict[str, dict] = {}
    portfolios: dict[str, dict] = {}
    yearly_portfolios: dict[str, list[dict]] = {}
    for name, score in scores.items():
        logger.info("评估 %s", name)
        model_metrics = evaluate_score_frame(
            score,
            labels,
            eligible,
            quantiles=int(manifest["quantiles"]),
        )
        portfolio, yearly = _portfolio_with_yearly(
            score,
            labels,
            eligible,
            start=start,
            end=end,
            top_n=int(manifest["top_n"]),
        )
        model_level[name] = model_metrics
        portfolios[name] = portfolio
        yearly_portfolios[name] = yearly
        comparison[name] = _combined_row(model_metrics, portfolio)
    decision = _decision(manifest, comparison)
    comparison_frame = pd.DataFrame(
        [{"name": name, **row} for name, row in comparison.items()]
    )
    comparison_frame.to_csv(folder / "comparison.csv", index=False)
    yearly_rows = []
    for name, rows in yearly_portfolios.items():
        for row in rows:
            yearly_rows.append({"name": name, **row})
    pd.DataFrame(yearly_rows).to_csv(folder / "yearly_comparison.csv", index=False)
    current = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    seed_stability = _seed_stability(manifest, comparison)
    if seed_stability is not None:
        pd.DataFrame(seed_stability["members"]).to_csv(
            folder / "seed_stability.csv",
            index=False,
        )
        write_json(folder / "seed_stability.json", seed_stability)
    metrics = {
        "model_id": model_id,
        "dataset_id": manifest["dataset_id"],
        "evaluated": True,
        "start": start,
        "end": end,
        "validation": current.get("validation", {}),
        "test_model": model_level["MODEL"],
        "portfolio": portfolios["MODEL"],
        "yearly_portfolio": yearly_portfolios["MODEL"],
        "baselines": {
            name: {
                "model_level": model_level[name],
                "portfolio": portfolios[name],
                "yearly": yearly_portfolios[name],
            }
            for name in scores
            if name != "MODEL"
        },
        "comparison": comparison,
        "ridge_model_id": ridge_model_id,
        "seed_stability": seed_stability,
        "decision": decision,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metrics_path, metrics)
    manifest.update(
        {
            "status": "evaluated",
            "decision": decision["status"],
            "evaluated_at": metrics["evaluated_at"],
            "evaluation_sha256": file_sha256(metrics_path),
        }
    )
    write_json(folder / "manifest.json", manifest)
    _upsert_experiment(manifest, metrics)
    logger.info(
        "评估完成：%s 决策=%s RankIC=%.4f",
        model_id,
        decision["status"],
        comparison["MODEL"]["rank_ic_mean"],
    )
    return metrics


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    metrics = evaluate_model(
        args.model_id,
        ridge_model_id=args.ridge_model_id,
        force=args.force,
    )
    summary = {
        "model_id": args.model_id,
        "start": metrics["start"],
        "end": metrics["end"],
        "rank_ic_mean": metrics["comparison"]["MODEL"]["rank_ic_mean"],
        "icir": metrics["comparison"]["MODEL"]["icir"],
        "annual_return": metrics["comparison"]["MODEL"]["annual_return"],
        "max_drawdown": metrics["comparison"]["MODEL"]["max_drawdown"],
        "turnover_two_way_annual": metrics["comparison"]["MODEL"][
            "turnover_two_way_annual"
        ],
        "decision": metrics["decision"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
