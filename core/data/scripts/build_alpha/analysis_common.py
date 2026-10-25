"""P3-4 分析的数据读取、指标与抽样公共逻辑。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256, model_dir
from alpha.errors import AlphaArtifactError, AlphaEvaluationError
from alpha.evaluate import (
    evaluate_score_frame,
    rank_ic_series,
    rank_ic_summary,
    yearly_rank_ic,
)
from alpha.explain import ModelRef, tree_shap_feature_summary
from alpha.gbdt import load_gbdt_model
from alpha.predict import (
    load_model_manifest as load_artifact_manifest,
    load_prediction_parts,
)
from scripts.build_alpha.weekly_sim import simulate_score_frame, yearly_summary
from common.paths import PROJECT_ROOT

ANALYSIS_SCHEMA_VERSION = "alpha_analysis_v1"


def load_model_manifest(model_id: str) -> tuple[Path, dict]:
    """读取并校验 LightGBM 集成模型 manifest。"""

    manifest = load_artifact_manifest(model_id)
    folder = model_dir(model_id)
    if manifest.get("model_type") != "lightgbm_ensemble":
        raise AlphaArtifactError("P3-4 解释入口只接受 LightGBM 多种子集成模型")
    return folder, manifest


def load_models(
    folder: Path,
    manifest: dict,
) -> tuple[tuple[ModelRef, ...], dict[tuple[str, int], object], dict[str, object]]:
    """读取全部成员模型并校验文件哈希。"""

    refs: list[ModelRef] = []
    by_key: dict[tuple[str, int], object] = {}
    seed_zero: dict[str, object] = {}
    for fold in manifest["folds"]:
        for member in fold["members"]:
            seed = int(member["seed"])
            relative = str(member["model_file"])
            path = folder / relative
            if not path.is_file():
                raise AlphaArtifactError(f"成员模型不存在：{path}")
            if file_sha256(path) != str(member["model_sha256"]):
                raise AlphaArtifactError(f"成员模型哈希不一致：{path}")
            booster = load_gbdt_model(path)
            refs.append(
                ModelRef(
                    fold_id=str(fold["fold_id"]),
                    seed=seed,
                    booster=booster,
                    model_file=relative,
                )
            )
            by_key[(str(fold["fold_id"]), seed)] = booster
            if seed == 0:
                seed_zero[str(fold["fold_id"])] = booster
    if len(refs) != len(manifest["folds"]) * len(manifest["seeds"]):
        raise AlphaArtifactError("成员模型数量与 manifest 不一致")
    return tuple(refs), by_key, seed_zero


def code_sha256() -> str:
    """汇总解释链路的代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "research/modeling/alpha/ablation.py",
        "research/modeling/alpha/explain.py",
        "research/modeling/alpha/exposure.py",
        "research/modeling/alpha/feature_groups.py",
        "core/data/scripts/build_alpha/analysis_common.py",
        "core/data/scripts/build_alpha/analysis_report.py",
        "core/data/scripts/build_alpha/explain_model.py",
        "core/data/scripts/build_alpha/weekly_sim.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def analysis_id(manifest: dict, config: dict, code_hash: str) -> str:
    """由模型、预测、代码和参数生成确定性分析 ID。"""

    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "model_id": manifest["model_id"],
        "model_sha256": manifest["model_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "dataset_id": manifest["dataset_id"],
        "config": config,
        "code_sha256": code_hash,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"p3_4_{digest}"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def wide_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把标签和股票池转换为宽表。"""

    labels = data.pivot(
        index="trade_date", columns="stock_code", values="label_raw"
    )
    eligible = data.pivot(
        index="trade_date", columns="stock_code", values="eligible"
    ).astype("boolean").fillna(False).astype(bool)
    return labels, eligible


def wide_scores(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """把标准预测表转换为宽分数。"""

    return predictions.pivot(
        index="trade_date", columns="stock_code", values="score"
    ).reindex(index=dates)


def full_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
    manifest: dict,
    auxiliary,
    *,
    top_n: int,
) -> tuple[dict, list[dict]]:
    """计算模型层和 Top-N 组合层指标及分年结果。"""

    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(predictions["trade_date"]).unique())
    )
    scores = wide_scores(predictions, dates)
    model_level = evaluate_score_frame(
        scores,
        labels,
        eligible,
        quantiles=int(manifest["quantiles"]),
    )
    start = str(dates.min().date())
    end = str(dates.max().date())
    portfolio, daily, _ = simulate_score_frame(
        scores,
        labels,
        eligible,
        start=start,
        end=end,
        top_n=int(top_n),
        auxiliary=auxiliary,
    )
    yearly_model = {
        int(row["year"]): row for row in model_level.get("yearly", [])
    }
    yearly_rows = []
    for row in yearly_summary(daily):
        year = int(row["year"])
        yearly_rows.append(
            {
                "year": year,
                "rank_ic_mean": yearly_model.get(year, {}).get("rank_ic_mean"),
                "total_return": row.get("total_return"),
                "max_drawdown": row.get("max_drawdown"),
                "turnover_two_way_annual": row.get("turnover"),
            }
        )
    combined = {
        "rank_ic_mean": model_level.get("rank_ic_mean"),
        "icir": model_level.get("icir"),
        "rank_ic_t": model_level.get("rank_ic_t"),
        "rank_ic_win_rate": model_level.get("rank_ic_win_rate"),
        "monotonicity": model_level.get("monotonicity"),
        "annual_return": portfolio.get("annual_return"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "turnover_two_way_annual": portfolio.get("turnover_two_way_annual"),
        "total_cost": portfolio.get("total_cost"),
    }
    return combined, yearly_rows


def rank_metrics(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """只计算模型层 RankIC 和分年结果。"""

    labeled = predictions.merge(
        data.loc[:, ["trade_date", "stock_code", "label_raw"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"label_raw": "label"})
    series = rank_ic_series(labeled)
    return rank_ic_summary(series), yearly_rank_ic(series)


def shap_records(
    preprocessed,
    manifest: dict,
    models: dict[tuple[str, int], object],
    sample_rows: int,
) -> pd.DataFrame:
    """在每折的确定性测试样本上计算所有种子的 TreeSHAP。"""

    features = tuple(str(name) for name in manifest["features"])
    rows: list[dict] = []
    folds = sorted(manifest["folds"], key=lambda item: item["test_start"])
    for fold_index, fold in enumerate(folds):
        dates = pd.to_datetime(preprocessed.data["trade_date"])
        mask = (
            (dates >= pd.Timestamp(fold["test_start"]))
            & (dates <= pd.Timestamp(fold["test_end"]))
            & preprocessed.data["eligible"].astype(bool)
        )
        eligible = preprocessed.data.loc[mask]
        if eligible.empty:
            raise AlphaEvaluationError(f"{fold['fold_id']} TreeSHAP 测试样本为空")
        count = min(int(sample_rows), len(eligible))
        rng = np.random.default_rng(20261020 + fold_index)
        selected = np.sort(rng.choice(len(eligible), size=count, replace=False))
        matrix = eligible.iloc[selected].loc[:, features].to_numpy(dtype="float32")
        for seed in manifest["seeds"]:
            booster = models.get((str(fold["fold_id"]), int(seed)))
            if booster is None:
                raise AlphaArtifactError(
                    f"TreeSHAP 缺少 {fold['fold_id']}/seed={seed} 模型"
                )
            summary = tree_shap_feature_summary(booster, matrix, features)
            summary["fold_id"] = str(fold["fold_id"])
            summary["seed"] = int(seed)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def delta(baseline: dict, current: dict, metric: str) -> float | None:
    """计算基线减消融值。"""

    left = baseline.get(metric)
    right = current.get(metric)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def yearly_deltas(
    baseline: list[dict],
    current: list[dict],
    group: str,
    *,
    model_level_only: bool,
) -> list[dict]:
    """按年计算消融差值。"""

    baseline_by_year = {int(row["year"]): row for row in baseline}
    rows = []
    for row in current:
        year = int(row["year"])
        base = baseline_by_year.get(year, {})
        item = {
            "group": group,
            "year": year,
            "rank_ic_mean": row.get("rank_ic_mean"),
            "delta_rank_ic": (
                None
                if base.get("rank_ic_mean") is None
                or row.get("rank_ic_mean") is None
                else float(base["rank_ic_mean"]) - float(row["rank_ic_mean"])
            ),
        }
        if not model_level_only:
            item.update(
                {
                    "total_return": row.get("total_return"),
                    "delta_total_return": (
                        None
                        if base.get("total_return") is None
                        or row.get("total_return") is None
                        else float(base["total_return"]) - float(row["total_return"])
                    ),
                }
            )
        rows.append(item)
    return rows


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "analysis_id",
    "code_sha256",
    "delta",
    "full_metrics",
    "load_model_manifest",
    "load_models",
    "load_prediction_parts",
    "rank_metrics",
    "shap_records",
    "wide_frames",
    "wide_scores",
    "write_csv",
    "yearly_deltas",
]
