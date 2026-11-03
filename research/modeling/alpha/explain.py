"""LightGBM 重要性、置换消融与特征相关性计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from alpha.errors import AlphaEvaluationError
from alpha.feature_groups import FeatureGroups


@dataclass(frozen=True)
class ModelRef:
    """用于解释的单个落盘 LightGBM 模型。"""

    fold_id: str
    seed: int
    booster: object
    model_file: str = ""


def compute_gain_importance(
    models: Sequence[ModelRef],
    features: tuple[str, ...],
) -> pd.DataFrame:
    """计算每个模型的 gain 及模型内份额。"""

    if not models:
        raise AlphaEvaluationError("gain 重要性缺少模型")
    if not features or len(set(features)) != len(features):
        raise AlphaEvaluationError("gain 重要性特征清单为空或含重复项")
    rows: list[dict] = []
    for model in models:
        gains = np.asarray(
            model.booster.feature_importance(importance_type="gain"),
            dtype="float64",
        )
        if gains.shape != (len(features),):
            raise AlphaEvaluationError(f"{model.fold_id}/seed={model.seed} gain 形状异常")
        if not np.isfinite(gains).all() or (gains < 0).any():
            raise AlphaEvaluationError("gain 含非有限值或负值")
        total = float(gains.sum())
        if total <= 0:
            raise AlphaEvaluationError("模型 gain 总和为零")
        for feature, gain in zip(features, gains):
            rows.append(
                {
                    "fold_id": model.fold_id,
                    "seed": model.seed,
                    "feature": feature,
                    "gain": float(gain),
                    "share": float(gain / total),
                }
            )
    return pd.DataFrame(rows)


def tree_shap_feature_summary(
    booster,
    matrix: np.ndarray,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """用 LightGBM 原生 TreeSHAP 汇总一批样本的特征贡献。"""

    values = np.asarray(matrix, dtype="float32")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(features):
        raise AlphaEvaluationError("TreeSHAP 输入矩阵与特征清单不一致")
    if not np.isfinite(values).all():
        raise AlphaEvaluationError("TreeSHAP 输入包含非有限值")
    try:
        contributions = np.asarray(
            booster.predict(values, pred_contrib=True),
            dtype="float64",
        )
        raw_scores = np.asarray(booster.predict(values), dtype="float64")
    except Exception as error:
        raise AlphaEvaluationError(f"TreeSHAP 计算失败：{error}") from error
    expected = (len(features) + 1,)
    if contributions.ndim != 2 or contributions.shape[1:] != expected:
        raise AlphaEvaluationError("TreeSHAP 输出形状异常")
    reconstructed = contributions[:, :-1].sum(axis=1) + contributions[:, -1]
    scale = np.maximum(1.0, np.abs(raw_scores))
    if not np.allclose(reconstructed, raw_scores, rtol=1e-5, atol=1e-5 * scale.max()):
        raise AlphaEvaluationError("TreeSHAP 贡献加基值与模型分数不一致")
    absolute = np.abs(contributions[:, :-1])
    total = float(absolute.sum())
    if total <= 0:
        raise AlphaEvaluationError("TreeSHAP 绝对值总和为零")
    return pd.DataFrame(
        {
            "feature": list(features),
            "mean_abs_shap": absolute.mean(axis=0),
            "mean_signed_shap": contributions[:, :-1].mean(axis=0),
            "share": absolute.sum(axis=0) / total,
            "rows": int(values.shape[0]),
        }
    )


def compute_tree_shap_importance(
    booster,
    matrix: np.ndarray,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """计算一批样本的特征级 TreeSHAP 重要性。"""

    return tree_shap_feature_summary(booster, matrix, features)


def aggregate_importance(
    records: pd.DataFrame,
    groups: FeatureGroups,
    *,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    """按转换特征、根特征和因子族汇总模型级重要性。"""

    required = {"feature", *value_columns}
    missing = sorted(required - set(records.columns))
    if missing:
        raise AlphaEvaluationError(f"重要性记录缺少列：{missing}")
    work = records.copy()
    work["root_feature"] = work["feature"].map(groups.root_by_transformed)
    if work["root_feature"].isna().any():
        raise AlphaEvaluationError("重要性记录含未登记转换特征")
    work["family"] = work["root_feature"].map(groups.family_by_root)
    if work["family"].isna().any():
        raise AlphaEvaluationError("重要性记录含未登记根特征")
    aggregations = {
        column: ["mean", "std"]
        for column in value_columns
    }
    feature_level = (
        work.groupby(["feature", "root_feature", "family"], sort=False)
        .agg(aggregations)
        .reset_index()
    )
    feature_level.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in feature_level.columns
    ]
    primary = value_columns[0]
    feature_level = feature_level.sort_values(
        [f"{primary}_mean", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    feature_level["rank"] = np.arange(1, len(feature_level) + 1)
    return feature_level


def aggregate_group_importance(
    feature_level: pd.DataFrame,
    groups: FeatureGroups,
    *,
    metric: str,
) -> pd.DataFrame:
    """按根特征和分族汇总已聚合的特征重要性。"""

    value = f"{metric}_mean"
    if value not in feature_level:
        raise AlphaEvaluationError(f"特征重要性缺少列：{value}")
    root = (
        feature_level.groupby("root_feature", as_index=False)[value]
        .sum()
        .sort_values([value, "root_feature"], ascending=[False, True])
    )
    root["rank"] = np.arange(1, len(root) + 1)
    family = (
        feature_level.groupby("family", as_index=False)[value]
        .sum()
        .sort_values([value, "family"], ascending=[False, True])
    )
    family["rank"] = np.arange(1, len(family) + 1)
    labels = {group.key: group.label for group in groups.families}
    family["label"] = family["family"].map(labels)
    return root, family


def permute_group_within_dates(
    matrix: np.ndarray,
    dates: np.ndarray,
    positions: Sequence[int],
    *,
    seed: int,
) -> np.ndarray:
    """在每个交易日截面内置换一组特征列。"""

    values = np.asarray(matrix, dtype="float32")
    day_values = pd.to_datetime(dates).to_numpy()
    if values.ndim != 2 or len(values) != len(day_values):
        raise AlphaEvaluationError("置换矩阵与日期长度不一致")
    columns = tuple(int(position) for position in positions)
    if not columns or len(set(columns)) != len(columns):
        raise AlphaEvaluationError("置换列不能为空或重复")
    if min(columns) < 0 or max(columns) >= values.shape[1]:
        raise AlphaEvaluationError("置换列超出矩阵范围")
    result = values.copy()
    rng = np.random.default_rng(int(seed))
    codes, levels = pd.factorize(day_values, sort=False)
    for code in range(len(levels)):
        rows = np.flatnonzero(codes == code)
        if rows.size < 2:
            continue
        order = rng.permutation(rows.size)
        result[np.ix_(rows, columns)] = values[np.ix_(rows[order], columns)]
    return result


def compute_feature_correlation(
    data: pd.DataFrame,
    root_features: tuple[str, ...],
    *,
    max_dates: int = 100,
    pair_threshold: float = 0.8,
    cluster_threshold: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在开发期确定性截面上计算根特征的逐日 Spearman 相关。"""

    if not root_features or len(set(root_features)) != len(root_features):
        raise AlphaEvaluationError("相关性特征清单为空或含重复项")
    missing = sorted(set(root_features) - set(data.columns))
    if missing:
        raise AlphaEvaluationError(f"数据集缺少相关性特征：{missing}")
    if type(max_dates) is not int or max_dates < 1:
        raise AlphaEvaluationError("max_dates 必须是正整数")
    if not 0.0 < pair_threshold <= 1.0 or not 0.0 < cluster_threshold <= 1.0:
        raise AlphaEvaluationError("相关性阈值必须在 0 到 1 之间")
    work = data.loc[:, ["trade_date", "eligible", *root_features]].copy()
    work = work.loc[work["eligible"].astype(bool)]
    dates = np.sort(pd.to_datetime(work["trade_date"]).unique())
    if not len(dates):
        raise AlphaEvaluationError("相关性分析没有可用交易日")
    selected = np.unique(
        np.linspace(0, len(dates) - 1, min(int(max_dates), len(dates))).astype(int)
    )
    selected_dates = set(dates[selected])
    work = work.loc[pd.to_datetime(work["trade_date"]).isin(selected_dates)]
    count = len(root_features)
    sums = np.zeros((count, count), dtype="float64")
    valid = np.zeros((count, count), dtype="int64")
    for _, frame in work.groupby("trade_date", sort=True):
        correlation = frame.loc[:, list(root_features)].corr(
            method="spearman",
            min_periods=5,
        )
        values = correlation.to_numpy(dtype="float64")
        finite = np.isfinite(values)
        sums[finite] += values[finite]
        valid[finite] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid > 0, sums / valid, np.nan)
    rows: list[dict] = []
    for left in range(count):
        for right in range(left + 1, count):
            value = float(mean[left, right])
            if not np.isfinite(value) or abs(value) < pair_threshold:
                continue
            rows.append(
                {
                    "feature_a": root_features[left],
                    "feature_b": root_features[right],
                    "mean_spearman": value,
                    "abs_spearman": abs(value),
                    "days": int(valid[left, right]),
                }
            )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["abs_spearman", "feature_a", "feature_b"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    parent = {name: name for name in root_features}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    if not pairs.empty:
        for row in pairs.itertuples(index=False):
            if float(row.abs_spearman) >= cluster_threshold:
                union(str(row.feature_a), str(row.feature_b))
    members: dict[str, list[str]] = {}
    for name in root_features:
        members.setdefault(find(name), []).append(name)
    cluster_rows = []
    for root, names in sorted(members.items()):
        cluster = f"cluster_{root}" if len(names) > 1 else "isolated"
        for name in names:
            cluster_rows.append(
                {
                    "feature": name,
                    "cluster": cluster,
                    "cluster_size": len(names),
                }
            )
    clusters = pd.DataFrame(cluster_rows).sort_values(
        ["cluster_size", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return pairs, clusters


__all__ = [
    "ModelRef",
    "aggregate_group_importance",
    "aggregate_importance",
    "compute_feature_correlation",
    "compute_gain_importance",
    "compute_tree_shap_importance",
    "permute_group_within_dates",
    "tree_shap_feature_summary",
]
