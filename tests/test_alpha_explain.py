"""P3-4 特征分组、重要性、相关性和消融测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.ablation import fit_family_ablation
from alpha.artifacts import prediction_sha256
from alpha.explain import (
    ModelRef,
    compute_feature_correlation,
    compute_gain_importance,
    permute_group_within_dates,
    tree_shap_feature_summary,
)
from alpha.exposure import PortfolioExposureResult
from alpha.feature_groups import FeatureGroup, build_feature_groups
from alpha.gbdt import fit_gbdt_fold
from alpha.preprocess import transform
from scripts.build_alpha.analysis_report import write_report
from tests.test_alpha_gbdt import _config, _synthetic_dataset


def _fitted_booster():
    """构造一个小型确定性 LightGBM 模型。"""

    rng = np.random.default_rng(23)
    dates = np.repeat(pd.date_range("2020-01-03", periods=24, freq="W-FRI"), 30)
    matrix = rng.normal(size=(len(dates), 3))
    raw = 1.5 * matrix[:, 0] - 0.8 * matrix[:, 1] + 0.3 * matrix[:, 2] ** 2
    target = pd.Series(raw).groupby(dates).rank(pct=True).to_numpy(dtype="float32")
    weights = np.ones(len(target), dtype="float32")
    fitted = fit_gbdt_fold(
        matrix[:600],
        target[:600],
        weights[:600],
        matrix[600:],
        target[600:],
        weights[600:],
        dates[600:],
        ("F1", "F2", "F3"),
        _config(),
    )
    return fitted.booster, matrix[600:]


def test_feature_groups_cover_roots_families_and_missing_columns():
    """单特征组与分族组必须完整覆盖全部转换列。"""

    manifest = {
        "raw_features": ["MOM_20", "REV_5"],
        "context_features": ["log_mv"],
        "features": [
            "MOM_20",
            "REV_5",
            "log_mv",
            "MOM_20__missing",
            "REV_5__missing",
            "log_mv__missing",
        ],
    }
    groups = build_feature_groups(manifest)
    assert len(groups.singles) == 3
    assert {group.key for group in groups.families} == {"momentum", "context"}
    assert groups.root_by_transformed["MOM_20__missing"] == "MOM_20"
    covered = [
        column
        for group in groups.families
        for column in group.transformed_features
    ]
    assert sorted(covered) == sorted(manifest["features"])


def test_gain_and_tree_shap_are_normalized_and_reconstruct_scores():
    """gain 份额归一，TreeSHAP 贡献加基值等于模型分数。"""

    booster, matrix = _fitted_booster()
    features = ("F1", "F2", "F3")
    gain = compute_gain_importance(
        [ModelRef("wf_1", 0, booster, "m.txt")],
        features,
    )
    assert set(gain["feature"]) == set(features)
    assert gain["share"].sum() == pytest.approx(1.0)
    shap = tree_shap_feature_summary(booster, matrix, features)
    assert set(shap["feature"]) == set(features)
    assert shap["share"].sum() == pytest.approx(1.0)
    assert (shap["rows"] == len(matrix)).all()
    contributions = booster.predict(matrix, pred_contrib=True)
    reconstructed = contributions[:, :-1].sum(axis=1) + contributions[:, -1]
    assert np.allclose(reconstructed, booster.predict(matrix), atol=1e-5)

def test_permutation_happens_only_within_each_date_and_is_deterministic():
    """整组特征按日共用一个行置换。"""

    matrix = np.arange(24, dtype="float32").reshape(8, 3)
    dates = np.array(["2020-01-03"] * 4 + ["2020-01-10"] * 4)
    left = permute_group_within_dates(matrix, dates, (0, 1), seed=17)
    right = permute_group_within_dates(matrix, dates, (0, 1), seed=17)
    assert np.array_equal(left, right)
    assert np.array_equal(left[:, 2], matrix[:, 2])
    for start, end in ((0, 4), (4, 8)):
        original = {tuple(row[:2]) for row in matrix[start:end]}
        changed = {tuple(row[:2]) for row in left[start:end]}
        assert original == changed


def test_feature_correlation_reports_near_duplicate_pair():
    """完全同向的两列应进入同一相关簇。"""

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=30)
    rows = []
    for day in dates:
        values = rng.normal(size=20)
        for index, value in enumerate(values):
            rows.append(
                {
                    "trade_date": day,
                    "eligible": True,
                    "F1": value,
                    "F2": 2.0 * value,
                    "F3": rng.normal(),
                }
            )
    pairs, clusters = compute_feature_correlation(
        pd.DataFrame(rows),
        ("F1", "F2", "F3"),
        max_dates=30,
    )
    assert list(pairs.iloc[0][["feature_a", "feature_b"]]) == ["F1", "F2"]
    assert pairs.iloc[0]["abs_spearman"] == pytest.approx(1.0)
    by_feature = clusters.set_index("feature")["cluster"]
    assert by_feature["F1"] == by_feature["F2"]


def test_family_ablation_ignores_test_labels_and_removes_group(tmp_path):
    """删除特征组后重训，修改最后测试年份标签不改变模型和预测。"""

    frame, manifest, folds = _synthetic_dataset()
    config = _config()
    manifest = dict(manifest)
    manifest["model_params"] = config.to_dict()
    manifest["folds"] = [
        {
            **fold.to_dict(),
            "members": [],
        }
        for fold in folds
    ]
    manifest["features"] = ["F1", "F2", "F1__missing", "F2__missing"]
    preprocessed = transform(
        frame,
        config.preprocess,
        feature_columns=("F1", "F2"),
    )
    group = FeatureGroup(
        key="family_f1",
        label="F1",
        kind="family",
        root_features=("F1",),
        transformed_features=("F1", "F1__missing"),
    )
    left = fit_family_ablation(
        preprocessed,
        manifest,
        group,
        output_dir=tmp_path / "left",
        seed=0,
        num_threads=1,
    )
    assert "F1" not in left.remaining_features
    assert "F1__missing" not in left.remaining_features

    changed = frame.copy()
    last_test = pd.to_datetime(changed["trade_date"]).between(
        pd.Timestamp(folds[-1].test_start),
        pd.Timestamp(folds[-1].test_end),
    )
    changed.loc[last_test, ["label_raw", "label_rank"]] *= -5
    changed_preprocessed = transform(
        changed,
        config.preprocess,
        feature_columns=("F1", "F2"),
    )
    right = fit_family_ablation(
        changed_preprocessed,
        manifest,
        group,
        output_dir=tmp_path / "right",
        seed=0,
        num_threads=1,
    )
    assert prediction_sha256(left.predictions) == prediction_sha256(right.predictions)
    for fold in folds:
        assert (tmp_path / "left" / f"{fold.fold_id}.txt").read_bytes() == (
            tmp_path / "right" / f"{fold.fold_id}.txt"
        ).read_bytes()


def test_explain_report_uses_family_table_and_seed_zero_label(tmp_path):
    """可读报告必须标出 seed=0 基线并使用真正的分族汇总表。"""

    gain_features = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "F1",
                "root_feature": "F1",
                "family": "liquidity",
                "share_mean": 0.6,
            }
        ]
    )
    gain_family = pd.DataFrame(
        [{"rank": 1, "family": "liquidity", "label": "流动性", "share_mean": 0.6}]
    )
    shap_features = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "F1",
                "root_feature": "F1",
                "family": "liquidity",
                "mean_abs_shap_mean": 0.4,
            }
        ]
    )
    shap_family = pd.DataFrame(
        [
            {
                "rank": 1,
                "family": "liquidity",
                "label": "流动性",
                "mean_abs_shap_mean": 0.4,
            }
        ]
    )
    family_table = pd.DataFrame(
        [
            {
                "group": "liquidity",
                "rank_ic_mean": 0.09,
                "delta_rank_ic": 0.01,
                "delta_icir": 0.02,
                "delta_annual_return": 0.01,
            }
        ]
    )
    permutation_table = pd.DataFrame(
        [
            {
                "group": "F1",
                "rank_ic_mean": 0.08,
                "delta_rank_ic": 0.02,
                "delta_icir": 0.03,
            }
        ]
    )
    exposure = PortfolioExposureResult(
        style_summary=pd.DataFrame(
            [
                {
                    "style": "SIZE",
                    "label": "规模",
                    "mean": -0.1,
                    "t": -1.0,
                    "positive_share": 0.4,
                    "mean_coverage": 1.0,
                }
            ]
        ),
        style_series=pd.DataFrame(),
        industry_summary=pd.DataFrame(
            [
                {
                    "industry": "I1",
                    "mean_weight": 0.5,
                    "max_weight": 0.6,
                    "mean_active_weight": 0.1,
                }
            ]
        ),
        industry_series=pd.DataFrame(),
        summary={"days": 1},
    )
    write_report(
        tmp_path,
        model_id="model",
        analysis_id="analysis",
        baseline={
            "start": "2020-01-01",
            "end": "2020-01-31",
            "rank_ic_mean": 0.1,
            "icir": 0.5,
            "annual_return": 0.1,
            "max_drawdown": -0.1,
            "turnover_two_way_annual": 1.0,
        },
        gain_features=gain_features,
        gain_family=gain_family,
        shap_features=shap_features,
        shap_family=shap_family,
        family_table=family_table,
        permutation_table=permutation_table,
        correlation_pairs=pd.DataFrame(
            columns=["feature_a", "feature_b", "mean_spearman", "abs_spearman", "days"]
        ),
        exposure=exposure,
        config={},
    )
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## seed=0 消融基线" in report
    assert "## 分族 Gain 重要性" in report
    assert "| liquidity | 流动性 | 0.600000 |" in report
