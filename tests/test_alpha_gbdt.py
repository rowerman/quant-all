"""LightGBM 训练、产物、预测重放与无泄漏测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import alpha.modeling.pipeline as training_pipeline
from alpha.artifacts import prediction_sha256
from alpha.errors import AlphaModelError
from alpha.gbdt import (
    GbdtConfig,
    GbdtEnsembleConfig,
    fit_gbdt_fold,
    load_gbdt_model,
    predict_gbdt,
    save_gbdt_model,
)
from alpha.predict import _predict_loaded
from alpha.split import Fold
from alpha.modeling import TrainingSpec
from alpha.modeling.pipeline import fit_model_loaded
from scripts.build_alpha.evaluate import _decision


def _training_arrays():
    """构造带非线性和交互关系的多日训练数据。"""

    rng = np.random.default_rng(11)
    dates = np.repeat(pd.date_range("2020-01-03", periods=30, freq="W-FRI"), 40)
    X = rng.normal(size=(len(dates), 4))
    raw = 2.0 * X[:, 0] ** 2 - 1.5 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    y = pd.Series(raw).groupby(dates).rank(pct=True).to_numpy(dtype="float32")
    weights = np.ones(len(y), dtype="float32")
    return X[:800], y[:800], weights[:800], X[800:], y[800:], weights[800:], dates[800:]


def _config() -> GbdtConfig:
    """返回可在小样本上快速收敛的配置。"""

    return GbdtConfig(
        learning_rate=0.1,
        num_leaves=7,
        max_depth=3,
        min_data_in_leaf=10,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        lambda_l2=1.0,
        max_rounds=60,
        early_stopping_rounds=15,
        num_threads=1,
    )


def _synthetic_dataset():
    """构造两折、八只股票的周频数据集。"""

    dates = pd.date_range("2012-01-06", periods=90, freq="W-FRI")
    rows = []
    for day_index, day in enumerate(dates):
        for stock_index in range(8):
            f1 = float(stock_index) + 0.01 * day_index
            f2 = float(7 - stock_index)
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": f"S{stock_index}",
                    "eligible": True,
                    "sample_weight": 1.0,
                    "label_raw": 0.02 * f1 + 0.001 * day_index,
                    "label_rank": (stock_index + 1) / 8,
                    "industry_l1": "I1",
                    "log_mv": 10.0 + stock_index,
                    "turnover_rate": 1.0,
                    "listed_days": 1000.0,
                    "F1": f1,
                    "F2": f2,
                }
            )
    frame = pd.DataFrame(rows)
    manifest = {
        "dataset_id": "synthetic",
        "features": ["F1", "F2"],
        "context_features": [],
        "versions": {"data_version": "d1", "factor_version": "f1"},
        "code_sha256": "dataset-code",
    }
    folds = (
        Fold(
            fold_id="wf_2013",
            train_start=str(dates[0].date()),
            train_end=str(dates[39].date()),
            valid_start=str(dates[40].date()),
            valid_end=str(dates[49].date()),
            test_start=str(dates[50].date()),
            test_end=str(dates[60].date()),
            purge_days=6,
            embargo_days=6,
        ),
        Fold(
            fold_id="wf_2014",
            train_start=str(dates[0].date()),
            train_end=str(dates[49].date()),
            valid_start=str(dates[50].date()),
            valid_end=str(dates[60].date()),
            test_start=str(dates[61].date()),
            test_end=str(dates[75].date()),
            purge_days=6,
            embargo_days=6,
        ),
    )
    return frame, manifest, folds


def test_gbdt_fit_is_deterministic_and_model_round_trip(tmp_path):
    """同一种子训练一致，落盘后预测不变。"""

    X_train, y_train, weight_train, X_valid, y_valid, weight_valid, dates = (
        _training_arrays()
    )
    features = ("F1", "F2", "F3", "F4")
    left = fit_gbdt_fold(
        X_train,
        y_train,
        weight_train,
        X_valid,
        y_valid,
        weight_valid,
        dates,
        features,
        _config(),
    )
    right = fit_gbdt_fold(
        X_train,
        y_train,
        weight_train,
        X_valid,
        y_valid,
        weight_valid,
        dates,
        features,
        _config(),
    )
    scores = predict_gbdt(left.booster, X_valid)
    assert np.corrcoef(scores, y_valid)[0, 1] > 0.75
    assert np.array_equal(scores, predict_gbdt(right.booster, X_valid))

    path = tmp_path / "model.txt"
    save_gbdt_model(left.booster, path, num_iteration=left.best_iteration)
    loaded = load_gbdt_model(path)
    assert np.array_equal(scores, predict_gbdt(loaded, X_valid))


def test_gbdt_rejects_feature_shape_mismatch():
    """预测矩阵必须与训练特征数一致。"""

    X_train, y_train, weight_train, X_valid, y_valid, weight_valid, dates = (
        _training_arrays()
    )
    fitted = fit_gbdt_fold(
        X_train,
        y_train,
        weight_train,
        X_valid,
        y_valid,
        weight_valid,
        dates,
        ("F1", "F2", "F3", "F4"),
        _config(),
    )
    with pytest.raises(AlphaModelError, match="特征数"):
        predict_gbdt(fitted.booster, np.ones((10, 3), dtype="float32"))


def test_gbdt_artifact_and_replay_ignore_test_labels(tmp_path, monkeypatch):
    """测试标签变化不改变模型、预测或重放哈希。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    config = _config()
    spec = TrainingSpec(backend="lightgbm", config=config.to_dict())
    left = fit_model_loaded(
        frame,
        manifest,
        "synthetic",
        spec,
        folds=folds,
        output_root=tmp_path / "left",
    )
    stored = json.loads((left.path / "manifest.json").read_text(encoding="utf-8"))
    assert stored["model_type"] == "lightgbm"
    assert len(stored["fold_models"]) == len(folds)
    assert (left.path / "fold_models.csv").is_file()

    replayed = _predict_loaded(
        frame,
        stored,
        root=tmp_path / "left",
        start=stored["prediction_start"],
        end=stored["prediction_end"],
    )
    assert len(replayed) == left.row_count
    assert prediction_sha256(replayed) == left.prediction_sha256

    changed = frame.copy()
    last_test = pd.to_datetime(changed["trade_date"]).between(
        pd.Timestamp(folds[-1].test_start),
        pd.Timestamp(folds[-1].test_end),
    )
    changed.loc[last_test, ["label_raw", "label_rank"]] *= -5
    right = fit_model_loaded(
        changed,
        manifest,
        "synthetic",
        spec,
        folds=folds,
        output_root=tmp_path / "right",
    )
    assert left.model_id == right.model_id
    assert left.prediction_sha256 == right.prediction_sha256
    for left_record, right_record in zip(
        left.manifest["fold_models"], right.manifest["fold_models"], strict=True
    ):
        left_model = (left.path / left_record["model_file"]).read_bytes()
        right_model = (right.path / right_record["model_file"]).read_bytes()
        assert left_model == right_model


def test_lightgbm_evaluation_stays_pending_p3():
    """P3-1/P3-2 评估不提前执行 P3-5 决策门。"""

    decision = _decision(
        {"model_type": "lightgbm"},
        {"MODEL": {"rank_ic_mean": 0.1}},
    )
    assert decision["status"] == "pending_p3"
    assert decision["enter_p3"] is None


def test_lightgbm_evaluation_preserves_recorded_p3_decision():
    """已落盘的 P3-5 决策不会被普通评估覆盖。"""

    decision = _decision(
        {
            "model_type": "lightgbm_ensemble",
            "p3_decision": {
                "status": "pass_p3",
                "reference_model_id": "ridge",
                "reasons": ["通过固定决策门"],
                "analysis_id": "p3_5_test",
                "selected_candidate": "LIGHTGBM_RAW",
            },
        },
        {"MODEL": {"rank_ic_mean": 0.1}},
    )
    assert decision["status"] == "pass_p3"
    assert decision["enter_p3"]
    assert decision["selected_candidate"] == "LIGHTGBM_RAW"


def test_gbdt_ensemble_artifact_replays_all_seeds(tmp_path, monkeypatch):
    """五种子集成保存全部成员，预测可重放且不读取测试标签。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    ensemble_config = GbdtEnsembleConfig(
        seeds=(0, 1, 2, 3, 4),
        member=_config(),
    )
    left = fit_model_loaded(
        frame,
        manifest,
        "synthetic",
        TrainingSpec(
            backend="lightgbm",
            config=ensemble_config.member.to_dict(),
            seeds=ensemble_config.seeds,
        ),
        folds=folds,
        output_root=tmp_path / "left",
    )
    stored = json.loads((left.path / "manifest.json").read_text(encoding="utf-8"))
    assert stored["model_type"] == "lightgbm_ensemble"
    assert stored["seeds"] == [0, 1, 2, 3, 4]
    assert len(stored["fold_models"]) == len(folds) * 5
    assert set(stored["member_predictions"]) == {str(seed) for seed in range(5)}

    replayed = _predict_loaded(
        frame,
        stored,
        root=tmp_path / "left",
        start=stored["prediction_start"],
        end=stored["prediction_end"],
    )
    assert prediction_sha256(replayed) == left.prediction_sha256

    changed = frame.copy()
    last_test = pd.to_datetime(changed["trade_date"]).between(
        pd.Timestamp(folds[-1].test_start),
        pd.Timestamp(folds[-1].test_end),
    )
    changed.loc[last_test, ["label_raw", "label_rank"]] *= -5
    right = fit_model_loaded(
        changed,
        manifest,
        "synthetic",
        TrainingSpec(
            backend="lightgbm",
            config=ensemble_config.member.to_dict(),
            seeds=ensemble_config.seeds,
        ),
        folds=folds,
        output_root=tmp_path / "right",
    )
    assert left.model_id == right.model_id
    assert left.prediction_sha256 == right.prediction_sha256
    for left_record, right_record in zip(
        left.manifest["fold_models"], right.manifest["fold_models"], strict=True
    ):
        assert (left.path / left_record["model_file"]).read_bytes() == (
            right.path / right_record["model_file"]
        ).read_bytes()
