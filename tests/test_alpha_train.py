"""P2 训练、落盘、重放与无泄漏联动测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha.modeling.pipeline as training_pipeline
from alpha.evaluate import evaluate_score_frame
from alpha.errors import AlphaArtifactError, AlphaModelError
from alpha.modeling import TrainingSpec
from alpha.modeling.pipeline import fit_model_loaded
from alpha.predict import _predict_loaded
from alpha.split import Fold


def _synthetic_dataset():
    """构造两折、六只股票的周频数据集。"""

    dates = pd.date_range("2012-01-06", periods=90, freq="W-FRI")
    rows = []
    for day_index, day in enumerate(dates):
        for stock_index in range(6):
            f1 = float(stock_index) + 0.01 * day_index
            f2 = float(5 - stock_index)
            label = 0.02 * f1 + 0.001 * day_index
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": f"S{stock_index}",
                    "eligible": True,
                    "sample_weight": 1.0,
                    "label_raw": label,
                    "label_rank": (stock_index + 1) / 6,
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


def _fit(tmp_path: Path, frame: pd.DataFrame, manifest: dict, folds):
    """运行合成数据训练。"""

    return fit_model_loaded(
        frame,
        manifest,
        "synthetic",
        TrainingSpec(
            backend="linear",
            config={
                "model_type": "ridge",
                "alpha_grid": [0.1, 1.0],
                "seed": 0,
            },
        ),
        folds=folds,
        output_root=tmp_path,
    )


def test_train_persist_predict_and_evaluate(tmp_path, monkeypatch):
    """训练产物可重载，预测哈希一致，并能完成模型层评估。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    artifact = _fit(tmp_path, frame, manifest, folds)
    stored = json.loads(
        (artifact.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert artifact.row_count == stored["prediction_rows"] > 0
    replayed = _predict_loaded(
        frame,
        stored,
        root=tmp_path,
        start=stored["prediction_start"],
        end=stored["prediction_end"],
    )
    assert set(replayed.columns) == {"trade_date", "stock_code", "score"}
    assert not replayed.duplicated(["trade_date", "stock_code"]).any()
    evaluation = evaluate_score_frame(
        replayed.pivot(index="trade_date", columns="stock_code", values="score"),
        frame.pivot(index="trade_date", columns="stock_code", values="label_raw"),
        frame.pivot(index="trade_date", columns="stock_code", values="eligible"),
        quantiles=3,
    )
    assert evaluation["days"] > 0


def test_test_labels_do_not_change_model_or_predictions(tmp_path, monkeypatch):
    """改变测试标签不改变折内选参、模型文件和预测。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    left = _fit(tmp_path / "left", frame, manifest, folds)
    changed = frame.copy()
    last_test = pd.to_datetime(changed["trade_date"]).between(
        pd.Timestamp(folds[-1].test_start),
        pd.Timestamp(folds[-1].test_end),
    )
    changed.loc[last_test, ["label_raw", "label_rank"]] *= -5
    right = _fit(tmp_path / "right", changed, manifest, folds)
    assert left.model_id == right.model_id
    assert left.prediction_sha256 == right.prediction_sha256
    left_record = left.manifest["fold_models"][0]
    right_record = right.manifest["fold_models"][0]
    left_model = (left.path / left_record["model_file"]).read_bytes()
    right_model = (right.path / right_record["model_file"]).read_bytes()
    assert left_model == right_model


def test_valid_model_artifact_is_reused(tmp_path, monkeypatch):
    """完整训练产物再次运行时直接复用。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    first = _fit(tmp_path, frame, manifest, folds)
    second = _fit(tmp_path, frame, manifest, folds)
    assert first.model_id == second.model_id
    assert first.prediction_sha256 == second.prediction_sha256


def test_resume_rejects_corrupt_model_file(tmp_path, monkeypatch):
    """已有模型文件损坏时不能把训练恢复报告为成功。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    artifact = _fit(tmp_path, frame, manifest, folds)
    record = artifact.manifest["fold_models"][0]
    path = artifact.path / record["model_file"]
    path.write_bytes(b"corrupt")
    with pytest.raises(AlphaModelError, match="哈希不一致"):
        _fit(tmp_path, frame, manifest, folds)


def test_resume_rejects_corrupt_prediction_part(tmp_path, monkeypatch):
    """已有预测分片损坏时不能把训练恢复报告为成功。"""

    monkeypatch.setattr(
        training_pipeline, "append_experiment", lambda *args, **kwargs: None
    )
    frame, manifest, folds = _synthetic_dataset()
    artifact = _fit(tmp_path, frame, manifest, folds)
    part = next((artifact.path / "predictions").glob("year=*/part.parquet"))
    part.write_bytes(b"corrupt")
    with pytest.raises(AlphaArtifactError, match="无法读取"):
        _fit(tmp_path, frame, manifest, folds)
