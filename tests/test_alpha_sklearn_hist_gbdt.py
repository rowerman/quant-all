"""Scikit-learn 直方图 GBDT 后端契约测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.errors import AlphaModelError
from alpha.modeling import FoldFitResult, TrainingSpec
from alpha.modeling.backends.sklearn_hist_gbdt import (
    HistGradientBoostingConfig,
    SklearnHistGbdtBackend,
)
from alpha.modeling.pipeline import fit_model_loaded
from alpha.split import Fold


def _rows() -> pd.DataFrame:
    """构造可完成单折训练的截面数据。"""

    dates = pd.date_range("2020-01-03", periods=45, freq="W-FRI")
    records = []
    for day_index, day in enumerate(dates):
        for stock_index in range(20):
            signal = stock_index / 19 + day_index * 0.001
            records.append(
                {
                    "trade_date": day,
                    "stock_code": f"S{stock_index:02d}",
                    "eligible": True,
                    "sample_weight": 1.0,
                    "label_raw": signal / 100,
                    "label_rank": signal,
                    "industry_l1": "I1",
                    "log_mv": 10.0,
                    "turnover_rate": 1.0,
                    "listed_days": 1000.0,
                    "F1": signal,
                }
            )
    return pd.DataFrame(records)


def _fold(dates: pd.DatetimeIndex) -> Fold:
    """构造训练、验证和测试互不重叠的一折。"""

    return Fold(
        "wf_2020",
        str(dates[0].date()),
        str(dates[24].date()),
        str(dates[25].date()),
        str(dates[32].date()),
        str(dates[33].date()),
        str(dates[43].date()),
        6,
        6,
    )


def test_backend_config_validation_rejects_invalid_values():
    """非法模型类型和参数在构建配置时失败。"""

    backend = SklearnHistGbdtBackend()
    with pytest.raises(AlphaModelError, match="model_type"):
        backend.build_config({"model_type": "other"})
    with pytest.raises(AlphaModelError, match="learning_rate"):
        backend.build_config({"learning_rate": 0})
    with pytest.raises(AlphaModelError, match="validation_fraction"):
        backend.build_config({"validation_fraction": 1.0})


def test_hist_gbdt_trains_and_writes_v2_manifest(tmp_path, monkeypatch):
    """新后端无需修改共享流水线即可写标准 v2 产物。"""

    import alpha.modeling.pipeline as pipeline

    monkeypatch.setattr(pipeline, "append_experiment", lambda *args, **kwargs: None)
    frame = _rows()
    dates = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    manifest = {
        "dataset_id": "synthetic_hist",
        "features": ["F1"],
        "context_features": [],
        "versions": {"data_version": "d1", "factor_version": "f1"},
        "code_sha256": "dataset-code",
    }
    artifact = fit_model_loaded(
        frame,
        manifest,
        "synthetic_hist",
        TrainingSpec(
            backend="sklearn_hist_gbdt",
            config={
                "max_iter": 12,
                "min_samples_leaf": 5,
                "max_leaf_nodes": 7,
                "validation_fraction": 0.2,
                "n_iter_no_change": 2,
                "seed": 3,
            },
        ),
        folds=(_fold(dates),),
        output_root=tmp_path,
    )

    assert artifact.manifest["schema_version"] == "alpha_model_v2"
    assert artifact.manifest["backend"]["name"] == "sklearn_hist_gbdt"
    assert artifact.manifest["model_type"] == "sklearn_hist_gbdt"
    model_files = list((artifact.path / "models").glob("*.joblib"))
    assert len(model_files) == 1

    backend = SklearnHistGbdtBackend()
    loaded = backend.load_model(model_files[0], fold_id="wf_2020")
    matrix = np.column_stack(
        [
            frame.loc[:, "F1"].head(20).to_numpy(dtype="float32"),
            np.zeros(20, dtype="float32"),
        ]
    )
    scores = backend.predict(
        loaded,
        matrix,
    )
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_refit_requires_iteration_selection():
    """缺少早停轮数时不能静默重拟合。"""

    backend = SklearnHistGbdtBackend()
    config = HistGradientBoostingConfig(min_samples_leaf=2)
    with pytest.raises(AlphaModelError, match="早停轮数"):
        backend.refit(
            _rows(),
            ("F1",),
            config,
            FoldFitResult(model=object()),
        )
