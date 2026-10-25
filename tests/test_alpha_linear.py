"""线性模型拟合与数组化落盘的单元测试。"""

import numpy as np
import pytest

from alpha.errors import AlphaModelError
from alpha.linear import fit_linear, load_model_bundle, predict_linear, save_model_bundle


def _training_data():
    """构造可线性拟合的小样本。"""

    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.2
    weights = np.ones(len(y), dtype="float32")
    return X, y, weights


def test_ridge_fit_and_bundle_round_trip(tmp_path):
    """Ridge 能恢复主要线性关系，落盘后预测不变。"""

    X, y, weights = _training_data()
    model = fit_linear(
        X,
        y,
        weights,
        model_type="ridge",
        alpha=0.1,
    )
    prediction = predict_linear(model, X)
    assert np.corrcoef(prediction, y)[0, 1] > 0.99
    path = tmp_path / "model.npz"
    save_model_bundle({"wf_2020": model}, path)
    loaded = load_model_bundle(path)["wf_2020"]
    assert np.allclose(predict_linear(loaded, X), prediction)


def test_elastic_net_smoke():
    """ElasticNet 共用同一预测接口。"""

    X, y, weights = _training_data()
    model = fit_linear(
        X,
        y,
        weights,
        model_type="elastic_net",
        alpha=0.001,
        l1_ratio=0.5,
        seed=3,
    )
    prediction = predict_linear(model, X)
    assert np.corrcoef(prediction, y)[0, 1] > 0.95


def test_linear_model_rejects_non_finite_features():
    """预测特征必须有限。"""

    X, y, weights = _training_data()
    model = fit_linear(X, y, weights, model_type="ridge", alpha=1.0)
    broken = X.copy()
    broken[0, 0] = np.nan
    with pytest.raises(AlphaModelError, match="非有限"):
        predict_linear(model, broken)
