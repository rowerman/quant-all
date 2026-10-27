"""截面预处理与中性化的单元测试。"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from alpha.preprocess import PreprocessConfig, transform

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


def _frame() -> pd.DataFrame:
    """构造两个日期、四只股票的小数据集。"""

    rows = []
    for day in pd.to_datetime(["2024-01-05", "2024-01-12"]):
        for index, stock in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "eligible": True,
                    "sample_weight": 1.0,
                    "label_raw": 0.01 * (index + 1),
                    "label_rank": (index + 1) / 4,
                    "industry_l1": ["I1", "I1", "I2", "I2"][index],
                    "X": float(index + 1),
                    "log_mv": float(index + 1),
                    "turnover_rate": 1.0,
                    "listed_days": 100.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[0, "X"] = np.nan
    return frame


def test_rank_transform_uses_daily_cross_section_and_missing_fill():
    """rank 在当日截面内计算，缺失填 0.5 并生成指示列。"""

    frame = _frame()
    result = transform(frame, PreprocessConfig(method="rank"))
    first = result.data.iloc[:4]
    assert np.allclose(first["X"].to_numpy(), [0.5, 1 / 3, 2 / 3, 1.0])
    assert first["X__missing"].tolist() == [1, 0, 0, 0]
    assert "X__missing" in result.feature_names


def test_future_values_do_not_change_past_transform():
    """改变未来日期不改变过去样本的截面变换。"""

    frame = _frame()
    changed = frame.copy()
    changed.loc[changed["trade_date"] == pd.Timestamp("2024-01-12"), "X"] += 100
    left = transform(frame, PreprocessConfig(method="zscore"))
    right = transform(changed, PreprocessConfig(method="zscore"))
    past = left.data["trade_date"] == pd.Timestamp("2024-01-05")
    assert np.allclose(
        left.data.loc[past, "X"].to_numpy(),
        right.data.loc[past, "X"].to_numpy(),
    )


def test_industry_neutralization_removes_group_means():
    """行业中性化后每个行业组的当日均值接近零。"""

    frame = _frame()
    result = transform(
        frame,
        PreprocessConfig(method="rank", neutralize=("industry",)),
    )
    first = result.data.iloc[:4]
    means = first.groupby("industry_l1")["X"].mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-6)


@requires_native
@pytest.mark.parametrize("method", ["rank", "zscore"])
@pytest.mark.parametrize("winsorize", [False, True])
@pytest.mark.parametrize("neutralize", [(), ("industry",), ("log_mv",)])
def test_preprocess_backends_match(
    monkeypatch,
    method,
    winsorize,
    neutralize,
):
    """三后端保持预处理值、报告、缺失列和中性化语义。"""

    frame = _frame()
    config = PreprocessConfig(
        method=method,
        winsorize=winsorize,
        neutralize=neutralize,
        clip_z=2.5,
    )
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = transform(frame, config)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    native = transform(frame, config)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "auto")
    automatic = transform(frame, config)

    pd.testing.assert_frame_equal(native.data, expected.data)
    pd.testing.assert_frame_equal(automatic.data, expected.data)
    assert native.feature_names == expected.feature_names
    assert native.missing_columns == expected.missing_columns
    assert native.report == expected.report


@requires_native
def test_preprocess_restores_shuffled_row_order(monkeypatch):
    """日期和行顺序打乱时仍按日期计算并恢复原位置。"""

    frame = pd.concat([_frame(), _frame().assign(trade_date=pd.Timestamp("2024-01-19"))])
    shuffled = frame.sample(frac=1.0, random_state=72)
    config = PreprocessConfig(method="zscore", winsorize=True)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = transform(shuffled, config)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual = transform(shuffled, config)

    pd.testing.assert_frame_equal(actual.data, expected.data)
