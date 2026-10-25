"""截面预处理与中性化的单元测试。"""

import numpy as np
import pandas as pd

from alpha.preprocess import PreprocessConfig, transform


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
