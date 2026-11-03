"""日截面 rank 组合和集成配置测试。"""

import numpy as np
import pandas as pd
import pytest

from alpha.ensemble import combine_rank_predictions, rank_by_date
from alpha.errors import AlphaModelError
from alpha.gbdt import GbdtEnsembleConfig


def _prediction_frame(scores: list[float]) -> pd.DataFrame:
    """构造两日三股票预测表。"""

    dates = pd.to_datetime(["2020-01-03"] * 3 + ["2020-01-10"] * 3)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "stock_code": ["A", "B", "C", "A", "B", "C"],
            "score": scores,
        }
    )


def test_rank_average_uses_each_date_cross_section():
    """组合结果是逐日 rank 的等权平均。"""

    left = _prediction_frame([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    right = _prediction_frame([30.0, 20.0, 10.0, 10.0, 20.0, 30.0])
    result = combine_rank_predictions([left, right])
    expected = (
        rank_by_date(left)["score"].to_numpy()
        + rank_by_date(right)["score"].to_numpy()
    ) / 2.0
    assert np.array_equal(result["score"].to_numpy(), expected)
    assert result.groupby("trade_date").size().eq(3).all()
    fixed_blend = combine_rank_predictions(
        [left, right],
        weights=(0.5, 0.5),
    )
    assert np.array_equal(fixed_blend["score"].to_numpy(), expected)


def test_rank_average_rejects_bad_members():
    """成员键不一致、重复行和非有限分数必须报错。"""

    left = _prediction_frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    missing = left.iloc[:-1].copy()
    with pytest.raises(AlphaModelError, match="日期"):
        combine_rank_predictions([left, missing])

    duplicated = pd.concat([left, left.iloc[[0]]], ignore_index=True)
    with pytest.raises(AlphaModelError, match="重复"):
        rank_by_date(duplicated)

    invalid = left.copy()
    invalid.loc[0, "score"] = np.inf
    with pytest.raises(AlphaModelError, match="非有限"):
        rank_by_date(invalid)


def test_ensemble_config_requires_fixed_sorted_seeds():
    """集成种子必须是 3-5 个严格递增的非负整数。"""

    assert GbdtEnsembleConfig().seeds == (0, 1, 2, 3, 4)
    with pytest.raises(AlphaModelError, match="3 到 5"):
        GbdtEnsembleConfig(seeds=(0, 1))
    with pytest.raises(AlphaModelError, match="严格递增"):
        GbdtEnsembleConfig(seeds=(0, 2, 1))

