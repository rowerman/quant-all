"""P0 基线配置的单元测试。"""

import pytest

from alpha.errors import AlphaConfigError
from scripts.build_alpha.baselines import (
    BASELINE_IDS,
    _guard_development,
    baseline_strategy,
    baseline_weights,
    scores_for_baseline,
)


def test_baseline_weights_are_direction_adjusted():
    """B2 和 B3 的方向与因子注册表一致。"""

    assert baseline_weights("B2_MOM_EP_VOL") == {
        "MOM_20": 1.0,
        "EP_TTM": 1.0,
        "VOL_20": -1.0,
    }
    weights = baseline_weights("B3_ALL_ZSCORE")
    assert len(weights) == 47
    assert weights["VOL_20"] == -1.0


def test_b4_strategy_uses_provider_and_custom_rebalance():
    """B4 通过 factor_rank 提供者输出目标权重。"""

    strategy = baseline_strategy("B4_ALL_RANK", ("MOM_20", "VOL_20"))
    assert strategy.signal.provider.name == "factor_rank"
    assert strategy.portfolio.selection == "provider"
    assert strategy.rebalance.frequency == "custom"


def test_development_guard_rejects_holdout():
    """基线脚本不能跨入最终测试集。"""

    _guard_development("2012-01-01", "2023-12-31")
    with pytest.raises(AlphaConfigError, match="最终测试集"):
        _guard_development("2012-01-01", "2024-01-01")


def test_all_baseline_ids_have_strategy():
    """四条基线都能构造配置。"""

    for baseline_id in BASELINE_IDS:
        strategy = baseline_strategy(baseline_id, ("MOM_20", "VOL_20"))
        assert strategy.universe == "all_a"
        assert strategy.benchmark.index_code == "H00300.CSI"
