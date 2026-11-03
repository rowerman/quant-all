"""ml_rank v3 生命周期状态切换和权重测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha.lifecycle import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_PRIMARY_MODEL_ID,
)
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import get_provider
from backtest.specs import ProviderSpec
from backtest.strategies import ml_rank_v3
from backtest.strategies.ml_rank_v3 import MlRankLifecycleProvider

PRIMARY = "primary"
FALLBACK = "fallback"
STOCKS = ["A", "B", "C", "D"]
DATES = pd.date_range("2023-01-06", periods=6, freq="W-FRI")


def _predictions(offset: float = 0.0) -> pd.DataFrame:
    """构造确定性主模型或回滚模型预测。"""

    rows = []
    for day_index, day in enumerate(DATES):
        for stock_index, stock in enumerate(STOCKS):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "score": offset - stock_index * 0.1 + day_index * 0.01,
                }
            )
    return pd.DataFrame(rows)


def _data():
    """构造 provider 所需的最小行情对象。"""

    return type(
        "Data",
        (),
        {"index": DATES, "columns": pd.Index(STOCKS)},
    )()


def _manifest(model_id: str) -> dict:
    """构造预测 manifest。"""

    return {
        "model_id": model_id,
        "prediction_start": "2023-01-06",
        "prediction_end": "2023-02-10",
    }


def _lifecycle(states: list[str], dates: pd.DatetimeIndex | None = None) -> dict:
    """构造与预测日期对齐的生命周期包。"""

    multipliers = {"normal": 1.0, "reduced": 0.5, "rollback": 1.0, "paused": 0.0}
    active = {
        "normal": PRIMARY,
        "reduced": PRIMARY,
        "rollback": FALLBACK,
        "paused": None,
    }
    selected_dates = DATES if dates is None else dates
    if len(selected_dates) != len(states):
        raise ValueError("states 和 dates 长度必须一致")
    frame = pd.DataFrame(
        {
            "trade_date": selected_dates,
            "state": states,
            "active_model_id": [active[state] for state in states],
            "exposure_multiplier": [multipliers[state] for state in states],
        }
    )
    return {
        "timeseries": frame,
        "manifest": {"sources": {"fallback_model_id": FALLBACK}},
    }


def _mock_inputs(monkeypatch, states: list[str]):
    """替换生命周期和两个模型读取。"""

    monkeypatch.setattr(
        ml_rank_v3,
        "load_model_lifecycle",
        lambda lifecycle_id, model_id: _lifecycle(states),
    )

    def loader(model_id):
        return (
            None,
            _manifest(model_id),
            _predictions(1.0 if model_id == PRIMARY else 0.5),
        )

    monkeypatch.setattr(ml_rank_v3, "load_predictions", loader)


def test_v3_registration_and_parameter_validation():
    """v3 注册、默认参数和非法模型组合明确失败。"""

    assert get_provider(ProviderSpec("ml_rank", "3")).version == "3"
    provider = MlRankLifecycleProvider()
    normalized = provider.validate_params(
        {
            "lifecycle_id": "p5_3_test",
            "model_id": DEFAULT_PRIMARY_MODEL_ID,
            "fallback_model_id": DEFAULT_FALLBACK_MODEL_ID,
            "top_n": 2,
            "rebalance_dates": ["2023-01-20", "2023-01-06"],
        }
    )
    assert normalized["model_id"] == DEFAULT_PRIMARY_MODEL_ID
    assert normalized["fallback_model_id"] == DEFAULT_FALLBACK_MODEL_ID
    assert normalized["rebalance_dates"] == ["2023-01-06", "2023-01-20"]
    with pytest.raises(SpecError, match="model_id"):
        provider.validate_params({"lifecycle_id": "p5_3_test"})
    with pytest.raises(SpecError, match="不能相同"):
        provider.validate_params(
            {
                "lifecycle_id": "p5_3_test",
                "model_id": PRIMARY,
                "fallback_model_id": PRIMARY,
            }
        )


def test_normal_matches_primary_and_reduced_holds_cash(monkeypatch):
    """normal 使用主模型满仓，reduced 将目标权重缩为 0.5。"""

    states = ["normal", "reduced", "normal", "reduced", "normal", "reduced"]
    _mock_inputs(monkeypatch, states)
    params = {
        "lifecycle_id": "p5_3_test",
        "model_id": PRIMARY,
        "fallback_model_id": FALLBACK,
        "top_n": 2,
        "smoothing_weeks": 4,
        "rebalance_dates": [str(day.date()) for day in DATES],
    }
    weights = MlRankLifecycleProvider().build(
        _data(),
        pd.DataFrame(True, index=DATES, columns=STOCKS),
        params,
    )
    assert weights.loc[DATES[0]].sum() == pytest.approx(1.0)
    assert weights.loc[DATES[1]].sum() == pytest.approx(0.5)
    assert set(weights.loc[DATES[0]][weights.loc[DATES[0]] > 0].index) == {
        "A",
        "B",
    }


def test_rollback_uses_fallback_and_paused_clears_position(monkeypatch):
    """rollback 使用 Ridge 分数，paused 返回全 0 行。"""

    states = ["normal", "rollback", "rollback", "paused", "normal", "normal"]
    _mock_inputs(monkeypatch, states)
    params = {
        "lifecycle_id": "p5_3_test",
        "model_id": PRIMARY,
        "fallback_model_id": FALLBACK,
        "top_n": 2,
        "smoothing_weeks": 1,
        "rebalance_dates": [str(day.date()) for day in DATES],
    }
    weights = MlRankLifecycleProvider().build(
        _data(),
        pd.DataFrame(True, index=DATES, columns=STOCKS),
        params,
    )
    assert weights.loc[DATES[1]].sum() == pytest.approx(1.0)
    assert weights.loc[DATES[3]].eq(0.0).all()
    assert set(weights.loc[DATES[1]][weights.loc[DATES[1]] > 0].index) == set(
        ["A", "B"]
    )


def test_provider_rejects_missing_lifecycle_date(monkeypatch):
    """生命周期包缺少调仓日时失败关闭。"""

    _mock_inputs(monkeypatch, ["normal"] * 6)
    monkeypatch.setattr(
        ml_rank_v3,
        "load_model_lifecycle",
        lambda lifecycle_id, model_id: _lifecycle(
            ["normal"] * 5,
            DATES[:-1],
        ),
    )
    with pytest.raises(DataAlignmentError, match="缺少信号日"):
        MlRankLifecycleProvider().build(
            _data(),
            pd.DataFrame(True, index=DATES, columns=STOCKS),
            {
                "lifecycle_id": "p5_3_test",
                "model_id": PRIMARY,
                "fallback_model_id": FALLBACK,
                "rebalance_dates": [str(day.date()) for day in DATES],
            },
        )


def test_artifact_snapshot_contains_both_models_and_lifecycle(monkeypatch):
    """输入快照同时绑定生命周期、主模型和回滚模型。"""

    monkeypatch.setattr(
        ml_rank_v3,
        "load_model_lifecycle",
        lambda lifecycle_id, model_id: {
            "manifest": {
                "files": {"timeseries.csv": "a" * 64},
                "sources": {"fallback_model_id": FALLBACK},
            }
        },
    )
    monkeypatch.setattr(
        ml_rank_v3,
        "model_artifact_snapshot",
        lambda model_id: {"model_id": model_id, "artifact_sha256": "b" * 64},
    )
    payload = MlRankLifecycleProvider().artifact_snapshot(
        {
            "lifecycle_id": "p5_3_test",
            "model_id": PRIMARY,
            "fallback_model_id": FALLBACK,
        }
    )
    assert payload["lifecycle_id"] == "p5_3_test"
    assert payload["primary"]["model_id"] == PRIMARY
    assert payload["fallback"]["model_id"] == FALLBACK


def test_artifact_snapshot_is_cached_for_one_provider_run(monkeypatch):
    """同一 provider 的重复快照读取复用已校验产物。"""

    calls = {"count": 0}
    monkeypatch.setattr(
        ml_rank_v3,
        "load_model_lifecycle",
        lambda lifecycle_id, model_id: {
            "manifest": {
                "files": {"timeseries.csv": "a" * 64},
                "sources": {"fallback_model_id": FALLBACK},
            }
        },
    )

    def snapshot(model_id):
        calls["count"] += 1
        return {"model_id": model_id}

    monkeypatch.setattr(ml_rank_v3, "model_artifact_snapshot", snapshot)
    provider = MlRankLifecycleProvider()
    params = {
        "lifecycle_id": "p5_3_test",
        "model_id": PRIMARY,
        "fallback_model_id": FALLBACK,
    }
    first = provider.artifact_snapshot(params)
    second = provider.artifact_snapshot(params)
    assert first == second
    assert calls["count"] == 2


def test_v3_is_chunk_safe_and_respects_eligibility(monkeypatch):
    """不同行情面板长度不改变调仓日权重，且不可交易股票无权。"""

    states = ["normal", "reduced", "rollback", "normal", "reduced", "normal"]
    _mock_inputs(monkeypatch, states)
    params = {
        "lifecycle_id": "p5_3_test",
        "model_id": PRIMARY,
        "fallback_model_id": FALLBACK,
        "top_n": 2,
        "smoothing_weeks": 4,
        "rebalance_dates": [str(day.date()) for day in DATES],
    }
    eligible = pd.DataFrame(True, index=DATES, columns=STOCKS)
    eligible.loc[DATES[2], "A"] = False
    provider = MlRankLifecycleProvider()
    full = provider.build(_data(), eligible, params)
    first = provider.build(
        type(
            "Data",
            (),
            {"index": DATES[:3], "columns": pd.Index(STOCKS)},
        )(),
        eligible.loc[DATES[:3]],
        params,
    )
    pd.testing.assert_frame_equal(full.loc[DATES[:3]], first)
    assert full.loc[DATES[2], "A"] == 0.0
