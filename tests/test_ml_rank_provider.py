"""ml_rank 目标权重提供者的平滑、过滤和联动测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.errors import AlphaArtifactError
from alpha.portfolio import smooth_rank_scores
from backtest.errors import DataAlignmentError, SpecError
from backtest.provider_schema import provider_param_schema
from backtest.providers import get_provider
from backtest.signals import build_signal_targets
from backtest.snapshot import build_input_snapshot, data_snapshot_from_market_data
from backtest.specs import (
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.strategies import ml_rank
from backtest.strategies import ml_rank_v2
from backtest.strategies._factor_rank import weekly_last_dates
from backtest.strategies.ml_rank import MlRankProvider
from backtest.strategies.ml_rank_v2 import MlRankV2Provider
from backtest.store import BacktestStore

MODEL_ID = "test_ml_model"
STOCKS = ["SH600000", "SZ000001", "SZ000002", "SZ000003"]
DATES = pd.DatetimeIndex(
    [
        "2023-01-06",
        "2023-01-13",
        "2023-01-20",
        "2023-01-27",
        "2023-02-03",
        "2023-02-10",
    ]
)


def _predictions() -> pd.DataFrame:
    """构造六周、四只股票的确定性预测。"""

    rows = []
    for position, day in enumerate(DATES):
        for stock_position, stock in enumerate(STOCKS):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "score": float((position + 1) * (stock_position + 2)),
                }
            )
    return pd.DataFrame(rows)


def _manifest() -> dict:
    """构造最小预测 manifest。"""

    return {
        "model_id": MODEL_ID,
        "prediction_start": "2023-01-06",
        "prediction_end": "2023-02-10",
    }


def _data(index: pd.DatetimeIndex):
    """构造 provider 所需的最小行情索引。"""

    return type(
        "Data",
        (),
        {
            "index": index,
            "columns": pd.Index(STOCKS, name="stock_code"),
        },
    )()


def _mock_predictions(monkeypatch, frame: pd.DataFrame | None = None):
    """替换落盘预测读取。"""

    payload = _predictions() if frame is None else frame

    def loader(model_id):
        assert model_id == MODEL_ID
        return Path("/tmp") / model_id, _manifest(), payload

    monkeypatch.setattr(ml_rank, "load_predictions", loader)


def test_params_and_schema():
    """参数只接受模型 ID 和正整数 Top-N。"""

    provider = MlRankProvider()
    assert provider.validate_params({"model_id": MODEL_ID, "top_n": 3}) == {
        "model_id": MODEL_ID,
        "top_n": 3,
    }
    assert provider.validate_params({"model_id": MODEL_ID})["top_n"] == 30
    with pytest.raises(SpecError, match="model_id"):
        provider.validate_params({})
    with pytest.raises(SpecError, match="top_n"):
        provider.validate_params({"model_id": MODEL_ID, "top_n": 0})
    with pytest.raises(SpecError, match="未知字段"):
        provider.validate_params({"model_id": MODEL_ID, "smoothing_weeks": 2})

    schema = provider_param_schema(provider)
    assert [item["name"] for item in schema] == ["model_id", "top_n"]
    assert isinstance(schema[0]["default"], str) and schema[0]["default"]
    assert schema[1]["default"] == 30


def test_v1_and_v2_registration_and_explicit_dates(monkeypatch):
    """v1 保持严格周历，v2 可按显式日期避开无预测的自然周。"""

    assert get_provider(ProviderSpec("ml_rank", "1")).version == "1"
    assert get_provider(ProviderSpec("ml_rank", "2")).version == "2"
    provider = MlRankV2Provider()
    assert provider.validate_params(
        {
            "model_id": MODEL_ID,
            "top_n": 2,
            "rebalance_dates": ["2023-01-13", "2023-01-06", "2023-01-13"],
        }
    )["rebalance_dates"] == ["2023-01-06", "2023-01-13"]

    missing = _predictions().loc[
        pd.to_datetime(_predictions()["trade_date"]) != pd.Timestamp("2023-01-20")
    ]
    def loader(model_id):
        assert model_id == MODEL_ID
        return Path("/tmp") / model_id, _manifest(), missing

    monkeypatch.setattr(ml_rank_v2, "load_predictions", loader)
    index = pd.bdate_range("2023-01-02", "2023-01-27")
    weights = provider.build(
        _data(index),
        pd.DataFrame(True, index=index, columns=STOCKS),
        {
            "model_id": MODEL_ID,
            "top_n": 2,
            "rebalance_dates": ["2023-01-06", "2023-01-13"],
        },
    )
    assert weights.loc["2023-01-06"].notna().all()
    assert weights.loc["2023-01-13"].notna().all()
    assert weights.loc["2023-01-20"].isna().all()


def test_v2_smoothing_weeks_defaults_to_four_and_supports_raw(monkeypatch):
    """v2 默认保持四周平滑，窗口 1 只使用当日 rank。"""

    provider = MlRankV2Provider()
    assert provider.validate_params({"model_id": MODEL_ID})["smoothing_weeks"] == 4
    with pytest.raises(SpecError, match="smoothing_weeks"):
        provider.validate_params(
            {"model_id": MODEL_ID, "smoothing_weeks": 0}
        )

    _mock_predictions(monkeypatch)
    monkeypatch.setattr(
        ml_rank_v2,
        "load_predictions",
        ml_rank.load_predictions,
    )
    index = pd.bdate_range("2023-01-02", "2023-02-10")
    params = {
        "model_id": MODEL_ID,
        "top_n": 2,
        "smoothing_weeks": 1,
    }
    weights = provider.build(
        _data(index),
        pd.DataFrame(True, index=index, columns=STOCKS),
        params,
    )
    day = pd.Timestamp("2023-02-10")
    wide = _predictions().pivot(
        index="trade_date",
        columns="stock_code",
        values="score",
    )
    score = smooth_rank_scores(wide, wide.notna(), window=1)
    chosen = (
        score.loc[day]
        .dropna()
        .sort_values(ascending=False, kind="mergesort")
        .head(2)
        .index
    )
    assert weights.loc[day, chosen].tolist() == pytest.approx([0.5, 0.5])


def test_artifact_snapshot_extracts_model_and_errors(monkeypatch):
    """产物快照使用规范化参数，失败转换为回测数据错误。"""

    captured = {}

    def snapshot(model_id):
        captured["model_id"] = model_id
        return {"artifact_sha256": "a" * 64, "model_id": model_id}

    monkeypatch.setattr(ml_rank, "model_artifact_snapshot", snapshot)
    payload = MlRankProvider().artifact_snapshot({"model_id": MODEL_ID})
    assert payload["model_id"] == MODEL_ID
    assert captured["model_id"] == MODEL_ID

    def broken(model_id):
        raise AlphaArtifactError(f"broken {model_id}")

    monkeypatch.setattr(ml_rank, "model_artifact_snapshot", broken)
    with pytest.raises(DataAlignmentError, match="无法读取模型产物"):
        MlRankProvider().artifact_snapshot({"model_id": MODEL_ID})


def test_artifacts_enter_snapshot_and_change_snapshot_id(
    monkeypatch,
    make_data,
    tmp_path,
):
    """产物摘要进入 rules.provider，并参与最终快照 ID。"""

    monkeypatch.setattr(
        ml_rank,
        "model_artifact_snapshot",
        lambda model_id: {"artifact_sha256": "a" * 64, "model_id": model_id},
    )
    data = make_data()
    strategy = StrategySpec(
        name="ML artifact snapshot",
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=ProviderSpec(
                "ml_rank",
                "1",
                {"model_id": MODEL_ID, "top_n": 2},
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=2),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    first = build_input_snapshot(
        strategy,
        data_snapshot_from_market_data(data),
    )
    assert first.components["rules"]["provider"]["artifacts"]["model_id"] == MODEL_ID
    store = BacktestStore(tmp_path / "backtest.duckdb")
    assert store.record_snapshot(first.to_dict()) == first.snapshot_id
    restored = store.get_snapshot(first.snapshot_id)
    assert restored["components"]["rules"]["provider"]["artifacts"]["model_id"] == MODEL_ID

    monkeypatch.setattr(
        ml_rank,
        "model_artifact_snapshot",
        lambda model_id: {"artifact_sha256": "b" * 64, "model_id": model_id},
    )
    second = build_input_snapshot(
        strategy,
        data_snapshot_from_market_data(data),
    )
    assert second.snapshot_id != first.snapshot_id


def test_build_matches_four_week_causal_smoothing(monkeypatch):
    """权重等于当前及此前 3 个信号日的 rank 等权平均。"""

    _mock_predictions(monkeypatch)
    index = pd.bdate_range("2023-01-02", "2023-02-10")
    eligible = pd.DataFrame(True, index=index, columns=STOCKS)
    eligible.loc["2023-02-10", "SZ000002"] = False
    weights = MlRankProvider().build(
        _data(index),
        eligible,
        {"model_id": MODEL_ID, "top_n": 2},
    )

    signals = weekly_last_dates(index)
    wide = _predictions().pivot(
        index="trade_date",
        columns="stock_code",
        values="score",
    )
    score = smooth_rank_scores(wide, wide.notna(), window=4)
    day = pd.Timestamp("2023-02-10")
    chosen = (
        score.loc[day]
        .where(eligible.loc[day])
        .dropna()
        .sort_values(ascending=False, kind="mergesort")
        .head(2)
        .index
    )

    assert weights.loc[~weights.index.isin(signals)].isna().all().all()
    assert weights.loc[day, chosen].tolist() == pytest.approx([0.5, 0.5])
    assert weights.loc[day, "SZ000002"] == 0.0
    assert np.allclose(weights.loc[day].sum(), 1.0)


def test_build_is_chunk_independent(monkeypatch):
    """截取后段行情不改变同一天的目标权重。"""

    _mock_predictions(monkeypatch)
    provider = MlRankProvider()
    full_index = pd.bdate_range("2023-01-02", "2023-02-10")
    chunk_index = pd.bdate_range("2023-01-30", "2023-02-10")
    params = {"model_id": MODEL_ID, "top_n": 2}
    full_weights = provider.build(
        _data(full_index),
        pd.DataFrame(True, index=full_index, columns=STOCKS),
        params,
    )
    chunk_weights = provider.build(
        _data(chunk_index),
        pd.DataFrame(True, index=chunk_index, columns=STOCKS),
        params,
    )
    active = chunk_weights.notna().any(axis=1)
    pd.testing.assert_frame_equal(
        chunk_weights.loc[active],
        full_weights.loc[chunk_weights.index[active]],
        check_freq=False,
    )


def test_build_rejects_uncovered_signal_and_bad_artifact(monkeypatch):
    """预测覆盖不足和产物错误都转换成回测数据错误。"""

    _mock_predictions(monkeypatch)
    index = pd.bdate_range("2022-12-26", "2023-01-13")
    with pytest.raises(DataAlignmentError, match="超出模型预测覆盖"):
        MlRankProvider().build(
            _data(index),
            pd.DataFrame(True, index=index, columns=STOCKS),
            {"model_id": MODEL_ID, "top_n": 2},
        )

    missing = _predictions().loc[
        pd.to_datetime(_predictions()["trade_date"]) != pd.Timestamp("2023-01-20")
    ]
    _mock_predictions(monkeypatch, missing)
    index = pd.bdate_range("2023-01-16", "2023-01-27")
    with pytest.raises(DataAlignmentError, match="缺少回测信号日预测"):
        MlRankProvider().build(
            _data(index),
            pd.DataFrame(True, index=index, columns=STOCKS),
            {"model_id": MODEL_ID, "top_n": 2},
        )

    def broken_loader(model_id):
        raise AlphaArtifactError(f"broken {model_id}")

    monkeypatch.setattr(ml_rank, "load_predictions", broken_loader)
    with pytest.raises(DataAlignmentError, match="无法读取模型预测"):
        MlRankProvider().build(
            _data(index),
            pd.DataFrame(True, index=index, columns=STOCKS),
            {"model_id": MODEL_ID, "top_n": 2},
        )


def test_build_signal_targets_accepts_provider(monkeypatch, make_data):
    """provider 权重通过回测信号层的矩阵与可交易性校验。"""

    _mock_predictions(monkeypatch)
    index = pd.bdate_range("2023-01-02", "2023-02-10")
    columns = pd.Index(STOCKS, name="stock_code")
    data = make_data(
        dates=index,
        stocks=STOCKS,
        list_date=pd.Series(pd.Timestamp("2022-01-01"), index=columns),
    )
    strategy = StrategySpec(
        name="ML rank test",
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=ProviderSpec(
                "ml_rank",
                "1",
                {"model_id": MODEL_ID, "top_n": 2},
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=2),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    weights, ranks = build_signal_targets(data, strategy)
    signals = weekly_last_dates(index)

    assert weights.loc[signals].notna().all(axis=1).all()
    assert (weights.loc[signals] >= 0).all().all()
    assert np.allclose(weights.loc[signals].sum(axis=1), 1.0)
    assert ranks.loc[signals].notna().sum(axis=1).eq(2).all()
