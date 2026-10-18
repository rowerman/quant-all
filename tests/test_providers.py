"""目标权重提供者的注册、配置和权重校验。"""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest.providers as providers
from backtest.errors import DataAlignmentError, SpecError
from backtest.engine import run_backtest
from backtest.providers import (
    get_provider,
    list_providers,
    provider_metadata,
    register_provider,
    signal_metadata,
    validate_strategy_provider,
)
from backtest.signals import (
    build_signal_targets,
    eligible_mask,
    rank_provider_weights,
    validate_provider_weights,
)
from backtest.specs import (
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    RiskSpec,
    SignalSpec,
    BacktestParams,
    StrategySpec,
)


class TopTwoProvider:
    """按 MOM_20 每日选前两只等权。"""

    name = "test_top_two"
    version = "1"
    description = "测试用 Top-2 等权"
    required_factors = ("MOM_20",)

    def build(self, data, eligible, params):
        score = data.factors["MOM_20"].astype("float64")
        weights = pd.DataFrame(
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        for day in data.index:
            chosen = (
                score.loc[day]
                .where(eligible.loc[day])
                .dropna()
                .sort_values(ascending=False)
                .head(2)
                .index
            )
            weights.loc[day] = 0.0
            if len(chosen):
                weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


register_provider(TopTwoProvider())


class CashNanProvider:
    """覆盖现金、NaN 行和全 0 行的测试提供者。"""

    name = "test_cash_nan"
    version = "1"
    description = "测试用现金/NaN/清仓目标"
    required_factors = ()

    def build(self, data, eligible, params):
        weights = pd.DataFrame(
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        if len(data.index):
            weights.iloc[0] = 0.0
            weights.iloc[0, 0] = 0.5
        if len(data.index) > 2:
            weights.iloc[2] = 0.0
        if len(data.index) > 3:
            weights.iloc[3] = 0.0
            weights.iloc[3, 1] = 0.25
        return weights


register_provider(CashNanProvider())


def _provider_class(module: str, name: str):
    """构造指定模块来源的测试提供者类。"""
    return type(
        "NamedProvider",
        (),
        {
            "__module__": module,
            "name": name,
            "version": "1",
            "description": "测试来源",
            "required_factors": (),
            "build": lambda self, data, eligible, params: pd.DataFrame(),
        },
    )


def _unregister(name: str) -> None:
    """移除测试临时注册的提供者。"""
    providers._REGISTRY.pop(name, None)
    providers._ORIGINS.pop(name, None)


def _provider_strategy(**params) -> StrategySpec:
    return StrategySpec(
        name="测试提供者",
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec("test_top_two", "1", params),
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
    )


def _frame(make_data):
    return make_data(
        factors={
            "MOM_20": pd.DataFrame(
                [[3.0, 2.0, 1.0]] * 4,
                index=pd.date_range("2024-01-29", periods=4, freq="B"),
                columns=["SH600000", "SZ000001", "SZ000002"],
                dtype="float32",
            )
        }
    )


def test_provider_config_round_trip_and_hash():
    spec = _provider_strategy(top_n=2)
    restored = StrategySpec.from_dict(json.loads(spec.to_json()))
    assert restored == spec
    assert restored.config_hash == spec.config_hash
    assert restored.config_hash != _provider_strategy(top_n=3).config_hash


def test_provider_config_requires_provider_mode():
    with pytest.raises(SpecError, match="selection"):
        StrategySpec(
            signal=SignalSpec(
                factors=("MOM_20",),
                weights=(),
                provider=ProviderSpec("test_top_two"),
            )
        )
    with pytest.raises(SpecError, match="custom"):
        StrategySpec(
            signal=SignalSpec(
                factors=("MOM_20",),
                weights=(),
                provider=ProviderSpec("test_top_two"),
            ),
            portfolio=PortfolioSpec(selection="provider"),
        )
    with pytest.raises(SpecError, match="weights"):
        SignalSpec(
            factors=("MOM_20",),
            weights=(1.0,),
            provider=ProviderSpec("test_top_two"),
        )
    with pytest.raises(SpecError, match="custom"):
        StrategySpec(rebalance=RebalanceSpec(frequency="custom"))


def test_provider_registry_and_metadata():
    provider = get_provider(ProviderSpec("test_top_two"))
    assert provider.name == "test_top_two"
    assert "test_top_two" in {item["name"] for item in list_providers()}
    metadata = provider_metadata(ProviderSpec("test_top_two"))
    assert metadata["required_factors"] == ["MOM_20"]
    assert len(metadata["code_sha256"]) == 64
    assert signal_metadata(_provider_strategy().signal)["kind"] == "provider"
    assert signal_metadata(StrategySpec().signal)["kind"] == "factor"


def test_provider_duplicate_error_includes_both_modules_and_logs():
    """重复注册要指出两个来源模块，并记录名称、版本和模块。"""
    first = _provider_class("providers.alpha", "duplicate_origin")
    second = _provider_class("providers.beta", "duplicate_origin")
    try:
        register_provider(first)
        with pytest.raises(SpecError, match=r"providers\.alpha.*providers\.beta"):
            register_provider(second)

        messages = []

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = Capture()
        providers.logger.addHandler(handler)
        try:
            register_provider(_provider_class("providers.gamma", "logged_provider"))
        finally:
            providers.logger.removeHandler(handler)
        assert any(
            "logged_provider" in message
            and "v1" in message
            and "providers.gamma" in message
            for message in messages
        )
    finally:
        _unregister("duplicate_origin")
        _unregister("logged_provider")


def test_provider_import_failure_includes_module_and_cause(monkeypatch):
    """提供者模块导入失败要给出模块名、异常类型和原始原因。"""
    real_import = providers.importlib.import_module

    def broken_import(name, *args, **kwargs):
        if name == "backtest.strategies.broken_test":
            raise RuntimeError("模块初始化失败")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(providers, "_DISCOVERED", False)
    monkeypatch.setattr(
        providers.pkgutil,
        "iter_modules",
        lambda path: [SimpleNamespace(name="broken_test")],
    )
    monkeypatch.setattr(providers.importlib, "import_module", broken_import)
    with pytest.raises(
        SpecError,
        match=r"backtest\.strategies\.broken_test.*RuntimeError.*模块初始化失败",
    ):
        providers.discover_providers()


def test_momentum_timing_sample_provider(make_data):
    data = _frame(make_data)
    provider = get_provider(ProviderSpec("momentum_timing"))
    weights = provider.build(
        data, eligible_mask(data), {"top_n": 2, "min_breadth": 0.5}
    )
    assert np.allclose(weights.iloc[0].to_numpy(), [0.5, 0.5, 0.0])
    with pytest.raises(SpecError, match="top_n"):
        provider.build(data, eligible_mask(data), {"top_n": 0})
    with pytest.raises(SpecError, match="min_breadth"):
        provider.build(data, eligible_mask(data), {"min_breadth": 1.5})


def test_provider_unknown_version_and_missing_factor():
    with pytest.raises(SpecError, match="未注册"):
        get_provider(ProviderSpec("does_not_exist"))
    with pytest.raises(SpecError, match="版本不匹配"):
        get_provider(ProviderSpec("test_top_two", "2"))
    spec = _provider_strategy()
    bad = StrategySpec(
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=spec.signal.provider,
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    with pytest.raises(SpecError, match="需要因子"):
        validate_strategy_provider(bad)


def test_provider_weight_validation(make_data):
    data = _frame(make_data)
    good = pd.DataFrame(
        [[0.5, 0.5, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        index=data.index,
        columns=data.columns,
    )
    checked = validate_provider_weights(good, data)
    assert checked.loc[data.index[0], "SH600000"] == 0.5
    assert checked.iloc[1].isna().all()

    partial = good.copy()
    partial.iloc[0, 1] = np.nan
    with pytest.raises(DataAlignmentError, match="部分 NaN"):
        validate_provider_weights(partial, data)
    negative = good.copy()
    negative.iloc[0, 0] = -0.1
    with pytest.raises(DataAlignmentError, match="负权重"):
        validate_provider_weights(negative, data)
    oversized = good.copy()
    oversized.iloc[0, :] = [0.7, 0.7, 0.0]
    with pytest.raises(DataAlignmentError, match="超过 1"):
        validate_provider_weights(oversized, data)
    infinite = good.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(DataAlignmentError, match="无穷"):
        validate_provider_weights(infinite, data)
    wrong_shape = good.rename(columns={"SH600000": "SH600001"})
    with pytest.raises(DataAlignmentError, match="不一致"):
        validate_provider_weights(wrong_shape, data)


def test_provider_rejects_ineligible_positive_weight(make_data):
    data = _frame(make_data)
    suspended = data.is_suspended.copy()
    suspended.iloc[0, 0] = True
    data = __import__("dataclasses").replace(data, is_suspended=suspended)
    weights = pd.DataFrame(
        0.0, index=data.index, columns=data.columns, dtype="float64"
    )
    weights.iloc[0, 0] = 1.0
    with pytest.raises(DataAlignmentError, match="不可买入"):
        validate_provider_weights(weights, data)


def test_provider_ranks_follow_positive_weights():
    frame = pd.DataFrame(
        [[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]],
        columns=["A", "B", "C"],
    )
    ranks = rank_provider_weights(frame)
    assert ranks.loc[0, "B"] == 1.0
    assert ranks.loc[0, "A"] == 2.0
    assert np.isnan(ranks.loc[0, "C"])
    assert ranks.loc[1].isna().all()


def test_build_signal_targets_dispatches_provider(make_data):
    data = _frame(make_data)
    weights, ranks = build_signal_targets(data, _provider_strategy())
    assert np.allclose(weights.iloc[0].to_numpy(), [0.5, 0.5, 0.0])
    assert ranks.loc[data.index[0], "SH600000"] == 1.0
    with pytest.raises(SpecError, match="rebalance_days"):
        build_signal_targets(
            data,
            _provider_strategy(),
            pd.Series(True, index=data.index),
        )


def test_provider_runs_through_engine_and_risk_cap(make_data):
    data = _frame(make_data)
    strategy = StrategySpec(
        name="测试提供者风控",
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec("test_top_two"),
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
        risk=RiskSpec(max_single_weight=0.4),
    )
    params = BacktestParams(
        str(data.index[0].date()), str(data.index[-1].date()), 1_000_000.0
    )
    result = run_backtest(data, strategy, params)
    assert result.metadata["signal"]["kind"] == "provider"
    assert result.metadata["signal"]["name"] == "test_top_two"
    assert not result.orders.empty
    assert float(np.nanmax(result.target_weights.to_numpy())) <= 0.4 + 1e-9
    assert result.metadata["calibration"]["single_cap_days"] > 0


def test_provider_cash_nan_and_flat_rows_feed_report(make_data):
    data = _frame(make_data)
    strategy = StrategySpec(
        name="测试提供者现金与空行",
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            standardize=False,
            provider=ProviderSpec("test_cash_nan"),
        ),
        portfolio=PortfolioSpec(selection="provider"),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    params = BacktestParams(
        str(data.index[0].date()), str(data.index[-1].date()), 1_000_000.0
    )
    result = run_backtest(data, strategy, params)
    assert "turnover_attribution" in result.metrics
    assert "cost_attribution" in result.metrics
    assert "industry_exposure" in result.metrics
    assert result.metrics["industry_exposure"]["available"] is False
    assert result.metrics["turnover_attribution"]["one_way_annual"] <= (
        result.metrics["turnover_attribution"]["two_way_annual"] + 1e-9
    )
