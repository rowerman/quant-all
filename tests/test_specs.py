import json

import pytest

from backtest.specs import BacktestParams, ExecutionSpec, PortfolioSpec, SpecError, StrategySpec


def test_strategy_json_round_trip_is_stable():
    spec = StrategySpec()
    restored = StrategySpec.from_dict(json.loads(spec.to_json()))
    assert restored == spec
    assert restored.config_hash == spec.config_hash


def test_strategy_rejects_unknown_field():
    with pytest.raises(SpecError, match="未知字段"):
        StrategySpec.from_dict({"unknown": 1})


def test_backtest_params_validate_dates_and_cash():
    with pytest.raises(SpecError):
        BacktestParams(start="2024-02-01", end="2024-01-01")
    with pytest.raises(SpecError):
        BacktestParams(start="2024-01-01", end="2024-01-02", init_cash=0)


def test_weighting_supports_market_cap():
    assert PortfolioSpec(weighting="market_cap").weighting == "market_cap"
    with pytest.raises(SpecError, match="weighting"):
        PortfolioSpec(weighting="risk_parity")


def test_universe_accepts_index_pool():
    assert StrategySpec(universe="index:000905.SH").universe == "index:000905.SH"
    with pytest.raises(SpecError, match="universe"):
        StrategySpec(universe="hs300")


def test_multi_factor_weights_must_match_factors():
    from backtest.specs import SignalSpec

    spec = SignalSpec(factors=("MOM_20", "EP_TTM", "VOL_20"), weights=(1.0, 1.0, -1.0))
    assert spec.weights == (1.0, 1.0, -1.0)
    with pytest.raises(SpecError, match="weights"):
        SignalSpec(factors=("MOM_20", "VOL_20"), weights=(1.0,))


def test_index_universe_survives_json_round_trip():
    spec = StrategySpec(universe="index:000300.SH", portfolio=PortfolioSpec(weighting="market_cap"))
    restored = StrategySpec.from_dict(json.loads(spec.to_json()))
    assert restored == spec
    assert restored.config_hash == spec.config_hash


def test_execution_lot_size_is_fixed():
    assert ExecutionSpec().lot_size == 100
    with pytest.raises(SpecError, match="100 股"):
        ExecutionSpec(lot_size=10)
