import json

import pytest

from backtest.specs import (
    BacktestParams,
    BenchmarkSpec,
    ExecutionSpec,
    PortfolioSpec,
    SpecError,
    StrategySpec,
)
from backtest.pipeline import RunRequest, effective_spec


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


def test_execution_cash_buffer_is_configurable_and_bounded():
    """执行层预留的现金缓冲默认 2%，越界直接报错。"""
    assert ExecutionSpec().cash_buffer == 0.02
    assert ExecutionSpec(cash_buffer=0.0).cash_buffer == 0.0
    with pytest.raises(SpecError, match="cash_buffer"):
        ExecutionSpec(cash_buffer=0.5)
    with pytest.raises(SpecError, match="cash_buffer"):
        ExecutionSpec(cash_buffer=-0.01)


def test_cash_buffer_enters_config_hash():
    """缓冲是执行口径的一部分，必须体现在策略哈希里。"""
    assert StrategySpec(execution=ExecutionSpec(cash_buffer=0.0)).config_hash != StrategySpec().config_hash


def test_default_benchmark_is_total_return():
    """默认超额必须与含分红的组合收益使用同口径基准。"""
    assert BenchmarkSpec().index_code == "H00300.CSI"
    assert StrategySpec().benchmark.index_code == "H00300.CSI"
    with pytest.raises(SpecError, match="benchmark.index_code"):
        BenchmarkSpec(index_code="")


def test_run_request_can_override_total_return_with_price_index():
    """运行级覆盖仍直接使用指数代码，不改变已保存策略版本。"""
    original = StrategySpec()
    request = RunRequest(
        strategy_id="demo", start="2024-01-01", end="2024-01-31", benchmark="000300.SH"
    )
    effective = effective_spec(original, request)
    assert original.benchmark.index_code == "H00300.CSI"
    assert effective.benchmark.index_code == "000300.SH"
