"""策略版本与回测结果的落库行为。"""

import pandas as pd
import pytest

from backtest import BacktestParams, BacktestResult, StrategySpec
from backtest.builtin import BUILTINS, MOM20_ID
from backtest.errors import BacktestError
from backtest.specs import PortfolioSpec
from backtest.store import BacktestStore


def _result(daily=None, **overrides):
    """构造一个最小的回测结果，用于落库测试。"""
    fields = {
        "portfolio": object(),
        "daily": daily if daily is not None else pd.DataFrame({"nav": [1.0, 1.01]}, index=pd.date_range("2024-01-01", periods=2)),
        "orders": pd.DataFrame(),
        "positions": pd.DataFrame(),
        "holdings": pd.DataFrame(),
        "target_weights": pd.DataFrame(),
        "rejections": pd.DataFrame(),
        "metrics": {"total_return": 0.01},
        "metadata": {"strategy_hash": "abc", "universe": "all_a"},
    }
    fields.update(overrides)
    return BacktestResult(**fields)


@pytest.fixture
def store(tmp_path):
    """临时库上的回测存储。"""
    return BacktestStore(tmp_path / "backtest.duckdb")


def test_strategy_versions_are_immutable(store):
    first = StrategySpec()
    strategy_id, version = store.create_strategy(first)
    assert version == 1
    second = StrategySpec(name="MOM_20 Top30 v2")
    assert store.save_version(strategy_id, second) == 2
    assert store.get_version(strategy_id, 1) == first
    current = store.get_version(strategy_id)
    assert current.name == second.name
    assert current.version == 2
    assert store.list_strategies()[0]["current_version"] == 2


def test_ensure_builtins_is_idempotent(store):
    first = store.ensure_builtins()
    assert {row["strategy_id"] for row in first} == {item[0] for item in BUILTINS}
    version = store.get_version(MOM20_ID).version
    again = store.ensure_builtins()
    assert len(again) == len(first)
    assert store.get_version(MOM20_ID).version == version


def test_ensure_builtins_appends_version_when_definition_changes(store):
    store.ensure_builtins()
    changed = StrategySpec(name="MOM_20 Top30", portfolio=PortfolioSpec(top_n=20))
    version, created = store.ensure_strategy(MOM20_ID, changed, "改了 Top-N")
    assert created
    assert version == 2
    assert store.get_version(MOM20_ID, 1).portfolio.top_n == 30


def test_run_lifecycle_reports_status_and_list(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    assert store.get_run(run_id)["status"] == "running"
    store.finish_run(run_id, _result())
    assert store.get_run(run_id)["status"] == "success"
    runs = store.list_runs()
    assert runs[0]["run_id"] == run_id
    assert runs[0]["strategy_hash"] == "abc"
    assert runs[0]["metrics"] == {"total_return": 0.01}


def test_failed_and_cancelled_runs_are_recorded(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    failed = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    store.fail_run(failed, BacktestError("炸了"))
    assert store.get_run(failed)["status"] == "failed"
    assert "炸了" in store.get_run(failed)["error_message"]
    cancelled = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    store.cancel_run(cancelled)
    assert store.get_run(cancelled)["status"] == "cancelled"


def test_daily_table_keeps_trade_date_column(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    store.finish_run(run_id, _result())
    rows = store.get_result_table(run_id, "daily")
    assert "trade_date" in rows[0]
    assert len(rows) == 2


def test_target_weights_and_holdings_are_saved_as_long_tables(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    result = _result(
        target_weights=pd.DataFrame({"SH600000": [0.5, None]}, index=pd.date_range("2024-01-01", periods=2)),
        holdings=pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2024-01-01")],
                "stock_code": ["SH600000"],
                "market_value": [5000.0],
                "weight": [0.5],
            }
        ),
    )
    store.finish_run(run_id, result)
    assert store.get_result_table(run_id, "target_weight")[0]["stock_code"] == "SH600000"
    holding = store.get_result_table(run_id, "holding")[0]
    assert holding["stock_code"] == "SH600000"
    assert holding["weight"] == 0.5


def test_get_metrics_recomputes_from_daily(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_010_000.0],
            "cash": [0.0, 0.0],
            "nav": [1.0, 1.01],
            "return": [0.0, 0.01],
            "drawdown": [0.0, 0.0],
            "cost": [10.0, 10.0],
            "turnover": [0.0, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    store.finish_run(run_id, _result(daily=daily))
    metrics = store.get_metrics(run_id)
    assert metrics["total_return"] == pytest.approx(0.01)
    assert metrics["trading_days"] == 2


def test_get_metrics_without_daily_raises(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    with pytest.raises(BacktestError, match="没有日结果"):
        store.get_metrics(run_id)


def test_unknown_result_table_lists_options(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    with pytest.raises(BacktestError, match="不支持的回测结果表"):
        store.get_result_table(run_id, "unknown")

