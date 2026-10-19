"""策略版本与回测结果的落库行为。"""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import duckdb
import pandas as pd
import pytest

from backtest import BacktestParams, BacktestResult, StrategySpec
from backtest.builtin import BUILTINS, MOM20_ID
from backtest.errors import (
    BacktestError,
    RunConflictError,
    RunNotFoundError,
    SnapshotConflictError,
)
from backtest.providers import provider_metadata
from backtest.specs import (
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
)
from backtest.store import RESULT_TABLES, BacktestStore


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


def _provider_strategy(name: str = "提供者绑定测试") -> StrategySpec:
    return StrategySpec(
        name=name,
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            provider=ProviderSpec("momentum_timing"),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=30),
        rebalance=RebalanceSpec(frequency="custom"),
    )


def test_create_provider_strategy_binds_source_revision(store):
    spec = _provider_strategy()
    strategy_id, version = store.create_strategy(spec)
    metadata = provider_metadata(spec.signal.provider)
    with store._connect() as conn:
        row = conn.execute(
            "SELECT source_sha256, module FROM bt.provider_revision "
            "WHERE provider_name = ? AND provider_version = ?",
            [metadata["name"], metadata["version"]],
        ).fetchone()
    assert row == (metadata["code_sha256"], metadata["module"])
    assert store.get_version(strategy_id, version) == spec


def test_provider_revision_conflict_rolls_back_strategy(store):
    spec = _provider_strategy("源码冲突策略")
    metadata = provider_metadata(spec.signal.provider)
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO bt.provider_revision VALUES (?, ?, ?, ?, ?, ?)",
            [
                metadata["name"],
                metadata["version"],
                "f" * 64,
                metadata["module"],
                '["MOM_20"]',
                store._now(),
            ],
        )
    with pytest.raises(SnapshotConflictError, match="提升 version"):
        store.create_strategy(spec)
    assert spec.name not in {row["name"] for row in store.list_strategies()}


def test_record_snapshot_rolls_back_on_provider_conflict(store):
    snapshot = {
        "snapshot_id": "d" * 64,
        "schema_version": "backtest_input_snapshot_v5",
        "created_at": "2026-10-19T00:00:00+00:00",
        "scope": {"start": "2024-01-01", "end": "2024-01-05", "universe": "all_a"},
        "components": {
            "rules": {
                "provider": {
                    "name": "momentum_timing",
                    "version": "1",
                    "source_sha256": "e" * 64,
                    "module": "backtest.strategies.momentum_timing",
                    "required_factors": ["MOM_20"],
                }
            }
        },
    }
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO bt.provider_revision VALUES (?, ?, ?, ?, ?, ?)",
            [
                "momentum_timing",
                "1",
                "f" * 64,
                "backtest.strategies.momentum_timing",
                '["MOM_20"]',
                store._now(),
            ],
        )
    with pytest.raises(SnapshotConflictError, match="提升 version"):
        store.record_snapshot(snapshot)
    with store._connect() as conn:
        count = conn.execute(
            "SELECT count(*) FROM bt.input_snapshot WHERE snapshot_id = ?",
            [snapshot["snapshot_id"]],
        ).fetchone()[0]
    assert count == 0


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


def test_finish_run_rejects_missing_and_finished_runs(store):
    """完成写入只能发生一次，缺失或已结束的运行不能被追加结果。"""
    with pytest.raises(RunNotFoundError, match="不存在"):
        store.finish_run("missing", _result())

    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    store.finish_run(run_id, _result())
    with pytest.raises(RunConflictError, match="不能重复写入"):
        store.finish_run(run_id, _result())
    assert len(store.get_result_table(run_id, "daily")) == 2


def test_terminal_status_is_not_rewritten_by_late_cleanup(store):
    """成功、失败和取消都是终态，迟到的清理不能覆盖已有状态。"""
    strategy_id, version = store.create_strategy(StrategySpec())

    success = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    store.finish_run(success, _result())
    store.fail_run(success, BacktestError("迟到的失败"))
    store.cancel_run(success)
    record = store.get_run(success)
    assert record["status"] == "success"
    assert record["error_message"] is None
    assert record["metadata_json"] is not None

    failed = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    store.fail_run(failed, BacktestError("原始失败"))
    store.cancel_run(failed)
    assert store.get_run(failed)["status"] == "failed"
    with pytest.raises(RunConflictError, match="当前状态为 failed"):
        store.finish_run(failed, _result())

    cancelled = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    store.cancel_run(cancelled)
    store.fail_run(cancelled, BacktestError("迟到的失败"))
    assert store.get_run(cancelled)["status"] == "cancelled"


def test_concurrent_finish_writes_results_once(store):
    """两个线程同时完成同一运行时，只有一个能提交结果。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    barrier = threading.Barrier(2)

    def finish():
        barrier.wait()
        try:
            store.finish_run(run_id, _result())
            return "success"
        except RunConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: finish(), range(2)))

    assert sorted(outcomes) == ["conflict", "success"]
    assert store.get_run(run_id)["status"] == "success"
    assert len(store.get_result_table(run_id, "daily")) == 2


def test_finish_run_rolls_back_every_result_table(store, monkeypatch):
    """任一张结果表写入失败时，本次运行的其余结果与成功状态都要回滚。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05")
    )
    orders = pd.DataFrame(
        {
            "stock_code": ["SH600000"],
            "trade_date": [pd.Timestamp("2024-01-02")],
            "Size": [100.0],
        }
    )
    original = BacktestStore._write_frame

    def fail_on_order(conn, table, frame):
        if table == "order":
            raise RuntimeError("注入的落库失败")
        return original(conn, table, frame)

    monkeypatch.setattr(BacktestStore, "_write_frame", staticmethod(fail_on_order))
    with pytest.raises(RuntimeError, match="注入的落库失败"):
        store.finish_run(run_id, _result(orders=orders))

    with store._connect() as conn:
        for table in RESULT_TABLES:
            count = conn.execute(
                f"SELECT count(*) FROM bt.{table} WHERE run_id = ?", [run_id]
            ).fetchone()[0]
            assert count == 0
    assert store.get_run(run_id)["status"] == "running"
    store.fail_run(run_id, BacktestError("落库失败"))
    assert store.get_run(run_id)["status"] == "failed"


def test_run_lifecycle_persists_snapshot_id(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    snapshot = {
        "snapshot_id": "a" * 64,
        "schema_version": "backtest_input_snapshot_v1",
        "created_at": "2026-10-18T00:00:00+00:00",
        "scope": {"start": "2024-01-01", "end": "2024-01-05", "universe": "all_a"},
        "components": {"market": {"digest": "m"}},
    }
    store.record_snapshot(snapshot)
    run_id = store.start_run(
        strategy_id,
        version,
        BacktestParams("2024-01-01", "2024-01-05"),
        snapshot_id=snapshot["snapshot_id"],
    )
    store.finish_run(run_id, _result(metadata={"snapshot_id": snapshot["snapshot_id"]}))
    record = store.get_run(run_id)
    assert record["snapshot_id"] == snapshot["snapshot_id"]
    assert store.list_runs()[0]["snapshot_id"] == snapshot["snapshot_id"]
    assert store.get_snapshot(snapshot["snapshot_id"])["scope"]["universe"] == "all_a"


def test_snapshot_registration_is_immutable(store):
    snapshot = {
        "snapshot_id": "b" * 64,
        "schema_version": "backtest_input_snapshot_v1",
        "created_at": "2026-10-18T00:00:00+00:00",
        "scope": {"start": "2024-01-01", "end": "2024-01-05", "universe": "all_a"},
        "components": {"market": {"digest": "m"}},
    }
    store.record_snapshot(snapshot)
    store.record_snapshot({**snapshot, "created_at": "2026-10-18T01:00:00+00:00"})
    with pytest.raises(SnapshotConflictError, match="内容不同"):
        store.record_snapshot(
            {**snapshot, "components": {"market": {"digest": "changed"}}}
        )


def test_store_migrates_old_run_table_for_snapshot_id(tmp_path):
    path = tmp_path / "backtest.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA bt")
    con.execute(
        """CREATE TABLE bt.run (
            run_id VARCHAR PRIMARY KEY,
            strategy_id VARCHAR,
            strategy_version INTEGER,
            start_date DATE,
            end_date DATE,
            init_cash DOUBLE,
            status VARCHAR,
            metadata_json VARCHAR,
            error_message VARCHAR,
            started_at TIMESTAMP,
            finished_at TIMESTAMP
        )"""
    )
    con.close()

    store = BacktestStore(path)
    with store._connect() as conn:
        columns = {row[0] for row in conn.execute("DESCRIBE bt.run").fetchall()}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'bt'"
            ).fetchall()
        }
    assert {"universe", "benchmark", "snapshot_id"} <= columns
    assert {"input_snapshot", "provider_revision"} <= tables


def test_delete_run_removes_run_and_all_result_rows(store):
    """删除已结束运行时，所有结果表不能留下孤立行。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    store.finish_run(run_id, _result())
    with store._connect() as conn:
        for table in ("position", "order", "rejection", "target_weight", "holding", "liquidation", "allocation", "corporate_action"):
            conn.execute(f"INSERT INTO bt.{table} (run_id) VALUES (?)", [run_id])

    store.delete_run(run_id)

    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM bt.run WHERE run_id = ?", [run_id]).fetchone()[0] == 0
        for table in ("daily", "position", "order", "rejection", "target_weight", "holding", "liquidation", "allocation", "corporate_action"):
            assert conn.execute(f"SELECT count(*) FROM bt.{table} WHERE run_id = ?", [run_id]).fetchone()[0] == 0
    with pytest.raises(RunNotFoundError):
        store.get_run(run_id)


def test_delete_run_rejects_running_and_missing_runs(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    running = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    with pytest.raises(RunConflictError, match="仍在执行"):
        store.delete_run(running)
    with pytest.raises(RunNotFoundError, match="不存在"):
        store.delete_run("missing")


def test_delete_run_accepts_failed_and_cancelled_runs(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    failed = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    store.fail_run(failed, BacktestError("失败"))
    cancelled = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    store.cancel_run(cancelled)

    store.delete_run(failed)
    store.delete_run(cancelled)

    with pytest.raises(RunNotFoundError):
        store.get_run(failed)
    with pytest.raises(RunNotFoundError):
        store.get_run(cancelled)


def test_run_records_universe_and_benchmark(store):
    """股票池与基准要落成列，SQL 里能直接筛。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"),
        universe="index:000905.SH", benchmark="H00905.CSI",
    )
    record = store.get_run(run_id)
    assert record["universe"] == "index:000905.SH"
    assert record["benchmark"] == "H00905.CSI"
    listed = store.list_runs()[0]
    assert listed["universe"] == "index:000905.SH"
    assert listed["benchmark"] == "H00905.CSI"


def test_legacy_run_table_gets_new_columns(tmp_path):
    """老库的 bt.run 没有这两列：打开时自动补列，列表回落到元数据。"""
    path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS bt")
        conn.execute(
            """CREATE TABLE bt.run (
                run_id VARCHAR PRIMARY KEY, strategy_id VARCHAR, strategy_version INTEGER,
                start_date DATE, end_date DATE, init_cash DOUBLE, status VARCHAR,
                metadata_json VARCHAR, error_message VARCHAR, started_at TIMESTAMP,
                finished_at TIMESTAMP
            )"""
        )
        conn.execute(
            "INSERT INTO bt.run VALUES ('legacy', 'sid', 1, '2024-01-01', '2024-01-05', 100000.0, "
            "'success', ?, NULL, '2024-01-06 00:00:00', '2024-01-06 00:01:00')",
            [json.dumps({"universe": "all_a", "benchmark": {"code": "H00300.CSI"}})],
        )
    listed = BacktestStore(path).list_runs()[0]
    assert listed["universe"] == "all_a"
    assert listed["benchmark"] == "H00300.CSI"


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


def test_allocation_table_is_saved(store):
    """资金分配明细按逐格落库，供前端与自检查「补一手 / 未建仓」的格子。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    allocations = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "stock_code": ["SH600000"],
            "target_weight": [0.5],
            "price": [10.0],
            "baseline_shares": [0.0],
            "allocated_shares": [4900.0],
            "status": ["allocated"],
        }
    )
    store.finish_run(run_id, _result(allocations=allocations))
    row = store.get_result_table(run_id, "allocation")[0]
    assert row["stock_code"] == "SH600000"
    assert row["allocated_shares"] == 4900.0
    assert row["status"] == "allocated"


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


def test_cost_components_are_persisted_and_old_daily_still_summarizes(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 999_000.0],
            "cash": [0.0, 0.0],
            "nav": [1.0, 0.999],
            "return": [0.0, -0.001],
            "drawdown": [0.0, -0.001],
            "cost": [20.0, 10.0],
            "commission": [15.0, 5.0],
            "stamp_tax": [4.0, 4.0],
            "transfer_fee": [1.0, 1.0],
            "dividend_tax": [0.0, 0.0],
            "slippage_cost": [2.0, 1.0],
            "total_cost": [22.0, 11.0],
            "turnover": [0.0, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    orders = pd.DataFrame(
        {
            "stock_code": ["SH600000"],
            "trade_date": [pd.Timestamp("2024-01-02")],
            "Size": [100.0],
            "Price": [10.0],
            "Fees": [10.0],
            "Side": ["Sell"],
            "Board": ["主板"],
            "Order Valid": [True],
            "reference_price": [10.0],
            "gross_amount": [1000.0],
            "commission": [5.0],
            "stamp_tax": [4.0],
            "transfer_fee": [1.0],
            "slippage_rate": [0.0005],
            "slippage_cost": [0.5],
            "total_cost": [10.5],
        }
    )
    store.finish_run(run_id, _result(daily=daily, orders=orders))
    daily_rows = store.get_result_table(run_id, "daily")
    assert {"commission", "stamp_tax", "transfer_fee", "slippage_cost", "total_cost"} <= set(
        daily_rows[0]
    )
    order_rows = store.get_result_table(run_id, "order")
    assert {"stamp_tax", "transfer_fee", "slippage_cost", "total_cost"} <= set(order_rows[0])
    metrics = store.get_metrics(run_id)
    assert metrics["total_cost"] == pytest.approx(33.0)
    assert metrics["cash_cost"] == pytest.approx(30.0)


def test_get_metrics_without_daily_raises(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    with pytest.raises(BacktestError, match="没有日结果"):
        store.get_metrics(run_id)


def test_industry_exposure_is_persisted_and_metrics_use_run_risk_free_rate(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_010_000.0],
            "cash": [0.0, 0.0],
            "nav": [1.0, 1.01],
            "return": [0.0, 0.01],
            "drawdown": [0.0, 0.0],
            "cost": [0.0, 0.0],
            "turnover": [0.0, 0.01],
            "turnover_one_way": [0.0, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    industry = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "industry": ["SW2014 801010 农林牧渔"],
            "market_value": [500_000.0],
            "weight": [0.5],
            "is_cash": [False],
        }
    )
    store.finish_run(
        run_id,
        _result(
            daily=daily,
            industry_exposure=industry,
            metadata={
                "strategy_hash": "abc",
                "universe": "all_a",
                "params": {"risk_free_rate": 0.03},
            },
        ),
    )
    rows = store.get_result_table(run_id, "industry_exposure")
    assert rows[0]["industry"] == "SW2014 801010 农林牧渔"
    metrics = store.get_metrics(run_id)
    assert metrics["risk_free_rate"] == pytest.approx(0.03)


def test_unknown_result_table_lists_options(store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    with pytest.raises(BacktestError, match="不支持的回测结果表"):
        store.get_result_table(run_id, "unknown")

