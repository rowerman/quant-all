"""回测输入快照的稳定性与分项敏感性。"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import backtest.providers as providers
import backtest.snapshot as snapshot_module
from backtest.config import PROJECT_ROOT
from backtest.errors import DataAlignmentError, SnapshotConflictError
from backtest.pipeline import RunRequest, run
from backtest.snapshot import (
    EXECUTION_MODEL_VERSION,
    EXECUTION_SOURCE_FILES,
    SNAPSHOT_SCHEMA_VERSION,
    _execution_model_component,
    build_data_snapshot,
    build_input_snapshot,
    data_snapshot_from_market_data,
)
from backtest.specs import (
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.store import BacktestStore


@pytest.fixture
def snapshot_db(tmp_path):
    """从主库复制一段最小行情，供快照范围测试使用。"""
    source = PROJECT_ROOT / "db" / "quant.duckdb"
    target = tmp_path / "quant.duckdb"
    con = duckdb.connect(str(target))
    con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
    statements = [
        "CREATE SCHEMA md",
        "CREATE SCHEMA ref",
        "CREATE SCHEMA meta",
        "CREATE TABLE md.daily_bar AS SELECT * FROM src.md.daily_bar "
        "WHERE trade_date BETWEEN DATE '2026-07-01' AND DATE '2026-09-30'",
        "CREATE TABLE md.calendar AS SELECT * FROM src.md.calendar",
        "CREATE TABLE ref.stock_basic AS SELECT * FROM src.ref.stock_basic",
        "CREATE TABLE ref.corporate_action AS SELECT * FROM src.ref.corporate_action "
        "WHERE ex_date BETWEEN DATE '2026-07-01' AND DATE '2026-09-30'",
        "CREATE TABLE ref.sw_industry_member AS SELECT * FROM src.ref.sw_industry_member",
        "CREATE TABLE md.index_daily AS SELECT * FROM src.md.index_daily "
        "WHERE index_code = 'H00300.CSI' "
        "AND trade_date BETWEEN DATE '2026-07-01' AND DATE '2026-09-30'",
        "CREATE TABLE ref.index_basic AS SELECT * FROM src.ref.index_basic",
        "CREATE TABLE ref.index_weight AS SELECT * FROM src.ref.index_weight "
        "WHERE index_code = '000300.SH' AND trade_date <= DATE '2026-09-30'",
        "CREATE TABLE meta.sw_industry_manifest AS SELECT * FROM src.meta.sw_industry_manifest",
        "CREATE TABLE meta.manifest AS SELECT * FROM src.meta.manifest",
        "CREATE TABLE meta.corporate_action_manifest AS SELECT * FROM src.meta.corporate_action_manifest",
    ]
    for statement in statements:
        con.execute(statement)
    con.execute("DETACH src")
    con.close()
    return target


def _snapshot(db_path: Path, factor_dir: Path | None = None, strategy=None):
    """按测试固定区间构建完整快照。"""
    strategy = strategy or StrategySpec(universe="index:000300.SH")
    data = build_data_snapshot(
        "2026-08-03",
        "2026-08-14",
        list(strategy.signal.factors),
        universe=strategy.universe,
        db_path=db_path,
        factor_dir=factor_dir,
        benchmark_code=strategy.benchmark.index_code,
        membership_end="2026-08-14",
        industry_level="L1" if strategy.risk.max_industry_weight is not None else None,
    )
    return build_input_snapshot(strategy, data)


def _mutable_connection(db_path: Path):
    """打开可写连接。"""
    return duckdb.connect(str(db_path))


def test_snapshot_is_deterministic_and_ignores_out_of_window_rows(snapshot_db):
    first = _snapshot(snapshot_db)
    second = _snapshot(snapshot_db)
    assert first.schema_version == SNAPSHOT_SCHEMA_VERSION
    assert first.snapshot_id == second.snapshot_id
    assert first.components == second.components

    execution = first.components["rules"]["execution_model"]
    assert execution["version"] == EXECUTION_MODEL_VERSION
    assert execution["backend"] == "ledger"
    assert execution["mode"] == "causal_segmented"
    assert set(execution["source_sha256"]) == set(EXECUTION_SOURCE_FILES)
    assert all(len(value) == 64 for value in execution["source_sha256"].values())
    assert len(execution["source_set_sha256"]) == 64

    con = _mutable_connection(snapshot_db)
    con.execute(
        "CREATE TABLE _out AS SELECT * FROM md.daily_bar "
        "WHERE trade_date = DATE '2026-08-14' LIMIT 1"
    )
    con.execute("UPDATE _out SET trade_date = DATE '2026-10-05'")
    con.execute("INSERT INTO md.daily_bar SELECT * FROM _out")
    con.execute("DROP TABLE _out")
    con.execute("INSERT INTO md.calendar VALUES (DATE '2026-10-05', 1)")
    con.execute(
        "CREATE TABLE _index AS SELECT * FROM md.index_daily "
        "WHERE trade_date = DATE '2026-08-14' LIMIT 1"
    )
    con.execute("UPDATE _index SET trade_date = DATE '2026-10-05'")
    con.execute("INSERT INTO md.index_daily SELECT * FROM _index")
    con.execute("DROP TABLE _index")
    con.close()

    third = _snapshot(snapshot_db)
    assert third.snapshot_id == first.snapshot_id


def test_execution_source_change_changes_digest(tmp_path):
    """核心执行模块源码变化必须改变执行模型摘要。"""
    for name in EXECUTION_SOURCE_FILES:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    first = _execution_model_component(tmp_path)

    engine = tmp_path / "engine.py"
    engine.write_text("# engine changed\n", encoding="utf-8")
    second = _execution_model_component(tmp_path)

    assert (
        second["source_sha256"]["engine.py"]
        != first["source_sha256"]["engine.py"]
    )
    assert second["source_set_sha256"] != first["source_set_sha256"]


def test_execution_model_enters_input_snapshot_id(snapshot_db, monkeypatch):
    """执行模型摘要必须参与规则分项和最终 snapshot_id。"""
    first = _snapshot(snapshot_db)
    original = snapshot_module._execution_model_component

    def changed():
        item = original()
        item["version"] = "changed"
        item["source_set_sha256"] = "f" * 64
        return item

    monkeypatch.setattr(snapshot_module, "_execution_model_component", changed)
    second = _snapshot(snapshot_db)

    assert second.snapshot_id != first.snapshot_id
    assert second.components["rules"]["execution_model"]["version"] == "changed"


def test_in_window_market_change_changes_only_market_component(snapshot_db):
    first = _snapshot(snapshot_db)
    with duckdb.connect(str(snapshot_db), read_only=True) as conn:
        member = conn.execute(
            "SELECT stock_code FROM ref.index_weight "
            "WHERE index_code = '000300.SH' AND weight > 0 "
            "ORDER BY stock_code LIMIT 1"
        ).fetchone()[0]
    con = _mutable_connection(snapshot_db)
    con.execute(
        "UPDATE md.daily_bar SET close = close + 1 "
        "WHERE trade_date = DATE '2026-08-10' AND stock_code = ?",
        [member],
    )
    con.close()

    second = _snapshot(snapshot_db)
    assert second.snapshot_id != first.snapshot_id
    assert (
        second.components["market"]["digest"]
        != first.components["market"]["digest"]
    )
    assert (
        second.components["factors"]["MOM_20"]["digest"]
        == first.components["factors"]["MOM_20"]["digest"]
    )


def test_benchmark_action_and_industry_components_are_sensitive(snapshot_db):
    from backtest.specs import RiskSpec

    strategy = StrategySpec(
        universe="index:000300.SH",
        risk=RiskSpec(max_industry_weight=0.5),
    )
    first = _snapshot(snapshot_db, strategy=strategy)
    con = _mutable_connection(snapshot_db)
    con.execute(
        "UPDATE md.index_daily SET close = close + 1 "
        "WHERE index_code = 'H00300.CSI' AND trade_date = DATE '2026-08-10'"
    )
    con.execute(
        "CREATE TABLE _action AS SELECT * FROM ref.corporate_action LIMIT 1"
    )
    con.execute(
        "UPDATE _action SET stock_code = "
        "(SELECT stock_code FROM ref.index_weight "
        " WHERE index_code = '000300.SH' AND weight > 0 LIMIT 1), "
        "ex_date = DATE '2026-08-11', source_event_key = 'snapshot-test'"
    )
    con.execute("INSERT INTO ref.corporate_action SELECT * FROM _action")
    con.execute("DROP TABLE _action")
    con.execute(
        "UPDATE ref.sw_industry_member SET l1_name = l1_name || 'X' "
        "WHERE stock_code = (SELECT stock_code FROM ref.sw_industry_member "
        " WHERE stock_code IN (SELECT stock_code FROM ref.index_weight "
        " WHERE index_code = '000300.SH' AND weight > 0) LIMIT 1)"
    )
    con.close()
    second = _snapshot(snapshot_db, strategy=strategy)
    assert (
        second.components["benchmark"]["digest"]
        != first.components["benchmark"]["digest"]
    )
    assert (
        second.components["corporate_actions"]["digest"]
        != first.components["corporate_actions"]["digest"]
    )
    assert (
        second.components["industry"]["digest"]
        != first.components["industry"]["digest"]
    )
    assert second.snapshot_id != first.snapshot_id


def test_factor_change_is_isolated_to_selected_factor(snapshot_db, tmp_path):
    dates = pd.to_datetime(["2026-08-03", "2026-08-04"])
    stocks = ["SH600000", "SZ000001"]
    for family, name, value in (
        ("momentum", "MOM_20", 1.0),
        ("volatility", "VOL_20", 2.0),
    ):
        directory = tmp_path / family / "year=2026"
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "stock_code": stocks * len(dates),
                "trade_date": [date for date in dates for _ in stocks],
                name: [value] * (len(stocks) * len(dates)),
                "year": [2026] * (len(stocks) * len(dates)),
            }
        ).to_parquet(directory / "part.parquet", index=False)

    strategy = StrategySpec(
        universe="index:000300.SH",
        signal=SignalSpec(factors=("MOM_20", "VOL_20"), weights=(1.0, 1.0)),
    )
    first = _snapshot(snapshot_db, tmp_path, strategy)
    pd.DataFrame(
        {
            "stock_code": stocks * len(dates),
            "trade_date": [date for date in dates for _ in stocks],
            "MOM_20": [3.0] * (len(stocks) * len(dates)),
            "year": [2026] * (len(stocks) * len(dates)),
        }
    ).to_parquet(tmp_path / "momentum" / "year=2026" / "part.parquet", index=False)

    second = _snapshot(snapshot_db, tmp_path, strategy)
    assert second.snapshot_id != first.snapshot_id
    assert (
        second.components["factors"]["MOM_20"]["digest"]
        != first.components["factors"]["MOM_20"]["digest"]
    )
    assert (
        second.components["factors"]["VOL_20"]["digest"]
        == first.components["factors"]["VOL_20"]["digest"]
    )


def test_provider_source_enters_snapshot_and_source_change_changes_id(
    snapshot_db, monkeypatch
):
    strategy = StrategySpec(
        universe="index:000300.SH",
        signal=SignalSpec(
            factors=("MOM_20",),
            weights=(),
            provider=ProviderSpec(
                "momentum_timing",
                "1",
                {"top_n": 30, "min_breadth": 0.1},
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=30),
        rebalance=RebalanceSpec(frequency="custom"),
    )
    first = _snapshot(snapshot_db, strategy=strategy)
    provider = first.components["rules"]["provider"]
    assert provider["name"] == "momentum_timing"
    assert provider["version"] == "1"
    assert provider["params"] == {"top_n": 30, "min_breadth": 0.1}
    assert provider["required_factors"] == ["MOM_20"]
    assert "class" in provider["source"]
    assert len(provider["source_sha256"]) == 64

    original = providers._code_source
    monkeypatch.setattr(
        providers,
        "_code_source",
        lambda provider: (original(provider)[0] + "\n# changed", "f" * 64),
    )
    second = _snapshot(snapshot_db, strategy=strategy)
    assert second.snapshot_id != first.snapshot_id
    assert (
        second.components["rules"]["provider"]["source_sha256"]
        != first.components["rules"]["provider"]["source_sha256"]
    )


def test_synthetic_market_data_snapshot_is_deterministic(make_data):
    strategy = StrategySpec()
    data = make_data()
    first = build_input_snapshot(strategy, data_snapshot_from_market_data(data))
    second = build_input_snapshot(strategy, data_snapshot_from_market_data(data))
    assert first.snapshot_id == second.snapshot_id
    assert first.components["market"]["algorithm"] == "pandas_hash_v1"


def test_store_records_immutable_snapshot_and_rejects_provider_revision(tmp_path):
    store = BacktestStore(tmp_path / "backtest.duckdb")
    base = {
        "snapshot_id": "a" * 64,
        "schema_version": "backtest_input_snapshot_v1",
        "created_at": "2026-10-18T00:00:00+00:00",
        "scope": {"start": "2024-01-01", "end": "2024-01-02", "universe": "all_a"},
        "components": {
            "market": {"digest": "m"},
            "rules": {
                "provider": {
                    "name": "demo",
                    "version": "1",
                    "source_sha256": "1" * 64,
                    "module": "demo",
                    "required_factors": ["MOM_20"],
                }
            },
        },
    }
    assert store.record_snapshot(base) == base["snapshot_id"]
    same = {**base, "created_at": "2026-10-18T01:00:00+00:00"}
    assert store.record_snapshot(same) == base["snapshot_id"]
    assert store.get_snapshot(base["snapshot_id"])["snapshot_id"] == base["snapshot_id"]

    conflicted = {**base, "components": {**base["components"], "market": {"digest": "x"}}}
    with pytest.raises(SnapshotConflictError, match="内容不同"):
        store.record_snapshot(conflicted)

    second = {
        **base,
        "snapshot_id": "b" * 64,
        "components": {
            **base["components"],
            "rules": {
                "provider": {
                    **base["components"]["rules"]["provider"],
                    "source_sha256": "2" * 64,
                }
            },
        },
    }
    with pytest.raises(SnapshotConflictError, match="提升 version"):
        store.record_snapshot(second)

    bumped = {
        **second,
        "snapshot_id": "d" * 64,
        "components": {
            **second["components"],
            "rules": {
                "provider": {
                    **second["components"]["rules"]["provider"],
                    "version": "2",
                }
            },
        },
    }
    assert store.record_snapshot(bumped) == bumped["snapshot_id"]


def test_data_snapshot_requires_known_factor(snapshot_db):
    with pytest.raises(DataAlignmentError, match="未知因子"):
        build_data_snapshot(
            "2026-08-03",
            "2026-08-14",
            ["NOT_A_FACTOR"],
            universe="index:000300.SH",
            db_path=snapshot_db,
        )


def test_pipeline_reuses_snapshot_for_repeated_run(tmp_path):
    store = BacktestStore(tmp_path / "backtest.duckdb")
    strategy = StrategySpec(
        universe="index:000300.SH",
        portfolio=PortfolioSpec(top_n=5),
        rebalance=RebalanceSpec(frequency="daily"),
    )
    strategy_id, _ = store.create_strategy(strategy)
    request = RunRequest(strategy_id, "2026-08-03", "2026-08-04")
    first = run(store, request)
    second = run(store, request)
    assert first["snapshot_id"] == second["snapshot_id"]
    record = store.get_run(first["run_id"])
    assert record["snapshot_id"] == first["snapshot_id"]
    snapshot = store.get_snapshot(record["snapshot_id"])
    assert snapshot["scope"]["universe"] == "index:000300.SH"
    assert snapshot["scope"]["price_adjust"] == "raw"
