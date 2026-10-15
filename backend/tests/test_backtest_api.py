"""策略与回测接口。"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import backtest_service
from backend.app import create_app
from backtest.engine import BacktestResult
from backtest.builtin import MOM20_ID
from backtest.specs import BacktestParams, StrategySpec


@pytest.fixture
def store(tmp_path):
    """临时库上的回测存储。"""
    return backtest_service.BacktestStore(tmp_path / "backtest.duckdb")


@pytest.fixture
def client(store, monkeypatch):
    """把服务层固定到临时库上的测试客户端。"""
    monkeypatch.setattr(backtest_service, "_store", lambda ignored=None: store)
    with TestClient(create_app()) as test_client:
        yield test_client


def test_strategy_api_round_trip(client):
    created = client.post("/api/strategies", json={"name": "api-test"})
    assert created.status_code == 200
    strategy_id = created.json()["strategy_id"]
    listed = client.get("/api/strategies").json()
    assert strategy_id in {row["strategy_id"] for row in listed}
    # 内置策略与用户策略在同一条列表里
    assert MOM20_ID in {row["strategy_id"] for row in listed}
    detail = client.get(f"/api/strategies/{strategy_id}").json()
    assert detail["spec"]["name"] == "api-test"
    assert client.get(f"/api/strategies/{strategy_id}?version=99").status_code == 400


def test_backtest_options_expose_strategies_universes_and_limits(client):
    payload = client.get("/api/backtest/options").json()
    assert {row["strategy_id"] for row in payload["strategies"]} >= {MOM20_ID}
    assert {"all_a", "index:000300.SH"} <= {row["key"] for row in payload["universes"]}
    assert {"name", "direction"} <= set(payload["factors"][0])
    assert payload["limits"]["trade_days"] > 0


def test_run_tables_and_metrics_are_readable(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"))
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_020_000.0],
            "cash": [0.0, 100.0],
            "nav": [1.0, 1.02],
            "return": [0.0, 0.02],
            "drawdown": [0.0, 0.0],
            "cost": [15.0, 0.0],
            "turnover": [0.0, 0.3],
        },
        index=pd.date_range("2024-01-02", periods=2),
    )
    holding = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "stock_code": ["SH600000"],
            "market_value": [500_000.0],
            "weight": [0.5],
        }
    )
    liquidation = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-03")],
            "stock_code": ["SZ000001"],
            "delist_date": [pd.Timestamp("2024-01-03")],
            "size": [4900.0],
            "price": [9.995],
            "value": [48975.5],
        }
    )
    result = BacktestResult(
        portfolio=object(),
        daily=daily,
        orders=pd.DataFrame(),
        positions=pd.DataFrame(),
        holdings=holding,
        target_weights=pd.DataFrame(),
        rejections=pd.DataFrame(),
        metrics={"total_return": 0.02},
        metadata={"strategy_hash": "hash", "universe": "all_a", "elapsed_seconds": 3.0},
        liquidations=liquidation,
    )
    store.finish_run(run_id, result)

    runs = client.get("/api/backtests").json()
    assert runs[0]["run_id"] == run_id
    assert runs[0]["universe"] == "all_a"

    detail = client.get(f"/api/backtests/{run_id}").json()
    assert detail["status"] == "success"
    assert detail["metadata"]["strategy_hash"] == "hash"

    metrics = client.get(f"/api/backtests/{run_id}/metrics").json()
    assert metrics["total_return"] == pytest.approx(0.02)
    assert metrics["trading_days"] == 2

    assert client.get(f"/api/backtests/{run_id}/holding").json()[0]["stock_code"] == "SH600000"
    assert len(client.get(f"/api/backtests/{run_id}/daily").json()) == 2
    forced = client.get(f"/api/backtests/{run_id}/liquidation").json()
    assert len(forced) == 1
    assert forced[0]["stock_code"] == "SZ000001"
    assert forced[0]["value"] == pytest.approx(48975.5)
    assert client.get(f"/api/backtests/{run_id}/unknown").status_code == 400
    assert client.get("/api/backtests/missing-run").status_code == 400


def test_backtest_stream_pushes_stage_events(client, monkeypatch):
    """流式接口逐阶段推送，且没有被 {run_id} 通配路由吃掉。"""

    def fake_stream(request, should_cancel, store_path=None):
        yield "meta", {"run_id": "r1", "name": "MOM_20 Top30"}
        yield "load", {"trading_days": 10, "stocks": 100}
        yield "done", {"run_id": "r1", "orders": 5}

    monkeypatch.setattr(backtest_service, "stream_strategy", fake_stream)
    with client.stream(
        "GET",
        "/api/backtests/stream",
        params={"strategy_id": MOM20_ID, "start": "2024-01-01", "end": "2024-02-01"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "event: queued" in body
    assert "event: meta" in body and '"name": "MOM_20 Top30"' in body
    assert "event: load" in body
    assert "event: done" in body


def test_backtest_stream_rejects_bad_request(client):
    response = client.get(
        "/api/backtests/stream",
        params={"strategy_id": MOM20_ID, "start": "2024-02-01", "end": "2024-01-01"},
    )
    assert response.status_code == 400
    assert response.json()["code"].startswith("BT-")


def test_run_strategy_rejects_unknown_strategy(client):
    response = client.post(
        "/api/backtests",
        json={"strategy_id": "missing", "start": "2024-01-01", "end": "2024-01-05"},
    )
    assert response.status_code == 400
    assert "策略不存在" in response.json()["message"]

