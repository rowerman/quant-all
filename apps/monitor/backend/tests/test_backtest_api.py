"""策略与回测接口。"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import backtest_service
from backend.app import create_app
from backtest.coverage import benchmark_coverage, market_coverage
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


def test_strategy_api_accepts_complete_supported_configuration(client):
    payload = {
        "description": "完整配置接口测试",
        "spec": {
            "name": "周频双因子",
            "universe": "index:000905.SH",
            "signal": {
                "factors": ["MOM_20", "VOL_20"],
                "weights": [1.0, -0.5],
                "winsorize": True,
                "standardize": False,
            },
            "portfolio": {"selection": "top_n", "top_n": 20, "weighting": "market_cap", "long_only": True},
            "rebalance": {"frequency": "weekly", "signal_at": "close", "execute_at": "next_open"},
            "execution": {
                "lot_size": 100,
                "cash_buffer": 0.03,
                "max_participation_rate": 0.1,
                "reject_suspended": True,
                "reject_limit_up_buy": True,
                "reject_limit_down_sell": False,
                "enforce_t_plus_1": True,
            },
            "risk": {
                "max_single_weight": 0.1,
                "max_industry_weight": 0.3,
                "max_one_way_turnover": 0.3,
            },
            "costs": {
                "commission_rate": 0.0002,
                "minimum_commission": 3.0,
                "stamp_tax_rate": 0.0005,
                "slippage": 0.0008,
            },
            "benchmark": {"index_code": "H00905.CSI"},
        },
    }
    created = client.post("/api/strategies", json=payload)
    assert created.status_code == 200
    detail = client.get(f"/api/strategies/{created.json()['strategy_id']}").json()
    assert detail["spec"]["signal"]["weights"] == [1.0, -0.5]
    assert detail["spec"]["rebalance"]["frequency"] == "weekly"
    assert detail["spec"]["costs"]["slippage"] == pytest.approx(0.0008)
    assert detail["spec"]["execution"]["max_participation_rate"] == pytest.approx(0.1)
    assert detail["spec"]["risk"]["max_single_weight"] == pytest.approx(0.1)
    assert detail["spec"]["risk"]["max_industry_weight"] == pytest.approx(0.3)
    assert detail["spec"]["risk"]["max_one_way_turnover"] == pytest.approx(0.3)


def test_strategy_api_uses_ten_percent_participation_by_default(client):
    created = client.post("/api/strategies", json={"name": "默认容量"})
    assert created.status_code == 200
    detail = client.get(f"/api/strategies/{created.json()['strategy_id']}").json()
    assert detail["spec"]["execution"]["max_participation_rate"] == pytest.approx(0.1)
    assert detail["spec"]["costs"]["impact_model"] == "sqrt"


def test_strategy_api_rejects_invalid_weights_and_costs(client):
    mismatch = client.post(
        "/api/strategies",
        json={"name": "坏权重", "signal": {"factors": ["MOM_20", "VOL_20"], "weights": [1.0]}},
    )
    assert mismatch.status_code == 400
    assert "数量必须" in mismatch.json()["message"]
    negative = client.post(
        "/api/strategies",
        json={"name": "坏成本", "costs": {"commission_rate": -0.1}},
    )
    assert negative.status_code == 400
    assert "不能为负数" in negative.json()["message"]


def test_backtest_options_expose_strategies_universes_and_limits(client):
    payload = client.get("/api/backtest/options").json()
    assert {row["strategy_id"] for row in payload["strategies"]} >= {MOM20_ID}
    providers = {row["name"]: row for row in payload["providers"]}
    assert "momentum_timing" in providers
    assert [item["name"] for item in providers["momentum_timing"]["param_schema"]] == [
        "top_n",
        "min_breadth",
    ]
    assert {"all_a", "index:000300.SH"} <= {row["key"] for row in payload["universes"]}
    assert {"H00300.CSI", "H00905.CSI", "000300.SH", "000905.SH"} == {
        row["code"] for row in payload["benchmarks"]
    }
    assert payload["defaults"]["benchmark"] == "H00300.CSI"
    assert payload["defaults"]["risk_free_rate"] == pytest.approx(0.02)
    market = market_coverage()
    benchmark = benchmark_coverage("H00300.CSI")
    assert payload["defaults"]["end"] == min(str(market.end), str(benchmark.end))
    assert payload["coverage"]["market_end"] == market.end
    assert all(
        {"start", "end", "default_end"} <= set(row) for row in payload["benchmarks"]
    )
    assert {"name", "direction"} <= set(payload["factors"][0])
    assert payload["limits"]["trade_days"] > 0
    assert payload["limits"]["chunk_cells"] < payload["limits"]["panel_cells"]
    assert payload["limits"]["max_chunks"] >= 2
    assert payload["costs"]["stamp_tax"] == "historical"
    assert payload["costs"]["transfer_fee"] == "historical"
    assert payload["costs"]["fee_rounding"] == "cent"
    assert payload["costs"]["impact_model"] == "sqrt"
    assert payload["costs"]["impact_coefficient"] == pytest.approx(0.5)
    assert "stamp_tax_rate" not in payload["costs"]


def test_provider_validation_returns_normalized_defaults(client):
    response = client.post(
        "/api/providers/validate",
        json={
            "provider": {"name": "momentum_timing", "version": "1", "params": {}},
            "factors": ["MOM_20"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["params"] == {"top_n": 30, "min_breadth": 0.5}
    assert payload["provider"]["code_sha256"]
    assert payload["provider"]["param_schema"][0]["name"] == "top_n"


@pytest.mark.parametrize(
    "provider,factors,message",
    [
        (
            {"name": "momentum_timing", "version": "1", "params": {"top_n": 0}},
            ["MOM_20"],
            "top_n",
        ),
        (
            {"name": "momentum_timing", "version": "1", "params": {"unknown": 1}},
            ["MOM_20"],
            "未知字段",
        ),
        (
            {"name": "momentum_timing", "version": "1", "params": {}},
            [],
            "需要因子",
        ),
    ],
)
def test_provider_validation_rejects_invalid_input(client, provider, factors, message):
    response = client.post(
        "/api/providers/validate",
        json={"provider": provider, "factors": factors},
    )
    assert response.status_code == 400
    assert message in response.json()["message"]


def test_backtest_detail_exposes_input_snapshot(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    snapshot = {
        "snapshot_id": "c" * 64,
        "schema_version": "backtest_input_snapshot_v1",
        "created_at": "2026-10-18T00:00:00+00:00",
        "scope": {"start": "2024-01-01", "end": "2024-01-02", "universe": "all_a"},
        "components": {"market": {"digest": "m"}},
    }
    store.record_snapshot(snapshot)
    run_id = store.start_run(
        strategy_id,
        version,
        BacktestParams("2024-01-01", "2024-01-02"),
        snapshot_id=snapshot["snapshot_id"],
    )
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_010_000.0],
            "cash": [0.0, 0.0],
            "nav": [1.0, 1.01],
            "return": [0.0, 0.01],
            "drawdown": [0.0, 0.0],
            "cost": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "benchmark_return": [0.0, 0.0],
            "benchmark_nav": [1.0, 1.0],
            "excess_return": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    result = BacktestResult(
        portfolio=object(),
        daily=daily,
        orders=pd.DataFrame(),
        positions=pd.DataFrame(),
        holdings=pd.DataFrame(),
        target_weights=pd.DataFrame(),
        rejections=pd.DataFrame(),
        metrics={"total_return": 0.01},
        metadata={
            "strategy_hash": "hash",
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot": snapshot,
        },
    )
    store.finish_run(run_id, result)

    assert client.get(f"/api/backtests/{run_id}").json()["snapshot_id"] == snapshot["snapshot_id"]
    listed = client.get("/api/backtests").json()
    assert listed[0]["snapshot_id"] == snapshot["snapshot_id"]


def test_backtest_metrics_handles_empty_benchmark_columns(client, store):
    """没有基准数据时，指标接口仍返回组合指标。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id,
        version,
        BacktestParams("2024-01-01", "2024-01-03"),
    )
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0, 1_010_000.0, 1_020_000.0],
            "cash": [1_000_000.0] * 3,
            "nav": [1.0, 1.01, 1.02],
            "return": [0.0, 0.01, 1.02 / 1.01 - 1.0],
            "drawdown": [0.0, 0.0, 0.0],
            "cost": [0.0] * 3,
            "turnover": [0.0] * 3,
            "benchmark_return": [None] * 3,
            "benchmark_nav": [None] * 3,
            "excess_return": [None] * 3,
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    result = BacktestResult(
        portfolio=object(),
        daily=daily,
        orders=pd.DataFrame(),
        positions=pd.DataFrame(),
        holdings=pd.DataFrame(),
        target_weights=pd.DataFrame(),
        rejections=pd.DataFrame(),
        metrics={},
        metadata={},
    )
    store.finish_run(run_id, result)

    response = client.get(f"/api/backtests/{run_id}/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_return"] == pytest.approx(0.02)
    assert "information_ratio" not in payload


def test_backtest_list_serializes_unfinished_run_with_missing_finished_at(client, store):
    """未完成运行的空完成时间应作为 JSON null 返回。"""
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id,
        version,
        BacktestParams("2024-01-01", "2024-01-02"),
    )

    response = client.get("/api/backtests")

    assert response.status_code == 200
    listed = response.json()
    assert listed[0]["run_id"] == run_id
    assert listed[0]["finished_at"] is None


def test_strategy_api_creates_provider_strategy(client):
    payload = {
        "description": "提供者策略接口测试",
        "spec": {
            "name": "广度择时",
            "signal": {
                "provider": {
                    "name": "momentum_timing",
                    "version": "1",
                    "params": {"top_n": 5, "min_breadth": 0.5},
                },
                "factors": ["MOM_20"],
                "weights": [],
            },
            "portfolio": {
                "selection": "provider",
                "top_n": 5,
                "weighting": "equal",
                "long_only": True,
            },
            "rebalance": {
                "frequency": "custom",
                "signal_at": "close",
                "execute_at": "next_open",
            },
        },
    }
    created = client.post("/api/strategies", json=payload)
    assert created.status_code == 200
    assert created.json()["provider"]["code_sha256"]
    assert created.json()["provider"]["param_schema"]
    detail = client.get(f"/api/strategies/{created.json()['strategy_id']}").json()
    assert detail["spec"]["signal"]["provider"]["name"] == "momentum_timing"
    assert detail["spec"]["rebalance"]["frequency"] == "custom"


def test_strategy_api_rejects_unknown_provider(client):
    payload = {
        "name": "未知提供者",
        "signal": {
            "provider": {"name": "missing_provider", "version": "1", "params": {}},
            "factors": ["MOM_20"],
            "weights": [],
        },
        "portfolio": {"selection": "provider", "top_n": 5},
        "rebalance": {"frequency": "custom"},
    }
    created = client.post("/api/strategies", json=payload)
    assert created.status_code == 400
    assert "未注册" in created.json()["message"]


def test_backtest_prepare_accepts_explicit_no_benchmark(client):
    """null 或空字符串表示本次关闭基准，而不是沿用策略里的默认值。"""
    for value in (None, ""):
        request = backtest_service.prepare(
            {
                "strategy_id": MOM20_ID,
                "start": "2024-01-01",
                "end": "2024-02-01",
                "benchmark": value,
            }
        )
        assert request.clear_benchmark is True
        assert request.benchmark is None


def test_backtest_preflight_rejects_benchmark_start_gap(client):
    """全历史请求必须在开跑前提示基准起始日。"""
    response = client.get(
        "/api/backtests/stream",
        params={
            "strategy_id": MOM20_ID,
            "start": "2000-01-04",
            "end": market_coverage().end,
        },
    )
    assert response.status_code == 400
    message = response.json()["message"]
    assert "基准指数 H00300.CSI" in message
    assert "benchmark=None" in message


def test_strategy_without_stamp_tax_rate_uses_historical_schedule(client):
    payload = {
        "spec": {
            "name": "历史成本",
            "costs": {
                "commission_rate": 0.0003,
                "minimum_commission": 5.0,
                "slippage": 0.0005,
            },
        }
    }
    created = client.post("/api/strategies", json=payload)
    assert created.status_code == 200
    detail = client.get(f"/api/strategies/{created.json()['strategy_id']}").json()
    assert detail["spec"]["costs"]["stamp_tax_rate"] is None
    assert detail["spec"]["costs"]["transfer_fee_enabled"] is True
    assert detail["spec"]["costs"]["fee_rounding"] == "cent"


def test_run_tables_and_metrics_are_readable(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = store.start_run(
        strategy_id, version, BacktestParams("2024-01-01", "2024-01-05"),
        universe="index:000905.SH", benchmark="H00905.CSI",
    )
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
        industry_exposure=pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2024-01-02")],
                "industry": ["SW2014 801010 农林牧渔"],
                "market_value": [500_000.0],
                "weight": [0.5],
                "is_cash": [False],
            }
        ),
    )
    store.finish_run(run_id, result)

    runs = client.get("/api/backtests").json()
    assert runs[0]["run_id"] == run_id
    assert runs[0]["universe"] == "index:000905.SH"
    assert runs[0]["benchmark"] == "H00905.CSI"

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
    industry = client.get(f"/api/backtests/{run_id}/industry_exposure").json()
    assert industry[0]["industry"] == "SW2014 801010 农林牧渔"
    assert client.get(f"/api/backtests/{run_id}/unknown").status_code == 400
    assert client.get("/api/backtests/missing-run").status_code == 404


def test_run_request_accepts_risk_free_rate(store):
    strategy_id, _ = store.create_strategy(StrategySpec())
    request = backtest_service.prepare(
        {
            "strategy_id": strategy_id,
            "start": "2024-01-01",
            "end": "2024-01-05",
            "risk_free_rate": 0.031,
        },
        store_path=str(store.path),
    )
    assert request.risk_free_rate == pytest.approx(0.031)
    assert request.params.risk_free_rate == pytest.approx(0.031)


def _finish_run(store, strategy_id, version, start, dates, nav, excess):
    """创建一条带可对比日结果的成功运行。"""
    run_id = store.start_run(
        strategy_id,
        version,
        BacktestParams(start, str(pd.Timestamp(dates[-1]).date())),
        universe="all_a",
        benchmark="H00300.CSI",
    )
    daily = pd.DataFrame(
        {
            "portfolio_value": [1_000_000.0 * item for item in nav],
            "cash": [0.0] * len(dates),
            "nav": nav,
            "return": [0.0] + [nav[index] / nav[index - 1] - 1.0 for index in range(1, len(nav))],
            "drawdown": [0.0] * len(dates),
            "cost": [0.0] * len(dates),
            "turnover": [0.0] * len(dates),
            "benchmark_return": [0.0] * len(dates),
            "benchmark_nav": [1.0] * len(dates),
            "excess_return": excess,
        },
        index=pd.to_datetime(dates),
    )
    result = BacktestResult(
        portfolio=object(),
        daily=daily,
        orders=pd.DataFrame(),
        positions=pd.DataFrame(),
        holdings=pd.DataFrame(),
        target_weights=pd.DataFrame(),
        rejections=pd.DataFrame(),
        metrics={"total_return": nav[-1] / nav[0] - 1.0},
        metadata={
            "strategy": {"name": "测试策略"},
            "strategy_hash": "hash",
            "universe": "all_a",
            "benchmark": {"code": "H00300.CSI"},
        },
    )
    store.finish_run(run_id, result)
    return run_id


def test_backtest_compare_uses_common_dates_and_normalized_curves(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    first = _finish_run(
        store, strategy_id, version, "2024-01-01",
        ["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, 1.1, 1.21], [0.0, 0.01, 0.02],
    )
    second = _finish_run(
        store, strategy_id, version, "2024-01-02",
        ["2024-01-02", "2024-01-03", "2024-01-04"], [2.0, 2.1, 2.2], [0.0, 0.03, 0.04],
    )

    response = client.get("/api/backtests/compare", params={"run_ids": f"{first},{second}"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dates"] == ["2024-01-02", "2024-01-03"]
    assert [row["run_id"] for row in payload["runs"]] == [first, second]
    assert payload["runs"][0]["nav"] == pytest.approx([1.0, 1.1])
    assert payload["runs"][0]["cumulative_excess"] == pytest.approx([0.0, 0.02])
    assert payload["runs"][1]["cumulative_excess"] == pytest.approx([0.0, 0.03])


def test_backtest_compare_validates_selection(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    run_id = _finish_run(
        store, strategy_id, version, "2024-01-01",
        ["2024-01-01", "2024-01-02"], [1.0, 1.01], [0.0, 0.01],
    )
    assert client.get("/api/backtests/compare", params={"run_ids": run_id}).status_code == 400
    assert client.get("/api/backtests/compare", params={"run_ids": f"{run_id},{run_id}"}).status_code == 400
    assert client.get("/api/backtests/compare", params={"run_ids": "a,b,c,d,e"}).status_code == 400

    running = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    response = client.get("/api/backtests/compare", params={"run_ids": f"{run_id},{running}"})
    assert response.status_code == 400
    assert "只有成功运行" in response.json()["message"]

    disjoint = _finish_run(
        store, strategy_id, version, "2025-01-01",
        ["2025-01-01", "2025-01-02"], [1.0, 1.01], [0.0, 0.01],
    )
    response = client.get("/api/backtests/compare", params={"run_ids": f"{run_id},{disjoint}"})
    assert response.status_code == 400
    assert "共同交易日不足" in response.json()["message"]


def test_delete_backtest_api_removes_finished_but_rejects_running(client, store):
    strategy_id, version = store.create_strategy(StrategySpec())
    finished = _finish_run(
        store, strategy_id, version, "2024-01-01",
        ["2024-01-01", "2024-01-02"], [1.0, 1.01], [0.0, 0.01],
    )
    response = client.delete(f"/api/backtests/{finished}")
    assert response.status_code == 200
    assert response.json() == {"run_id": finished, "deleted": True}
    assert client.get(f"/api/backtests/{finished}").status_code == 404

    running = store.start_run(strategy_id, version, BacktestParams("2024-01-01", "2024-01-02"))
    response = client.delete(f"/api/backtests/{running}")
    assert response.status_code == 409
    assert response.json()["code"] == "BT-602"
    assert client.delete("/api/backtests/missing").status_code == 404


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
    assert '"status": "queued"' in body
    assert "event: status" in body
    assert '"status": "preflight"' in body
    assert "event: meta" in body and '"name": "MOM_20 Top30"' in body
    assert "event: load" in body
    assert "event: done" in body


def test_backtest_sync_api_uses_scheduler_and_drains_events(client, monkeypatch):
    """同步接口经过回测调度器，且不会因事件队列无人消费而停住。"""

    def fake_stream(request, should_cancel, store_path=None):
        for number in range(8):
            yield "progress", {"number": number}
        yield "done", {"run_id": "r-sync", "orders": 3}

    def direct_path_is_forbidden(*args, **kwargs):
        raise AssertionError("同步接口绕过了调度器")

    monkeypatch.setattr(backtest_service, "stream_strategy", fake_stream)
    monkeypatch.setattr(backtest_service, "run_strategy", direct_path_is_forbidden)
    response = client.post(
        "/api/backtests",
        json={"strategy_id": MOM20_ID, "start": "2024-01-01", "end": "2024-02-01"},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.json() == {"run_id": "r-sync", "orders": 3}


def test_backtest_sync_api_rejects_before_preflight_when_capacity_is_full(client, monkeypatch):
    """同步回测容量已满时应直接429，并显示回测调度器的真实容量。"""
    called = False

    def prepare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("容量已满仍进入预检")

    monkeypatch.setattr(backtest_service, "prepare", prepare)
    scheduler = client.app.state.backtest_scheduler
    jobs = [scheduler.reserve() for _ in range(3)]
    try:
        response = client.post(
            "/api/backtests",
            json={"strategy_id": MOM20_ID, "start": "2024-01-01", "end": "2024-02-01"},
        )
        assert response.status_code == 429
        assert "1个运行任务及2个等待名额" in response.json()["message"]
        assert not called
    finally:
        for job in jobs:
            scheduler.discard(job)


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

