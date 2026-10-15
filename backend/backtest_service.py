"""策略和回测 HTTP 接口使用的服务函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from backtest import (
    BacktestParams,
    BacktestStore,
    RunRequest,
    StrategySpec,
    iter_run,
    run,
)
from backtest.config import MAX_PANEL_CELLS, MAX_TRADE_DAYS
from backtest.errors import BacktestError
from backtest.specs import BENCHMARK_OPTIONS, SpecError
from backtest.universe import list_universes, resolve_universe


def _store(path: str | None = None) -> BacktestStore:
    """取得回测存储。测试可传入临时库路径。"""
    return BacktestStore(Path(path)) if path else BacktestStore()


def _universe_options() -> list[dict]:
    """给出可选股票池。"""
    return [{"key": item.key, "label": item.label, "detail": item.detail} for item in list_universes()]


def options(store_path: str | None = None) -> dict:
    """回测页需要的枚举、默认值与资源上限。"""
    store = _store(store_path)
    strategies = store.ensure_builtins()
    import quant

    factors = quant.load_factor_list(status="implemented")
    return {
        "strategies": strategies,
        "universes": _universe_options(),
        "benchmarks": [dict(item) for item in BENCHMARK_OPTIONS],
        "factors": factors[["name", "family", "direction", "description"]].to_dict("records"),
        "defaults": {
            "start": "2024-01-01", "end": "2026-10-14", "init_cash": 1_000_000.0,
            "benchmark": "H00300.CSI",
        },
        "limits": {"trade_days": MAX_TRADE_DAYS, "panel_cells": MAX_PANEL_CELLS, "workers": 1, "waiting": 2},
        "costs": {
            "commission_rate": 0.0003,
            "minimum_commission": 5.0,
            "stamp_tax_rate": 0.0005,
            "slippage": 0.0005,
        },
    }


def create_strategy(payload: dict, store_path: str | None = None) -> dict:
    """创建策略第一版。"""
    if not isinstance(payload, dict):
        raise BacktestError("策略请求必须是对象")
    spec = StrategySpec.from_dict(payload.get("spec", payload))
    strategy_id, version = _store(store_path).create_strategy(spec, payload.get("description", ""))
    return {"strategy_id": strategy_id, "version": version, "config_hash": spec.config_hash}


def list_strategies(store_path: str | None = None) -> list[dict]:
    """返回策略列表，内置策略与用户策略一起给出。"""
    return _store(store_path).ensure_builtins()


def get_strategy(strategy_id: str, version: int | None = None, store_path: str | None = None) -> dict:
    """返回指定策略版本。"""
    spec = _store(store_path).get_version(strategy_id, version)
    return {"strategy_id": strategy_id, "version": spec.version, "config_hash": spec.config_hash, "spec": spec.to_dict()}


def list_runs(limit: int = 50, store_path: str | None = None) -> list[dict]:
    """返回回测运行列表。"""
    return _store(store_path).list_runs(limit)


def get_run(run_id: str, store_path: str | None = None) -> dict:
    """读取回测状态与指标。"""
    store = _store(store_path)
    record = store.get_run(run_id)
    import json

    metadata = json.loads(record["metadata_json"]) if record.get("metadata_json") else {}
    record["metadata"] = metadata
    record.pop("metadata_json", None)
    record["metrics"] = metadata.get("metrics")
    return record


def get_run_table(run_id: str, table: str, store_path: str | None = None) -> list[dict]:
    """读取回测运行的标准化结果表。"""
    return _store(store_path).get_result_table(run_id, table)


def get_run_metrics(run_id: str, store_path: str | None = None) -> dict:
    """按已落库日结果返回绩效指标。"""
    return _store(store_path).get_metrics(run_id)


def prepare(payload: dict) -> RunRequest:
    """校验回测请求，非法参数在开流前就报错。

    参数:
        payload: HTTP 请求体。
    """
    if not isinstance(payload, dict):
        raise BacktestError("回测请求必须是对象")
    strategy_id = payload.get("strategy_id")
    if not strategy_id:
        raise BacktestError("缺少 strategy_id")
    version = payload.get("version")
    if version is not None and (not isinstance(version, int) or version <= 0):
        raise SpecError("version 必须是正整数")
    init_cash = payload.get("init_cash", 1_000_000.0)
    start, end = payload.get("start", ""), payload.get("end", "")
    universe = payload.get("universe")
    if universe is not None:
        resolve_universe(universe)
    benchmark = payload.get("benchmark")
    request = RunRequest(
        strategy_id=str(strategy_id),
        start=start,
        end=end,
        version=version,
        init_cash=float(init_cash),
        universe=universe,
        benchmark=benchmark,
    )
    request.params  # 触发区间与初始资金校验
    return request


def run_strategy(payload: dict, store_path: str | None = None) -> dict:
    """同步跑完一次回测并保存结果。"""
    store = _store(store_path)
    request = prepare(payload)
    return run(store, request)


def stream_strategy(request: RunRequest, should_cancel, store_path: str | None = None) -> Iterator[tuple[str, dict]]:
    """返回供调度器消费的阶段事件生成器。"""
    return iter_run(_store(store_path), request, should_cancel)


def backtest_params(payload: dict) -> BacktestParams:
    """把请求体转成回测参数（兼容旧调用）。"""
    return prepare(payload).params

