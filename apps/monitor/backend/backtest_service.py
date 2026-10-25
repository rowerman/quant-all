"""策略和回测 HTTP 接口使用的服务函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from backtest import (
    BacktestParams,
    BacktestStore,
    RunRequest,
    StrategySpec,
    iter_run,
    run,
)
from backtest.config import CHUNK_CELL_BUDGET, MAX_CHUNKS, MAX_PANEL_CELLS, MAX_TRADE_DAYS
from backtest.coverage import benchmark_coverages, default_run_end, market_coverage, validate_window
from backtest.costs import CostSpec, cost_model_summary
from backtest.errors import BacktestError
from backtest.pipeline import effective_spec
from backtest.providers import (
    list_providers,
    provider_metadata,
    validate_provider_config,
    validate_strategy_provider,
)
from backtest.specs import BENCHMARK_OPTIONS, ProviderSpec, SpecError
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
    market = market_coverage()
    coverages = benchmark_coverages([item["code"] for item in BENCHMARK_OPTIONS])
    benchmarks = [
        {
            **item,
            "start": coverages[item["code"]].start,
            "end": coverages[item["code"]].end,
            "default_end": (
                None
                if coverages[item["code"]].end is None
                else min(str(market.end), str(coverages[item["code"]].end))
            ),
        }
        for item in BENCHMARK_OPTIONS
    ]
    return {
        "strategies": strategies,
        "providers": list_providers(),
        "universes": _universe_options(),
        "benchmarks": benchmarks,
        "factors": factors[["name", "family", "direction", "description"]].to_dict("records"),
        "defaults": {
            "start": "2024-01-01", "end": default_run_end("H00300.CSI"), "init_cash": 1_000_000.0,
            "benchmark": "H00300.CSI", "risk_free_rate": 0.02,
        },
        "coverage": {
            "market_start": market.start,
            "market_end": market.end,
            "trading_days": market.trading_days,
        },
        "limits": {
            "trade_days": MAX_TRADE_DAYS,
            "panel_cells": MAX_PANEL_CELLS,
            "chunk_cells": CHUNK_CELL_BUDGET,
            "max_chunks": MAX_CHUNKS,
            "workers": 1,
            "waiting": 2,
        },
        "costs": {
            "commission_rate": 0.0003,
            "minimum_commission": 5.0,
            "slippage": 0.0005,
            "dividend_tax_mode": "fifo_deferred",
            "stamp_tax": "historical",
            "transfer_fee": "historical",
            "fee_rounding": "cent",
            "schedule": cost_model_summary(StrategySpec().costs),
            "impact_model": StrategySpec().costs.impact_model,
            "impact_coefficient": StrategySpec().costs.impact_coefficient,
            "impact_volatility_window": StrategySpec().costs.impact_volatility_window,
            "impact_max_rate": StrategySpec().costs.impact_max_rate,
        },
    }


def create_strategy(payload: dict, store_path: str | None = None) -> dict:
    """创建策略第一版。"""
    if not isinstance(payload, dict):
        raise BacktestError("策略请求必须是对象")
    raw = payload.get("spec", payload)
    if isinstance(raw, dict) and "costs" not in raw:
        raw = {**raw, "costs": StrategySpec().to_dict()["costs"]}
    spec = StrategySpec.from_dict(raw)
    strategy_id, version = _store(store_path).create_strategy(spec, payload.get("description", ""))
    provider = (
        None
        if spec.signal.provider is None
        else provider_metadata(spec.signal.provider, include_param_schema=True)
    )
    return {
        "strategy_id": strategy_id,
        "version": version,
        "config_hash": spec.config_hash,
        "provider": provider,
    }


def validate_provider(payload: dict) -> dict:
    """校验提供者参数和必需因子，返回规范化结果。"""
    if not isinstance(payload, dict):
        raise SpecError("提供者校验请求必须是对象")
    raw = payload.get("provider")
    spec = ProviderSpec.from_dict(raw)
    factors = payload.get("factors")
    if not isinstance(factors, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in factors
    ):
        raise SpecError("factors 必须是因子名序列")
    params = validate_provider_config(spec, factors)
    return {
        "provider": provider_metadata(spec, include_param_schema=True),
        "params": params,
    }


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


def delete_run(run_id: str, store_path: str | None = None) -> dict:
    """删除一次已结束运行及其全部结果。"""
    _store(store_path).delete_run(run_id)
    return {"run_id": run_id, "deleted": True}


def compare_runs(run_ids: list[str], store_path: str | None = None) -> dict:
    """按共同交易日对齐 2～4 次成功运行并生成可比曲线。

    参数:
        run_ids: 按页面选择顺序传入的运行标识。
        store_path: 测试使用的数据库路径。
    """
    if not 2 <= len(run_ids) <= 4:
        raise BacktestError("运行对比必须选择 2～4 次运行")
    if len(set(run_ids)) != len(run_ids):
        raise BacktestError("运行对比不能包含重复运行")
    store = _store(store_path)
    records = [store.get_run(run_id) for run_id in run_ids]
    for record in records:
        if record["status"] != "success":
            raise BacktestError(f"只有成功运行可以对比：{record['run_id']} 当前为 {record['status']}")

    frames = []
    for run_id in run_ids:
        frame = pd.DataFrame(store.get_result_table(run_id, "daily"))
        required = {"trade_date", "nav", "excess_return"}
        missing = required - set(frame.columns)
        if missing:
            raise BacktestError(f"运行 {run_id} 缺少对比字段：{sorted(missing)}")
        frame = frame[["trade_date", "nav", "excess_return"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        frames.append(frame)

    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    if len(common) < 2:
        raise BacktestError("所选运行的共同交易日不足 2 天，无法对比")

    payload = []
    for record, frame in zip(records, frames, strict=True):
        aligned = frame.loc[common]
        nav = pd.to_numeric(aligned["nav"], errors="coerce")
        excess = pd.to_numeric(aligned["excess_return"], errors="coerce")
        if nav.isna().any() or excess.isna().any() or float(nav.iloc[0]) == 0.0:
            raise BacktestError(f"运行 {record['run_id']} 的共同区间存在无效净值或超额收益")
        relative_nav = nav / float(nav.iloc[0])
        excess_growth = (1.0 + excess).cumprod()
        cumulative_excess = excess_growth / float(excess_growth.iloc[0]) - 1.0
        metadata = json.loads(record["metadata_json"]) if record.get("metadata_json") else {}
        strategy = metadata.get("strategy") or {}
        benchmark = metadata.get("benchmark") or {}
        payload.append(
            {
                "run_id": record["run_id"],
                "strategy_id": record["strategy_id"],
                "strategy_name": strategy.get("name") or record["strategy_id"],
                "strategy_version": record["strategy_version"],
                "start_date": record["start_date"],
                "end_date": record["end_date"],
                "universe": record.get("universe") or metadata.get("universe"),
                "benchmark": record.get("benchmark") or benchmark.get("code"),
                "snapshot_id": record.get("snapshot_id") or metadata.get("snapshot_id"),
                "metrics": metadata.get("metrics") or {},
                "nav": relative_nav.tolist(),
                "cumulative_excess": cumulative_excess.tolist(),
            }
        )
    return {
        "start": common[0].date().isoformat(),
        "end": common[-1].date().isoformat(),
        "dates": [item.date().isoformat() for item in common],
        "runs": payload,
    }


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


def prepare(payload: dict, store_path: str | None = None) -> RunRequest:
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
    try:
        risk_free_rate = float(payload.get("risk_free_rate", 0.02))
    except (TypeError, ValueError) as error:
        raise SpecError("risk_free_rate 必须是数字") from error
    start, end = payload.get("start", ""), payload.get("end", "")
    universe = payload.get("universe")
    if universe is not None:
        resolve_universe(universe)
    has_benchmark = "benchmark" in payload
    raw_benchmark = payload.get("benchmark")
    clear_benchmark = has_benchmark and (
        raw_benchmark is None or str(raw_benchmark).strip() == ""
    )
    benchmark = None if clear_benchmark else raw_benchmark
    request = RunRequest(
        strategy_id=str(strategy_id),
        start=start,
        end=end,
        version=version,
        init_cash=float(init_cash),
        risk_free_rate=float(risk_free_rate),
        universe=universe,
        benchmark=benchmark,
        clear_benchmark=clear_benchmark,
    )
    request.params  # 触发区间与初始资金校验
    store = _store(store_path)
    store.ensure_builtins()
    spec = effective_spec(store.get_version(request.strategy_id, request.version), request)
    validate_strategy_provider(spec)
    validate_window(spec.benchmark.index_code, request.start, request.end)
    return request


def run_strategy(payload: dict, store_path: str | None = None) -> dict:
    """同步跑完一次回测并保存结果。"""
    store = _store(store_path)
    request = prepare(payload, store_path)
    return run(store, request)


def stream_strategy(request: RunRequest, should_cancel, store_path: str | None = None) -> Iterator[tuple[str, dict]]:
    """返回供调度器消费的阶段事件生成器。"""
    return iter_run(_store(store_path), request, should_cancel)


def backtest_params(payload: dict) -> BacktestParams:
    """把请求体转成回测参数（兼容旧调用）。"""
    return prepare(payload).params

