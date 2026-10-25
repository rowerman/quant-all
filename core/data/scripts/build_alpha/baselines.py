"""四条 P0 基线的定义、回测执行与结果落盘。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import factors
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.errors import AlphaConfigError, AlphaDataError
from backtest.chunked import run_chunked
from backtest.providers import provider_metadata
from backtest.snapshot import build_data_snapshot
from backtest.specs import (
    BacktestParams,
    BenchmarkSpec,
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.strategies._factor_rank import ensure_registered
from scripts.build_alpha.panel import (
    data_versions,
    implemented_factor_names,
    linear_zscore_score,
    rank_score,
)

BASELINE_IDS: tuple[str, ...] = (
    "B1_MOM20",
    "B2_MOM_EP_VOL",
    "B3_ALL_ZSCORE",
    "B4_ALL_RANK",
)


def _json_default(value: Any) -> Any:
    """把路径、时间与 numpy 标量转换成 JSON 可写对象。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """以稳定格式写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def baseline_weights(baseline_id: str) -> dict[str, float]:
    """返回基线使用的线性分数权重。"""

    if baseline_id == "B1_MOM20":
        return {"MOM_20": 1.0}
    if baseline_id == "B2_MOM_EP_VOL":
        return {"MOM_20": 1.0, "EP_TTM": 1.0, "VOL_20": -1.0}
    if baseline_id == "B3_ALL_ZSCORE":
        return {
            name: float(factors.resolve(name).direction)
            for name in implemented_factor_names()
        }
    raise AlphaConfigError(f"{baseline_id} 没有线性权重定义")


def baseline_strategy(
    baseline_id: str,
    factor_names: tuple[str, ...],
) -> StrategySpec:
    """构造四条基线之一的可复现策略配置。"""

    if baseline_id == "B4_ALL_RANK":
        return StrategySpec(
            name="P0 全因子 rank 等权",
            universe=DEFAULT_CONTRACT.universe,
            signal=SignalSpec(
                factors=tuple(factor_names),
                weights=(),
                provider=ProviderSpec(
                    name="factor_rank",
                    version="1",
                    params={"top_n": DEFAULT_CONTRACT.top_n, "factor_names": list(factor_names)},
                ),
            ),
            portfolio=PortfolioSpec(selection="provider", top_n=DEFAULT_CONTRACT.top_n),
            rebalance=RebalanceSpec(frequency="custom"),
            benchmark=BenchmarkSpec(DEFAULT_CONTRACT.benchmark),
        )
    weights = baseline_weights(baseline_id)
    return StrategySpec(
        name=f"P0 基线 {baseline_id}",
        universe=DEFAULT_CONTRACT.universe,
        signal=SignalSpec(
            factors=tuple(weights),
            weights=tuple(weights.values()),
            standardize=True,
            winsorize=False,
        ),
        portfolio=PortfolioSpec(
            selection="top_n",
            top_n=DEFAULT_CONTRACT.top_n,
            weighting=DEFAULT_CONTRACT.weighting,
        ),
        rebalance=RebalanceSpec(frequency=DEFAULT_CONTRACT.rebalance),
        benchmark=BenchmarkSpec(DEFAULT_CONTRACT.benchmark),
    )


def scores_for_baseline(
    frame: pd.DataFrame,
    baseline_id: str,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """在一年面板上计算某条基线的截面分数。"""

    if baseline_id == "B4_ALL_RANK":
        return rank_score(frame, factor_names)
    return linear_zscore_score(frame, baseline_weights(baseline_id))


def _guard_development(start: str, end: str) -> None:
    """拒绝把最终测试集带进研究运行。"""

    if start < DEFAULT_CONTRACT.dev_start:
        raise AlphaConfigError(
            f"基线起始日 {start} 早于研究契约 {DEFAULT_CONTRACT.dev_start}"
        )
    if end > DEFAULT_CONTRACT.dev_end:
        raise AlphaConfigError(
            f"基线截止日 {end} 超过开发期 {DEFAULT_CONTRACT.dev_end}；最终测试集未开放"
        )


def _shared_snapshot(
    factor_names: tuple[str, ...],
    start: str,
    end: str,
) -> dict:
    """为 47 因子基线构建并缓存同一份数据快照。"""

    versions = data_versions()
    cache = (
        MODEL_DATA_DIR
        / "baselines"
        / "_snapshots"
        / (
            f"all47_{start}_{end}_"
            f"{versions['data_version']}_{versions['factor_version']}.json"
        )
    )
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    snapshot = build_data_snapshot(
        start,
        end,
        list(factor_names),
        universe=DEFAULT_CONTRACT.universe,
        benchmark_code=DEFAULT_CONTRACT.benchmark,
        industry_report=True,
        impact_volatility_window=20,
    )
    write_json(cache, snapshot)
    return snapshot


def _yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """从日结果计算逐年收益、超额、换手和回撤。"""

    frame = daily.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    rows = []
    for year, group in frame.groupby(frame.index.year, sort=True):
        total_return = float((1.0 + group["return"].astype(float)).prod() - 1.0)
        row = {
            "year": int(year),
            "days": int(len(group)),
            "total_return": total_return,
            "max_drawdown": float(group["drawdown"].astype(float).min()),
            "turnover_two_way": float(group["turnover"].astype(float).sum()),
        }
        if "benchmark_return" in group.columns:
            benchmark = float(
                (1.0 + group["benchmark_return"].astype(float)).prod() - 1.0
            )
            row["benchmark_total_return"] = benchmark
            row["excess_total_return"] = total_return - benchmark
        rows.append(row)
    return pd.DataFrame(rows)


def _read_ic_summary(baseline_id: str) -> dict | None:
    """读取审计阶段生成的基线 IC 汇总。"""

    path = MODEL_DATA_DIR / "audit" / "baseline_ic" / baseline_id / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _append_experiment(row: dict) -> None:
    """幂等追加实验记录。"""

    path = MODEL_DATA_DIR / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if not existing.empty and row["experiment_id"] in set(existing["experiment_id"]):
        return
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.is_file(), index=False)


def run_baseline(
    baseline_id: str,
    start: str,
    end: str,
    *,
    force: bool = False,
    engine: str = "weekly",
) -> dict:
    """运行一条基线并写入 compact 结果。"""

    if baseline_id not in BASELINE_IDS:
        raise AlphaConfigError(f"未知基线 {baseline_id}；可选 {list(BASELINE_IDS)}")
    _guard_development(start, end)
    target = MODEL_DATA_DIR / "baselines" / baseline_id
    metrics_path = target / "metrics.json"
    if metrics_path.is_file() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if engine == "weekly":
        from scripts.build_alpha.weekly_sim import run_and_write

        return run_and_write(baseline_id, start, end)
    if engine != "ledger":
        raise AlphaConfigError("engine 只支持 weekly/ledger")
    factor_names = implemented_factor_names()

    strategy = baseline_strategy(baseline_id, factor_names)
    if baseline_id == "B4_ALL_RANK":
        ensure_registered()
    snapshot = (
        _shared_snapshot(factor_names, start, end)
        if baseline_id in ("B3_ALL_ZSCORE", "B4_ALL_RANK")
        else None
    )
    started = time.perf_counter()
    result = run_chunked(
        strategy,
        BacktestParams(start=start, end=end),
        data_snapshot=snapshot,
    )
    elapsed = time.perf_counter() - started
    versions = data_versions()
    ic = _read_ic_summary(baseline_id)
    metrics = {
        "baseline_id": baseline_id,
        "start": start,
        "end": end,
        "engine": "ledger",
        "portfolio": result.metrics,
        "yearly": _yearly_summary(result.daily).to_dict(orient="records"),
        "ic": ic,
        "elapsed_seconds": round(elapsed, 3),
        "data_versions": versions,
        "snapshot_id": (result.metadata or {}).get("snapshot_id"),
    }
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "config.json", strategy.to_dict())
    write_json(target / "metrics.json", metrics)
    write_json(target / "manifest.json", result.metadata or {})
    result.daily.reset_index().to_parquet(target / "daily.parquet", index=False)
    result.orders.to_parquet(target / "orders.parquet", index=False)
    if baseline_id == "B4_ALL_RANK":
        write_json(
            target / "provider.json",
            provider_metadata(strategy.signal.provider),
        )
    config_hash = strategy.config_hash
    experiment_id = hashlib.sha256(
        f"{baseline_id}|{config_hash}|{metrics['snapshot_id']}".encode("utf-8")
    ).hexdigest()[:16]
    _append_experiment(
        {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "batch": "P0",
            "name": baseline_id,
            "kind": "baseline",
            "stage": "development",
            "start": start,
            "end": end,
            "config_hash": config_hash,
            "feature_set": "all_implemented" if baseline_id.startswith("B3") or baseline_id.startswith("B4") else "linear_subset",
            "data_version": versions["data_version"],
            "factor_version": versions["factor_version"],
            "snapshot_id": metrics["snapshot_id"],
            "metrics_json": json.dumps(
                {
                    "annual_return": result.metrics.get("annual_return"),
                    "excess_annual_return": result.metrics.get("excess_annual_return"),
                    "rank_ic_mean": (ic or {}).get("rank_ic_mean"),
                    "icir": (ic or {}).get("rank_ic_ir"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "notes": "",
        }
    )
    return metrics


def write_comparison() -> pd.DataFrame:
    """汇总四条基线为一张同口径对照表。"""

    rows = []
    for baseline_id in BASELINE_IDS:
        path = MODEL_DATA_DIR / "baselines" / baseline_id / "metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        portfolio = payload["portfolio"]
        ic = payload.get("ic") or {}
        rows.append(
            {
                "baseline_id": baseline_id,
                "annual_return": portfolio.get("annual_return"),
                "excess_annual_return": portfolio.get("excess_annual_return"),
                "rank_ic_mean": ic.get("rank_ic_mean"),
                "icir": ic.get("rank_ic_ir"),
                "rank_ic_t": ic.get("rank_ic_t"),
                "rank_ic_win_rate": ic.get("rank_ic_win_rate"),
                "turnover_two_way_annual": portfolio.get("turnover_two_way_annual"),
                "max_drawdown": portfolio.get("max_drawdown"),
                "total_cost": portfolio.get("total_cost"),
                "engine": payload.get("engine", "ledger"),
                "snapshot_id": payload.get("snapshot_id"),
            }
        )
    frame = pd.DataFrame(rows)
    path = MODEL_DATA_DIR / "baselines" / "baseline_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
