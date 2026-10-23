"""最终留出集的候选策略、完整账本执行与复用。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.errors import AlphaFinalHoldoutError
from alpha.final_holdout_data import FinalHoldoutSpec, read_final_holdout_json
from scripts.build_alpha.backtest_accept_artifacts import write_candidate_artifacts
from scripts.build_alpha.backtest_accept_report import (
    acceptance_checks,
    capacity_summary,
    yearly_summary,
)


def _strategy(
    *,
    holdout_id: str,
    candidate_name: str,
    prediction_name: str,
    smoothing_weeks: int,
    top_n: int,
    dates: list[str],
):
    """构造留出集完整账本策略。"""

    from alpha.config import DEFAULT_CONTRACT
    from backtest.specs import (
        BenchmarkSpec,
        PortfolioSpec,
        ProviderSpec,
        RebalanceSpec,
        SignalSpec,
        StrategySpec,
    )

    return StrategySpec(
        name=f"Final holdout {candidate_name} Top{top_n}",
        universe=DEFAULT_CONTRACT.universe,
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=ProviderSpec(
                name="final_holdout_rank",
                version="1",
                params={
                    "holdout_id": holdout_id,
                    "candidate": prediction_name,
                    "top_n": top_n,
                    "smoothing_weeks": smoothing_weeks,
                    "rebalance_dates": dates,
                },
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=top_n),
        rebalance=RebalanceSpec(frequency="custom"),
        benchmark=BenchmarkSpec(DEFAULT_CONTRACT.benchmark),
    )


def run_candidate(
    *,
    root: Path,
    holdout_id: str,
    candidate_name: str,
    prediction_name: str,
    smoothing_weeks: int,
    spec: FinalHoldoutSpec,
    dates: pd.DatetimeIndex,
):
    """运行并落盘一个完整账本候选。"""

    from alpha.config import DEFAULT_CONTRACT
    from backtest.chunked import run_chunked
    from backtest.specs import BacktestParams
    from backtest.strategies._final_holdout import ensure_registered

    ensure_registered()
    strategy = _strategy(
        holdout_id=holdout_id,
        candidate_name=candidate_name,
        prediction_name=prediction_name,
        smoothing_weeks=smoothing_weeks,
        top_n=spec.top_n,
        dates=[str(day.date()) for day in dates],
    )
    params = BacktestParams(
        start=str(dates.min().date()),
        end=spec.end,
        init_cash=float(spec.init_cash),
        risk_free_rate=float(spec.risk_free_rate),
    )
    result = run_chunked(strategy, params)
    yearly = yearly_summary(result.daily)
    capacity = capacity_summary(result.metadata)
    daily_dates = pd.DatetimeIndex(result.daily.index).sort_values()
    positions = daily_dates.get_indexer(dates)
    execution_dates = [
        daily_dates[position + 1]
        for position in positions
        if position >= 0 and position + 1 < len(daily_dates)
    ]
    actual_signals = int(
        (result.metadata.get("signal") or {}).get("rebalance_days", -1)
    )
    checks = acceptance_checks(
        result.metrics,
        result.metadata,
        result.daily,
        dev_start=spec.start,
        dev_end=spec.end,
        expected_signal_days=len(dates),
        expected_execution_days=len(execution_dates),
        actual_signal_days=actual_signals,
    )
    scope = {
        "start": str(dates.min().date()),
        "end": spec.end,
        "init_cash": float(spec.init_cash),
        "risk_free_rate": float(spec.risk_free_rate),
        "top_n": spec.top_n,
        "benchmark": DEFAULT_CONTRACT.benchmark,
        "max_participation_rate": float(spec.max_participation_rate),
        "rebalance_dates": [str(day.date()) for day in dates],
        "rebalance_dates_count": len(dates),
    }
    metrics = {
        "candidate": candidate_name,
        "engine": "ledger",
        "source": prediction_name,
        "scope": scope,
        "strategy_hash": strategy.config_hash,
        "snapshot_id": result.metadata.get("snapshot_id"),
        "metrics": result.metrics,
        "yearly": yearly.to_dict(orient="records"),
        "capacity": capacity,
        "checks": checks,
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
    }
    run_manifest = {
        **result.metadata,
        "candidate": candidate_name,
        "source": prediction_name,
        "acceptance_checks": checks,
    }
    folder = root / "acceptance" / candidate_name
    write_candidate_artifacts(
        folder,
        analysis_id=holdout_id,
        candidate=candidate_name,
        metrics=metrics,
        strategy=strategy.to_dict(),
        yearly=yearly,
        daily=result.daily,
        orders=result.orders,
        holdings=result.holdings,
        run_manifest=run_manifest,
    )
    return {
        "metrics": result.metrics,
        "yearly": yearly,
        "capacity": capacity,
        "checks": checks,
        "run_manifest": run_manifest,
        "result": result,
    }


def load_candidate_payload(root: Path, candidate_name: str) -> dict:
    """读取已完成的候选账本，供失败续跑复用。"""

    folder = root / "acceptance" / candidate_name
    metrics_path = folder / "metrics.json"
    yearly_path = folder / "yearly.csv"
    run_manifest_path = folder / "run_manifest.json"
    complete_path = folder / "complete.json"
    for path in (metrics_path, yearly_path, run_manifest_path, complete_path):
        if not path.is_file():
            raise AlphaFinalHoldoutError(f"续跑候选产物缺失：{path}")
    payload = read_final_holdout_json(metrics_path, f"{candidate_name} 指标")
    complete = read_final_holdout_json(
        complete_path,
        f"{candidate_name} 完成标记",
    )
    expected_files = complete.get("files") or {}
    for relative, expected in expected_files.items():
        path = folder / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise AlphaFinalHoldoutError(
                f"续跑候选文件缺失或哈希不一致：{path}"
            )
    result = None
    if (folder / "daily.parquet").is_file() and (
        folder / "holdings.parquet"
    ).is_file():
        daily = pd.read_parquet(folder / "daily.parquet")
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        daily = daily.set_index("trade_date")
        holdings = pd.read_parquet(folder / "holdings.parquet")
        result = SimpleNamespace(daily=daily, holdings=holdings)
    return {
        "metrics": payload["metrics"],
        "yearly": pd.read_csv(yearly_path),
        "capacity": payload["capacity"],
        "checks": payload["checks"],
        "run_manifest": read_final_holdout_json(
            run_manifest_path,
            f"{candidate_name} run manifest",
        ),
        "result": result,
    }


__all__ = ["load_candidate_payload", "run_candidate"]
