"""P0 快速周频研究模拟器：Top-N 周收益、换手和成本。

这不是回测执行账本的替代品。它用下一周的开盘到开盘标签计算组合收益，
用 ``CostSpec`` 的费率估算换手成本，适合四条基线的同口径排序；
P4 接入策略时仍必须使用 ``backtest`` 的完整账本。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import quant
from alpha.calendar import load_calendar, weekly_signal_dates
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.errors import AlphaDataError
from alpha.portfolio import HoldingPolicy, select_holdings
from backtest.costs import CostSpec
from backtest.trading_rules import BOARD_BSE, BSE_TRADING_START
from scripts.build_alpha.baselines import BASELINE_IDS, scores_for_baseline
from scripts.build_alpha.panel import (
    data_versions,
    eligible_wide,
    implemented_factor_names,
    load_year_panel,
    to_wide,
)

PERIODS_PER_YEAR = 52
INITIAL_CASH = 1_000_000.0


def _auxiliary_panels(
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """读取成交额、20 日波动、基准周收益和股票板块。"""

    amount = quant.load_panel(
        start=start, end=end, fields=["amount_cny"], adjust="raw"
    )
    amount_wide = amount.pivot(
        index="trade_date", columns="stock_code", values="amount_cny"
    ).astype("float32")
    returns = quant.load_panel(
        start=start, end=end, fields=["ret"], adjust="hfq"
    )
    ret_wide = returns.pivot(
        index="trade_date", columns="stock_code", values="ret"
    ).astype("float32")
    impact = ret_wide.rolling(20, min_periods=10).std(ddof=1).shift(1)
    benchmark_raw = quant.load_index_daily(
        index_code=DEFAULT_CONTRACT.benchmark, start=start, end=end
    )
    benchmark = benchmark_raw.sort_values("trade_date").set_index("trade_date")
    benchmark_returns = (
        pd.to_numeric(benchmark["open"], errors="coerce").shift(-6)
        / pd.to_numeric(benchmark["open"], errors="coerce").shift(-1)
        - 1.0
    )
    basic = quant.load_stock_basic()
    board = basic.set_index("stock_code")["market"].astype(str)
    return amount_wide, impact, benchmark_returns, board


def _select_top(
    score: pd.Series,
    eligible: pd.Series,
    top_n: int,
) -> pd.Index:
    """按分数选当日可交易的前 N 只。"""

    return select_holdings(
        score,
        eligible,
        None,
        top_n=top_n,
        policy=HoldingPolicy(),
    )


def _cost_for_trades(
    day: pd.Timestamp,
    delta: pd.Series,
    amount: pd.DataFrame,
    impact: pd.DataFrame,
    board: pd.Series,
    cost: CostSpec,
    portfolio_value: float,
) -> tuple[float, dict]:
    """按目标权重变化估算一次调仓的现金成本。"""

    active = delta[delta.abs() > 1e-12]
    if active.empty:
        return 0.0, {}
    codes = active.index.astype(str).to_numpy()
    sides = np.sign(active.to_numpy(dtype="float64")).astype("int8")
    quantities = np.abs(active.to_numpy(dtype="float64")) * portfolio_value
    reference = np.ones(len(codes), dtype="float64")
    boards = board.reindex(codes).fillna("主板").to_numpy(dtype=object)
    amount_values = (
        amount.loc[day].reindex(codes).to_numpy(dtype="float64")
        if day in amount.index
        else np.full(len(codes), np.nan)
    )
    impact_values = (
        impact.loc[day].reindex(codes).to_numpy(dtype="float64")
        if day in impact.index
        else np.full(len(codes), np.nan)
    )
    charges = cost.charge_arrays(
        day,
        sides,
        quantities,
        reference,
        codes,
        boards,
        amount=amount_values,
        volatility=impact_values,
    )
    breakdown = {
        "commission": float(np.nansum(charges.commission)),
        "stamp_tax": float(np.nansum(charges.stamp_tax)),
        "transfer_fee": float(np.nansum(charges.transfer_fee)),
        "slippage_cost": float(np.nansum(charges.slippage_cost)),
        "impact_cost": float(np.nansum(charges.impact_cost)),
    }
    return float(np.nansum(charges.total_cost)), breakdown


def _metrics(
    daily: pd.DataFrame,
    benchmark: pd.Series,
) -> dict:
    """从周频结果计算年化、回撤、换手和成本。"""

    returns = daily["net_return"].astype(float)
    nav = daily["nav"].astype(float)
    periods = len(daily)
    total = float(nav.iloc[-1] - 1.0)
    annual = (
        (1.0 + total) ** (PERIODS_PER_YEAR / periods) - 1.0
        if periods and total > -1.0
        else None
    )
    drawdown = nav / nav.cummax() - 1.0
    bench = benchmark.reindex(daily.index).fillna(0.0)
    bench_nav = (1.0 + bench).cumprod()
    bench_total = float(bench_nav.iloc[-1] - 1.0) if periods else None
    bench_annual = (
        (1.0 + bench_total) ** (PERIODS_PER_YEAR / periods) - 1.0
        if bench_total is not None
        else None
    )
    volatility = float(returns.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    cost = float(daily["cost"].sum())
    return {
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "periods": periods,
        "total_return": total,
        "annual_return": annual,
        "annual_volatility": volatility,
        "max_drawdown": float(drawdown.min()),
        "turnover_two_way_annual": float(daily["turnover"].sum() * PERIODS_PER_YEAR / periods),
        "turnover_one_way_annual": float(
            daily["turnover_one_way"].sum() * PERIODS_PER_YEAR / periods
        ),
        "total_cost": cost,
        "total_cost_bps_of_value": cost / INITIAL_CASH * 10_000.0,
        "benchmark_total_return": bench_total,
        "benchmark_annual_return": bench_annual,
        "excess_annual_return": (
            None if annual is None or bench_annual is None else annual - bench_annual
        ),
        "sharpe": (
            float(returns.mean() / returns.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
            if returns.std(ddof=1) > 0
            else None
        ),
    }


def simulate_score_frame(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    start: str,
    end: str,
    top_n: int = DEFAULT_CONTRACT.top_n,
    auxiliary: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None = None,
    holding_policy: HoldingPolicy | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """用宽分数面板运行同一套周频 Top-N 研究模拟。"""

    if end > DEFAULT_CONTRACT.dev_end:
        raise AlphaDataError("快速模拟器不能在最终测试集上运行")
    if top_n <= 0:
        raise AlphaDataError("top_n 必须为正整数")
    dates = pd.DatetimeIndex(scores.index).sort_values()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if dates.empty:
        raise AlphaDataError("分数面板没有目标区间的信号日")
    amount, impact, benchmark, board = (
        auxiliary if auxiliary is not None else _auxiliary_panels(start, end)
    )
    policy = holding_policy or HoldingPolicy()
    cost = CostSpec(impact_model="sqrt")
    nav = 1.0
    previous = pd.Series(dtype="float64")
    bse_columns = board.index[board.eq(BOARD_BSE)]
    rows: list[dict] = []
    orders: list[dict] = []
    for day in dates:
        score = scores.loc[day]
        eligible_day = eligible.reindex(columns=score.index).loc[day].copy()
        if day < pd.Timestamp(BSE_TRADING_START):
            eligible_day.loc[
                eligible_day.index.intersection(bse_columns)
            ] = False
        chosen = select_holdings(
            score,
            eligible_day,
            previous.index,
            top_n=top_n,
            policy=policy,
        )
        returns = labels.reindex(index=[day], columns=chosen).iloc[0].dropna()
        chosen = returns.index
        weights = pd.Series(0.0, index=score.index, dtype="float64")
        if len(chosen):
            weights.loc[chosen] = 1.0 / len(chosen)
        columns = previous.index.union(weights.index)
        delta = weights.reindex(columns).fillna(0.0) - previous.reindex(
            columns
        ).fillna(0.0)
        portfolio_value = nav * INITIAL_CASH
        trade_cost, breakdown = _cost_for_trades(
            day,
            delta,
            amount,
            impact,
            board,
            cost,
            portfolio_value,
        )
        gross = float((returns * (1.0 / len(chosen))).sum()) if len(chosen) else 0.0
        net = max(gross - trade_cost / portfolio_value, -0.999)
        nav = max(nav * (1.0 + net), 1e-9)
        rows.append(
            {
                "trade_date": day,
                "gross_return": gross,
                "net_return": net,
                "benchmark_return": float(
                    benchmark.reindex([day]).fillna(0.0).iloc[0]
                ),
                "cost": trade_cost,
                "turnover": float(delta.abs().sum()),
                "turnover_one_way": float(delta.clip(lower=0.0).sum()),
                "nav": nav,
                **breakdown,
            }
        )
        for code in delta.index[delta.abs() > 1e-12]:
            orders.append(
                {
                    "trade_date": day,
                    "stock_code": code,
                    "delta_weight": float(delta.loc[code]),
                    "notional": float(delta.loc[code] * portfolio_value),
                }
            )
        previous = weights.loc[weights > 0.0]
    daily = pd.DataFrame(rows).set_index("trade_date").sort_index()
    metrics = _metrics(daily, benchmark.reindex(daily.index))
    return metrics, daily, pd.DataFrame(orders)


def simulate_baseline(
    baseline_id: str,
    start: str,
    end: str,
    *,
    top_n: int = DEFAULT_CONTRACT.top_n,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """运行一条快速周频基线，返回指标、周结果和订单明细。"""

    if baseline_id not in BASELINE_IDS:
        raise AlphaDataError(f"未知基线 {baseline_id}")
    if end > DEFAULT_CONTRACT.dev_end:
        raise AlphaDataError("快速模拟器不能在最终测试集上运行")
    factor_names = implemented_factor_names()
    amount, impact, benchmark, board = _auxiliary_panels(start, end)
    cost = CostSpec(impact_model="sqrt")
    nav = 1.0
    previous = pd.Series(dtype="float64")
    bse_columns = board.index[board.eq(BOARD_BSE)]
    rows: list[dict] = []
    orders: list[dict] = []
    first_year = pd.Timestamp(start).year
    last_year = pd.Timestamp(end).year
    full_weekly = weekly_signal_dates(load_calendar(start, end))
    for year in range(first_year, last_year + 1):
        panel = load_year_panel(year, factor_names)
        year_start = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
        year_end = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31"))
        dates = full_weekly[
            (full_weekly >= year_start) & (full_weekly <= year_end)
        ]
        if dates.empty:
            continue
        score = scores_for_baseline(panel.frame, baseline_id, factor_names)
        eligible = eligible_wide(panel.frame)
        labels = to_wide(panel.frame, "fwd_ret_open_5").reindex(
            index=score.index, columns=score.columns
        )
        cutoff = pd.Timestamp(BSE_TRADING_START)
        for day in dates:
            eligible_day = eligible.loc[day].copy()
            if day < cutoff:
                eligible_day.loc[
                    eligible_day.index.intersection(bse_columns)
                ] = False
            chosen = _select_top(score.loc[day], eligible_day, top_n)
            returns = labels.loc[day].reindex(chosen).dropna()
            chosen = returns.index
            weights = pd.Series(0.0, index=score.columns, dtype="float64")
            if len(chosen):
                weights.loc[chosen] = 1.0 / len(chosen)
            columns = previous.index.union(weights.index)
            delta = weights.reindex(columns).fillna(0.0) - previous.reindex(
                columns
            ).fillna(0.0)
            portfolio_value = nav * INITIAL_CASH
            trade_cost, breakdown = _cost_for_trades(
                day,
                delta,
                amount,
                impact,
                board,
                cost,
                portfolio_value,
            )
            gross = float((returns * (1.0 / len(chosen))).sum()) if len(chosen) else 0.0
            net = max(gross - trade_cost / portfolio_value, -0.999)
            nav = max(nav * (1.0 + net), 1e-9)
            rows.append(
                {
                    "trade_date": day,
                    "gross_return": gross,
                    "net_return": net,
                    "cost": trade_cost,
                    "turnover": float(delta.abs().sum()),
                    "turnover_one_way": float(delta.clip(lower=0.0).sum()),
                    "nav": nav,
                    **breakdown,
                }
            )
            for code in delta.index[delta.abs() > 1e-12]:
                orders.append(
                    {
                        "trade_date": day,
                        "stock_code": code,
                        "delta_weight": float(delta.loc[code]),
                        "notional": float(delta.loc[code] * portfolio_value),
                    }
                )
            previous = weights
    if not rows:
        raise AlphaDataError(f"{baseline_id} 没有生成任何周频结果")
    daily = pd.DataFrame(rows).set_index("trade_date").sort_index()
    benchmark_returns = benchmark.reindex(daily.index).fillna(0.0)
    metrics = _metrics(daily, benchmark_returns)
    return metrics, daily, pd.DataFrame(orders)


def run_and_write(baseline_id: str, start: str, end: str) -> dict:
    """运行快速模拟器并写入基线目录。"""

    metrics, daily, orders = simulate_baseline(baseline_id, start, end)
    folder = MODEL_DATA_DIR / "baselines" / baseline_id
    folder.mkdir(parents=True, exist_ok=True)
    versions = data_versions()
    payload = {
        "baseline_id": baseline_id,
        "start": start,
        "end": end,
        "engine": "weekly_research_simulator",
        "portfolio": metrics,
        "yearly": yearly_summary(daily),
        "data_versions": versions,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "ic": _read_ic(baseline_id),
        "snapshot_id": None,
    }
    from scripts.build_alpha.baselines import baseline_strategy

    strategy = baseline_strategy(baseline_id, implemented_factor_names())
    from scripts.build_alpha.baselines import write_json

    write_json(folder / "config.json", strategy.to_dict())
    write_json(
        folder / "manifest.json",
        {
            "engine": "weekly_research_simulator",
            "start": start,
            "end": end,
            "data_versions": versions,
            "rows": int(len(daily)),
            "orders": int(len(orders)),
        },
    )
    daily.reset_index().to_parquet(folder / "daily.parquet", index=False)
    orders.to_parquet(folder / "orders.parquet", index=False)
    (folder / "metrics.json").write_text(
        json_dumps(payload), encoding="utf-8"
    )
    experiment_id = hashlib.sha256(
        (
            f"{baseline_id}|weekly|{start}|{end}|"
            f"{versions['data_version']}|{payload['code_sha256']}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    from scripts.build_alpha.baselines import _append_experiment

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
            "config_hash": strategy.config_hash,
            "feature_set": "all_implemented"
            if baseline_id in ("B3_ALL_ZSCORE", "B4_ALL_RANK")
            else "linear_subset",
            "data_version": versions["data_version"],
            "factor_version": versions["factor_version"],
            "snapshot_id": None,
            "metrics_json": json.dumps(
                {
                    "annual_return": metrics.get("annual_return"),
                    "excess_annual_return": metrics.get("excess_annual_return"),
                    "rank_ic_mean": (payload["ic"] or {}).get("rank_ic_mean"),
                    "icir": (payload["ic"] or {}).get("rank_ic_ir"),
                    "engine": "weekly_research_simulator",
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            "notes": "research weekly simulator; not ledger execution",
        }
    )
    return payload


def json_dumps(payload: dict) -> str:
    """把 numpy 标量转成 JSON 文本。"""

    import json

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n"


def yearly_summary(daily: pd.DataFrame) -> list[dict]:
    """按年汇总周频净收益、成本、换手和回撤。"""

    frame = daily.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    rows = []
    for year, group in frame.groupby(frame.index.year, sort=True):
        nav = group["nav"].astype(float)
        drawdown = nav / nav.cummax() - 1.0
        total = float((1.0 + group["net_return"].astype(float)).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "periods": int(len(group)),
                "total_return": total,
                "max_drawdown": float(drawdown.min()),
                "cost": float(group["cost"].sum()),
                "turnover": float(group["turnover"].sum()),
            }
        )
    return rows


def _read_ic(baseline_id: str) -> dict | None:
    """读取 P0 审计阶段生成的基线 IC。"""

    path = MODEL_DATA_DIR / "audit" / "baseline_ic" / baseline_id / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
