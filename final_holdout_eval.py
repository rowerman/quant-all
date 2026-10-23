"""最终留出集的模型层指标、暴露和预注册决策门。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha.errors import AlphaFinalHoldoutError
from alpha.evaluate import evaluate_score_frame, p3_decision_gate
from alpha.exposure import PortfolioExposureResult, compute_portfolio_exposure
from alpha.portfolio import smooth_rank_scores

HOLDOUT_YEARS = (2024, 2025, 2026)
MODEL_METRIC_KEYS = (
    "rank_ic_mean",
    "rank_ic_std",
    "icir",
    "rank_ic_t",
    "rank_ic_win_rate",
    "days",
    "monotonicity",
    "top_minus_bottom_mean",
    "score_autocorrelation_1w",
)


def _panel_frames(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把留出预测、标签和股票池对齐为宽表。"""

    work = data.loc[pd.to_datetime(data["trade_date"]).isin(dates)].copy()
    columns = pd.Index(sorted(work["stock_code"].astype(str).unique()))
    score = predictions.pivot(
        index="trade_date",
        columns="stock_code",
        values="score",
    ).reindex(index=dates, columns=columns)
    labels = work.pivot(
        index="trade_date",
        columns="stock_code",
        values="label_raw",
    ).reindex(index=dates, columns=columns)
    eligible = (
        work.pivot(
            index="trade_date",
            columns="stock_code",
            values="eligible",
        )
        .reindex(index=dates, columns=columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    if score.isna().all(axis=1).any():
        raise AlphaFinalHoldoutError("留出预测存在整日无分数")
    return score.astype("float64"), labels.astype("float64"), eligible


def holdout_score_matrix(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    smoothing_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回因果 rank 平滑后的分数、标签和股票池宽表。"""

    raw, labels, eligible = _panel_frames(predictions, data, dates)
    scores = smooth_rank_scores(
        raw,
        eligible,
        window=smoothing_weeks,
        min_periods=1,
    )
    return scores, labels, eligible


def evaluate_holdout_models(
    predictions: Mapping[str, pd.DataFrame],
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    smoothing_weeks: int,
    selected_name: str,
    reference_name: str,
    control_name: str,
) -> dict[str, dict]:
    """计算计划候选和控制的模型层指标。"""

    required = {"lightgbm", "ridge", "b4"}
    missing = sorted(required - set(predictions))
    if missing:
        raise AlphaFinalHoldoutError(f"缺少留出预测：{missing}")
    candidates = {
        selected_name: (predictions["lightgbm"], smoothing_weeks),
        reference_name: (predictions["ridge"], 1),
        control_name: (predictions["b4"], 1),
    }
    output = {}
    for name, (frame, weeks) in candidates.items():
        scores, labels, eligible = holdout_score_matrix(
            frame,
            data,
            dates,
            smoothing_weeks=weeks,
        )
        metrics = evaluate_score_frame(scores, labels, eligible, quantiles=10)
        missing_metrics = [key for key in MODEL_METRIC_KEYS if key not in metrics]
        if missing_metrics:
            raise AlphaFinalHoldoutError(
                f"{name} 模型指标缺失：{missing_metrics}"
            )
        output[name] = metrics
    return output


def compute_holdout_exposure(
    selected_predictions: pd.DataFrame,
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    top_n: int,
    smoothing_weeks: int,
) -> PortfolioExposureResult:
    """计算主候选 Top-N 目标组合的留出期风格和行业暴露。"""

    scores, _, eligible = holdout_score_matrix(
        selected_predictions,
        data,
        dates,
        smoothing_weeks=smoothing_weeks,
    )
    long = (
        scores.where(eligible)
        .stack(future_stack=True)
        .rename("score")
        .reset_index()
    )
    long.columns = ["trade_date", "stock_code", "score"]
    long = long.dropna(subset=["score"])
    return compute_portfolio_exposure(
        long,
        data,
        top_n=top_n,
        allow_holdout=True,
    )


def _merge_model_and_ledger(
    model_metrics: dict,
    ledger_metrics: dict,
) -> dict:
    """合并用于 P3 决策的模型层和组合层指标。"""

    payload = {
        key: model_metrics.get(key)
        for key in MODEL_METRIC_KEYS
    }
    payload.update(
        {
            key: ledger_metrics.get(key)
            for key in (
                "annual_return",
                "excess_annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "turnover_two_way_annual",
                "total_cost",
            )
        }
    )
    return payload


def evaluate_final_holdout_gate(
    model_metrics: Mapping[str, dict],
    ledger_metrics: Mapping[str, dict],
    yearly: pd.DataFrame,
    *,
    selected_name: str,
    reference_name: str,
) -> dict:
    """按预注册 P3 门判断四周平滑主候选。"""

    try:
        selected_model = model_metrics[selected_name]
        ridge_model = model_metrics[reference_name]
        selected_ledger = ledger_metrics[selected_name]
        ridge_ledger = ledger_metrics[reference_name]
    except KeyError as error:
        raise AlphaFinalHoldoutError(
            f"最终留出决策缺少候选：{error.args[0]}"
        ) from error
    selected = _merge_model_and_ledger(selected_model, selected_ledger)
    ridge = _merge_model_and_ledger(ridge_model, ridge_ledger)
    model_yearly = {
        int(item["year"]): item for item in selected_model.get("yearly") or []
    }
    ledger_yearly = yearly.loc[
        yearly["candidate"].eq(selected_name)
    ]
    yearly_rows = []
    for year in HOLDOUT_YEARS:
        model_row = model_yearly.get(year)
        ledger_row = ledger_yearly.loc[ledger_yearly["year"].eq(year)]
        yearly_rows.append(
            {
                "year": year,
                "rank_ic_mean": (
                    None if model_row is None else model_row.get("rank_ic_mean")
                ),
                "excess_total_return": (
                    None if ledger_row.empty else ledger_row.iloc[0]["excess_total_return"]
                ),
            }
        )
    gate = p3_decision_gate(
        selected,
        ridge,
        yearly_rows,
        expected_years=HOLDOUT_YEARS,
        required_improvements=2,
        turnover_multiple_limit=2.0,
    )
    return {
        "status": gate["status"],
        "passed": bool(gate["passed"]),
        "candidate": selected_name,
        "reference": reference_name,
        "metrics": selected,
        "reference_metrics": ridge,
        "yearly": yearly_rows,
        "gate": gate,
    }


def model_metrics_for_registry(model_metrics: Mapping[str, dict]) -> dict:
    """转成不依赖 numpy 类型的 JSON 结构。"""

    output = {}
    for name, payload in model_metrics.items():
        output[name] = {
            key: (
                float(value)
                if isinstance(value, np.generic)
                else value
            )
            for key, value in payload.items()
            if key != "bins"
        }
        output[name]["bins"] = payload.get("bins") or []
    return output


__all__ = [
    "HOLDOUT_YEARS",
    "compute_holdout_exposure",
    "evaluate_final_holdout_gate",
    "evaluate_holdout_models",
    "holdout_score_matrix",
    "model_metrics_for_registry",
]
