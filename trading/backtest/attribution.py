"""回测成本、换手和行业暴露归因。

所有归因都从最终成交、逐日持仓和标准化日结果派生，不假设目标权重合计为 1，
也不假设每个交易日都有调仓。提供者保留现金、整行 ``NaN`` 或全 0 行时同样适用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import TRADING_DAYS, summarize


COST_COMPONENTS = (
    "commission",
    "stamp_tax",
    "transfer_fee",
    "dividend_tax",
    "slippage_cost",
    "impact_cost",
)


@dataclass(frozen=True)
class Report:
    """一次回测的指标与行业暴露明细。"""

    metrics: dict
    industry_exposure: pd.DataFrame


def _order_value(orders: pd.DataFrame, side: str) -> float:
    """返回指定方向的成交额；缺列时返回 0。"""
    if orders is None or orders.empty or "Side" not in orders.columns:
        return 0.0
    mask = orders["Side"].astype(str).str.lower().eq(side)
    if not mask.any():
        return 0.0
    return float(
        (
            orders.loc[mask, "Size"].astype(float).abs()
            * orders.loc[mask, "Price"].astype(float)
        ).sum()
    )


def _turnover_series(daily: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """按实际成交额生成逐日买卖和单双边换手。"""
    index = pd.DatetimeIndex(daily.index)
    result = pd.DataFrame(0.0, index=index, columns=["buy", "sell", "forced_exit"])
    if orders is not None and not orders.empty and "trade_date" in orders.columns:
        frame = orders.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["value"] = frame["Size"].astype(float).abs() * frame["Price"].astype(float)
        side = frame["Side"].astype(str).str.lower()
        buy = frame.loc[side.eq("buy")].groupby("trade_date")["value"].sum()
        sell = frame.loc[side.eq("sell")].groupby("trade_date")["value"].sum()
        result["buy"] = buy.reindex(index).fillna(0.0)
        result["sell"] = sell.reindex(index).fillna(0.0)
        if "forced_exit" in frame.columns:
            forced = frame.loc[
                side.eq("sell") & frame["forced_exit"].fillna(False).astype(bool)
            ]
            result["forced_exit"] = (
                forced.groupby("trade_date")["value"].sum().reindex(index).fillna(0.0)
            )
    previous = daily["portfolio_value"].astype(float).shift(1)
    if len(previous):
        previous.iloc[0] = float(daily["portfolio_value"].astype(float).iloc[0])
    previous = previous.replace(0, np.nan)
    result["buy_turnover"] = result["buy"] / previous
    result["sell_turnover"] = result["sell"] / previous
    result["forced_exit_turnover"] = result["forced_exit"] / previous
    result["one_way"] = result[["buy_turnover", "sell_turnover"]].max(axis=1)
    result["two_way"] = result["buy_turnover"] + result["sell_turnover"]
    return result


def turnover_attribution(daily: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """给出买卖成交额、强制退出和单双边换手。"""
    series = _turnover_series(daily, orders)
    days = max(len(series), 1)
    buy = float(series["buy"].sum())
    sell = float(series["sell"].sum())
    forced = float(series["forced_exit"].sum())
    one_way = float(series["one_way"].sum(skipna=True))
    two_way = float(series["two_way"].sum(skipna=True))
    return {
        "buy_value": buy,
        "sell_value": sell,
        "forced_exit_sell_value": forced,
        "one_way_annual": one_way * TRADING_DAYS / days,
        "two_way_annual": two_way * TRADING_DAYS / days,
        "buy_annual": float(series["buy_turnover"].sum(skipna=True)) * TRADING_DAYS / days,
        "sell_annual": float(series["sell_turnover"].sum(skipna=True)) * TRADING_DAYS / days,
        "forced_exit_annual": (
            float(series["forced_exit_turnover"].sum(skipna=True)) * TRADING_DAYS / days
        ),
        "forced_exit_share_of_sell": 0.0 if sell <= 0 else forced / sell,
    }


def _daily_component(daily: pd.DataFrame, name: str) -> float:
    """读取一个日度成本分项；缺失时返回 0。"""
    if name not in daily.columns:
        return 0.0
    return float(daily[name].fillna(0.0).astype(float).sum())


def _side_cost(orders: pd.DataFrame, side: str) -> dict:
    """按买卖方向汇总订单成本。"""
    result = {
        "commission": 0.0,
        "stamp_tax": 0.0,
        "transfer_fee": 0.0,
        "dividend_tax": 0.0,
        "slippage_cost": 0.0,
        "impact_cost": 0.0,
        "total_cost": 0.0,
    }
    if orders is None or orders.empty or "Side" not in orders.columns:
        return result
    mask = orders["Side"].astype(str).str.lower().eq(side)
    if not mask.any():
        return result
    frame = orders.loc[mask]
    for name in ("commission", "stamp_tax", "transfer_fee", "slippage_cost", "impact_cost"):
        if name in frame.columns:
            result[name] = float(frame[name].fillna(0.0).astype(float).sum())
    result["total_cost"] = (
        result["commission"]
        + result["stamp_tax"]
        + result["transfer_fee"]
        + result["slippage_cost"]
        + result["impact_cost"]
    )
    return result


def cost_attribution(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """给出成本分项、占净值/成交额 bps 和买卖方向拆分。"""
    average_value = float(daily["portfolio_value"].astype(float).mean())
    traded = (
        float(daily["buy_value"].fillna(0.0).astype(float).sum())
        + float(daily["sell_value"].fillna(0.0).astype(float).sum())
        if "buy_value" in daily.columns and "sell_value" in daily.columns
        else _order_value(orders, "buy") + _order_value(orders, "sell")
    )
    totals = {name: _daily_component(daily, name) for name in COST_COMPONENTS}
    cash_cost = sum(
        totals[name]
        for name in ("commission", "stamp_tax", "transfer_fee", "dividend_tax")
    )
    total_cost = cash_cost + totals["slippage_cost"] + totals["impact_cost"]
    bps_nav = {
        name: (0.0 if average_value <= 0 else value / average_value * 10_000.0)
        for name, value in totals.items()
    }
    bps_nav["cash_cost"] = 0.0 if average_value <= 0 else cash_cost / average_value * 10_000.0
    bps_nav["total_cost"] = 0.0 if average_value <= 0 else total_cost / average_value * 10_000.0
    bps_traded = {
        name: (0.0 if traded <= 0 else value / traded * 10_000.0)
        for name, value in totals.items()
    }
    bps_traded["cash_cost"] = 0.0 if traded <= 0 else cash_cost / traded * 10_000.0
    bps_traded["total_cost"] = 0.0 if traded <= 0 else total_cost / traded * 10_000.0
    by_side = {"buy": _side_cost(orders, "buy"), "sell": _side_cost(orders, "sell")}
    # 红利税按卖出批次结算，归到卖出侧；订单表本身没有这项。
    by_side["sell"]["dividend_tax"] = totals["dividend_tax"]
    by_side["sell"]["total_cost"] += totals["dividend_tax"]
    return {
        "risk_free_rate": float(risk_free_rate),
        "total": {
            **totals,
            "cash_cost": cash_cost,
            "total_cost": total_cost,
        },
        "bps_of_nav": bps_nav,
        "bps_of_traded_value": bps_traded,
        "by_side": by_side,
    }


def industry_exposure_frame(
    daily: pd.DataFrame,
    holdings: pd.DataFrame,
    industry_l1: pd.DataFrame | None,
    industry_labels: dict[int, str] | None,
) -> pd.DataFrame:
    """按实际持仓生成逐日行业、现金和未分类暴露。"""
    columns = ["trade_date", "industry", "market_value", "weight", "is_cash"]
    if industry_l1 is None:
        return pd.DataFrame(columns=columns)
    labels = industry_labels or {}
    index = pd.DatetimeIndex(daily.index)
    values = industry_l1.reindex(index=index).to_numpy(dtype="int16", copy=False)
    stock_lookup = {str(code): position for position, code in enumerate(industry_l1.columns)}
    portfolio_value = daily["portfolio_value"].astype(float)
    rows: list[dict] = []
    grouped = (
        holdings.copy()
        if holdings is not None and not holdings.empty
        else pd.DataFrame(columns=["trade_date", "stock_code", "market_value", "weight"])
    )
    if not grouped.empty:
        grouped["trade_date"] = pd.to_datetime(grouped["trade_date"])
    for row, day in enumerate(index):
        day_rows = grouped.loc[grouped["trade_date"].eq(day)] if not grouped.empty else grouped
        total = float(portfolio_value.loc[day])
        industry_value: dict[str, float] = {}
        for item in day_rows.itertuples(index=False):
            position = stock_lookup.get(str(item.stock_code))
            code = 0 if position is None else int(values[row, position])
            label = "未分类" if code <= 0 else labels.get(code, f"行业{code}")
            value = 0.0 if pd.isna(item.market_value) else float(item.market_value)
            industry_value[label] = industry_value.get(label, 0.0) + value
        for label, value in sorted(industry_value.items(), key=lambda pair: (-pair[1], pair[0])):
            weight = 0.0 if total <= 0 else value / total
            rows.append(
                {
                    "trade_date": day,
                    "industry": label,
                    "market_value": value,
                    "weight": weight,
                    "is_cash": False,
                }
            )
        cash = max(
            0.0,
            1.0 - sum(0.0 if total <= 0 else value / total for value in industry_value.values()),
        )
        rows.append(
            {
                "trade_date": day,
                "industry": "现金",
                "market_value": cash * total,
                "weight": cash,
                "is_cash": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_industry_exposure(
    exposure: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict:
    """把逐日行业暴露压缩成页面可直接展示的摘要。"""
    if exposure is None or exposure.empty:
        return {"available": False}
    index = pd.DatetimeIndex(daily.index)
    pivot = (
        exposure.pivot_table(
            index="trade_date",
            columns="industry",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index)
        .fillna(0.0)
    )
    industry_columns = [name for name in pivot.columns if name != "现金"]
    cash = pivot["现金"] if "现金" in pivot.columns else pd.Series(0.0, index=index)
    stats = []
    for name in industry_columns:
        series = pivot[name]
        stats.append(
            {
                "industry": str(name),
                "avg_weight": float(series.mean()),
                "max_weight": float(series.max()),
                "end_weight": float(series.iloc[-1]),
            }
        )
    stats.sort(key=lambda item: (-item["avg_weight"], item["industry"]))
    unclassified = pivot["未分类"] if "未分类" in pivot.columns else pd.Series(0.0, index=index)
    hhi = np.square(pivot.to_numpy(dtype="float64")).sum(axis=1).mean() if len(pivot) else 0.0
    return {
        "available": True,
        "days": int(len(index)),
        "cash_avg_weight": float(cash.mean()),
        "cash_max_weight": float(cash.max()),
        "cash_end_weight": float(cash.iloc[-1]) if len(cash) else 0.0,
        "unclassified_avg_weight": float(unclassified.mean()),
        "max_single_industry_weight": float(
            max((item["max_weight"] for item in stats), default=0.0)
        ),
        "hhi_avg": float(hhi),
        "top": stats[:10],
    }


def build_report(
    daily: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    holdings: pd.DataFrame | None = None,
    industry_l1: pd.DataFrame | None = None,
    industry_labels: dict[int, str] | None = None,
    risk_free_rate: float = 0.02,
    impact_model: str = "none",
    industry_exposure_rows: pd.DataFrame | None = None,
) -> Report:
    """从最终结果生成指标和归因明细。"""
    metrics = summarize(daily, risk_free_rate=risk_free_rate)
    if not metrics:
        return Report(metrics={}, industry_exposure=pd.DataFrame())
    orders = pd.DataFrame() if orders is None else orders
    holdings = pd.DataFrame() if holdings is None else holdings
    metrics["cost_attribution"] = cost_attribution(daily, orders, risk_free_rate)
    metrics["turnover_attribution"] = turnover_attribution(daily, orders)
    metrics["cost_attribution"]["impact_model"] = str(impact_model)
    exposure = (
        industry_exposure_rows
        if industry_exposure_rows is not None
        else industry_exposure_frame(daily, holdings, industry_l1, industry_labels)
    )
    metrics["industry_exposure"] = summarize_industry_exposure(exposure, daily)
    return Report(metrics=metrics, industry_exposure=exposure)
