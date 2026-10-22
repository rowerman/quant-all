"""模型监控的 Markdown 与 JSON 报告。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256, write_json
from alpha.monitor_policy import MONITOR_SCHEMA_VERSION, MonitorResult


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """渲染紧凑 Markdown 表。"""

    if frame.empty:
        return "无数据。\n"
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _finite(value: object) -> object:
    """把 NumPy 和 NaN 转换为 JSON 安全值。"""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _payload_records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict]:
    """把 DataFrame 转为 JSON 记录。"""

    selected = frame if limit is None else frame.head(limit)
    return [
        {str(key): _finite(value) for key, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def write_monitor_package(
    folder: Path,
    *,
    monitor_id: str,
    identity: dict,
    result: MonitorResult,
    decay: dict,
    sources: dict,
) -> dict:
    """写入 JSON、Markdown 并返回监控 manifest。"""

    latest = result.timeseries.iloc[-1]
    current = pd.DataFrame(
        [
            {
                "signal_days": result.summary["signal_days"],
                "rolling_deployed_rank_ic": latest["rolling_deployed_rank_ic"],
                "rolling_excess": latest["rolling_excess"],
                "turnover_two_way_annual": latest[
                    "rolling_turnover_two_way_annual"
                ],
                "drawdown": latest["drawdown"],
                "missing_rate": latest["missing_rate"],
                "max_style_abs_exposure": latest["max_style_abs_exposure"],
                "max_abs_industry_active_weight": latest[
                    "max_abs_industry_active_weight"
                ],
            }
        ]
    )
    recent_alerts = result.alerts.tail(20).copy()
    report = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "monitor_id": monitor_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": result.summary,
        "policy": identity["policy"],
        "sources": sources,
        "decay_simulation": {
            key: value for key, value in decay.items() if key != "timeseries"
        },
        "style_summary": _payload_records(result.style_summary),
        "industry_summary": _payload_records(result.industry_summary),
        "recent_alerts": _payload_records(recent_alerts),
    }
    write_json(folder / "report.json", report)

    alert_table = _markdown_table(
        recent_alerts,
        ["trade_date", "metric", "value", "threshold", "severity", "reason"],
    )
    style_table = _markdown_table(
        result.style_summary.head(7),
        [
            "style",
            "mean_exposure",
            "mean_abs_exposure",
            "min_exposure",
            "max_exposure",
            "mean_coverage",
        ],
    )
    industry_table = _markdown_table(
        result.industry_summary.head(10),
        [
            "industry",
            "max_abs_active_weight",
            "mean_top_active_weight",
            "appearances",
        ],
    )
    lines = [
        f"# P5-2 模型监控：{sources['model_id']}",
        "",
        f"- 监控 ID：`{monitor_id}`",
        f"- 区间：{result.summary['start']} ~ {result.summary['end']}",
        f"- 信号日：{result.summary['signal_days']}",
        f"- 最新状态：`{result.summary['status']}`",
        "- 2024-2026 最终测试集未读取。",
        "",
        "## 最新监控值",
        "",
        _markdown_table(
            current,
            [
                "signal_days",
                "rolling_deployed_rank_ic",
                "rolling_excess",
                "turnover_two_way_annual",
                "drawdown",
                "missing_rate",
                "max_style_abs_exposure",
                "max_abs_industry_active_weight",
            ],
        ),
        "## 告警摘要",
        "",
        f"区间内 warning {result.summary['warning_count']} 条，"
        f"alert {result.summary['alert_count']} 条。",
        "",
        alert_table,
        "## 模型衰减演练",
        "",
        "对最后 13 个信号周的原始分数做截面排名反转，再按正式 4 周平滑监控；"
        "该演练只用于验证告警链路，不代表经济回测结果。",
        "",
        f"- 状态：`{decay['status']}`",
        f"- 最新衰减滚动 RankIC："
        f"{decay['latest_decayed_rolling_deployed_rank_ic']}",
        f"- 区间：{decay['start']} ~ {decay['end']}",
        "",
        "## 风格暴露",
        "",
        "风格暴露复用 P3-4 的原始 Top-30 目标持仓口径，不等同于完整账本成交持仓。",
        "",
        style_table,
        "## 实际行业主动暴露",
        "",
        industry_table,
        "## 口径与限制",
        "",
        "- 模型层 RankIC 使用部署分数：原始五种子分数按日 rank 后做 4 周因果平滑。",
        "- 净超额和换手来自 P4-4A 完整账本，不把日频收益重新解释为周频信号收益。",
        "- 风格暴露沿用 P3-4 目标持仓；行业暴露使用完整账本实际持仓和数据集行业列。",
        "- 缺失率只统计可交易股票中缺少模型分数的比例。",
        "- 监控是 2018-2023 研究期离线报告，不包含生产实时推送。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")

    files = {
        name: file_sha256(folder / name)
        for name in ("alerts.csv", "decay_simulation.csv", "report.json", "report.md", "timeseries.csv")
    }
    return {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "monitor_id": monitor_id,
        "created_at": report["generated_at"],
        "summary": result.summary,
        "decay_simulation": report["decay_simulation"],
        "sources": sources,
        "policy": identity["policy"],
        "code_sha256": identity["code_sha256"],
        "files": files,
    }


__all__ = ["write_monitor_package"]
