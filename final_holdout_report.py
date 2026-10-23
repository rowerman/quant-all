"""最终留出集横向比较和中文报告。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEDGER_METRICS = (
    "annual_return",
    "excess_annual_return",
    "sharpe",
    "max_drawdown",
    "turnover_two_way_annual",
    "total_cost",
)


def comparison_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """生成最终留出候选横向对照。"""

    rows = []
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "candidate": name,
                **{key: metrics.get(key) for key in LEDGER_METRICS},
                "acceptance_passed": bool(payload["checks"]["passed"]),
                "snapshot_id": payload["run_manifest"].get("snapshot_id"),
            }
        )
    return pd.DataFrame(rows)


def combined_yearly_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """合并逐候选分年结果。"""

    rows = [
        payload["yearly"].assign(candidate=name)
        for name, payload in candidates.items()
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _percent(value) -> str:
    """格式化为百分比。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else f"{number:.2%}"


def _number(value, digits: int = 4) -> str:
    """格式化普通数字。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else f"{number:.{digits}f}"


def _display_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """格式化横向对照。"""

    display = frame.copy()
    for name in ("annual_return", "excess_annual_return", "max_drawdown"):
        if name in display:
            display[name] = display[name].map(_percent)
    for name in ("sharpe", "turnover_two_way_annual"):
        if name in display:
            display[name] = display[name].map(_number)
    if "total_cost" in display:
        display["total_cost"] = display["total_cost"].map(
            lambda value: f"{float(value) / 10_000.0:,.2f} 万"
            if pd.notna(value)
            else ""
        )
    return display.rename(
        columns={
            "candidate": "候选",
            "annual_return": "年化净收益",
            "excess_annual_return": "年化净超额",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
            "acceptance_passed": "结构验收",
        }
    )


def _display_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    """格式化分年结果。"""

    columns = [
        "candidate",
        "year",
        "total_return",
        "benchmark_total_return",
        "excess_total_return",
        "max_drawdown",
        "turnover_two_way_annual",
    ]
    display = frame.loc[:, [name for name in columns if name in frame]].copy()
    for name in (
        "total_return",
        "benchmark_total_return",
        "excess_total_return",
        "max_drawdown",
    ):
        if name in display:
            display[name] = display[name].map(_percent)
    if "turnover_two_way_annual" in display:
        display["turnover_two_way_annual"] = display[
            "turnover_two_way_annual"
        ].map(_number)
    return display.rename(
        columns={
            "candidate": "候选",
            "year": "年份",
            "total_return": "净收益",
            "benchmark_total_return": "基准收益",
            "excess_total_return": "净超额",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
        }
    )


def build_final_holdout_report(
    *,
    holdout_id: str,
    status: str,
    scope: dict,
    sources: dict,
    model_metrics: dict,
    gate: dict,
    candidates: dict[str, dict],
    monitor_summary: dict,
    lifecycle_summary: dict,
    exposure_summary: dict,
) -> str:
    """生成最终留出集中文报告。"""

    comparison = comparison_frame(candidates)
    yearly = combined_yearly_frame(candidates)
    model_rows = []
    for name in model_metrics:
        metrics = model_metrics.get(name, {})
        model_rows.append(
            {
                "候选": name,
                "RankIC": _number(metrics.get("rank_ic_mean")),
                "ICIR": _number(metrics.get("icir")),
                "胜率": _percent(metrics.get("rank_ic_win_rate")),
                "天数": metrics.get("days"),
            }
        )
    gate_text = (
        "通过"
        if gate.get("passed")
        else "未通过"
    )
    lines = [
        "# 2024-2026 ML 模型最终留出集验证",
        "",
        f"- 验证 ID：`{holdout_id}`",
        f"- 状态：`{status}`",
        f"- 数据区间：{scope['data_start']} ~ {scope['data_end']}",
        f"- 完整标签信号日：{scope['signal_start']} ~ {scope['signal_end']}，"
        f"{scope['signal_days']} 个",
        f"- 完整账本区间：{scope['ledger_start']} ~ {scope['ledger_end']}",
        "- 冻结口径：不重新训练、不调参；模型折由评估计划指定。",
        "",
        "## 预注册判定",
        "",
        f"- 主候选：`{gate.get('candidate')}`，参照：`{gate.get('reference')}`。",
        f"- P3 门槛：**{gate_text}**。",
        *[f"- {reason}" for reason in gate.get("gate", {}).get("reasons", [])],
        "",
        "## 模型层",
        "",
        pd.DataFrame(model_rows).to_markdown(index=False),
        "",
        "## 完整账本",
        "",
        _display_comparison(comparison).to_markdown(index=False),
        "",
        "## 分年结果",
        "",
        _display_yearly(yearly).to_markdown(index=False),
        "",
        "## 监控与生命周期",
        "",
        f"- 监控最新状态：`{monitor_summary.get('status')}`。",
        f"- 监控 warning：{monitor_summary.get('warning_count')}；"
        f"alert：{monitor_summary.get('alert_count')}。",
        f"- 生命周期状态分布：{lifecycle_summary.get('state_counts')}。",
        f"- 生命周期动作：{lifecycle_summary.get('event_counts')}。",
        "",
        "## 组合暴露",
        "",
        f"- 风格最低覆盖率：{_percent(exposure_summary.get('style_min_coverage'))}。",
        f"- 行业未知平均权重：{_percent(exposure_summary.get('unknown_industry_mean_weight'))}。",
        f"- 平均持仓数：{_number(exposure_summary.get('holdings_mean'), 2)}。",
        "",
        "## 边界",
        "",
        "- 本轮只验证一次，不根据留出集结果重新调参或改变门槛。",
        "- 生命周期沿用 P5 固定规则；模型冻结，因此不自动训练或晋升挑战者。",
        "- 完整账本固定 100 万元和 10% 成交额参与率，不代表更大资金容量。",
        "- 通过只表示开发模型在本次留出期保住了预注册指标，不表示生产批准。",
        "",
        "## 来源",
        "",
        "```json",
        pd.Series(sources).to_json(force_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def incomplete_report(holdout_id: str, error: str) -> str:
    """生成失败关闭报告。"""

    return "\n".join(
        [
            "# 2024-2026 ML 模型最终留出集验证",
            "",
            f"- 验证 ID：`{holdout_id}`",
            "- 状态：`incomplete_final_holdout`",
            f"- 错误：{error}",
            "",
            "本次运行没有形成通过或失败结论，也未根据部分结果调整模型。",
            "",
        ]
    )


def result_files(root: Path) -> dict[str, str]:
    """列出最终结果文件。"""

    names = [
        "model_metrics.json",
        "sources.json",
        "gate.json",
        "comparison.csv",
        "yearly.csv",
        "report.md",
        "exposure/style_summary.csv",
        "exposure/style_series.csv",
        "exposure/industry_summary.csv",
        "monitor/timeseries.csv",
        "monitor/alerts.csv",
        "lifecycle/timeseries.csv",
        "lifecycle/events.csv",
    ]
    return {
        name: str(root / name)
        for name in names
        if (root / name).is_file()
    }


__all__ = [
    "build_final_holdout_report",
    "combined_yearly_frame",
    "comparison_frame",
    "incomplete_report",
    "result_files",
]
