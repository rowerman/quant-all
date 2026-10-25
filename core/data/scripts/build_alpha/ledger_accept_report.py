"""通用完整账本候选对照和 Markdown 报告。"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

CORE_METRICS = (
    "annual_return",
    "excess_annual_return",
    "sharpe",
    "max_drawdown",
    "turnover_two_way_annual",
    "total_cost",
)


def comparison_frame(candidates: Mapping[str, dict]) -> pd.DataFrame:
    """展开候选核心指标和结构检查状态。"""

    rows = []
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "candidate": name,
                "source": payload["source"],
                "score_policy": payload["score_policy"],
                "holding_policy": payload["holding_policy"],
                **{metric: metrics.get(metric) for metric in CORE_METRICS},
                "acceptance_passed": bool(
                    payload["checks"].get("passed")
                ),
                "snapshot_id": payload["snapshot_id"],
            }
        )
    return pd.DataFrame(rows)


def yearly_frame(candidates: Mapping[str, dict]) -> pd.DataFrame:
    """合并全部候选的完整账本分年结果。"""

    rows = [
        payload["yearly"].assign(candidate=name)
        for name, payload in candidates.items()
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _number(value) -> float | None:
    """把值规范为有限浮点数。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percent(value) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.2%}"


def _float(value, digits: int = 4) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.{digits}f}"


def build_report(
    *,
    analysis_id: str,
    evaluation_id: str,
    scope: dict,
    candidates: Mapping[str, dict],
    status: str,
) -> str:
    """生成可审计的通用完整账本报告。"""

    lines = [
        "# 通用模型完整账本验收",
        "",
        f"- 账本 ID：`{analysis_id}`",
        f"- 评估 ID：`{evaluation_id}`",
        f"- 区间：{scope['start']}～{scope['end']}",
        f"- 初始资金：{scope['init_cash']:,.0f}",
        f"- 状态：`{status}`",
        "",
        "本轮候选完全来自评估计划和不可变候选分数。",
        "",
        "| 候选 | 年化净收益 | 净超额 | 夏普 | 最大回撤 | 双边年换手 | 检查 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        lines.append(
            f"| {name} | {_percent(metrics.get('annual_return'))} | "
            f"{_percent(metrics.get('excess_annual_return'))} | "
            f"{_float(metrics.get('sharpe'))} | "
            f"{_percent(metrics.get('max_drawdown'))} | "
            f"{_float(metrics.get('turnover_two_way_annual'), 2)} | "
            f"{'通过' if payload['checks'].get('passed') else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- 信号日、调仓日和分数哈希绑定评估产物。",
            "- 资金分配、容量、成本、日期和快照沿用完整账本统一检查。",
            "- 本报告是执行结构验收，不替代独立最终留出验证。",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = ["build_report", "comparison_frame", "yearly_frame"]
