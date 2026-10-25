"""P5-3 生命周期完整账本验收报告。"""

from __future__ import annotations

import json

import pandas as pd


def build_report(
    manifest: dict,
    metrics: dict,
    comparison: pd.DataFrame,
) -> str:
    """生成 P5-3 验收 Markdown。"""

    lifecycle_summary = manifest["lifecycle"]["summary"]
    current = metrics["metrics"]
    lines = [
        "# P5-3 模型生命周期完整账本验收",
        "",
        f"- 验收 ID：`{manifest['analysis_id']}`",
        f"- 生命周期 ID：`{manifest['lifecycle']['lifecycle_id']}`",
        f"- 状态：`{manifest['status']}`",
        f"- 区间：{metrics['scope']['start']} ~ {metrics['scope']['end']}",
        f"- 信号日：{metrics['checks']['actual_signal_days']}",
        "",
        "## 生命周期动作",
        "",
        f"- 状态分布：{json.dumps(lifecycle_summary['state_counts'], ensure_ascii=False)}",
        f"- 事件数量：{json.dumps(lifecycle_summary['event_counts'], ensure_ascii=False)}",
        "",
        "## 完整账本指标",
        "",
        f"- 年化净收益：{current.get('annual_return')}",
        f"- 年化净超额：{current.get('excess_annual_return')}",
        f"- Sharpe：{current.get('sharpe')}",
        f"- 最大回撤：{current.get('max_drawdown')}",
        f"- 双边年换手：{current.get('turnover_two_way_annual')}",
        f"- 总成本：{current.get('total_cost')}",
        "",
        "## 对照",
        "",
        comparison.to_markdown(index=False),
        "",
        "本验收只判断生命周期规则能否在完整账本中执行。生命周期可能降低收益，",
        "不能据此事后调整阈值或把经济差异解释为规则失败。",
        "",
        "2024-2026 最终测试集未读取。",
        "",
    ]
    return "\n".join(lines)


__all__ = ["build_report"]
