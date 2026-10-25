"""P4-4B1 完整账本对照的指标、容量表和报告。"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256

CORE_METRICS = (
    "annual_return",
    "excess_annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "turnover_two_way_annual",
    "total_cost",
)


def _number(value) -> float | None:
    """把指标转换成有限浮点数；无效时返回 None。"""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percent(value) -> str:
    """把小数指标格式化百分比。"""

    number = _number(value)
    return "" if number is None else f"{number:.2%}"


def _float(value, digits: int = 4) -> str:
    """格式化普通浮点数。"""

    number = _number(value)
    return "" if number is None else f"{number:.{digits}f}"


def _money(value) -> str:
    """把金额格式化万元。"""

    number = _number(value)
    return "" if number is None else f"{number / 10_000.0:,.2f} 万"


def comparison_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """生成八候选横向对照。"""

    rows = []
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "candidate": name,
                "engine": "ledger",
                "origin": payload.get(
                    "origin",
                    "reused_p4_4a" if payload.get("reused") else "new_run",
                ),
                **{metric: metrics.get(metric) for metric in CORE_METRICS},
                "total_cost_bps_of_nav": metrics.get("total_cost_bps_of_nav"),
                "acceptance_passed": bool(payload["checks"]["passed"]),
                "snapshot_id": payload["run_manifest"].get("snapshot_id"),
            }
        )
    return pd.DataFrame(rows)


def combined_yearly_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """合并逐候选年度结果。"""

    rows = [
        payload["yearly"].assign(candidate=name, engine="ledger")
        for name, payload in candidates.items()
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def capacity_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """生成容量、现金和收敛诊断表。"""

    rows = []
    for name, payload in candidates.items():
        capacity = payload["capacity"]
        rows.append(
            {
                "candidate": name,
                "liquidity_limited_cells": capacity["liquidity_limited_cells"],
                "liquidity_limited_value": capacity["liquidity_limited_value"],
                "liquidity_unavailable_cells": capacity["liquidity_unavailable_cells"],
                "liquidity_shortfall_shares": capacity["liquidity_shortfall_shares"],
                "cash_clipped_cells": capacity["cash_clipped_cells"],
                "cash_carry_over_cells": capacity["cash_carry_over_cells"],
                "pending_at_end_cells": capacity["pending_at_end_cells"],
                "allocation_converged": capacity["allocation_converged"],
                "carry_converged": capacity["carry_converged"],
                "allocation_iterations_total": capacity[
                    "allocation_iterations_total"
                ],
            }
        )
    return pd.DataFrame(rows)


def _display_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告用横向对照。"""

    if frame.empty:
        return frame
    display = frame[
        [
            "candidate",
            "origin",
            "annual_return",
            "excess_annual_return",
            "sharpe",
            "max_drawdown",
            "turnover_two_way_annual",
            "total_cost",
        ]
    ].copy()
    for name in ("annual_return", "excess_annual_return", "max_drawdown"):
        display[name] = display[name].map(_percent)
    display["sharpe"] = display["sharpe"].map(_float)
    display["turnover_two_way_annual"] = display[
        "turnover_two_way_annual"
    ].map(_float)
    display["total_cost"] = display["total_cost"].map(_money)
    return display.rename(
        columns={
            "candidate": "候选",
            "origin": "来源",
            "annual_return": "年化收益",
            "excess_annual_return": "年化净超额",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
        }
    )


def _display_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告用分年表。"""

    if frame.empty:
        return frame
    display = frame.copy()
    columns = [
        "candidate",
        "year",
        "total_return",
        "benchmark_total_return",
        "excess_total_return",
        "max_drawdown",
        "turnover_two_way_annual",
        "total_cost",
    ]
    display = display[[name for name in columns if name in display]].copy()
    for name in (
        "total_return",
        "benchmark_total_return",
        "excess_total_return",
        "max_drawdown",
    ):
        display[name] = display[name].map(_percent)
    display["turnover_two_way_annual"] = display[
        "turnover_two_way_annual"
    ].map(_float)
    display["total_cost"] = display["total_cost"].map(_money)
    return display.rename(
        columns={
            "candidate": "候选",
            "year": "年份",
            "total_return": "净收益",
            "benchmark_total_return": "基准收益",
            "excess_total_return": "净超额",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
        }
    )


def _display_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告用容量表。"""

    if frame.empty:
        return frame
    display = frame[
        [
            "candidate",
            "liquidity_limited_cells",
            "liquidity_limited_value",
            "liquidity_unavailable_cells",
            "cash_clipped_cells",
            "cash_carry_over_cells",
            "pending_at_end_cells",
            "allocation_converged",
            "carry_converged",
        ]
    ].copy()
    display["liquidity_limited_value"] = display[
        "liquidity_limited_value"
    ].map(_money)
    display["allocation_converged"] = display["allocation_converged"].map(
        lambda value: "是" if bool(value) else "否"
    )
    display["carry_converged"] = display["carry_converged"].map(
        lambda value: "是" if bool(value) else "否"
    )
    return display.rename(
        columns={
            "candidate": "候选",
            "liquidity_limited_cells": "成交额限制格数",
            "liquidity_limited_value": "成交额限制金额",
            "liquidity_unavailable_cells": "成交额不可用格数",
            "cash_clipped_cells": "现金裁剪格数",
            "cash_carry_over_cells": "补回跨日格数",
            "pending_at_end_cells": "区间末待补格数",
            "allocation_converged": "资金分配收敛",
            "carry_converged": "补回收敛",
        }
    )


def build_report(
    *,
    analysis_id: str,
    scope: dict,
    candidates: dict[str, dict],
    expected_candidates: int,
    source_analysis_id: str,
) -> str:
    """生成 P4-4B1 Markdown 报告。"""

    comparison = comparison_frame(candidates)
    yearly = combined_yearly_frame(candidates)
    capacity = capacity_frame(candidates)
    all_complete = len(candidates) == expected_candidates
    all_passed = bool(candidates) and all(
        payload["checks"]["passed"] for payload in candidates.values()
    )
    status = (
        "验收通过"
        if all_complete and all_passed
        else "部分候选"
        if not all_complete
        else "存在未通过检查"
    )
    check_rows = [
        {
            "候选": name,
            "检查": check,
            "结果": "通过" if passed else "失败",
        }
        for name, payload in candidates.items()
        for check, passed in payload["checks"]["checks"].items()
    ]
    lines = [
        "# P4-4B1 固定资金完整账本横向对照",
        "",
        f"- 分析 ID：`{analysis_id}`",
        f"- P4-4A 复用来源：`{source_analysis_id}`",
        f"- 区间：{scope['start']} ~ {scope['end']}",
        f"- 状态：**{status}**",
        f"- 候选完成数：{len(candidates)} / {expected_candidates}",
        "",
        "## 固定口径",
        "",
        "- 全 A 可交易池、Top-30 等权、周频调仓，收盘生成信号、次一交易日开盘执行。",
        (
            f"- 八候选统一使用 {scope['rebalance_dates_count']} 个显式调仓日，"
            f"日期哈希 `{scope['rebalance_dates_sha256']}`。"
        ),
        f"- 初始资金 {scope['init_cash']:,.0f} 元，无风险利率 {scope['risk_free_rate']:.2%}。",
        "- 基准 `H00300.CSI`；默认成本、平方根冲击、10% 成交额参与率。",
        "- Ridge、单 LightGBM 和集成为当日 rank 原始分数；选中模型为四周 rank 平滑。",
        "",
        "## 同口径对照",
        "",
        _display_comparison(comparison).to_markdown(index=False)
        if not comparison.empty
        else "无结果。",
        "",
        "## 容量与资金分配",
        "",
        _display_capacity(capacity).to_markdown(index=False)
        if not capacity.empty
        else "无结果。",
        "",
        "## 分年结果",
        "",
        _display_yearly(yearly).to_markdown(index=False)
        if not yearly.empty
        else "无结果。",
        "",
        "## 验收检查",
        "",
        pd.DataFrame(check_rows).to_markdown(index=False) if check_rows else "无结果。",
        "",
        "## 解释边界",
        "",
        "- 完整账本包含实际持仓、现金、交易规则、公司行动、历史税费、滑点与冲击。",
        "- 本轮只对应 100 万初始资金和 10% 成交额参与率。",
        "- 多资金档位容量曲线属于 P4-4B2，本轮未运行。",
        "- 本轮结果不用于重新训练模型、调参或改变 P3 选择。",
    ]
    return "\n".join(lines) + "\n"


def output_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """计算报告级输出文件的 SHA256。"""

    return {
        name: file_sha256(root / name)
        for name in names
        if (root / name).is_file()
    }
