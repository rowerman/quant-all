"""P4-4B2 单候选多资金档位容量曲线和报告。"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.errors import AlphaConfigError

BASE_CASH = 1_000_000.0
CAPITAL_TIERS = (1_000_000.0, 10_000_000.0, 100_000_000.0)
TIER_CANDIDATE = "LIGHTGBM_SMOOTH_4W"


def _number(value) -> float | None:
    """把值转换成有限浮点数；无效时返回 None。"""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percent(value) -> str:
    """格式化百分比。"""

    number = _number(value)
    return "" if number is None else f"{number:.2%}"


def _float(value, digits: int = 4) -> str:
    """格式化普通浮点数。"""

    number = _number(value)
    return "" if number is None else f"{number:.{digits}f}"


def _money(value) -> str:
    """把金额格式化为万元。"""

    number = _number(value)
    return "" if number is None else f"{number / 10_000.0:,.2f} 万"


def _difference(value, baseline) -> float | None:
    """计算两个有限数值的差。"""

    left = _number(value)
    right = _number(baseline)
    return None if left is None or right is None else left - right


def capacity_curve(candidates: dict[float, dict]) -> pd.DataFrame:
    """生成三档资金容量曲线。"""

    baseline = candidates.get(BASE_CASH)
    if baseline is None:
        raise AlphaConfigError("P4-4B2 容量曲线缺少 100 万基准档")
    baseline_metrics = baseline["metrics"]
    rows = []
    for cash in sorted(candidates):
        payload = candidates[cash]
        metrics = payload["metrics"]
        capacity = payload["capacity"]
        calibration = payload["run_manifest"].get("calibration") or {}
        turnover = metrics.get("turnover_attribution") or {}
        traded_value = _number(turnover.get("buy_value")) or 0.0
        traded_value += _number(turnover.get("sell_value")) or 0.0
        limited_value = _number(capacity.get("liquidity_limited_value")) or 0.0
        rows.append(
            {
                "candidate": TIER_CANDIDATE,
                "init_cash": float(cash),
                "init_cash_wan": float(cash) / 10_000.0,
                "annual_return": metrics.get("annual_return"),
                "excess_annual_return": metrics.get("excess_annual_return"),
                "annual_return_change_vs_1m": _difference(
                    metrics.get("annual_return"),
                    baseline_metrics.get("annual_return"),
                ),
                "excess_return_change_vs_1m": _difference(
                    metrics.get("excess_annual_return"),
                    baseline_metrics.get("excess_annual_return"),
                ),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "turnover_two_way_annual": metrics.get(
                    "turnover_two_way_annual"
                ),
                "total_cost": metrics.get("total_cost"),
                "total_cost_bps_of_nav": metrics.get("total_cost_bps_of_nav"),
                "liquidity_limited_cells": capacity.get(
                    "liquidity_limited_cells"
                ),
                "liquidity_limited_value": limited_value,
                "liquidity_limited_rate": (
                    limited_value / traded_value if traded_value > 0 else None
                ),
                "liquidity_shortfall_shares": capacity.get(
                    "liquidity_shortfall_shares"
                ),
                "cash_carry_over_cells": capacity.get("cash_carry_over_cells"),
                "liquidity_retry_filled_cells": capacity.get(
                    "liquidity_retry_filled_cells"
                ),
                "pending_at_end_cells": capacity.get("pending_at_end_cells"),
                "planned_orders": calibration.get("planned_orders"),
                "final_orders": calibration.get("final_orders"),
                "unaffordable_min_order_cells": calibration.get(
                    "unaffordable_min_order_cells"
                ),
                "weight_deviation_p95": calibration.get(
                    "weight_deviation_p95"
                ),
                "weight_deviation_max": calibration.get(
                    "weight_deviation_max"
                ),
                "allocation_converged": capacity.get("allocation_converged"),
                "carry_converged": capacity.get("carry_converged"),
                "acceptance_passed": payload["checks"].get("passed"),
            }
        )
    return pd.DataFrame(rows)


def yearly_curve(candidates: dict[float, dict]) -> pd.DataFrame:
    """合并三档资金的逐年度账本结果。"""

    rows = []
    for cash in sorted(candidates):
        payload = candidates[cash]
        frame = payload["yearly"].copy()
        frame.insert(0, "candidate", TIER_CANDIDATE)
        frame.insert(1, "init_cash", float(cash))
        frame.insert(2, "init_cash_wan", float(cash) / 10_000.0)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def economic_capacity(curve: pd.DataFrame) -> float | None:
    """返回年化净超额仍为正的最大已测资金档位。"""

    if curve.empty:
        return None
    qualified = curve[
        curve["excess_annual_return"].map(
            lambda value: _number(value) is not None
            and float(value) > 0
        )
    ]
    return None if qualified.empty else float(qualified["init_cash"].max())


def _display_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """生成报告中的容量曲线表。"""

    columns = [
        "init_cash_wan",
        "annual_return",
        "annual_return_change_vs_1m",
        "excess_annual_return",
        "excess_return_change_vs_1m",
        "sharpe",
        "max_drawdown",
        "turnover_two_way_annual",
        "total_cost",
        "liquidity_limited_cells",
        "liquidity_limited_value",
        "liquidity_limited_rate",
        "liquidity_shortfall_shares",
        "cash_carry_over_cells",
        "weight_deviation_p95",
        "weight_deviation_max",
        "pending_at_end_cells",
        "allocation_converged",
        "carry_converged",
        "acceptance_passed",
    ]
    display = curve[columns].copy()
    for name in (
        "annual_return",
        "annual_return_change_vs_1m",
        "excess_annual_return",
        "excess_return_change_vs_1m",
        "max_drawdown",
        "liquidity_limited_rate",
    ):
        display[name] = display[name].map(_percent)
    for name in (
        "sharpe",
        "turnover_two_way_annual",
        "weight_deviation_p95",
        "weight_deviation_max",
    ):
        display[name] = display[name].map(_float)
    display["total_cost"] = display["total_cost"].map(_money)
    display["liquidity_limited_value"] = display[
        "liquidity_limited_value"
    ].map(_money)
    for name in (
        "allocation_converged",
        "carry_converged",
        "acceptance_passed",
    ):
        display[name] = display[name].map(
            lambda value: "是" if bool(value) else "否"
        )
    return display.rename(
        columns={
            "init_cash_wan": "资金（万元）",
            "annual_return": "年化净收益",
            "annual_return_change_vs_1m": "较 100 万变化",
            "excess_annual_return": "年化净超额",
            "excess_return_change_vs_1m": "超额较 100 万变化",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
            "liquidity_limited_cells": "成交额限制格数",
            "liquidity_limited_value": "成交额限制金额",
            "liquidity_limited_rate": "限制金额/成交额",
            "liquidity_shortfall_shares": "容量缺口股数",
            "cash_carry_over_cells": "跨日补单格数",
            "weight_deviation_p95": "权重偏差 P95",
            "weight_deviation_max": "权重偏差最大",
            "pending_at_end_cells": "区间末待补格数",
            "allocation_converged": "资金分配收敛",
            "carry_converged": "补回收敛",
            "acceptance_passed": "结构验收",
        }
    )


def _display_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告中的分年结果表。"""

    columns = [
        "init_cash_wan",
        "year",
        "total_return",
        "benchmark_total_return",
        "excess_total_return",
        "max_drawdown",
        "turnover_two_way_annual",
        "total_cost",
    ]
    display = frame[[name for name in columns if name in frame]].copy()
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
        ].map(_float)
    if "total_cost" in display:
        display["total_cost"] = display["total_cost"].map(_money)
    return display.rename(
        columns={
            "init_cash_wan": "资金（万元）",
            "year": "年份",
            "total_return": "净收益",
            "benchmark_total_return": "基准收益",
            "excess_total_return": "净超额",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
        }
    )


def build_report(
    *,
    analysis_id: str,
    candidates: dict[float, dict],
    expected_tiers: int,
    source_analysis_id: str,
    source_manifest_sha256: str,
) -> str:
    """生成 P4-4B2 容量曲线报告。"""

    curve = capacity_curve(candidates)
    yearly = yearly_curve(candidates)
    all_complete = len(candidates) == expected_tiers
    all_passed = bool(candidates) and all(
        payload["checks"].get("passed") is True
        for payload in candidates.values()
    )
    structure_status = (
        "通过"
        if all_complete and all_passed
        else "部分档位"
        if not all_complete
        else "存在未通过检查"
    )
    economic_cash = economic_capacity(curve)
    economic_label = (
        "三个已测档位均为正"
        if economic_cash == max(CAPITAL_TIERS)
        else f"{economic_cash / 10_000.0:,.0f} 万元"
        if economic_cash is not None
        else "无"
    )
    check_rows = [
        {
            "资金（万元）": cash / 10_000.0,
            "检查": check,
            "结果": "通过" if passed else "失败",
        }
        for cash, payload in sorted(candidates.items())
        for check, passed in payload["checks"].get("checks", {}).items()
    ]
    lines = [
        "# P4-4B2 选中策略三档容量曲线",
        "",
        f"- 分析 ID：`{analysis_id}`",
        f"- P4-4B1 来源：`{source_analysis_id}`",
        f"- 来源 manifest SHA256：`{source_manifest_sha256}`",
        "- 候选：`LIGHTGBM_SMOOTH_4W`",
        f"- 档位完成数：{len(candidates)} / {expected_tiers}",
        f"- 执行结构结论：**{structure_status}**",
        f"- 经济容量观察：年化净超额为正的最大已测档位为 **{economic_label}**",
        "",
        "## 固定口径",
        "",
        "- 区间 2018-01-12 ~ 2023-12-29，使用 300 个显式调仓日。",
        "- 全 A 可交易池、Top-30 等权、周频、10% 成交额参与率。",
        "- 三档均使用相同模型预测、日期、成本和回测规则；100 万档复用 P4-4B1。",
        "- 经济容量只做观察，不把收益变化事后转成新的结构验收门槛。",
        "",
        "## 容量曲线",
        "",
        _display_curve(curve).to_markdown(index=False)
        if not curve.empty
        else "无结果。",
        "",
        "## 分年结果",
        "",
        _display_yearly(yearly).to_markdown(index=False)
        if not yearly.empty
        else "无结果。",
        "",
        "## 执行检查",
        "",
        pd.DataFrame(check_rows).to_markdown(index=False)
        if check_rows
        else "无结果。",
        "",
        "## 解释边界",
        "",
        "- 完整账本包含实际持仓、现金、交易规则、公司行动、历史税费、滑点与冲击。",
        "- 10% 参与率是固定测试条件，本轮不扫描其他参与率。",
        "- 结果不用于重新训练模型、调参或改变 P3 选择。",
        "- 2024-2026 最终测试集未读取、未训练、未预测、未评估。",
    ]
    return "\n".join(lines) + "\n"


def output_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """计算报告级输出文件的 SHA256。"""

    return {
        name: file_sha256(root / name)
        for name in names
        if (root / name).is_file()
    }
