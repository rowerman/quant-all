"""P4-4A 完整账本验收的指标、检查和报告。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256
from alpha.artifacts import model_dir
from alpha.evaluation_plan import load_current_plan
from backtest.metrics import TRADING_DAYS

_PLAN = load_current_plan()
_SELECTED = _PLAN.by_name(_PLAN.selected or _PLAN.candidates[0].name)
P3_WEEKLY_ROOT = (
    model_dir(_SELECTED.source.id)
    / "analysis"
    / "p3_5_5c02d3520f25"
)
CAPACITY_KEYS = (
    "liquidity_limited_cells",
    "liquidity_limited_value",
    "liquidity_unavailable_cells",
    "liquidity_shortfall_shares",
    "liquidity_retry_filled_cells",
    "liquidity_pending_at_end_cells",
    "cash_clipped_cells",
    "cash_limited_cells",
    "cash_limited_value",
    "cash_carry_over_cells",
    "pending_at_end_cells",
    "delisting_capacity_shortfall_shares",
)
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


def yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """从完整账本日结果生成分年收益、超额、回撤、换手和成本。"""

    frame = daily.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    rows: list[dict] = []
    for year, group in frame.groupby(frame.index.year, sort=True):
        total_return = float((1.0 + group["return"].astype(float)).prod() - 1.0)
        row = {
            "year": int(year),
            "days": int(len(group)),
            "total_return": total_return,
            "max_drawdown": float(group["drawdown"].astype(float).min()),
            "turnover_two_way_annual": (
                float(group["turnover"].astype(float).sum())
                * TRADING_DAYS
                / max(len(group), 1)
            ),
            "cash_cost": float(group["cost"].fillna(0.0).astype(float).sum()),
            "total_cost": float(
                group.get("total_cost", group["cost"])
                .fillna(group["cost"])
                .astype(float)
                .sum()
            ),
        }
        if "benchmark_return" in group.columns and group["benchmark_return"].notna().any():
            benchmark = float(
                (1.0 + group["benchmark_return"].astype(float)).prod() - 1.0
            )
            row["benchmark_total_return"] = benchmark
            row["excess_total_return"] = total_return - benchmark
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_summary(metadata: dict) -> dict:
    """汇总完整账本校准中的容量、现金和补回状态。"""

    calibration = metadata.get("calibration") or {}
    payload = {
        key: _number(calibration.get(key, 0.0)) or 0.0
        for key in CAPACITY_KEYS
    }
    payload.update(
        {
            "allocation_converged": bool(
                calibration.get("allocation_converged", False)
            ),
            "carry_converged": bool(calibration.get("carry_converged", False)),
            "allocation_iterations": int(
                calibration.get("allocation_iterations", 0) or 0
            ),
            "allocation_iterations_total": int(
                calibration.get("allocation_iterations_total", 0) or 0
            ),
            "nonconverged_segments": int(
                calibration.get("nonconverged_segments", 0) or 0
            ),
        }
    )
    return payload


def acceptance_checks(
    metrics: dict,
    metadata: dict,
    daily: pd.DataFrame,
    *,
    dev_start: str,
    dev_end: str,
    expected_signal_days: int,
    expected_execution_days: int,
    actual_signal_days: int,
) -> dict:
    """核对账本收敛、日期、快照、核心指标和成本闭环。"""

    capacity = capacity_summary(metadata)
    total_cost = _number(metrics.get("total_cost"))
    attributed = _number(
        (metrics.get("cost_attribution") or {}).get("total", {}).get("total_cost")
    )
    tolerance = max(1e-6, abs(total_cost or 0.0) * 1e-9)
    dates = pd.DatetimeIndex(daily.index)
    last_date = str(dates.max().date()) if len(dates) else ""
    checks = {
        "allocation_converged": capacity["allocation_converged"],
        "carry_converged": capacity["carry_converged"],
        "snapshot_present": isinstance(metadata.get("snapshot_id"), str)
        and len(metadata.get("snapshot_id", "")) == 64,
        "benchmark_present": metrics.get("benchmark_total_return") is not None,
        "core_metrics_present": all(metrics.get(name) is not None for name in CORE_METRICS),
        "cost_reconciled": total_cost is not None
        and attributed is not None
        and abs(total_cost - attributed) <= tolerance,
        "within_development": bool(dates.size)
        and dev_start <= str(dates.min().date())
        and str(dates.min().date()) <= last_date
        and last_date <= dev_end,
        "signal_days_match": (
            expected_signal_days > 0
            and expected_execution_days > 0
            and actual_signal_days == expected_signal_days
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "last_date": last_date,
        "expected_signal_days": int(expected_signal_days),
        "expected_execution_days": int(expected_execution_days),
        "actual_signal_days": int(actual_signal_days),
        "cost_difference": (
            None if total_cost is None or attributed is None else total_cost - attributed
        ),
    }


def comparison_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """把完整账本候选压缩成横向对照表。"""

    rows = []
    for name, payload in candidates.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "candidate": name,
                "engine": "ledger",
                **{metric: metrics.get(metric) for metric in CORE_METRICS},
                "total_cost_bps_of_nav": metrics.get("total_cost_bps_of_nav"),
                "acceptance_passed": payload["checks"]["passed"],
                "snapshot_id": payload["run_manifest"].get("snapshot_id"),
            }
        )
    return pd.DataFrame(rows)


def combined_yearly_frame(candidates: dict[str, dict]) -> pd.DataFrame:
    """合并逐候选完整账本年度结果。"""

    rows = [
        payload["yearly"].assign(candidate=name, engine="ledger")
        for name, payload in candidates.items()
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_p3_weekly_reference(names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 P3-5 固定候选和周频年度结果的同口径参照。"""

    candidate_path = P3_WEEKLY_ROOT / "candidates.csv"
    yearly_path = P3_WEEKLY_ROOT / "yearly.csv"
    if not candidate_path.is_file() or not yearly_path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    candidates = pd.read_csv(candidate_path)
    yearly = pd.read_csv(yearly_path)
    candidates = candidates.loc[candidates["name"].isin(names)].copy()
    candidates["engine"] = "weekly_reference"
    yearly = yearly.loc[yearly["candidate"].isin(names)].copy()
    yearly["engine"] = "weekly_reference"
    return candidates, yearly


def _percent(value) -> str:
    """把小数指标格式化为百分比。"""

    number = _number(value)
    return "" if number is None else f"{number:.2%}"


def _float(value, digits: int = 4) -> str:
    """把普通指标格式化为固定小数。"""

    number = _number(value)
    return "" if number is None else f"{number:.{digits}f}"


def _money(value) -> str:
    """把金额格式化为万元。"""

    number = _number(value)
    return "" if number is None else f"{number / 10_000.0:,.2f} 万"


def _display_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告使用的格式化对照表。"""

    if frame.empty:
        return frame
    display = frame[
        [
            "candidate",
            "engine",
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
    display["turnover_two_way_annual"] = display["turnover_two_way_annual"].map(_float)
    display["total_cost"] = display["total_cost"].map(_money)
    return display.rename(
        columns={
            "candidate": "候选",
            "engine": "引擎",
            "annual_return": "年化收益",
            "excess_annual_return": "年化净超额",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
        }
    )


def _display_yearly(frame: pd.DataFrame) -> pd.DataFrame:
    """生成报告使用的分年表。"""

    if frame.empty:
        return frame
    display = frame.copy()
    if "total_cost" not in display and "cost" in display:
        display["total_cost"] = display["cost"]
    columns = [
        "candidate",
        "engine",
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
    display["turnover_two_way_annual"] = display["turnover_two_way_annual"].map(_float)
    display["total_cost"] = display["total_cost"].map(_money)
    return display.rename(
        columns={
            "candidate": "候选",
            "engine": "引擎",
            "year": "年份",
            "total_return": "净收益",
            "benchmark_total_return": "基准收益",
            "excess_total_return": "净超额",
            "max_drawdown": "最大回撤",
            "turnover_two_way_annual": "双边年换手",
            "total_cost": "总成本",
        }
    )


def _display_capacity(candidates: dict[str, dict]) -> pd.DataFrame:
    """生成容量和收敛诊断表。"""

    rows = []
    for name, payload in candidates.items():
        capacity = payload["capacity"]
        rows.append(
            {
                "候选": name,
                "成交额限制格数": int(capacity["liquidity_limited_cells"]),
                "成交额限制金额": _money(capacity["liquidity_limited_value"]),
                "成交额不可用格数": int(capacity["liquidity_unavailable_cells"]),
                "成交额缺口股数": _float(capacity["liquidity_shortfall_shares"], 2),
                "现金裁剪格数": int(capacity["cash_clipped_cells"]),
                "补回跨日格数": int(capacity["cash_carry_over_cells"]),
                "区间末待补格数": int(capacity["pending_at_end_cells"]),
                "资金分配收敛": "是" if capacity["allocation_converged"] else "否",
                "补回收敛": "是" if capacity["carry_converged"] else "否",
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(
    *,
    analysis_id: str,
    scope: dict,
    candidates: dict[str, dict],
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
    weekly_candidates: pd.DataFrame,
    weekly_yearly: pd.DataFrame,
    complete: bool,
) -> str:
    """生成 P4-4A Markdown 报告。"""

    checks_passed = bool(candidates) and all(
        payload["checks"]["passed"] for payload in candidates.values()
    )
    status = (
        "结构验收通过"
        if complete and checks_passed
        else "部分候选，尚未完成 P4-4A"
        if not complete
        else "存在未通过检查"
    )
    weekly_rows = weekly_candidates[
        [
            "name",
            "engine",
            "annual_return",
            "excess_annual_return",
            "sharpe",
            "max_drawdown",
            "turnover_two_way_annual",
            "total_cost",
        ]
    ].rename(columns={"name": "candidate"}) if not weekly_candidates.empty else pd.DataFrame()
    all_comparison = pd.concat([comparison, weekly_rows], ignore_index=True)
    all_yearly = pd.concat([yearly, weekly_yearly], ignore_index=True)
    check_rows = []
    for name, payload in candidates.items():
        for check, passed in payload["checks"]["checks"].items():
            check_rows.append(
                {
                    "候选": name,
                    "检查": check,
                    "结果": "通过" if passed else "失败",
                }
            )
    lines = [
        "# P4-4A 首批完整账本验收",
        "",
        f"- 分析 ID：`{analysis_id}`",
        f"- 区间：{scope['start']} ~ {scope['end']}",
        f"- 状态：**{status}**",
        "- 范围：P3 选中的 `LIGHTGBM_SMOOTH_4W` 与 `B4_ALL_RANK`",
        "- P4-4B 未运行，P4-4 整体尚未完成；2024-2026 最终测试集保持封存。",
        "",
        "## 固定口径",
        "",
        "- 全 A 可交易池、Top-30 等权、周频调仓，收盘生成信号、次一交易日开盘执行。",
        (
            f"- 两个候选统一使用模型实际存在的 {scope['rebalance_dates_count']} 个显式"
            f"调仓日，日期哈希 `{scope['rebalance_dates_sha256']}`。"
        ),
        f"- 初始资金 {scope['init_cash']:,.0f} 元，无风险利率 {scope['risk_free_rate']:.2%}。",
        "- 基准 `H00300.CSI`；默认成本、平方根冲击、10% 成交额参与率。",
        "",
        "## 同口径对照",
        "",
        _display_comparison(all_comparison).to_markdown(index=False)
        if not all_comparison.empty
        else "无结果。",
        "",
        "## 容量与资金分配",
        "",
        _display_capacity(candidates).to_markdown(index=False)
        if candidates
        else "无结果。",
        "",
        "## 分年结果",
        "",
        _display_yearly(all_yearly).to_markdown(index=False)
        if not all_yearly.empty
        else "无结果。",
        "",
        "## 验收检查",
        "",
        pd.DataFrame(check_rows).to_markdown(index=False) if check_rows else "无结果。",
        "",
        "## 解释边界",
        "",
        "- 完整账本包含实际持仓、现金、交易规则、公司行动、历史税费、滑点与冲击。",
        "- 周频结果仅作为 P3-5 研究参照，不用完整账本结果反向调整模型。",
        "- 本轮容量诊断对应 100 万资金和 10% 参与率；多资金档位容量曲线属于 P4-4B。",
    ]
    return "\n".join(lines) + "\n"


def output_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """计算报告级输出文件的 SHA256。"""

    return {
        name: file_sha256(root / name)
        for name in names
        if (root / name).is_file()
    }
