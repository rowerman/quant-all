"""只读诊断：按 run_id 打印一次回测的执行校准与偏差，用于验收与回归排查。

用法：
    .venv/bin/python tools/monitor/backtest_audit.py                      # 最近一次运行
    .venv/bin/python tools/monitor/backtest_audit.py --run-id <run_id>
    .venv/bin/python tools/monitor/backtest_audit.py --run-id <run_id> --json

自检不通过时退出码为 1，方便接进验收流程。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from common.paths import BACKTEST_DB_FILE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = BACKTEST_DB_FILE

# 与前端「执行校准」卡片同一套标签
CALIBRATION_LABELS = {
    "cash_buffer": "现金缓冲（执行层预留）",
    "allocation_iterations": "资金分配迭代次数",
    "allocation_segments": "因果求解的调仓段数",
    "allocation_iterations_total": "各调仓段迭代次数合计",
    "allocation_iterations_max": "单个调仓段最大迭代次数",
    "nonconverged_segments": "未收敛的调仓段数",
    "allocation_converged": "资金分配固定点是否稳定",
    "carry_iterations": "现金补回的调整次数",
    "carry_converged": "现金补回是否收敛",
    "provider_state_passes": "提供者状态反馈轮次",
    "provider_state_converged": "提供者状态反馈是否收敛",
    "provider_state_seconds": "提供者状态反馈总耗时（秒）",
    "probe_orders": "探针成交笔数",
    "planned_orders": "计划下单格数",
    "rebalance_orders": "其中再平衡格数",
    "rejected_orders": "规则拒单笔数",
    "forced_liquidations": "退市零价值核销笔数",
    "forced_exit_orders": "退市整理期强制退出成交笔数",
    "delisting_writeoff_shares": "退市核销股数",
    "delisting_writeoff_value": "退市核销参考市值（元）",
    "delisting_residual_shares_at_end": "区间末退市残留股数",
    "delisting_buy_orders": "退市整理期买入成交笔数",
    "delisting_capacity_shortfall_shares": "退市退出容量缺口股数",
    "retry_attempts": "被拦交易的重试尝试次数",
    "retried_filled_cells": "非调仓日补单成交格数",
    "pending_at_end_cells": "区间末仍未解决的待重试格数",
    "cash_shrunk_cells": "被现金缩小的买单格数",
    "cash_limited_cells": "其中进入补回的格数",
    "cash_limited_value": "进入补回的缩水金额（元）",
    "cash_retry_attempts": "现金补回的尝试格次",
    "cash_retry_filled_cells": "现金补回补齐格数",
    "cash_pending_at_end_cells": "区间末仍未补回的格数（不影响固定点收敛）",
    "allocated_cells": "分配出目标股数的格数",
    "bumped_one_lot_cells": "用取整余量补到一手的格数",
    "unaffordable_lot_cells": "一手金额超出预算、未能建仓的格数",
    "bumped_min_order_cells": "用取整余量补到最小申报量的格数",
    "unaffordable_min_order_cells": "最小申报量超出预算、未能建仓的格数",
    "below_min_order_cells": "已有持仓因差额不足最小申报量而未成交的格数",
    "no_price_cells": "缺执行价、未下单的格数",
    "final_orders": "最终成交笔数",
    "final_buys": "最终买入笔数",
    "final_sells": "最终卖出笔数",
    "fallback_fee_orders": "没有成交估计、按意图方向计费笔数",
    "fee_matrix_max_diff": "逐笔费用与成本模型最大偏差（元）",
    "commission_total": "佣金合计（元）",
    "stamp_tax_total": "印花税合计（元）",
    "transfer_fee_total": "过户费合计（元）",
    "slippage_cost_total": "滑点成本合计（元）",
    "economic_cost_total": "订单经济成本合计（元）",
    "total_cost_total": "日结果经济成本合计（元）",
    "cash_clipped_cells": "被现金裁剪的格数",
    "allocation_gap_max_shares": "实际持股与目标股数的最大差额（股）",
    "target_cells": "目标下单格数",
    "weight_deviation_median": "权重偏差中位数",
    "weight_deviation_p95": "权重偏差 95 分位",
    "weight_deviation_max": "权重偏差最大值",
    "weight_deviation_over_1pct": "权重偏差超 1% 的格数",
    "t_plus_1_blocked_cells": "严格 T+1 拦截卖出格数",
    "t_plus_1_blocked_shares": "严格 T+1 拦截卖出股数",
    "t_plus_1_blocked_value": "严格 T+1 拦截卖出金额（元）",
    "single_cap_days": "触发单股上限的调仓日数",
    "single_cap_cells": "单股上限封顶格数",
    "single_cap_excess_weight": "单股上限截下的权重",
    "single_cap_residual_cash": "单股上限未再分配的现金权重",
    "single_cap_max_weight": "单股封顶后的最大目标权重",
    "industry_cap_days": "触发行业上限的调仓日数",
    "industry_cap_groups": "触发上限的行业次数",
    "industry_cap_cells": "行业上限影响的目标格数",
    "industry_cap_excess_weight": "行业上限截下的权重",
    "industry_cap_residual_cash": "行业上限未再分配的现金权重",
    "industry_cap_target_max_weight": "约束后的最大目标行业权重",
    "industry_post_turnover_max_weight": "换手约束后的最大行业权重",
    "industry_inherited_over_cap_days": "换手后继承超限的日期数",
    "industry_inherited_over_cap_groups": "换手后继承超限的行业次数",
    "industry_inherited_excess_weight": "换手后继承的超限权重",
    "industry_missing_cells": "缺失行业的目标格数",
    "risk_residual_cash": "集中度约束留下的现金权重",
    "turnover_limited_days": "触发换手上限的调仓日数",
    "turnover_limited_cells": "换手约束缩放格数",
    "turnover_before_max": "约束前最大计划单边换手",
    "turnover_after_max": "约束后最大计划单边换手",
    "turnover_limit_error_max": "计划换手超限最大误差",
}

RATE_KEYS = {
    "cash_buffer",
    "weight_deviation_median",
    "weight_deviation_p95",
    "weight_deviation_max",
    "single_cap_excess_weight",
    "single_cap_residual_cash",
    "single_cap_max_weight",
    "industry_cap_excess_weight",
    "industry_cap_residual_cash",
    "industry_cap_target_max_weight",
    "industry_post_turnover_max_weight",
    "industry_inherited_excess_weight",
    "risk_residual_cash",
    "turnover_before_max",
    "turnover_after_max",
    "turnover_limit_error_max",
}


def _format(key: str, value) -> str:
    """校准字段的展示格式：比例字段按百分比，其余原样。"""
    if value is None:
        return "—"
    if key in RATE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4%}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _resolve_run(conn, run_id: str | None) -> str:
    """给出要审计的 run_id；没指定就取最近一次运行。"""
    if run_id:
        found = conn.execute("SELECT run_id FROM bt.run WHERE run_id = ?", [run_id]).fetchone()
        if found is not None:
            return found[0]
        # 允许用前缀，方便从日志里复制前 8 位
        matches = conn.execute(
            "SELECT run_id FROM bt.run WHERE run_id LIKE ? ORDER BY started_at DESC", [f"{run_id}%"]
        ).fetchall()
        if len(matches) == 1:
            return matches[0][0]
        if not matches:
            raise SystemExit(f"没有这个 run_id：{run_id}")
        raise SystemExit(f"这个前缀匹配到 {len(matches)} 次运行，请写完整：{run_id}")
    row = conn.execute(
        "SELECT run_id FROM bt.run ORDER BY started_at DESC NULLS LAST LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("bt.run 里还没有运行记录")
    return row[0]


def _table(rows, headers) -> str:
    """把若干行对齐成文本表。"""
    text = [[str(item) for item in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    lines = ["  ".join(header.ljust(widths[position]) for position, header in enumerate(headers))]
    lines.extend(
        "  ".join(cell.ljust(widths[position]) for position, cell in enumerate(row))
        for row in text
    )
    return "\n".join(lines)


def audit(run_id: str | None, db_path: Path) -> dict:
    """读取一次运行的校准与明细，返回审计结果。"""
    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("SET memory_limit='2GB'")
    run_id = _resolve_run(conn, run_id)
    columns = [row[0] for row in conn.execute("DESCRIBE bt.run").fetchall()]
    row = conn.execute("SELECT * FROM bt.run WHERE run_id = ?", [run_id]).fetchone()
    record = dict(zip(columns, row))
    metadata = json.loads(record["metadata_json"]) if record.get("metadata_json") else {}
    calibration = metadata.get("calibration", {})
    table_columns = {
        table: {
            item[0]
            for item in conn.execute(f"DESCRIBE bt.{table}").fetchall()
        }
        for table in (
            "daily",
            "order",
            "rejection",
            "target_weight",
            "holding",
            "allocation",
            "liquidation",
            "corporate_action",
        )
    }

    rejections = (
        conn.execute(
            "SELECT reason, count(*) FROM bt.rejection WHERE run_id = ? GROUP BY 1 ORDER BY 2 DESC",
            [run_id],
        ).fetchall()
        if "reason" in table_columns["rejection"]
        else []
    )
    allocation = (
        conn.execute(
            "SELECT status, count(*), sum(target_weight), sum(allocated_shares) "
            "FROM bt.allocation WHERE run_id = ? GROUP BY 1 ORDER BY 2 DESC",
            [run_id],
        ).fetchall()
        if {"status", "target_weight", "allocated_shares"}
        <= table_columns["allocation"]
        else []
    )
    deviations = (
        conn.execute(
            "SELECT t.trade_date, t.stock_code, t.target_weight, coalesce(h.weight, 0.0) AS weight, "
            "abs(coalesce(h.weight, 0.0) - t.target_weight) AS deviation "
            "FROM bt.target_weight t LEFT JOIN bt.holding h ON h.run_id = t.run_id "
            "AND h.trade_date = t.trade_date AND h.stock_code = t.stock_code "
            "WHERE t.run_id = ? ORDER BY deviation DESC LIMIT 10",
            [run_id],
        ).fetchall()
        if {"trade_date", "stock_code", "target_weight"}
        <= table_columns["target_weight"]
        and {"trade_date", "stock_code", "weight"} <= table_columns["holding"]
        else []
    )
    counts = {
        table: conn.execute(f"SELECT count(*) FROM bt.{table} WHERE run_id = ?", [run_id]).fetchone()[0]
        for table in ("daily", "order", "rejection", "holding", "allocation", "liquidation", "corporate_action")
    }
    order_columns = table_columns["order"]
    daily_columns = table_columns["daily"]
    order_cost_error = None
    daily_cost_error = None
    if {"commission", "stamp_tax", "transfer_fee", "slippage_cost"} <= order_columns:
        order_cost_error = conn.execute(
            "SELECT coalesce(max(abs(Fees - (commission + stamp_tax + transfer_fee))), 0) "
            "FROM bt.order WHERE run_id = ?",
            [run_id],
        ).fetchone()[0]
    if {"commission", "stamp_tax", "transfer_fee", "dividend_tax", "slippage_cost", "total_cost"} <= daily_columns:
        impact_term = " + coalesce(impact_cost, 0)" if "impact_cost" in daily_columns else ""
        daily_cost_error = conn.execute(
            "SELECT coalesce(max(abs(cost - (commission + stamp_tax + transfer_fee + dividend_tax))), 0), "
            f"coalesce(max(abs(total_cost - (cost + slippage_cost{impact_term}))), 0) "
            "FROM bt.daily WHERE run_id = ?",
            [run_id],
        ).fetchone()
    has_board_audit = "Board" in order_columns
    invalid_board_buys = None
    board_orders = []
    liquidation_columns = table_columns["liquidation"]
    writeoff_cash_rows = None
    if {"price", "value"} <= liquidation_columns:
        reason_filter = (
            "AND coalesce(reason, '') = 'zero_writeoff' "
            if "reason" in liquidation_columns else ""
        )
        writeoff_cash_rows = conn.execute(
            "SELECT count(*) FROM bt.liquidation WHERE run_id = ? "
            f"{reason_filter}"
            "AND (coalesce(price, 0) <> 0 OR coalesce(value, 0) <> 0)",
            [run_id],
        ).fetchone()[0]
    forced_exit_orders = None
    if "forced_exit" in order_columns:
        forced_exit_orders = conn.execute(
            "SELECT count(*) FROM bt.order WHERE run_id = ? AND coalesce(forced_exit, FALSE)",
            [run_id],
        ).fetchone()[0]
    if has_board_audit:
        board_orders = conn.execute(
            "SELECT coalesce(Board, '未知'), Side, count(*), "
            "coalesce(sum(abs(Size) * Price), 0) "
            "FROM bt.order WHERE run_id = ? GROUP BY 1, 2 ORDER BY 1, 2",
            [run_id],
        ).fetchall()
        invalid_board_buys = conn.execute(
            "SELECT count(*) FROM bt.order WHERE run_id = ? AND Side = 'Buy' AND ("
            "Size IS NULL OR abs(Size - round(Size)) > 1e-9 OR "
            "(Board = '科创板' AND Size < 200) OR "
            "(Board = '北交所' AND Size < 100) OR "
            "(Board IN ('主板', '创业板') AND abs(Size % 100) > 1e-9))",
            [run_id],
        ).fetchone()[0]
    elif {"Side", "Size"} <= order_columns:
        off_lot = conn.execute(
            "SELECT count(*) FROM bt.order WHERE run_id = ? AND Side = 'Buy' "
            "AND (Size IS NULL OR abs(Size % 100) > 1e-9)",
            [run_id],
        ).fetchone()[0]
    else:
        off_lot = 0
    # 老库的 bt.allocation 没有 cash_limited 列，缺列时不查这一项
    allocation_columns = table_columns["allocation"]
    invalid_allocation_orders = None
    if "order_valid" in allocation_columns:
        invalid_allocation_orders = conn.execute(
            "SELECT count(*) FROM bt.allocation WHERE run_id = ? "
            "AND NOT coalesce(order_valid, TRUE)",
            [run_id],
        ).fetchone()[0]
    cash_limited = None
    if {
        "cash_limited",
        "cash_shrunk_shares",
        "trade_date",
        "price",
    } <= allocation_columns:
        cash_limited = conn.execute(
            "SELECT count(*) FILTER (WHERE cash_shrunk_shares > 0), "
            "count(*) FILTER (WHERE cash_limited), count(DISTINCT trade_date) "
            "FILTER (WHERE cash_limited), coalesce(sum(cash_shrunk_shares) "
            "FILTER (WHERE cash_limited), 0), coalesce(sum(cash_shrunk_shares * price) "
            "FILTER (WHERE cash_limited), 0) FROM bt.allocation WHERE run_id = ?",
            [run_id],
        ).fetchone()
    conn.close()

    final_orders = int(calibration.get("final_orders", 0) or 0)
    fallback = int(calibration.get("fallback_fee_orders", 0) or 0)
    target_cells = int(calibration.get("target_cells", 0) or 0)
    checks = [
        (
            "所有调仓段资金分配均已收敛",
            bool(calibration.get("allocation_converged"))
            and int(calibration.get("nonconverged_segments", 0) or 0) == 0,
            "未收敛段 {}，单段最大迭代 {}".format(
                calibration.get("nonconverged_segments", 0),
                calibration.get("allocation_iterations_max", calibration.get("allocation_iterations")),
            ),
        ),
        (
            "费用矩阵逐笔核对无偏差",
            float(calibration.get("fee_matrix_max_diff", 0.0) or 0.0) < 1e-6,
            f"最大偏差 {calibration.get('fee_matrix_max_diff')} 元",
        ),
        (
            "订单费用分项可加总",
            order_cost_error is None or float(order_cost_error) < 1e-6,
            f"最大偏差 {order_cost_error} 元" if order_cost_error is not None else "旧运行无成本分项",
        ),
        (
            "日成本分项可加总",
            daily_cost_error is None
            or (float(daily_cost_error[0]) < 1e-6 and float(daily_cost_error[1]) < 1e-6),
            f"现金成本偏差 {daily_cost_error[0]} 元，总成本偏差 {daily_cost_error[1]} 元"
            if daily_cost_error is not None else "旧运行无成本分项",
        ),
        (
            "没有被现金裁剪的格子",
            int(calibration.get("cash_clipped_cells", 0) or 0) == 0,
            f"裁剪格数 {calibration.get('cash_clipped_cells')}",
        ),
        (
            "兜底费率笔数占比低于 1%",
            (final_orders == 0 and fallback == 0)
            or (final_orders > 0 and fallback / final_orders < 0.01),
            f"无成交（兜底 {fallback} 笔）"
            if final_orders == 0 else f"兜底 {fallback}/{final_orders} 笔",
        ),
    ]
    if has_board_audit:
        checks.append(
            (
                "买入申报符合所属板块交易单位",
                int(invalid_board_buys or 0) == 0,
                f"非法买入 {invalid_board_buys} 笔",
            )
        )
        checks.append(
            (
                "资金分配订单逐格合法",
                int(invalid_allocation_orders or 0) == 0,
                f"非法分配订单 {invalid_allocation_orders} 格",
            )
        )
    else:
        checks.append(
            ("买入成交全是 100 股整数倍", int(off_lot) == 0, f"非整数倍买入 {off_lot} 笔")
        )
    checks.append(
        (
            "区间末没有已知退市残留持仓",
            float(calibration.get("delisting_residual_shares_at_end", 0.0) or 0.0) <= 1e-9,
            f"残留 {calibration.get('delisting_residual_shares_at_end')} 股",
        )
    )
    checks.append(
        (
            "退市整理期没有买入成交",
            int(calibration.get("delisting_buy_orders", 0) or 0) == 0,
            f"买入 {calibration.get('delisting_buy_orders')} 笔",
        )
    )
    if writeoff_cash_rows is not None:
        checks.append(
            (
                "退市核销不产生现金成交",
                int(writeoff_cash_rows) == 0,
                f"非零核销金额 {writeoff_cash_rows} 行",
            )
        )
    if forced_exit_orders is not None:
        checks.append(
            (
                "退市强制退出成交标记与校准一致",
                int(forced_exit_orders) == int(calibration.get("forced_exit_orders", 0) or 0),
                f"成交表 {forced_exit_orders} 笔，校准 "
                f"{calibration.get('forced_exit_orders', 0)} 笔",
            )
        )
    risk = metadata.get("risk") or {}
    if risk.get("max_single_weight") is not None:
        checks.append(
            (
                "单股目标封顶误差为零",
                float(calibration.get("single_cap_max_weight", 0.0) or 0.0)
                <= float(risk["max_single_weight"]) + 1e-9,
                f"最大目标权重 {calibration.get('single_cap_max_weight')}，上限 {risk['max_single_weight']}",
            )
        )
    if risk.get("max_one_way_turnover") is not None:
        checks.append(
            (
                "计划单边换手不超过上限",
                float(calibration.get("turnover_limit_error_max", 0.0) or 0.0) <= 1e-9,
                f"超限误差 {calibration.get('turnover_limit_error_max')}",
            )
        )
    if risk.get("max_industry_weight") is not None:
        checks.append(
            (
                "申万一级目标权重不超过上限",
                float(calibration.get("industry_cap_target_max_weight", 0.0) or 0.0)
                <= float(risk["max_industry_weight"]) + 1e-9,
                f"最大目标行业权重 {calibration.get('industry_cap_target_max_weight')}，"
                f"上限 {risk['max_industry_weight']}；换手后最大 "
                f"{calibration.get('industry_post_turnover_max_weight')}",
            )
        )
    policy = metadata.get("cash_retry_policy")
    if cash_limited is not None and policy == "all_cash_shortfall":
        checks.append(
            (
                "现金缩水格全部进入补回",
                int(cash_limited[0]) == int(cash_limited[1]),
                f"缩水 {cash_limited[0]} 格，补回 {cash_limited[1]} 格",
            )
        )
    elif cash_limited is not None and policy == "all_cash_shortfall_event":
        checks.append(
            (
                "现金缩水事件已全部进入补回",
                int(cash_limited[1]) == int(calibration.get("cash_limited_cells", 0) or 0)
                and abs(float(cash_limited[4]) - float(calibration.get("cash_limited_value", 0.0) or 0.0)) < 1e-6,
                f"缩水行 {cash_limited[0]} 行，补回事件 {cash_limited[1]} 格",
            )
        )
    return {
        "run_id": run_id,
        "run": {key: record[key] for key in columns if key != "metadata_json"},
        "metadata": {
            key: metadata.get(key)
            for key in (
                "strategy_hash",
                "universe",
                "benchmark",
                "price_adjust",
                "execution",
                "risk",
                "panel",
                "elapsed_seconds",
            )
        },
        "calibration": calibration,
        "rejections": [list(item) for item in rejections],
        "allocation": [list(item) for item in allocation],
        "cash_limited": None
        if cash_limited is None
        else {
            "shrunk_cells": int(cash_limited[0]),
            "event_cells": int(cash_limited[1]),
            "days": int(cash_limited[2]),
            "shares": float(cash_limited[3]),
            "value": float(cash_limited[4]),
            "target_cells": target_cells,
            "ratio": (int(cash_limited[1]) / target_cells) if target_cells else None,
        },
        "deviations": [list(item) for item in deviations],
        "board_orders": [list(item) for item in board_orders],
        "counts": counts,
        "checks": checks,
    }


def render(result: dict) -> str:
    """把审计结果渲染成文本。"""
    run, metadata = result["run"], result["metadata"]
    benchmark = metadata.get("benchmark") or {}
    lines = [
        f"== 运行 {result['run_id']} ==",
        f"策略        {run['strategy_id']} v{run['strategy_version']}（{metadata.get('strategy_hash')}）",
        f"区间        {run['start_date']} ~ {run['end_date']}",
        f"股票池      {run.get('universe') or metadata.get('universe') or '—'}",
        f"基准        {run.get('benchmark') or benchmark.get('code') or '—'}",
        f"状态        {run['status']}    耗时 {metadata.get('elapsed_seconds')} 秒",
        f"价格口径    {metadata.get('price_adjust')}",
        "执行口径    {}".format(json.dumps(metadata.get("execution"), ensure_ascii=False)),
        "面板        {}".format(json.dumps(metadata.get("panel"), ensure_ascii=False)),
        "",
        "== 执行校准 ==",
    ]
    for key, label in CALIBRATION_LABELS.items():
        if key in result["calibration"]:
            lines.append(f"{label:<32} {_format(key, result['calibration'][key])}")
    lines.append("")
    lines.append("== 规则拒单（按原因） ==")
    lines.append(
        _table(result["rejections"], ["原因", "条数"]) if result["rejections"] else "无"
    )
    lines.append("")
    lines.append("== 资金分配（按状态） ==")
    lines.append(
        _table(
            [
                (status, count, f"{float(weight or 0):.4f}", f"{float(shares or 0):.0f}")
                for status, count, weight, shares in result["allocation"]
            ],
            ["状态", "格数", "目标权重合计", "目标股数合计"],
        )
        if result["allocation"]
        else "无"
    )
    lines.append("")
    limited = result["cash_limited"]
    if limited is not None:
        ratio = "—" if limited["ratio"] is None else f"{limited['ratio']:.4%}"
        lines.append(
            "被现金缩小的买单 {} 行，其中进入补回事件 {} 格 / 目标格 {}（{}），"
            "涉及 {} 个交易日，合计 {} 股".format(
                limited["shrunk_cells"],
                limited["event_cells"],
                limited["target_cells"],
                ratio,
                limited["days"],
                f"{limited['shares']:.0f}",
            )
        )
        lines.append("")
    if result["board_orders"]:
        lines.append("== 成交板块分布 ==")
        lines.append(
            _table(
                [
                    (board, side, count, f"{float(value or 0):,.2f}")
                    for board, side, count, value in result["board_orders"]
                ],
                ["板块", "方向", "笔数", "成交金额（元）"],
            )
        )
        lines.append("")
    lines.append("== 权重偏差最大的目标格 ==")
    lines.append(
        _table(
            [
                (str(date)[:10], code, f"{float(target):.4%}", f"{float(weight):.4%}", f"{float(dev):.4%}")
                for date, code, target, weight, dev in result["deviations"]
            ],
            ["日期", "股票", "目标权重", "实际权重", "偏差"],
        )
        if result["deviations"]
        else "无"
    )
    lines.append("")
    lines.append("== 明细表行数 ==")
    lines.append(_table(sorted(result["counts"].items()), ["表", "行数"]))
    lines.append("")
    lines.append("== 自检结论 ==")
    for name, ok, detail in result["checks"]:
        lines.append(f"[{'通过' if ok else '不通过'}] {name}（{detail}）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="回测运行自检：打印校准与偏差")
    parser.add_argument("--run-id", default=None, help="不指定则取最近一次运行")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="回测库路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    result = audit(args.run_id, Path(args.db))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    else:
        print(render(result))
    return 0 if all(ok for _, ok, _ in result["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
