"""回测运行自检脚本的成本检查边界。"""

from datetime import datetime
import json
from pathlib import Path

import duckdb

from tools.monitor.backtest_audit import audit


def _make_db(
    path: Path,
    *,
    final_orders: int,
    fallback: int,
    with_impact: bool,
    legacy: bool = False,
) -> str:
    """构造只包含审计脚本所需表的最小回测库。"""
    run_id = "audit-test-run"
    metadata = {
        "calibration": {
            "allocation_converged": True,
            "nonconverged_segments": 0,
            "allocation_iterations_max": 1,
            "fee_matrix_max_diff": 0.0,
            "cash_clipped_cells": 0,
            "final_orders": final_orders,
            "fallback_fee_orders": fallback,
            "target_cells": final_orders,
            "delisting_residual_shares_at_end": 0.0,
            "delisting_buy_orders": 0,
        }
    }
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA bt")
    con.execute(
        "CREATE TABLE bt.run (run_id VARCHAR, status VARCHAR, metadata_json VARCHAR, started_at TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO bt.run VALUES (?, 'success', ?, ?)",
        [run_id, json.dumps(metadata), datetime(2026, 10, 19)],
    )
    impact_column = "" if legacy else ", impact_cost DOUBLE"
    con.execute(
        "CREATE TABLE bt.daily (run_id VARCHAR, cost DOUBLE, commission DOUBLE, "
        "stamp_tax DOUBLE, transfer_fee DOUBLE, dividend_tax DOUBLE, "
        f"slippage_cost DOUBLE, total_cost DOUBLE{impact_column})"
    )
    con.execute(
        "CREATE TABLE bt.order (run_id VARCHAR, Side VARCHAR, Size DOUBLE, Price DOUBLE, "
        "Fees DOUBLE, commission DOUBLE, stamp_tax DOUBLE, transfer_fee DOUBLE, "
        "slippage_cost DOUBLE)"
    )
    con.execute("CREATE TABLE bt.rejection (run_id VARCHAR, reason VARCHAR)")
    con.execute(
        "CREATE TABLE bt.allocation (run_id VARCHAR, status VARCHAR, "
        "target_weight DOUBLE, allocated_shares DOUBLE)"
    )
    con.execute(
        "CREATE TABLE bt.target_weight (run_id VARCHAR, trade_date DATE, "
        "stock_code VARCHAR, target_weight DOUBLE)"
    )
    con.execute(
        "CREATE TABLE bt.holding (run_id VARCHAR, trade_date DATE, "
        "stock_code VARCHAR, weight DOUBLE)"
    )
    con.execute("CREATE TABLE bt.liquidation (run_id VARCHAR)")
    con.execute("CREATE TABLE bt.corporate_action (run_id VARCHAR)")
    if final_orders:
        if legacy:
            con.execute(
                "INSERT INTO bt.daily VALUES (?, 10, 1, 2, 3, 4, 5, 15)",
                [run_id],
            )
        else:
            con.execute(
                "INSERT INTO bt.daily VALUES (?, 10, 1, 2, 3, 4, 5, ?, ?)",
                [run_id, 21.0 if with_impact else 15.0, 6.0 if with_impact else 0.0],
            )
        con.execute(
            "INSERT INTO bt.order VALUES (?, 'Buy', 100, 10, 6, 1, 2, 3, 5)",
            [run_id],
        )
    con.close()
    return run_id


def _check(result: dict, name: str) -> tuple[bool, str]:
    """取出指定自检项。"""
    return next((ok, detail) for item, ok, detail in result["checks"] if item == name)


def test_daily_cost_check_includes_impact_cost(tmp_path):
    """日结果总成本要能与现金成本、滑点和冲击成本加总对上。"""
    run_id = _make_db(tmp_path / "with_impact.duckdb", final_orders=1, fallback=0, with_impact=True)
    result = audit(run_id, tmp_path / "with_impact.duckdb")
    ok, detail = _check(result, "日成本分项可加总")
    assert ok, detail


def test_zero_order_run_passes_fallback_check(tmp_path):
    """无成交运行没有兜底费率可算，不应被比例检查判失败。"""
    run_id = _make_db(tmp_path / "no_orders.duckdb", final_orders=0, fallback=0, with_impact=False)
    result = audit(run_id, tmp_path / "no_orders.duckdb")
    ok, detail = _check(result, "兜底费率笔数占比低于 1%")
    assert ok, detail


def test_legacy_daily_without_impact_column_still_audits(tmp_path):
    """没有 impact_cost 列的旧运行继续按现金成本加滑点校验。"""
    run_id = _make_db(
        tmp_path / "legacy.duckdb",
        final_orders=1,
        fallback=0,
        with_impact=False,
        legacy=True,
    )
    result = audit(run_id, tmp_path / "legacy.duckdb")
    ok, detail = _check(result, "日成本分项可加总")
    assert ok, detail
