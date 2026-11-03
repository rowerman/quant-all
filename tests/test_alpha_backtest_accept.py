"""P4-4A 完整账本验收的固定口径、产物和检查测试。"""

from __future__ import annotations


import pandas as pd
import pytest

from alpha.errors import AlphaConfigError
from scripts.build_alpha.backtest_accept import (
    CANDIDATES,
    acceptance_analysis_id,
    acceptance_scope,
    attach_rebalance_dates,
    build_candidate_strategy,
    candidate_specs,
)
from scripts.build_alpha.backtest_accept_artifacts import (
    candidate_is_complete,
    read_json,
    write_candidate_artifacts,
)
from scripts.build_alpha.backtest_accept_report import (
    acceptance_checks,
    capacity_summary,
    comparison_frame,
    yearly_summary,
)


def _daily() -> pd.DataFrame:
    """构造两日完整账本日结果。"""

    index = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
    return pd.DataFrame(
        {
            "return": [0.01, -0.005],
            "portfolio_value": [1_010_000.0, 1_004_950.0],
            "nav": [1.01, 1.00495],
            "drawdown": [0.0, -0.005],
            "cost": [100.0, 80.0],
            "total_cost": [150.0, 120.0],
            "turnover": [0.1, 0.08],
            "benchmark_return": [0.002, -0.001],
            "benchmark_nav": [1.002, 1.000998],
            "excess_return": [0.008, -0.004],
        },
        index=index,
    )


def _metrics() -> dict:
    """构造通过结构检查的核心指标。"""

    return {
        "annual_return": 0.1,
        "excess_annual_return": 0.08,
        "annual_volatility": 0.2,
        "sharpe": 0.5,
        "max_drawdown": -0.1,
        "turnover_two_way_annual": 40.0,
        "total_cost": 270.0,
        "total_cost_bps_of_nav": 2.7,
        "benchmark_total_return": 0.001,
        "cost_attribution": {"total": {"total_cost": 270.0}},
    }


def _metadata() -> dict:
    """构造通过结构检查的运行元数据。"""

    return {
        "snapshot_id": "a" * 64,
        "calibration": {
            "allocation_converged": True,
            "carry_converged": True,
            "allocation_iterations": 2,
            "allocation_iterations_total": 3,
            "liquidity_limited_cells": 1,
            "liquidity_limited_value": 1000.0,
            "liquidity_unavailable_cells": 0,
            "cash_clipped_cells": 0,
            "cash_carry_over_cells": 0,
            "pending_at_end_cells": 0,
        },
    }


def test_scope_rejects_final_holdout_and_candidate_mapping_is_fixed():
    """验收不得进入 2024 年；候选映射保持 P4-4A 固定范围。"""

    with pytest.raises(AlphaConfigError, match="最终测试集"):
        acceptance_scope(end="2024-01-02")
    assert [spec.name for spec in candidate_specs("all")] == [
        "LIGHTGBM_SMOOTH_4W",
        "B4_ALL_RANK",
    ]
    assert candidate_specs("b4") == (CANDIDATES[1],)


def test_candidate_strategies_use_fixed_providers():
    """选中模型使用 ml_rank v2，B4 使用 factor_rank v2 和同一组日期。"""

    dates = ["2023-01-06", "2023-01-13"]
    selected = build_candidate_strategy(CANDIDATES[0], 30, dates)
    baseline = build_candidate_strategy(CANDIDATES[1], 30, dates)
    assert selected.signal.provider.name == "ml_rank"
    assert selected.signal.provider.version == "2"
    assert selected.signal.provider.params["top_n"] == 30
    assert selected.signal.provider.params["rebalance_dates"] == dates
    assert selected.portfolio.selection == "provider"
    assert baseline.signal.provider.name == "factor_rank"
    assert baseline.signal.provider.version == "2"
    assert baseline.signal.provider.params["rebalance_dates"] == dates
    assert baseline.rebalance.frequency == "custom"


def test_attach_rebalance_dates_records_hash_and_rejects_outside_dates():
    """显式日期进入范围和哈希，区间外日期直接拒绝。"""

    scope = acceptance_scope()
    dates = pd.DatetimeIndex(["2023-01-06", "2023-01-13"])
    attached = attach_rebalance_dates(scope, dates)
    assert attached["rebalance_dates"] == ["2023-01-06", "2023-01-13"]
    assert attached["rebalance_dates_count"] == 2
    assert len(attached["rebalance_dates_sha256"]) == 64
    with pytest.raises(AlphaConfigError, match="超出验收区间"):
        attach_rebalance_dates(scope, pd.DatetimeIndex(["2024-01-01"]))


def test_analysis_id_is_deterministic_and_model_sensitive():
    """模型或代码哈希变化会改变分析 ID。"""

    scope = acceptance_scope()
    manifest = {
        "model_id": "model",
        "model_sha256": "m",
        "prediction_sha256": "p",
        "dataset_id": "d",
    }
    left = acceptance_analysis_id(scope, CANDIDATES, manifest, "code")
    right = acceptance_analysis_id(scope, CANDIDATES, dict(manifest), "code")
    changed = acceptance_analysis_id(
        scope,
        CANDIDATES,
        {**manifest, "prediction_sha256": "p2"},
        "code",
    )
    assert left == right
    assert left != changed
    assert left.startswith("p4_4a_")


def test_yearly_capacity_and_acceptance_checks_reconcile():
    """分年指标、容量和成本闭环可从标准化结果重建。"""

    daily = _daily()
    yearly = yearly_summary(daily)
    capacity = capacity_summary(_metadata())
    checks = acceptance_checks(
        _metrics(),
        _metadata(),
        daily,
        dev_start="2012-01-01",
        dev_end="2023-12-29",
        expected_signal_days=1,
        expected_execution_days=1,
        actual_signal_days=1,
    )
    assert len(yearly) == 1
    assert yearly.iloc[0]["total_cost"] == pytest.approx(270.0)
    assert capacity["liquidity_limited_cells"] == 1
    assert capacity["allocation_converged"] is True
    assert checks["passed"] is True


def test_acceptance_checks_fail_closed_on_nonconvergence():
    """资金分配未收敛时不能给出通过结论。"""

    metadata = _metadata()
    metadata["calibration"]["allocation_converged"] = False
    checks = acceptance_checks(
        _metrics(),
        metadata,
        _daily(),
        dev_start="2012-01-01",
        dev_end="2023-12-29",
        expected_signal_days=1,
        expected_execution_days=1,
        actual_signal_days=1,
    )
    assert checks["passed"] is False
    assert checks["checks"]["allocation_converged"] is False


def test_candidate_artifacts_are_hashed_and_resume_validated(tmp_path):
    """完成标记覆盖全部候选文件，内容变化后恢复校验失败。"""

    folder = tmp_path / "candidate"
    yearly = yearly_summary(_daily())
    payload = {
        "candidate": "demo",
        "metrics": _metrics(),
        "capacity": capacity_summary(_metadata()),
        "checks": {"passed": True},
        "snapshot_id": "a" * 64,
    }
    write_candidate_artifacts(
        folder,
        analysis_id="p4_4a_demo",
        candidate="demo",
        metrics=payload,
        strategy={"name": "demo"},
        yearly=yearly,
        daily=_daily(),
        orders=pd.DataFrame({"Size": [100]}),
        holdings=pd.DataFrame({"stock_code": ["SH600000"]}),
        run_manifest={"snapshot_id": "a" * 64},
    )
    assert candidate_is_complete(folder, "p4_4a_demo", "demo")
    assert read_json(folder / "metrics.json")["candidate"] == "demo"
    (folder / "orders.parquet").write_bytes(b"broken")
    assert not candidate_is_complete(folder, "p4_4a_demo", "demo")


def test_comparison_keeps_acceptance_status():
    """汇总对照表保留候选通过状态和快照。"""

    frame = comparison_frame(
        {
            "demo": {
                "metrics": _metrics(),
                "checks": {"passed": True},
                "run_manifest": {"snapshot_id": "a" * 64},
            }
        }
    )
    assert frame.iloc[0]["candidate"] == "demo"
    assert bool(frame.iloc[0]["acceptance_passed"])
