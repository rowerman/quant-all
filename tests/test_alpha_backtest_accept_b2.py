"""P4-4B2 三档容量曲线、来源复用、产物和报告测试。"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.build_alpha.backtest_accept_b2 as b2
from scripts.build_alpha.backtest_accept_b2_artifacts import (
    SCHEMA_VERSION,
    read_tier_completion,
    tier_is_complete,
    write_tier_artifacts,
)
from scripts.build_alpha.backtest_accept_b2_report import (
    BASE_CASH,
    CAPITAL_TIERS,
    TIER_CANDIDATE,
    build_report,
    capacity_curve,
    economic_capacity,
    yearly_curve,
)


def _checks(passed: bool = True) -> dict:
    """构造结构验收结果。"""

    return {
        "passed": passed,
        "checks": {
            "allocation_converged": passed,
            "carry_converged": passed,
        },
    }


def _payload(
    cash: float,
    *,
    annual_return: float = 0.2,
    excess: float = 0.18,
    limited_value: float = 1_000_000.0,
    passed: bool = True,
) -> dict:
    """构造容量曲线所需的最小档位结果。"""

    return {
        "metrics": {
            "annual_return": annual_return,
            "excess_annual_return": excess,
            "sharpe": 1.0,
            "max_drawdown": -0.2,
            "turnover_two_way_annual": 40.0,
            "total_cost": cash * 0.3,
            "total_cost_bps_of_nav": 300.0,
            "turnover_attribution": {
                "buy_value": cash * 20.0,
                "sell_value": cash * 20.0,
            },
        },
        "capacity": {
            "liquidity_limited_cells": int(cash / BASE_CASH),
            "liquidity_limited_value": limited_value,
            "liquidity_shortfall_shares": 100.0 * cash / BASE_CASH,
            "liquidity_retry_filled_cells": 20.0,
            "cash_carry_over_cells": 10.0,
            "pending_at_end_cells": 0.0,
            "allocation_converged": passed,
            "carry_converged": passed,
        },
        "checks": _checks(passed),
        "run_manifest": {
            "calibration": {
                "planned_orders": 100,
                "final_orders": 98,
                "unaffordable_min_order_cells": 1,
                "weight_deviation_p95": 0.01,
                "weight_deviation_max": 0.05,
            }
        },
        "yearly": pd.DataFrame(
            {
                "year": [2023],
                "total_return": [0.1],
                "benchmark_total_return": [0.02],
                "excess_total_return": [0.08],
                "max_drawdown": [-0.1],
                "turnover_two_way_annual": [40.0],
                "total_cost": [1_000.0],
            }
        ),
        "snapshot_id": "a" * 64,
    }


def _daily() -> pd.DataFrame:
    """构造两日账本结果。"""

    return pd.DataFrame(
        {
            "return": [0.01, -0.005],
            "drawdown": [0.0, -0.005],
            "turnover": [0.1, 0.08],
            "cost": [60.0, 40.0],
            "total_cost": [60.0, 40.0],
            "benchmark_return": [0.002, -0.001],
        },
        index=pd.DatetimeIndex(["2023-01-03", "2023-01-04"]),
    )


def test_fixed_candidate_tiers_and_source():
    """只使用选中候选，三档顺序和 P4-4B1 来源固定。"""

    assert b2.TIER_CANDIDATE == TIER_CANDIDATE == "LIGHTGBM_SMOOTH_4W"
    assert b2.CAPITAL_TIERS == CAPITAL_TIERS == (
        1_000_000.0,
        10_000_000.0,
        100_000_000.0,
    )
    manifest, digest = b2._source_reference()
    assert manifest["status"] == "pass_p4_4b1"
    assert manifest["candidates"][TIER_CANDIDATE]["passed"] is True
    assert len(digest) == 64
    assert b2._load_source_tier.__name__ == "_load_source_tier"


def test_analysis_id_is_deterministic_and_source_sensitive():
    """模型、来源快照、档位和代码变化必须改变分析 ID。"""

    scope = {"rebalance_dates_sha256": "r"}
    scopes = {cash: scope for cash in CAPITAL_TIERS}
    model = {
        "model_sha256": "m",
        "prediction_sha256": "p",
        "dataset_id": "d",
        "prediction_rows": 3,
        "prediction_start": "2018-01-12",
        "prediction_end": "2023-12-29",
        "seeds": [0, 1, 2, 3, 4],
    }
    left = b2.acceptance_analysis_id(
        scopes,
        model,
        "source",
        "snapshot",
        "code",
    )
    right = b2.acceptance_analysis_id(
        scopes,
        model,
        "source",
        "snapshot",
        "code",
    )
    changed_model = b2.acceptance_analysis_id(
        scopes,
        {**model, "prediction_sha256": "p2"},
        "source",
        "snapshot",
        "code",
    )
    changed_source = b2.acceptance_analysis_id(
        scopes,
        model,
        "source2",
        "snapshot",
        "code",
    )
    assert left.startswith("p4_4b2_")
    assert left == right
    assert left != changed_model
    assert left != changed_source


def test_capacity_curve_reports_changes_limits_and_economic_capacity():
    """容量曲线包含相对 100 万变化、成交额限制率和经济容量。"""

    candidates = {
        1_000_000.0: _payload(1_000_000.0, annual_return=0.28, excess=0.29),
        10_000_000.0: _payload(10_000_000.0, annual_return=0.20, excess=0.21),
        100_000_000.0: _payload(100_000_000.0, annual_return=-0.05, excess=-0.06),
    }
    curve = capacity_curve(candidates)
    assert curve["init_cash"].tolist() == list(CAPITAL_TIERS)
    assert curve.iloc[1]["annual_return_change_vs_1m"] == pytest.approx(-0.08)
    assert curve.iloc[1]["excess_return_change_vs_1m"] == pytest.approx(-0.08)
    assert curve.iloc[1]["liquidity_limited_rate"] == pytest.approx(
        1_000_000.0 / 400_000_000.0
    )
    assert economic_capacity(curve) == 10_000_000.0
    yearly = yearly_curve(candidates)
    assert set(yearly["init_cash"]) == set(CAPITAL_TIERS)


def test_b2_artifacts_complete_only_after_hash_validation(tmp_path):
    """B2 档位完成标记覆盖全部文件，损坏后拒绝恢复。"""

    folder = tmp_path / "cash_10000000"
    write_tier_artifacts(
        folder,
        analysis_id="p4_4b2_demo",
        candidate=TIER_CANDIDATE,
        metrics={"candidate": TIER_CANDIDATE},
        strategy={"name": TIER_CANDIDATE},
        yearly=pd.DataFrame({"year": [2023]}),
        daily=_daily(),
        orders=pd.DataFrame({"Size": [100]}),
        holdings=pd.DataFrame({"stock_code": ["SH600000"]}),
        run_manifest={"snapshot_id": "a" * 64},
    )
    completion = read_tier_completion(folder)
    assert completion["schema_version"] == SCHEMA_VERSION
    assert tier_is_complete(folder, "p4_4b2_demo", TIER_CANDIDATE)
    (folder / "orders.parquet").write_bytes(b"broken")
    assert not tier_is_complete(folder, "p4_4b2_demo", TIER_CANDIDATE)


def test_report_separates_structural_and_economic_conclusions():
    """报告同时包含结构状态、三档曲线和经济容量观察。"""

    candidates = {
        1_000_000.0: _payload(1_000_000.0, annual_return=0.28, excess=0.29),
        10_000_000.0: _payload(10_000_000.0, annual_return=0.20, excess=0.21),
        100_000_000.0: _payload(100_000_000.0, annual_return=0.10, excess=0.11),
    }
    report = build_report(
        analysis_id="p4_4b2_demo",
        candidates=candidates,
        expected_tiers=3,
        source_analysis_id="p4_4b1_demo",
        source_manifest_sha256="b" * 64,
    )
    assert "P4-4B2" in report
    assert "执行结构结论：**通过**" in report
    assert "三个已测档位均为正" in report
    assert "100.00" in report
    assert "成交额限制格数" in report
    assert "2024-2026 最终测试集未读取" in report


def test_execute_reuses_base_and_skips_complete_tiers(monkeypatch, tmp_path):
    """执行器不重跑 100 万档，并在续跑时跳过已完成新档位。"""

    source = _payload(BASE_CASH)
    source["source"] = {"analysis_id": "source"}
    monkeypatch.setattr(
        b2,
        "_source_reference",
        lambda: ({"status": "pass_p4_4b1"}, "a" * 64),
    )
    monkeypatch.setattr(
        b2,
        "_load_model_input",
        lambda: ({"model": 1}, pd.DatetimeIndex([])),
    )
    monkeypatch.setattr(
        b2,
        "_build_scopes",
        lambda dates: {cash: {"init_cash": cash} for cash in CAPITAL_TIERS},
    )
    monkeypatch.setattr(b2, "_load_source_tier", lambda *args: source)
    monkeypatch.setattr(
        b2,
        "acceptance_analysis_id",
        lambda *args: "p4_4b2_demo",
    )
    monkeypatch.setattr(b2, "tier_is_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        b2,
        "_load_complete_tier",
        lambda folder, analysis_id, scope: {
            **_payload(scope["init_cash"]),
            "source": {"path": str(folder)},
        },
    )

    def fail_if_run(*args, **kwargs):
        raise AssertionError("已完成档位不应重跑")

    monkeypatch.setattr(b2, "_run_tier", fail_if_run)
    monkeypatch.setattr(
        b2,
        "_write_outputs",
        lambda **kwargs: {
            "status": "pass_p4_4b2",
            "economic_capacity_cash": 100_000_000.0,
        },
    )
    result = b2.execute_acceptance(resume=True, output_root=tmp_path)
    assert result["analysis_id"] == "p4_4b2_demo"
    assert result["status"] == "pass_p4_4b2"
    assert result["tiers"] == {
        BASE_CASH: "reused_p4_4b1",
        10_000_000.0: "skipped_complete",
        100_000_000.0: "skipped_complete",
    }
    assert (tmp_path / "p4_4b2_demo").is_dir()
