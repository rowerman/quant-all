"""P4-4B1 固定候选、来源复用、产物和报告测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_alpha.backtest_accept import (
    acceptance_scope,
    attach_rebalance_dates,
)
from scripts.build_alpha.backtest_accept_b1 import (
    CANDIDATES,
    _resume_source_fingerprint,
    _source_reference,
    acceptance_analysis_id,
    build_candidate_strategy,
    candidate_specs,
    load_reused_candidate,
)
from scripts.build_alpha.backtest_accept_b1_artifacts import (
    candidate_is_complete,
    write_candidate_artifacts,
)
from scripts.build_alpha.backtest_accept_b1_report import build_report


def _metrics() -> dict:
    """构造报告所需的最小完整指标。"""

    return {
        "annual_return": 0.1,
        "excess_annual_return": 0.08,
        "annual_volatility": 0.2,
        "sharpe": 0.5,
        "max_drawdown": -0.1,
        "turnover_two_way_annual": 40.0,
        "total_cost": 10_000.0,
        "total_cost_bps_of_nav": 100.0,
    }


def _capacity() -> dict:
    """构造报告所需的最小容量结果。"""

    return {
        "liquidity_limited_cells": 1,
        "liquidity_limited_value": 1000.0,
        "liquidity_unavailable_cells": 0,
        "liquidity_shortfall_shares": 100.0,
        "cash_clipped_cells": 0,
        "cash_carry_over_cells": 0,
        "pending_at_end_cells": 0,
        "allocation_converged": True,
        "carry_converged": True,
        "allocation_iterations_total": 2,
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


def test_candidate_order_groups_and_fixed_providers():
    """八候选顺序固定，原始、平滑和线性基线使用正确 provider。"""

    assert [spec.name for spec in candidate_specs("all")] == [
        "RIDGE_RAW",
        "LIGHTGBM_SINGLE_RAW",
        "LIGHTGBM_ENSEMBLE_RAW",
        "LIGHTGBM_SMOOTH_4W",
        "B1_MOM20",
        "B2_MOM_EP_VOL",
        "B3_ALL_ZSCORE",
        "B4_ALL_RANK",
    ]
    assert [spec.name for spec in candidate_specs("raw-models")] == [
        "RIDGE_RAW",
        "LIGHTGBM_SINGLE_RAW",
        "LIGHTGBM_ENSEMBLE_RAW",
    ]
    dates = ["2023-01-06"]
    raw = build_candidate_strategy(CANDIDATES[0], 30, dates)
    assert raw.signal.provider.params["smoothing_weeks"] == 1
    smooth = build_candidate_strategy(CANDIDATES[3], 30, dates)
    assert smooth.signal.provider.params["smoothing_weeks"] == 4
    linear = build_candidate_strategy(CANDIDATES[4], 30, dates)
    assert linear.signal.provider.name == "factor_linear"
    assert linear.signal.factors == ("MOM_20",)
    assert linear.signal.provider.params["factor_names"] == ["MOM_20"]
    rank = build_candidate_strategy(CANDIDATES[7], 30, dates)
    assert rank.signal.provider.name == "factor_rank"
    assert rank.signal.provider.version == "2"
    assert len(rank.signal.factors) == 47


def test_analysis_id_is_deterministic_and_source_sensitive():
    """来源或模型预测变化必须改变分析 ID。"""

    scope = acceptance_scope()
    specs = CANDIDATES[:3]
    manifests = {
        "model": {
            "model_sha256": "a",
            "prediction_sha256": "p",
            "dataset_id": "d",
        }
    }
    left = acceptance_analysis_id(
        scope, specs, manifests, "source", "", "", "code"
    )
    right = acceptance_analysis_id(
        scope, specs, manifests, "source", "", "", "code"
    )
    changed_source = acceptance_analysis_id(
        scope,
        specs,
        manifests,
        "source2",
        "",
        "",
        "code",
    )
    changed_prediction = acceptance_analysis_id(
        scope,
        specs,
        {"model": {**manifests["model"], "prediction_sha256": "p2"}},
        "source",
        "",
        "",
        "code",
    )
    assert left.startswith("p4_4b1_")
    assert left == right
    assert left != changed_source
    assert left != changed_prediction


def test_reuses_only_hash_validated_p4_4a_candidates():
    """当前 P4-4A 正式来源可读取，且保留来源文件和快照。"""

    source, digest = _source_reference()
    scope = acceptance_scope()
    scope = attach_rebalance_dates(scope, source["scope"]["rebalance_dates"])
    payload = load_reused_candidate(CANDIDATES[3], source, scope)
    assert payload["reused"] is True
    assert payload["checks"]["passed"] is True
    assert payload["source"]["analysis_id"] == source["analysis_id"]
    assert len(payload["source"]["files"]) == 7
    assert len(digest) == 64


def test_resume_source_reuses_only_complete_passed_candidates():
    """续跑来源只收录完成且哈希有效的候选，失败候选留待重跑。"""

    markers, digest = _resume_source_fingerprint(
        "p4_4b1_e9c029453c2d",
        CANDIDATES,
    )
    assert "B3_ALL_ZSCORE" in markers
    assert "RIDGE_RAW" not in markers
    assert len(digest) == 64


def test_b1_artifacts_are_complete_only_after_hash_validation(tmp_path):
    """候选完成标记覆盖全部文件，内容损坏后拒绝恢复。"""

    folder = tmp_path / "candidate"
    write_candidate_artifacts(
        folder,
        analysis_id="p4_4b1_demo",
        candidate="demo",
        metrics={"candidate": "demo"},
        strategy={"name": "demo"},
        yearly=pd.DataFrame({"year": [2023]}),
        daily=_daily(),
        orders=pd.DataFrame({"Size": [100]}),
        holdings=pd.DataFrame({"stock_code": ["SH600000"]}),
        run_manifest={"snapshot_id": "a" * 64},
    )
    assert candidate_is_complete(folder, "p4_4b1_demo", "demo")
    (folder / "orders.parquet").write_bytes(b"broken")
    assert not candidate_is_complete(folder, "p4_4b1_demo", "demo")


def test_report_contains_comparison_capacity_and_yearly_results():
    """报告包含净收益、容量、分年和检查结果。"""

    candidates = {
        "DEMO": {
            "metrics": _metrics(),
            "yearly": pd.DataFrame(
                {
                    "year": [2023],
                    "total_return": [0.1],
                    "benchmark_total_return": [0.02],
                    "excess_total_return": [0.08],
                    "max_drawdown": [-0.1],
                    "turnover_two_way_annual": [40.0],
                    "total_cost": [10_000.0],
                }
            ),
            "capacity": _capacity(),
            "checks": {"passed": True, "checks": {"allocation": True}},
            "run_manifest": {"snapshot_id": "a" * 64},
            "reused": False,
        }
    }
    scope = {
        "start": "2018-01-12",
        "end": "2023-12-29",
        "init_cash": 1_000_000.0,
        "risk_free_rate": 0.02,
        "rebalance_dates_count": 300,
        "rebalance_dates_sha256": "a" * 64,
    }
    report = build_report(
        analysis_id="p4_4b1_demo",
        scope=scope,
        candidates=candidates,
        expected_candidates=1,
        source_analysis_id="p4_4a_demo",
    )
    assert "P4-4B1" in report
    assert "DEMO" in report
    assert "容量与资金分配" in report
    assert "分年结果" in report
    assert "验收通过" in report
