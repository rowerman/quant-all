"""P5-3 完整账本验收的构造、产物和报告测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.build_alpha.lifecycle_accept import (
    CANDIDATE,
    SCHEMA_VERSION,
    _analysis_id,
    _strategy,
)
from scripts.build_alpha.lifecycle_accept_artifacts import (
    candidate_is_complete,
    comparison_frame,
    write_candidate,
)
from scripts.build_alpha.lifecycle_accept_report import build_report
from common.paths import MODEL_DATA_DIR


def test_lifecycle_accept_strategy_uses_v3_and_explicit_dates():
    """正式策略固定使用 v3、主模型、Ridge 回滚和显式调仓日。"""

    dates = ["2023-12-15", "2023-12-22", "2023-12-29"]
    strategy = _strategy("p5_3_test", dates)
    provider = strategy.signal.provider
    assert provider.name == "ml_rank"
    assert provider.version == "3"
    assert provider.params["lifecycle_id"] == "p5_3_test"
    assert provider.params["rebalance_dates"] == dates
    custom = _strategy(
        "p5_3_test",
        dates,
        model_id="new_model",
        fallback_model_id="fallback",
        smoothing_weeks=1,
        top_n=20,
    )
    assert custom.signal.provider.params["model_id"] == "new_model"
    assert custom.signal.provider.params["fallback_model_id"] == "fallback"
    assert custom.signal.provider.params["smoothing_weeks"] == 1
    assert custom.portfolio.top_n == 20


def test_lifecycle_accept_analysis_id_is_deterministic():
    """相同生命周期、范围和代码产生相同验收 ID。"""

    lifecycle = {
        "lifecycle_id": "p5_3_test",
        "files": {"timeseries.csv": "a" * 64},
        "sources": {"model_id": "model"},
    }
    scope = {"start": "2023-12-15", "end": "2023-12-29"}
    first = _analysis_id(lifecycle, scope, "b" * 64)
    second = _analysis_id(lifecycle, scope, "b" * 64)
    changed = _analysis_id(lifecycle, scope, "c" * 64)
    assert first == second
    assert first != changed
    assert first.startswith("p5_3_")


def test_candidate_artifacts_are_written_and_hash_checked(tmp_path: Path):
    """候选产物完整后可复用，任一文件变化后不再视为完整。"""

    strategy = _strategy("p5_3_test", ["2023-12-29"])
    dates = pd.DatetimeIndex(["2023-12-29"])
    result = SimpleNamespace(
        daily=pd.DataFrame({"return": [0.01]}, index=dates),
        orders=pd.DataFrame({"side": ["buy"]}),
        holdings=pd.DataFrame({"stock_code": ["A"], "weight": [1.0]}),
        metadata={"snapshot_id": "snapshot", "signal": {"rebalance_days": 1}},
    )
    folder = tmp_path / "acceptance" / CANDIDATE
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "scope": {"rebalance_dates": ["2023-12-29"]},
    }
    write_candidate(
        folder,
        analysis_id="p5_3_test",
        strategy=strategy,
        result=result,
        metrics=metrics,
        yearly=pd.DataFrame({"year": [2023]}),
    )
    assert candidate_is_complete(folder, "p5_3_test")
    with (folder / "metrics.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert not candidate_is_complete(folder, "p5_3_test")


def test_lifecycle_comparison_and_report_include_economic_result():
    """对照和报告包含未启用与启用生命周期的正式指标。"""

    root = MODEL_DATA_DIR
    lifecycle = {
        "sources": {
            "acceptance_id": "p4_4a_23c7e101aee9",
            "acceptance_candidate": "LIGHTGBM_SMOOTH_4W",
        }
    }
    metrics = {
        "metrics": {
            "annual_return": 0.2,
            "excess_annual_return": 0.1,
            "annual_volatility": 0.2,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "turnover_two_way_annual": 20.0,
            "total_cost": 10000.0,
        },
        "scope": {"start": "2018-01-12", "end": "2023-12-29"},
        "checks": {"actual_signal_days": 300},
    }
    comparison = comparison_frame(metrics, lifecycle, root)
    assert set(comparison["lifecycle_enabled"]) == {True, False}
    manifest = {
        "analysis_id": "p5_3_test",
        "status": "pass_p5_3",
        "lifecycle": {
            "lifecycle_id": "p5_3_lifecycle",
            "summary": {
                "state_counts": {"normal": 300},
                "event_counts": {"pause": 1},
            },
        },
    }
    report = build_report(manifest, metrics, comparison)
    assert "P5-3 模型生命周期完整账本验收" in report
    assert "pass_p5_3" in report
    assert "2024-2026 最终测试集未读取" in report
