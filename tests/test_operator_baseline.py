"""算子优化基准的契约、序列化和确定性测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.monitor.baseline_cases import run_case
from tools.monitor.baseline_contract import (
    CASE_DEFINITIONS,
    CASE_ORDER,
    GoldenMismatch,
    atomic_write_json,
    canonical_json,
    compare_golden,
    file_record,
    load_golden,
    strip_volatile,
    update_manifest,
)
from tools.monitor.operator_baseline import case_paths


def test_fixed_cases_lock_dates_and_parameters():
    """后续优化不能在不更新基准契约的情况下悄悄改变案例口径。"""

    assert CASE_ORDER == (
        "evaluation",
        "correlation",
        "correlation_target",
        "history",
        "factor_build",
        "mining",
        "edge",
    )
    assert CASE_DEFINITIONS["evaluation"]["start"] == "2023-10-15"
    assert CASE_DEFINITIONS["evaluation"]["end"] == "2026-10-15"
    assert CASE_DEFINITIONS["correlation_target"]["formal"] is False
    assert CASE_DEFINITIONS["correlation_target"]["stages"] == ("correlation",)
    assert CASE_DEFINITIONS["history"]["start"] == "2000-01-04"
    assert CASE_DEFINITIONS["factor_build"] == {
        "families": ("momentum",),
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert CASE_DEFINITIONS["mining"]["expression"] == "mean(close,5)"
    assert CASE_DEFINITIONS["mining"]["run_full"] is False


def test_nonfinite_values_are_distinguished_and_timings_removed():
    """稳定金标准要区分 NaN/正负无穷，并移除易变计时。"""

    payload = {
        "nan": float("nan"),
        "positive": float("inf"),
        "negative": float("-inf"),
        "nested": {"elapsed_ms": 12.0, "value": 1.0},
    }
    stable = strip_volatile(payload)
    assert stable == {
        "nan": "__NaN__",
        "positive": "__+Inf__",
        "negative": "__-Inf__",
        "nested": {"value": 1.0},
    }
    assert canonical_json(stable) == (
        '{"nan":"__NaN__","negative":"__-Inf__",'
        '"nested":{"value":1.0},"positive":"__+Inf__"}'
    )


def test_compare_golden_tolerance_and_exact_path():
    """普通浮点允许约定容差，排名类路径可要求完全相等。"""

    compare_golden({"value": 1.0}, {"value": 1.0 + 1e-12})
    with pytest.raises(GoldenMismatch):
        compare_golden(
            {"rank": 0.5},
            {"rank": 0.500000000001},
            exact_paths=("$.rank",),
        )
    with pytest.raises(GoldenMismatch):
        compare_golden({"items": [1, 2]}, {"items": [1]})


def test_edge_case_is_deterministic():
    """边界案例重复执行必须生成完全相同的稳定结果。"""

    first, first_timing = run_case("edge", {})
    second, second_timing = run_case("edge", {})
    assert first_timing.keys() == second_timing.keys()
    assert canonical_json(first) == canonical_json(second)
    assert all(first["checks"].values())


def test_file_record_and_manifest_are_stable(tmp_path: Path):
    """manifest 记录证据文件哈希，并排除自身和报告。"""

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    atomic_write_json(first, {"b": 2, "a": 1})
    atomic_write_json(second, {"value": 3})
    (tmp_path / "report.md").write_text("报告", encoding="utf-8")

    assert first.read_text(encoding="utf-8") == '{"a":1,"b":2}'
    assert load_golden(first) == {"a": 1, "b": 2}
    record = file_record(first)
    assert record["bytes"] == first.stat().st_size
    assert len(record["sha256"]) == 64

    manifest = update_manifest(tmp_path)
    assert {item["path"].split("/")[-1] for item in manifest["files"]} == {
        "first.json",
        "second.json",
    }
    assert len(manifest["tree_sha256"]) == 64


def test_case_paths_are_fixed_by_case_name(tmp_path: Path):
    """每个案例的稳定结果、计时和失败证据使用独立文件。"""

    golden, timing, error = case_paths(tmp_path, "edge")
    assert golden == tmp_path / "edge.golden.json"
    assert timing == tmp_path / "edge.timing.json"
    assert error == tmp_path / "edge.error.json"
