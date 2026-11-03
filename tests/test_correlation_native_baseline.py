"""相关性 C++ 接入门禁的确定性测试。"""

import numpy as np
import pytest

from tools.monitor.correlation_native_baseline import _array_digest, summarize_gate


def _stage(actual: str, elapsed: float, rss: float = 100.0) -> dict:
    """构造一个 stage worker 摘要。"""

    return {
        "actual_backend": actual,
        "requested_backend": actual,
        "fallback_reason": None,
        "stage_elapsed_ms": elapsed,
        "peak_rss_mb": rss,
    }


def _kernel(full_s: float, target_s: float, digest: str = "same") -> dict:
    """构造一个核 worker 摘要。"""

    return {
        "results": {
            "full": {
                "median_s": full_s,
                "correlation_sha256": digest,
                "count_sha256": digest,
            },
            "target": {
                "median_s": target_s,
                "correlation_sha256": digest,
                "count_sha256": digest,
            },
        }
    }


def _stages(native_full: float = 800.0, native_target: float = 90.0) -> dict:
    """构造满足正确性和 auto 选择的基本 stage 结果。"""

    return {
        "full": {
            "python": _stage("python", 1000.0),
            "native": _stage("native", native_full),
            "auto": _stage("native", native_full),
        },
        "target": {
            "python": _stage("python", 100.0),
            "native": _stage("native", native_target),
            "auto": _stage("native", native_target),
        },
    }


def test_gate_passes_only_when_all_cpp_replacement_conditions_hold():
    result = summarize_gate(
        _stages(),
        {"python": _kernel(0.15, 0.03), "native": _kernel(0.05, 0.01)},
        python_matches=True,
        payloads_match=True,
        kernel_results_match=True,
    )

    assert result["gate_pass"] is True
    assert result["full_stage_improvement"] == pytest.approx(0.2)
    assert result["auto_full_stage_improvement"] == pytest.approx(0.2)
    assert result["target_stage_speedup"] > 1
    assert result["auto_target_stage_speedup"] > 1
    assert result["kernel_speedup"] == pytest.approx(3)
    assert result["cpp_modules_replaced"] == 1


def test_gate_stops_for_stage_regression_missing_auto_and_mismatch():
    stages = _stages(native_full=810.0, native_target=101.0)
    stages["full"]["auto"] = _stage("python", 1000.0)
    result = summarize_gate(
        stages,
        {"python": _kernel(0.15, 0.03), "native": _kernel(0.11, 0.01, "other")},
        python_matches=True,
        payloads_match=True,
        kernel_results_match=False,
    )

    assert result["gate_pass"] is False
    assert result["cpp_modules_replaced"] == 0
    assert "auto_did_not_select_native" in result["reason"]
    assert "full_stage_improvement_below_20pct" in result["reason"]
    assert "auto_full_stage_improvement_below_20pct" in result["reason"]
    assert "target_stage_regression" in result["reason"]
    assert "kernel_speedup_below_1.5x" in result["reason"]
    assert "kernel_result_mismatch" in result["reason"]


def test_array_digest_normalizes_nan_sign_bits():
    positive = np.array([[1.0, np.nan], [2.0, 3.0]])
    negative = positive.copy()
    negative[0, 1] = -np.nan

    assert _array_digest(positive) == _array_digest(negative)
