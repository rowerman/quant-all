"""C++ Spearman 微基准门禁测试。"""

from __future__ import annotations

import pytest

from tools.monitor.native_kernel_baseline import summarize_gate


def _worker(median_s: float, peak_rss_mb: float) -> dict[str, float]:
    """构造最小 worker 结果。"""

    return {"median_s": median_s, "peak_rss_mb": peak_rss_mb}


def test_native_gate_passes_only_with_correct_fast_and_bounded_rss():
    """全部门槛满足时放行，任一失败时返回稳定原因。"""

    passed = summarize_gate(
        _worker(0.15, 100.0),
        _worker(0.05, 110.0),
        correct=True,
        counts_exact=True,
    )
    assert passed["gate_pass"] is True
    assert passed["reason"] is None
    assert passed["speedup"] == pytest.approx(3.0)
    assert passed["rss_ratio"] == pytest.approx(1.1)

    cases = [
        (False, True, 0.05, 100.0, "incorrect_correlation"),
        (True, False, 0.05, 100.0, "incorrect_count"),
        (True, True, 0.11, 100.0, "speedup_below_1.5x"),
        (True, True, 0.05, 121.0, "rss_above_1.2x"),
    ]
    for correct, counts_exact, native_s, native_rss, reason in cases:
        result = summarize_gate(
            _worker(0.15, 100.0),
            _worker(native_s, native_rss),
            correct=correct,
            counts_exact=counts_exact,
        )
        assert result["gate_pass"] is False
        assert result["reason"] == reason
