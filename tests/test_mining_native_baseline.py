"""P2-02 挖掘原生接入门禁的确定性测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.monitor.mining_native_baseline import summarize_gate


def _worker(
    *,
    backend: str,
    total: float,
    technical: float,
    correlation: float,
    rss: float = 100.0,
    rankic: bool = False,
) -> dict:
    """构造满足正确性条件的最小 worker 结果。"""

    return {
        "actual_backend": backend,
        "total_elapsed_s": total,
        "technical_elapsed_s": technical,
        "correlation_median_s": correlation,
        "peak_rss_mb": rss,
        "stable_evaluations": {"same": 1.0},
        "rankic": (
            {
                "sql_median_s": 0.65,
                "native_median_s": 0.90,
                "max_rank_ic_difference": 1e-14,
                "counts_equal": True,
                "retain_sql": True,
            }
            if rankic
            else None
        ),
    }


def _workers() -> dict[str, dict]:
    """构造通过全部门槛的三后端结果。"""

    return {
        "python": _worker(
            backend="python",
            total=40.0,
            technical=20.0,
            correlation=10.0,
        ),
        "native": _worker(
            backend="native",
            total=32.0,
            technical=17.0,
            correlation=7.0,
            rankic=True,
        ),
        "auto": _worker(
            backend="native",
            total=32.5,
            technical=17.2,
            correlation=7.1,
        ),
        "python_uncached": _worker(
            backend="python",
            total=45.0,
            technical=22.5,
            correlation=10.0,
        ),
    }


def test_mining_gate_passes_with_verified_improvement():
    """正确、自动选 native、达到收益和内存门槛时放行。"""

    result = summarize_gate(_workers())

    assert result["gate_pass"] is True
    assert result["correlation_speedup"] == pytest.approx(10.0 / 7.0)
    assert result["technical_improvement"] == pytest.approx(
        1.0 - 17.2 / 22.5
    )
    assert result["native_vs_cached_python_technical_speedup"] == (
        pytest.approx(20.0 / 17.0)
    )
    assert result["rss_ratios"] == {
        "native": 1.0,
        "auto": 1.0,
    }
    assert result["cpp_modules_replaced"] == 1


def test_mining_gate_rejects_payload_speed_rss_and_rankic_failures():
    """任一正确性、性能、后端或 RankIC 反例都必须关闭门禁。"""

    workers = _workers()
    workers["native"]["stable_evaluations"] = {"same": 2.0}
    workers["native"]["correlation_median_s"] = 9.0
    workers["native"]["technical_elapsed_s"] = 19.0
    workers["native"]["peak_rss_mb"] = 121.0
    workers["native"]["rankic"]["max_rank_ic_difference"] = 1e-6
    workers["auto"] = deepcopy(workers["auto"])
    workers["auto"]["actual_backend"] = "python"
    workers["python_uncached"]["technical_elapsed_s"] = 18.5

    result = summarize_gate(workers)

    assert result["gate_pass"] is False
    assert result["cpp_modules_replaced"] == 0
    assert {
        "stable_payload_mismatch",
        "auto_did_not_select_native",
        "correlation_speedup_below_1.2x",
        "technical_improvement_below_10pct",
        "rss_above_1.2x",
        "rankic_result_mismatch",
    } <= set(result["reason"])
