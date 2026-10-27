"""逐日 OLS 基准工具的纯函数测试。"""

from __future__ import annotations

from tools.monitor.ols_native_baseline import (
    PAYLOAD_TOLERANCE,
    _median_record,
    _worker_environment,
)


def test_stage_median_uses_median_time_and_peak_rss():
    """多个独立运行按耗时中位数和最保守 RSS 汇总。"""

    records = [
        {
            "actual_backend": "native",
            "elapsed_s": 10.0,
            "stage_elapsed_ms": {"exposure": 9000.0, "incremental_ic": 1000.0},
            "combined_stage_elapsed_ms": 10000.0,
            "peak_rss_mb": 200.0,
        },
        {
            "actual_backend": "native",
            "elapsed_s": 12.0,
            "stage_elapsed_ms": {"exposure": 11000.0, "incremental_ic": 1000.0},
            "combined_stage_elapsed_ms": 12000.0,
            "peak_rss_mb": 220.0,
        },
        {
            "actual_backend": "native",
            "elapsed_s": 11.0,
            "stage_elapsed_ms": {"exposure": 10000.0, "incremental_ic": 1000.0},
            "combined_stage_elapsed_ms": 11000.0,
            "peak_rss_mb": 210.0,
        },
    ]

    result = _median_record(records)

    assert result["combined_stage_elapsed_ms"] == 11000.0
    assert result["stage_elapsed_ms"]["exposure"] == 10000.0
    assert result["peak_rss_mb"] == 220.0


def test_worker_environment_forces_single_thread():
    """基准 worker 固定后端和单线程。"""

    environment = _worker_environment("native")

    assert environment["QUANT_KERNEL_BACKEND"] == "native"
    assert environment["QUANT_NATIVE_THREADS"] == "1"


def test_ols_payload_tolerance_is_user_approved_threshold():
    """OLS 业务对照使用用户确认的 1e-5 阈值。"""

    assert PAYLOAD_TOLERANCE == 1e-5
