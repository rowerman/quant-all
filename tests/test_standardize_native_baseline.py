"""逐日标准化 C++ 基准工具的纯函数测试。"""

from __future__ import annotations

import numpy as np

from tools.monitor.standardize_native_baseline import (
    KERNELS,
    _compare_micro,
    _median_record,
    _sha256_array,
    _worker_environment,
)


def test_standardize_micro_comparison_detects_each_kernel(tmp_path):
    """三个核分别比较，任一数组漂移都会使对应结果为 False。"""

    python_dir = tmp_path / "python"
    native_dir = tmp_path / "native"
    python_dir.mkdir()
    native_dir.mkdir()
    for index, name in enumerate(KERNELS):
        values = np.array([0.0, 1.0, np.nan, np.inf])
        np.save(python_dir / f"{name}.npy", values)
        changed = values.copy()
        if index == 1:
            changed[0] = 2.0
        np.save(native_dir / f"{name}.npy", changed)

    actual = _compare_micro(python_dir, native_dir)

    assert actual == {
        "daily_winsorize_mad": True,
        "daily_zscore": False,
        "daily_winsorize_zscore": True,
    }


def test_stage_median_record_uses_median_time_and_peak_rss():
    """重复 stage 以中位耗时汇总，并保留最保守的 RSS 峰值。"""

    records = [
        {
            "actual_backend": "native",
            "elapsed_s": 3.0,
            "stage_elapsed_ms": 3000.0,
            "peak_rss_mb": 120.0,
        },
        {
            "actual_backend": "native",
            "elapsed_s": 1.0,
            "stage_elapsed_ms": 1000.0,
            "peak_rss_mb": 140.0,
        },
        {
            "actual_backend": "native",
            "elapsed_s": 2.0,
            "stage_elapsed_ms": 2000.0,
            "peak_rss_mb": 130.0,
        },
    ]

    actual = _median_record(records)

    assert actual["actual_backend"] == "native"
    assert actual["elapsed_s"] == 2.0
    assert actual["stage_elapsed_ms"] == 2000.0
    assert actual["peak_rss_mb"] == 140.0
    assert actual["runs"] == records


def test_standardize_worker_environment_is_single_threaded():
    """基准子进程固定后端并避免线程预算叠加。"""

    environment = _worker_environment("native")

    assert environment["QUANT_KERNEL_BACKEND"] == "native"
    assert environment["QUANT_NATIVE_THREADS"] == "1"


def test_array_digest_includes_raw_float_bits():
    """输入哈希区分有限值、NaN 和正负零的原始字节。"""

    values = np.array([0.0, -0.0, np.nan], dtype=np.float64)

    assert len(_sha256_array(values)) == 64
    assert _sha256_array(values) != _sha256_array(values.copy()[::-1])
