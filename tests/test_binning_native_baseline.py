"""P1-06 基准工具的纯函数测试。"""

from __future__ import annotations

import numpy as np

from tools.monitor.binning_native_baseline import (
    AGGREGATE_STAGES,
    STAGES,
    _compare_micro,
    _sha256_array,
    _worker_environment,
    summarize_gate,
)


def _micro(scale: float) -> dict:
    """构造四个核的最小微基准记录。"""

    return {
        "kernels": {
            name: {"median_s": value * scale}
            for name, value in {
                "quantile_bins": 4.0,
                "returns_equal": 2.0,
                "returns_market_cap": 2.0,
                "turnover": 2.0,
                "row_spearman": 4.0,
            }.items()
        }
    }


def _stages(scale: float) -> dict[str, dict]:
    """构造全部 stage 的最小运行记录。"""

    return {
        stage: {
            "stage_elapsed_ms": 1000.0 * scale,
            "peak_rss_mb": 100.0,
            "actual_backend": "python" if scale == 1.0 else "native",
        }
        for stage in STAGES
    }


def test_binning_gate_requires_all_kernel_stage_rss_and_golden_checks():
    """任一核、stage、RSS 或金标准失败都不能放行。"""

    matches = {
        "quantile_bins": True,
        "returns_equal": True,
        "returns_market_cap": True,
        "turnover": True,
        "row_spearman": True,
    }
    golden = {
        stage: {"python": True, "native": True}
        for stage in STAGES
    }
    passed = summarize_gate(
        _micro(1.0),
        _micro(0.2),
        matches,
        _stages(1.0),
        _stages(0.4),
        _stages(0.4),
        golden,
    )
    assert passed["gate_pass"] is True
    assert passed["aggregate_stages"] == list(AGGREGATE_STAGES)
    assert passed["stage_improvement"] >= 0.20

    failed_golden = dict(
        golden,
        bins={"python": True, "native": False},
    )
    assert not summarize_gate(
        _micro(1.0),
        _micro(0.2),
        matches,
        _stages(1.0),
        _stages(0.4),
        _stages(0.4),
        failed_golden,
    )["gate_pass"]

    failed_rss = _stages(0.4)
    failed_rss["bins"] = dict(failed_rss["bins"], peak_rss_mb=121.0)
    assert not summarize_gate(
        _micro(1.0),
        _micro(0.2),
        matches,
        _stages(1.0),
        failed_rss,
        failed_rss,
        golden,
    )["rss_pass"]


def test_micro_comparison_checks_exact_and_float_arrays(tmp_path):
    """分位组和计数要求完全相等，其余数组允许 1e-10 浮点误差。"""

    python_dir = tmp_path / "python"
    native_dir = tmp_path / "native"
    python_dir.mkdir()
    native_dir.mkdir()
    names = (
        "returns_equal",
        "pools_equal",
        "returns_market_cap",
        "pools_market_cap",
        "turnover_per_bin",
        "turnover_overall",
        "row_spearman",
    )
    for name in names:
        values = np.array([1.0, np.nan, 2.0])
        np.save(python_dir / f"{name}.npy", values)
        np.save(native_dir / f"{name}.npy", values + 1e-12)
    for name in ("quantile_bins", "row_spearman_count"):
        values = np.array([0, -1, 2], dtype=np.int64)
        np.save(python_dir / f"{name}.npy", values)
        np.save(native_dir / f"{name}.npy", values)

    assert all(_compare_micro(python_dir, native_dir).values())
    changed = np.array([0, -1, 3], dtype=np.int64)
    np.save(native_dir / "quantile_bins.npy", changed)
    assert _compare_micro(python_dir, native_dir)["quantile_bins"] is False


def test_binning_worker_environment_is_single_threaded():
    """真实基准固定后端和单线程。"""

    environment = _worker_environment("native")
    assert environment["QUANT_KERNEL_BACKEND"] == "native"
    assert environment["QUANT_NATIVE_THREADS"] == "1"


def test_array_digest_uses_raw_float_bits():
    """输入哈希区分 NaN、正负零和顺序。"""

    values = np.array([0.0, -0.0, np.nan])
    assert len(_sha256_array(values)) == 64
    assert _sha256_array(values) != _sha256_array(values[::-1])
