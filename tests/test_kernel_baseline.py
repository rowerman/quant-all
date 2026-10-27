"""批量核基准工具的门禁和计时辅助测试。"""

from __future__ import annotations

import numpy as np

from tools.monitor import kernel_baseline
from tools.monitor.kernel_baseline import (
    _median,
    _numba_matrix,
    _timed_runs,
    finalize_gate,
)
from tools.monitor.baseline_contract import load_golden


def test_median_uses_hot_run_middle_value():
    """热运行汇总使用五次中位数。"""

    assert _median([5.0, 1.0, 3.0, 2.0, 4.0]) == 3.0


def test_timed_runs_returns_each_observation_and_last_value():
    """计时工具保留逐次证据并返回最后一次计算结果。"""

    counter = iter(range(3))
    timings, result = _timed_runs(lambda: next(counter), repeats=3)

    assert len(timings) == 3
    assert all(value >= 0 for value in timings)
    assert result == 2


def test_numba_matrix_wrapper_preserves_single_group_shape():
    """基准替换入口把单日矩阵转换为统一三维核结果的第一组。"""

    values = np.array(
        [
            [1., 2., 3.],
            [2., 1., 4.],
            [3., 4., 2.],
            [4., 3., 1.],
            [5., 6., 7.],
        ]
    )
    result = _numba_matrix()(values)
    assert result.shape == (3, 3)
    assert np.isfinite(result).all()


def test_gate_finalization_stops_after_microbench_failure(tmp_path):
    """Spearman 微基准失败时不运行真实 stage，并关闭后续 C++。"""

    micro = {
        "inputs": {
            "micro_date": "2026-08-03",
            "spearman_rows": 10,
            "spearman_columns": 3,
            "rank_start": "2026-08-01",
            "rank_end": "2026-08-31",
            "rank_groups": 2,
            "rank_rows": 10,
            "threads": 1,
        },
        "spearman": {
            "gate_pass": False,
            "speedup": 0.2,
            "python_median_s": 1.0,
            "numba_median_s": 5.0,
            "numba_cold_s": 10.0,
        },
        "rank": {
            "speedup": 2.0,
            "python_median_s": 1.0,
            "numba_median_s": .5,
            "numba_cold_s": 2.0,
        },
    }

    result = finalize_gate(tmp_path, micro, None)

    assert result["gate_pass"] is False
    assert result["stage_run"] is False
    assert result["reason"] == "spearman_microbench_below_1.5x"
    assert result["next_task"] == "P1-02 已停止"
    assert (tmp_path / "gate.json").is_file()


def test_gate_finalization_requires_stage_payload_and_rss(tmp_path):
    """微基准通过后仍要求真实 stage、正确性和内存同时达标。"""

    micro = {
        "inputs": {
            "micro_date": "2026-08-03",
            "spearman_rows": 10,
            "spearman_columns": 3,
            "rank_start": "2026-08-01",
            "rank_end": "2026-08-31",
            "rank_groups": 2,
            "rank_rows": 10,
            "threads": 1,
        },
        "spearman": {
            "gate_pass": True,
            "speedup": 1.6,
            "python_median_s": 5.0,
            "numba_median_s": 3.125,
            "numba_cold_s": 10.0,
        },
        "rank": {
            "speedup": 1.0,
            "python_median_s": 1.0,
            "numba_median_s": 1.0,
            "numba_cold_s": 2.0,
        },
    }
    stage = {
        "stage_gate_pass": True,
        "rss_gate_pass": False,
        "payload_match": True,
        "improvement": .30,
        "rss_ratio": 1.3,
    }

    result = finalize_gate(tmp_path, micro, stage)

    assert result["gate_pass"] is False
    assert result["stage_run"] is True
    assert result["reason"] == "spearman_stage_or_memory_gate_failed"


def test_stage_result_uses_v107_payload_and_baseline(monkeypatch, tmp_path):
    """stage 工具对照 V107 金标准和固定耗时，不改生产调用路径。"""

    expected = load_golden(kernel_baseline.V107_CORRELATION)["golden"]["events"]
    payload = next(
        event["payload"] for event in expected if event["stage"] == "correlation"
    )
    monkeypatch.setattr(
        kernel_baseline,
        "_run_correlation_stage",
        lambda: ([{"stage": "correlation", "payload": payload}], 100_000.0),
    )

    result = kernel_baseline.run_stage(tmp_path)

    assert result["payload_match"] is True
    assert result["stage_gate_pass"] is True
    assert result["rss_gate_pass"] is True
    assert (tmp_path / "stage.json").is_file()
