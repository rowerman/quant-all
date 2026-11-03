"""批量核基准工具的门禁和计时辅助测试。"""

from __future__ import annotations

from tools.monitor.kernel_baseline import (
    _median,
    _timed_runs,
    finalize_gate,
)


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


def _micro_result(*, rank_correct=True, rank_speedup=2.0):
    """构造逐日排名门禁测试输入；参数为排名正确性和加速比。"""

    return {
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
            "correct": True,
            "diagnostic_pass": False,
            "speedup": 0.2,
            "python_median_s": 1.0,
            "numba_median_s": 5.0,
            "numba_cold_s": 10.0,
        },
        "rank": {
            "correct": rank_correct,
            "gate_pass": rank_speedup >= 1.5,
            "speedup": rank_speedup,
            "python_median_s": 1.0,
            "numba_median_s": 1.0 / rank_speedup,
            "numba_cold_s": 2.0,
        },
    }


def test_gate_passes_on_rank_even_when_spearman_diagnostic_fails(tmp_path):
    """逐日排名达到门槛时放行，Spearman 失败只保留诊断状态。"""

    result = finalize_gate(tmp_path, _micro_result(rank_speedup=1.5))

    assert result["gate"] == "daily_rank"
    assert result["gate_pass"] is True
    assert result["micro_pass"] is True
    assert result["spearman_diagnostic_pass"] is False
    assert result["stage_run"] is False
    assert result["reason"] is None
    assert result["next_task"] == "P1-02"
    assert (tmp_path / "gate.json").is_file()


def test_gate_fails_when_rank_is_below_threshold(tmp_path):
    """逐日排名低于 1.5x 时 P1-01 不通过。"""

    result = finalize_gate(tmp_path, _micro_result(rank_speedup=1.49))

    assert result["gate_pass"] is False
    assert result["reason"] == "daily_rank_microbench_below_1.5x"
    assert result["next_task"] == "P1-02 保持待开始"


def test_gate_fails_when_rank_result_is_incorrect(tmp_path):
    """排名结果不正确时即使速度达标也不放行。"""

    result = finalize_gate(tmp_path, _micro_result(rank_correct=False))

    assert result["gate_pass"] is False
    assert result["reason"] == "daily_rank_incorrect"
