"""逐日 C++ 基准工具的比较与命令构造测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.monitor.daily_native_baseline import (
    _compare_micro,
    _stage_speedup,
    _subprocess_command,
    _worker_environment,
)


def _write_micro(root: Path, *, offset: float) -> None:
    """写入一套用于比较的微基准输出。"""

    root.mkdir(parents=True)
    np.save(root / "rank.npy", np.array([0.5, 1.0, np.nan]))
    np.save(root / "correlation.npy", np.array([0.25 + offset, np.nan]))
    np.save(root / "count.npy", np.array([5, 0], dtype=np.int64))


def test_compare_micro_accepts_tolerance_and_exact_counts(tmp_path):
    """Pearson 使用容差，排名和计数要求完全一致。"""

    python_dir = tmp_path / "python"
    native_dir = tmp_path / "native"
    _write_micro(python_dir, offset=0.0)
    _write_micro(native_dir, offset=1e-12)

    assert _compare_micro(python_dir, native_dir) == (True, True)


def test_compare_micro_rejects_rank_difference(tmp_path):
    """排名不是浮点近似量，任何差异都不能通过。"""

    python_dir = tmp_path / "python"
    native_dir = tmp_path / "native"
    _write_micro(python_dir, offset=0.0)
    _write_micro(native_dir, offset=0.0)
    np.save(native_dir / "rank.npy", np.array([0.5, 0.75, np.nan]))

    assert _compare_micro(python_dir, native_dir) == (False, True)


def test_stage_speedup_and_subprocess_contract(tmp_path):
    """stage 加速比按自报耗时计算，worker 命令携带固定后端输出。"""

    speedup = _stage_speedup(
        {"stage_elapsed_ms": 200.0},
        {"stage_elapsed_ms": 50.0},
    )
    command = _subprocess_command(
        backend="native",
        output_dir=tmp_path,
        stage="decay",
        context_path=tmp_path / "context.pkl",
    )

    assert speedup == 4.0
    assert command[1:3] == ["-m", "tools.monitor.daily_native_baseline"]
    assert "--worker-backend" in command
    assert "--worker-stage" in command
    assert "--worker-context" in command
    assert _worker_environment("native")["QUANT_NATIVE_THREADS"] == "1"
