"""验证批量核契约、Numba 原型和逐日排名放行门禁。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd

from calculate.correlation import available_factors, daily_spearman_matrix
from calculate.daily import MIN_CROSS_SECTION, rank_pct_by_day
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from kernels import get_backend
from kernels.backend import group_codes
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    load_golden,
    prepare_common,
    project_relative,
    update_manifest,
)

_BACKEND_ROOT = PROJECT_ROOT / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service

SCHEMA_VERSION = "kernel_baseline_v2"
EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "AiRecord/benchmarks/V112/p1-01-20261027-recheck"
)
V105_INPUTS = (
    PROJECT_ROOT
    / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
)
CORRELATION_QUERY = {
    "factor": "MOM_20",
    "start": "2023-10-15",
    "end": "2026-10-15",
    "universe": "all_a",
    "horizon": "5",
    "price": "close",
    "quantiles": "10",
    "winsorize": "true",
    "stages": "correlation",
    "metric_options": '{"correlation":{"top_n":10}}',
}
MICRO_DATE = "2026-08-03"
RANK_START = "2026-08-01"
RANK_END = "2026-08-31"
REPEATS = 5
MIN_KERNEL_SPEEDUP = 1.5


def _timed_runs(
    function: Callable[[], Any],
    repeats: int = REPEATS,
) -> tuple[list[float], Any]:
    """重复执行并返回逐次耗时和最后一次结果。"""

    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(round(time.perf_counter() - started, 6))
    return timings, result


def _median(values: list[float]) -> float:
    """返回有限观测的中位数；参数 values 为非空耗时列表。"""

    return float(np.median(np.asarray(values, dtype=np.float64)))


def _load_spearman_matrix(context) -> np.ndarray:
    """读取固定交易日的池内正式因子矩阵。"""

    panel = context.panel
    target = pd.Timestamp(MICRO_DATE)
    pool = panel.loc[
        (panel["trade_date"] == target) & panel["in_universe"],
        ["stock_code", "trade_date"],
    ].copy()
    if pool.empty:
        raise RuntimeError(f"固定微基准日期没有池内样本：{MICRO_DATE}")
    grouped = available_factors(context.factor_dir)
    names = [name for family in grouped.values() for name in family]
    with duckdb.connect() as conn:
        conn.register("pool_keys", pool)
        for family, family_names in grouped.items():
            path = str(
                Path(context.factor_dir) / family / "year=*" / "part.parquet"
            )
            selections = ",".join(
                'f."' + name.replace('"', '""') + '"' for name in family_names
            )
            frame = conn.execute(
                f"SELECT f.stock_code, f.trade_date, {selections} "
                "FROM read_parquet(?, hive_partitioning=1, union_by_name=1) f "
                "JOIN pool_keys k USING(stock_code, trade_date)",
                [path],
            ).df()
            pool = pool.merge(
                frame,
                on=["stock_code", "trade_date"],
                how="left",
                validate="one_to_one",
            )
        conn.unregister("pool_keys")
    pool = pool.sort_values(["trade_date", "stock_code"], kind="stable")
    return np.ascontiguousarray(pool[names].to_numpy(dtype=np.float64))


def _load_rank_values(context) -> tuple[np.ndarray, np.ndarray]:
    """读取固定月份的池内目标因子，返回按日期排序的值和边界。"""

    panel = context.panel
    pool = panel.loc[
        panel["in_universe"]
        & (panel["trade_date"] >= pd.Timestamp(RANK_START))
        & (panel["trade_date"] <= pd.Timestamp(RANK_END)),
        ["stock_code", "trade_date"],
    ].copy()
    if pool.empty:
        raise RuntimeError("固定排名微基准区间没有池内样本")
    target_column = '"' + context.params.factor.replace('"', '""') + '"'
    with duckdb.connect() as conn:
        conn.register("pool_keys", pool)
        values = conn.execute(
            "SELECT f.stock_code, f.trade_date, "
            + target_column
            + " FROM read_parquet(?, hive_partitioning=1) f "
            "JOIN pool_keys k USING(stock_code, trade_date)",
            [context.spec.parquet_glob],
        ).df()
        conn.unregister("pool_keys")
    frame = pool.merge(
        values,
        on=["stock_code", "trade_date"],
        how="left",
        validate="one_to_one",
    ).sort_values(["trade_date", "stock_code"], kind="stable")
    codes, _ = pd.factorize(frame["trade_date"], sort=True)
    sizes = np.bincount(codes).astype(np.int64)
    offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(sizes))
    ).astype(np.int64)
    return (
        np.ascontiguousarray(frame[context.params.factor].to_numpy(dtype=np.float64)),
        offsets,
    )


def run_micro(output: Path) -> dict[str, Any]:
    """测量真实截面上的 Python 与 Numba 排名及 Spearman。"""

    spec = service.prepare(CORRELATION_QUERY)
    context = build_panel(spec.params)
    matrix_values = _load_spearman_matrix(context)
    rank_values, rank_offsets = _load_rank_values(context)
    matrix_offsets = np.array([0, matrix_values.shape[0]], dtype=np.int64)
    rank_codes = group_codes(rank_offsets)

    python_backend = get_backend("python").backend
    numba_backend = get_backend("numba").backend
    spearman_python_started = time.perf_counter()
    spearman_baseline = daily_spearman_matrix(matrix_values)
    spearman_python_baseline = round(
        time.perf_counter() - spearman_python_started,
        6,
    )
    rank_python_started = time.perf_counter()
    rank_baseline = rank_pct_by_day(rank_codes, rank_values)
    rank_python_baseline = round(time.perf_counter() - rank_python_started, 6)

    spearman_started = time.perf_counter()
    spearman_numba, spearman_count = numba_backend.pairwise_spearman(
        matrix_values,
        matrix_offsets,
        min_n=MIN_CROSS_SECTION,
    )
    spearman_cold = round(time.perf_counter() - spearman_started, 6)
    rank_started = time.perf_counter()
    rank_numba = numba_backend.daily_rank_pct(rank_values, rank_offsets)
    rank_cold = round(time.perf_counter() - rank_started, 6)

    np.testing.assert_allclose(
        spearman_numba[0],
        spearman_baseline,
        atol=1e-10,
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(rank_numba, rank_baseline)
    expected_spearman_count = python_backend.pairwise_spearman(
        matrix_values,
        matrix_offsets,
        min_n=MIN_CROSS_SECTION,
    )[1]
    np.testing.assert_array_equal(spearman_count, expected_spearman_count)

    spearman_python_runs, _ = _timed_runs(
        lambda: daily_spearman_matrix(matrix_values)
    )
    spearman_numba_runs, _ = _timed_runs(
        lambda: numba_backend.pairwise_spearman(
            matrix_values,
            matrix_offsets,
            min_n=MIN_CROSS_SECTION,
        )
    )
    rank_python_runs, _ = _timed_runs(
        lambda: rank_pct_by_day(rank_codes, rank_values)
    )
    rank_numba_runs, _ = _timed_runs(
        lambda: numba_backend.daily_rank_pct(rank_values, rank_offsets)
    )
    spearman_python_median = _median(spearman_python_runs)
    spearman_numba_median = _median(spearman_numba_runs)
    rank_python_median = _median(rank_python_runs)
    rank_numba_median = _median(rank_numba_runs)
    spearman_speedup = spearman_python_median / spearman_numba_median
    rank_speedup = rank_python_median / rank_numba_median
    result = {
        "schema_version": SCHEMA_VERSION,
        "case": "micro",
        "inputs": {
            "micro_date": MICRO_DATE,
            "spearman_rows": int(matrix_values.shape[0]),
            "spearman_columns": int(matrix_values.shape[1]),
            "rank_start": RANK_START,
            "rank_end": RANK_END,
            "rank_rows": int(rank_values.size),
            "rank_groups": int(rank_offsets.size - 1),
            "threads": int(os.environ.get("QUANT_NATIVE_THREADS", "1")),
        },
        "spearman": {
            "correct": True,
            "counts_exact": True,
            "python_first_s": spearman_python_baseline,
            "python_runs_s": spearman_python_runs,
            "python_median_s": spearman_python_median,
            "numba_cold_s": spearman_cold,
            "numba_runs_s": spearman_numba_runs,
            "numba_median_s": spearman_numba_median,
            "speedup": spearman_speedup,
            "diagnostic_pass": bool(
                spearman_speedup >= MIN_KERNEL_SPEEDUP
            ),
        },
        "rank": {
            "correct": True,
            "python_first_s": rank_python_baseline,
            "python_runs_s": rank_python_runs,
            "python_median_s": rank_python_median,
            "numba_cold_s": rank_cold,
            "numba_runs_s": rank_numba_runs,
            "numba_median_s": rank_numba_median,
            "speedup": rank_speedup,
            "gate_pass": bool(rank_speedup >= MIN_KERNEL_SPEEDUP),
        },
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            3,
        ),
    }
    atomic_write_json(output / "micro.json", result)
    return result


def finalize_gate(
    output: Path,
    micro: dict[str, Any],
) -> dict[str, Any]:
    """按逐日排名核汇总 P1-01 门禁并写出机器可读结论。"""

    rank_correct = bool(micro["rank"]["correct"])
    rank_speedup = float(micro["rank"]["speedup"])
    if not rank_correct:
        reason = "daily_rank_incorrect"
        gate_pass = False
    elif rank_speedup < MIN_KERNEL_SPEEDUP:
        reason = "daily_rank_microbench_below_1.5x"
        gate_pass = False
    else:
        reason = None
        gate_pass = True
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": "daily_rank",
        "micro_pass": gate_pass,
        "gate_pass": gate_pass,
        "rank_correct": rank_correct,
        "rank_speedup": rank_speedup,
        "spearman_diagnostic_pass": micro["spearman"]["diagnostic_pass"],
        "spearman_speedup": micro["spearman"]["speedup"],
        "stage_run": False,
        "stage_scope": "deferred_to_p1_02",
        "reason": reason,
        "next_task": "P1-02" if gate_pass else "P1-02 保持待开始",
        "cpp_modules_replaced": 0,
    }
    atomic_write_json(output / "gate.json", result)
    lines = [
        "# P1-01 逐日排名放行门禁",
        "",
        f"- 逐日排名加速：{micro['rank']['speedup']:.3f}x",
        f"- 逐日排名正确：{'是' if micro['rank']['correct'] else '否'}",
        f"- Spearman 诊断加速：{micro['spearman']['speedup']:.3f}x",
        (
            "- Spearman 诊断结论："
            + ("达到 1.5x" if micro["spearman"]["diagnostic_pass"] else "未达到 1.5x")
        ),
        "- 真实 stage：本轮不执行，留给 P1-02 的 C++ POC 验证",
        f"- 门禁结论：{'通过' if gate_pass else '未通过'}",
        f"- 原因：{reason or '逐日排名门禁满足'}",
        "- 本轮实际 C++ 替换模块数：0",
    ]
    atomic_write_text(output / "gate.md", "\n".join(lines) + "\n")
    result_lines = [
        "# P1-01 逐日排名门禁返工结果",
        "",
        "## 口径",
        "",
        "- 数据：V105 固定输入，运行前校验输入树 SHA-256。",
        (
            f"- Spearman 截面：{micro['inputs']['micro_date']}，"
            f"{micro['inputs']['spearman_rows']} 行、"
            f"{micro['inputs']['spearman_columns']} 列。"
        ),
        (
            f"- 排名输入：{micro['inputs']['rank_start']}～"
            f"{micro['inputs']['rank_end']}，{micro['inputs']['rank_groups']} 组、"
            f"{micro['inputs']['rank_rows']} 行。"
        ),
        f"- 热运行：每项 {REPEATS} 次取中位数；线程数 {micro['inputs']['threads']}。",
        "",
        "## 结果",
        "",
        "| 内核 | Python 热运行中位 | Numba 热运行中位 | 加速 | Numba 冷启动 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Spearman | {micro['spearman']['python_median_s']:.6f}s | "
            f"{micro['spearman']['numba_median_s']:.6f}s | "
            f"{micro['spearman']['speedup']:.3f}x | "
            f"{micro['spearman']['numba_cold_s']:.3f}s |"
        ),
        (
            f"| 逐日排名 | {micro['rank']['python_median_s']:.6f}s | "
            f"{micro['rank']['numba_median_s']:.6f}s | "
            f"{micro['rank']['speedup']:.3f}x | "
            f"{micro['rank']['numba_cold_s']:.3f}s |"
        ),
        "",
        "逐日排名与 pandas/当前 NumPy 实现逐值一致，是 P1-01 放行门禁。Spearman"
        " 与当前 Python 实现在 `1e-10` 容差内一致、有效计数完全一致，但热运行仅为诊断，"
        "不参与本轮门禁，留给 P1-02 的 C++ POC 验证。",
        "",
        "## 门禁",
        "",
        "- 真实 stage：本轮不执行，留给 P1-02 的 C++ POC 验证。",
        f"- 结论：{'P1-02 可以继续' if gate_pass else 'P1-02 保持待开始'}",
        f"- 原因：{reason or '逐日排名门禁满足'}",
        "- 本轮实际 C++ 替换模块数：0。",
        "",
    ]
    atomic_write_text(output / "result.md", "\n".join(result_lines))
    return result


def run_suite(output: Path) -> int:
    """运行固定微基准并按逐日排名门禁生成结论。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    expected_inputs = load_golden(V105_INPUTS)
    current_hash = common["inputs"]["input_tree_sha256"]
    expected_hash = expected_inputs["input_tree_sha256"]
    if current_hash != expected_hash:
        raise RuntimeError(
            "输入数据相对 V105 已漂移，停止性能对照："
            f"{current_hash} != {expected_hash}"
        )
    with TemporaryDirectory(prefix="kernel-numba-micro-") as cache:
        environment = os.environ.copy()
        environment["NUMBA_CACHE_DIR"] = cache
        environment["QUANT_NATIVE_THREADS"] = "1"
        command = [
            sys.executable,
            "-m",
            "tools.monitor.kernel_baseline",
            "--case",
            "micro",
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    micro = load_golden(output / "micro.json")
    gate = finalize_gate(output, micro)
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "spearman_speedup": micro["spearman"]["speedup"],
                "rank_speedup": micro["rank"]["speedup"],
                "gate": gate,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析并执行 P1-01 核基准。"""

    parser = argparse.ArgumentParser(description="运行批量核 Numba 验证")
    parser.add_argument(
        "--case",
        choices=("suite", "micro"),
        default="suite",
    )
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.case == "micro":
        run_micro(output)
        return 0
    return run_suite(output)


if __name__ == "__main__":
    raise SystemExit(main())
