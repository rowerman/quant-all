"""生成算子优化与后续 C++ 核共用的固定基线证据。

用法::

    .venv/bin/python -m tools.monitor.operator_baseline --suite all
    .venv/bin/python -m tools.monitor.operator_baseline --case edge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

from tools.monitor.baseline_cases import run_case
from tools.monitor.baseline_contract import (
    CASE_DEFINITIONS,
    CASE_ORDER,
    EVIDENCE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    load_golden,
    prepare_common,
    project_relative,
    update_manifest,
)


def peak_rss_mb() -> float:
    """返回当前进程的峰值 RSS，单位 MB。"""

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def case_paths(output: Path, name: str) -> tuple[Path, Path, Path]:
    """返回一个案例的金标准、计时和失败证据路径。"""

    return (
        output / f"{name}.golden.json",
        output / f"{name}.timing.json",
        output / f"{name}.error.json",
    )


def execute_case(name: str, output: Path, *, refresh_common: bool = False) -> dict[str, Any]:
    """执行并落盘一个案例；参数 name 为固定案例名。"""

    common = prepare_common(output, refresh=refresh_common)
    golden_path, timing_path, error_path = case_paths(output, name)
    started = time.perf_counter()
    try:
        golden, case_timing = run_case(name, common)
    except Exception as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "case": name,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(error_path, error)
        raise

    timing = {
        "schema_version": SCHEMA_VERSION,
        "case": name,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_rss_mb": peak_rss_mb(),
        "golden_bytes": len(canonical_json(golden).encode("utf-8")),
        "case_timing": case_timing,
        "definition": CASE_DEFINITIONS[name],
    }
    atomic_write_json(
        golden_path,
        {
            "schema_version": SCHEMA_VERSION,
            "case": name,
            "definition": CASE_DEFINITIONS[name],
            "golden": golden,
        },
    )
    atomic_write_json(timing_path, timing)
    if error_path.exists():
        error_path.unlink()
    return timing


def run_suite(
    output: Path,
    *,
    refresh_common: bool = False,
    cases: tuple[str, ...] = CASE_ORDER,
) -> int:
    """用独立子进程顺序运行案例；参数 cases 为案例顺序。"""

    output.mkdir(parents=True, exist_ok=True)
    prepare_common(output, refresh=refresh_common)
    for name in cases:
        print(f"[operator-baseline] 开始 {name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "tools.monitor.operator_baseline",
            "--case",
            name,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, cwd=project_root(), check=False)
        if completed.returncode:
            print(f"[operator-baseline] 失败 {name}", file=sys.stderr, flush=True)
            return completed.returncode
        print(f"[operator-baseline] 完成 {name}", flush=True)
    finalize_output(output)
    return 0


def project_root() -> Path:
    """返回项目根目录。"""

    return Path(__file__).resolve().parents[2]


def finalize_output(output: Path) -> dict[str, Any]:
    """生成 manifest 和人类可读报告；参数 output 为证据目录。"""

    manifest = update_manifest(output)
    timings = []
    for name in CASE_ORDER:
        _, timing_path, _ = case_paths(output, name)
        if timing_path.is_file():
            timings.append(load_golden(timing_path))

    lines = [
        "# 算子优化固定基准",
        "",
        f"- 证据目录：`{project_relative(output)}`",
        f"- Schema：`{SCHEMA_VERSION}`",
        f"- 案例数：{len(timings)}",
        "- 完整评估只运行一次；稳定数值与计时字段分开保存。",
        "- 公共 payload 的非有限值保存为 `null`；底层边界案例使用显式字符串区分。",
        "- 浮点比较默认 `abs=1e-10, rel=1e-10`；日期、名称、顺序、有效计数和排名类输出必须完全一致。",
        "- 本工具记录 Python/DuckDB 路径；C++ 接入状态以对应阶段文档为准。",
        "",
        "| 案例 | 耗时(s) | 峰值RSS(MB) | 金标准大小(B) |",
        "|---|---:|---:|---:|",
    ]
    for timing in timings:
        lines.append(
            f"| {timing['case']} | {timing['elapsed_s']} | "
            f"{timing['peak_rss_mb']} | {timing['golden_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `environment.json`：Git、Python、依赖、CPU、内存和源码快照。",
            "- `inputs.json`：行情主库、因子目录、配置和全部正式因子 Parquet 的 SHA-256。",
            "- `case_definitions.json`：不可漂移的案例参数和比较规则。",
            "- `*.golden.json`：后续优化必须对照的稳定 Python 结果。",
            "- `*.timing.json`：逐案例、逐事件耗时和独立进程峰值 RSS。",
            "- `manifest.json`：除报告和 manifest 自身外的证据文件哈希。",
            "",
            "## 复现",
            "",
            "```bash",
            ".venv/bin/python -m tools.monitor.operator_baseline --suite all",
            "```",
        ]
    )
    atomic_write_text(output / "report.md", "\n".join(lines) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行固定基准；参数 argv 默认取命令行。"""

    parser = argparse.ArgumentParser(description="生成算子优化与 C++ 基线证据")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=CASE_ORDER)
    selection.add_argument("--suite", choices=("all",))
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--refresh-common", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()

    if args.case:
        timing = execute_case(args.case, output, refresh_common=args.refresh_common)
        finalize_output(output)
        print(json.dumps(timing, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    return run_suite(output, refresh_common=args.refresh_common)


if __name__ == "__main__":
    raise SystemExit(main())
