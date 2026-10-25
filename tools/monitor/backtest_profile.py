"""真实回测的耗时与峰值内存剖析。

每个案例在独立进程内运行，使用临时结果库和生产调度器；``dual`` 案例用两个
worker 同时提交两份回测，测量生产并发下的组合驻留内存。证据只追加不覆盖。

用法::

    .venv/bin/python -m tools.monitor.backtest_profile --case all_a_3y
    .venv/bin/python -m tools.monitor.backtest_profile --case index_10y
    .venv/bin/python -m tools.monitor.backtest_profile --case dual
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "apps" / "monitor"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend import backtest_service
from backend.scheduler import Scheduler
from backtest.builtin import MOM20_ID, MULTIFACTOR_ID
from backtest.store import BacktestStore

EVIDENCE_DIR = PROJECT_ROOT / "AiRecord/benchmarks/V57"


def case_payloads(case: str) -> list[dict]:
    """返回一个剖析案例包含的回测请求。"""
    all_a = {
        "strategy_id": MOM20_ID,
        "start": "2023-10-16",
        "end": "2026-10-15",
        "universe": "all_a",
        "benchmark": "H00300.CSI",
    }
    index_10y = {
        "strategy_id": MULTIFACTOR_ID,
        "start": "2016-10-17",
        "end": "2026-10-15",
        "universe": "index:000905.SH",
        "benchmark": "H00905.CSI",
    }
    if case == "all_a_3y":
        return [all_a]
    if case == "index_10y":
        return [index_10y]
    if case == "dual":
        return [all_a, index_10y]
    raise ValueError(f"未知剖析案例：{case}")


def read_vm_mb(key: str, pid: int | None = None) -> float | None:
    """读取 ``/proc/<pid>/status`` 中的内存字段，单位 MB。"""
    status = Path(f"/proc/{pid or 'self'}/status")
    try:
        match = re.search(rf"^{key}:\s+(\d+) kB$", status.read_text(), re.MULTILINE)
    except OSError:
        return None
    return round(int(match.group(1)) / 1024, 1) if match else None


def _sample_rss(stop: threading.Event, peak: list[float]) -> None:
    """持续采样当前进程 RSS，记录组合峰值。"""
    while not stop.wait(0.2):
        value = read_vm_mb("VmRSS")
        if value is not None:
            peak[0] = max(peak[0], value)


def summarize_stages(stages: list[dict]) -> dict[str, dict]:
    """把逐阶段事件聚合成次数、合计耗时和最大单次耗时。"""
    summary: dict[str, dict] = {}
    for item in stages:
        name = str(item["stage"])
        current = summary.setdefault(
            name, {"count": 0, "elapsed_s": 0.0, "max_elapsed_s": 0.0}
        )
        elapsed = float(item["elapsed_s"])
        current["count"] += 1
        current["elapsed_s"] = round(current["elapsed_s"] + elapsed, 3)
        current["max_elapsed_s"] = max(current["max_elapsed_s"], elapsed)
    return summary


def _write_evidence(record: dict, output: Path | None) -> Path:
    """把一条剖析结果追加到证据文件。"""
    path = output or (
        EVIDENCE_DIR / f"{date.today():%Y%m%d}-backtest-profile.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def profile(case: str, *, evidence: bool = True, output: Path | None = None) -> dict:
    """运行一个剖析案例并返回可落盘的记录。"""
    payloads = case_payloads(case)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    errors: list[BaseException | None] = [None] * len(payloads)
    results: list[dict | None] = [None] * len(payloads)
    stages: list[list[dict]] = [[] for _ in payloads]

    with tempfile.TemporaryDirectory(prefix="backtest-profile-") as directory:
        store_path = Path(directory) / "backtest.duckdb"
        requests = [
            backtest_service.prepare(payload, str(store_path))
            for payload in payloads
        ]
        scheduler = Scheduler(workers=max(1, len(requests)), waiting=1)
        jobs = []
        try:
            for index, request in enumerate(requests):
                job = scheduler.reserve()

                def factory(index=index, request=request, job=job):
                    last = time.perf_counter()
                    for stage, payload in backtest_service.stream_strategy(
                        request, job.cancelled.is_set, str(store_path)
                    ):
                        now = time.perf_counter()
                        stages[index].append(
                            {
                                "stage": stage,
                                "elapsed_s": round(now - last, 3),
                            }
                        )
                        last = now
                        yield stage, payload

                scheduler.start(job, factory)
                jobs.append(job)

            stop = threading.Event()
            peak = [read_vm_mb("VmRSS") or 0.0]
            monitor = threading.Thread(
                target=_sample_rss, args=(stop, peak), daemon=True
            )
            monitor.start()

            def consume(index: int, job) -> None:
                try:
                    results[index] = scheduler.wait(job)
                except BaseException as error:
                    errors[index] = error

            waiters = [
                threading.Thread(target=consume, args=(index, job), daemon=True)
                for index, job in enumerate(jobs)
            ]
            for waiter in waiters:
                waiter.start()
            for waiter in waiters:
                waiter.join()
            stop.set()
            monitor.join()
        finally:
            scheduler.close()

        store = BacktestStore(store_path)
        jobs_payload = []
        for index, (payload, result, error) in enumerate(
            zip(payloads, results, errors, strict=True)
        ):
            run_id = result.get("run_id") if result else None
            status = store.get_run(run_id)["status"] if run_id else None
            jobs_payload.append(
                {
                    **payload,
                    "run_id": run_id,
                    "status": status,
                    "orders": result.get("orders") if result else None,
                    "elapsed_seconds": (
                        result.get("elapsed_seconds") if result else None
                    ),
                    "stages": stages[index],
                    "stage_summary": summarize_stages(stages[index]),
                    "error": (
                        f"{type(error).__name__}: {error}" if error else None
                    ),
                }
            )

    record = {
        "case": case,
        "started_at": started_at,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_rss_mb": read_vm_mb("VmHWM"),
        "sampled_peak_rss_mb": round(peak[0], 1),
        "jobs": jobs_payload,
        "ok": all(
            item["status"] == "success" and item["error"] is None
            for item in jobs_payload
        ),
    }
    if evidence:
        record["evidence"] = str(_write_evidence(record, output))
    return record


def main() -> None:
    """解析参数并运行一个剖析案例。"""
    parser = argparse.ArgumentParser(description="回测耗时与峰值内存剖析")
    parser.add_argument(
        "--case", choices=("all_a_3y", "index_10y", "dual"), required=True
    )
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = profile(
        args.case,
        evidence=not args.no_evidence,
        output=args.output,
    )
    display = {
        **record,
        "jobs": [
            {key: value for key, value in job.items() if key != "stages"}
            for job in record["jobs"]
        ],
    }
    print(json.dumps(display, ensure_ascii=False, allow_nan=False), flush=True)
    if not record["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
