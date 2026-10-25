"""运行 Alpha 模型 P5-2 监控。"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha.monitor import (
    DEFAULT_ACCEPTANCE_ID,
    DEFAULT_EXPOSURE_ANALYSIS_ID,
    DEFAULT_MODEL_ID,
    run_model_monitor,
)


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Alpha 模型 P5-2 监控")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--acceptance-id", default=DEFAULT_ACCEPTANCE_ID)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--candidate")
    parser.add_argument(
        "--exposure-analysis-id",
        default=DEFAULT_EXPOSURE_ANALYSIS_ID,
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--force", action="store_true", help="强制重建已有监控包")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行监控并输出摘要。"""

    args = _parser().parse_args(argv)
    payload = run_model_monitor(
        args.model_id,
        acceptance_id=args.acceptance_id,
        exposure_analysis_id=args.exposure_analysis_id,
        evaluation_id=args.evaluation_id,
        candidate=args.candidate,
        start=args.start,
        end=args.end,
        force=args.force,
        root=args.root,
    )
    summary = payload["summary"]
    print(
        f"{payload['monitor_id']}\t{summary['status']}\t"
        f"{summary['start']}~{summary['end']}\t"
        f"warning={summary['warning_count']}\talert={summary['alert_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
