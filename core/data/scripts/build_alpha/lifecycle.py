"""运行 Alpha 模型 P5-3 生命周期规则。"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha.lifecycle import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_MONITOR_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    run_model_lifecycle,
)


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Alpha 模型 P5-3 生命周期")
    parser.add_argument("--model-id", default=DEFAULT_PRIMARY_MODEL_ID)
    parser.add_argument("--fallback-model-id", default=DEFAULT_FALLBACK_MODEL_ID)
    parser.add_argument("--monitor-id", default=DEFAULT_MONITOR_ID)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--force", action="store_true", help="强制重建生命周期包")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行生命周期规则并输出摘要。"""

    args = _parser().parse_args(argv)
    payload = run_model_lifecycle(
        args.model_id,
        fallback_model_id=args.fallback_model_id,
        monitor_id=args.monitor_id,
        start=args.start,
        end=args.end,
        force=args.force,
        root=args.root,
    )
    summary = payload["summary"]
    print(
        f"{payload['lifecycle_id']}\t{summary['latest_state']}\t"
        f"{summary['start']}~{summary['end']}\t"
        f"rollback={summary['event_counts']['rollback']}\t"
        f"pause={summary['event_counts']['pause']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
