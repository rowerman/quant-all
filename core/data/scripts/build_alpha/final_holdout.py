"""运行 2024-2026 最终留出集验证。"""

from __future__ import annotations

import argparse
import json

from alpha.final_holdout import FinalHoldoutSpec, load_final_holdout, run_final_holdout
from alpha.holdout import FUTURE_HOLDOUT_WINDOW_ID


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="ML 模型最终留出集验证")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load", metavar="HOLDOUT_ID")
    parser.add_argument("--resume-from", metavar="HOLDOUT_ID")
    parser.add_argument("--window-id", default=FUTURE_HOLDOUT_WINDOW_ID)
    parser.add_argument("--plan", help="本次留出验证使用的评估计划")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    if args.load:
        result = load_final_holdout(args.load)
        payload = {
            "holdout_id": result["manifest"]["holdout_id"],
            "status": result["manifest"]["status"],
            "path": str(result["folder"]),
        }
    else:
        payload = run_final_holdout(
            FinalHoldoutSpec(
                window_id=args.window_id,
                plan_path=args.plan,
            ),
            force=args.force,
            resume_from=args.resume_from,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    if payload.get("status") not in {
        "pass_final_holdout",
        "fail_final_holdout",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
