"""按通用评估计划执行候选比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_alpha.evaluation_executor import run_evaluation_plan


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="执行 Alpha 模型评估计划")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = run_evaluation_plan(Path(args.plan), force=args.force)
    print(
        json.dumps(
            {
                "analysis_id": result["analysis_id"],
                "status": result["status"],
                "selected_candidate": result["selected_candidate"],
                "selected_source": result["selected_source"],
                "candidate_count": result["candidate_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
