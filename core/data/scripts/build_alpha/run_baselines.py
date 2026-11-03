"""运行 P0 四条基线并写出对照表。"""

from __future__ import annotations

import argparse

from alpha.config import DEFAULT_CONTRACT
from alpha.logging_conf import get_logger
from scripts.build_alpha.baselines import (
    BASELINE_IDS,
    run_baseline,
    write_comparison,
)

logger = get_logger("scripts.build_alpha.run_baselines")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 ML 模型层 P0 线性基线")
    parser.add_argument(
        "--baseline",
        default="all",
        choices=("all", *BASELINE_IDS),
        help="运行单条基线或全部基线",
    )
    parser.add_argument("--start", default=DEFAULT_CONTRACT.dev_start)
    parser.add_argument("--end", default=DEFAULT_CONTRACT.dev_end)
    parser.add_argument(
        "--engine",
        choices=("weekly", "ledger"),
        default="weekly",
        help="weekly=快速研究模拟器；ledger=完整回测账本",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已有 compact 结果")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    selected = BASELINE_IDS if args.baseline == "all" else (args.baseline,)
    for baseline_id in selected:
        logger.info("开始基线 %s：%s ~ %s", baseline_id, args.start, args.end)
        result = run_baseline(
            baseline_id,
            args.start,
            args.end,
            force=args.force,
            engine=args.engine,
        )
        portfolio = result["portfolio"]
        logger.info(
            "%s 完成：年化=%s 超额=%s 回撤=%s",
            baseline_id,
            portfolio.get("annual_return"),
            portfolio.get("excess_annual_return"),
            portfolio.get("max_drawdown"),
        )
    comparison = write_comparison()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
