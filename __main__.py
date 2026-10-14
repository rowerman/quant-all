"""运行一次可保存的 VectorBT 回测。

用法示例::

    .venv/bin/python -m backtest --strategy mom20 --start 2024-01-01 --end 2026-09-28
    .venv/bin/python -m backtest --strategy multifactor --start 2016-01-01 --end 2026-09-28 \
        --universe index:000905.SH --benchmark 000905.SH
"""

from __future__ import annotations

import argparse
import json

from backtest.builtin import BUILTINS, MOM20_ID, MULTIFACTOR_ID
from backtest.errors import BacktestError
from backtest.pipeline import RunRequest, iter_run
from backtest.store import BacktestStore

ALIASES = {"mom20": MOM20_ID, "multifactor": MULTIFACTOR_ID}


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="VectorBT A 股回测")
    parser.add_argument("--strategy", choices=sorted(ALIASES), default="mom20", help="内置策略")
    parser.add_argument("--strategy-id", help="直接指定已保存的策略标识，覆盖 --strategy")
    parser.add_argument("--version", type=int, help="策略版本；默认取当前版本")
    parser.add_argument("--start", required=True, help="起始交易日 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="截止交易日 YYYY-MM-DD")
    parser.add_argument("--init-cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--universe", help="运行级股票池覆盖，如 index:000300.SH")
    parser.add_argument("--benchmark", help="运行级基准覆盖，如 000905.SH")
    return parser


def main() -> None:
    """解析参数并跑完一次回测。"""
    args = _parser().parse_args()
    store = BacktestStore()
    store.ensure_builtins()
    strategy_id = args.strategy_id or ALIASES[args.strategy]
    if args.strategy_id is None:
        name = dict((item[0], item[1].name) for item in BUILTINS)[strategy_id]
        print(f"内置策略：{name}（{strategy_id}）")
    request = RunRequest(
        strategy_id=strategy_id,
        start=args.start,
        end=args.end,
        version=args.version,
        init_cash=args.init_cash,
        universe=args.universe,
        benchmark=args.benchmark,
    )
    summary: dict = {}
    try:
        for stage, payload in iter_run(store, request):
            summary = payload if stage == "done" else summary
            print(f"[{stage}] {json.dumps(payload, ensure_ascii=False, default=str)}")
    except BacktestError as error:
        raise SystemExit(f"回测失败：{error}") from error
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

