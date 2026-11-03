"""在合成数据上运行一次因子评估和短区间回测。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """运行演示，不需要 Tushare 或模型 API。"""

    parser = argparse.ArgumentParser(description="运行合成量化演示")
    parser.add_argument("--data-dir", default="data/demo", help="演示数据目录")
    parser.add_argument("--start", default="2022-01-04", help="回测起始日")
    parser.add_argument("--end", default="2023-12-29", help="回测截止日")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    if not (data_dir / "databases" / "quant.duckdb").exists():
        raise SystemExit(
            f"演示数据不存在：{data_dir}；先运行 "
            f"python examples/build_demo_data.py --output {data_dir}"
        )
    os.environ["QUANT_DATA_DIR"] = str(data_dir.resolve())

    from backtest.data import load_market_data
    from backtest.engine import run_backtest
    from backtest.specs import BacktestParams, BenchmarkSpec, PortfolioSpec, StrategySpec
    from calculate.evaluate import evaluate

    print("1. 计算 MOM_20 因子评估")
    stages = {}
    for stage, payload in evaluate(
        "MOM_20",
        stages=("overview", "ic_series", "ic_dist"),
    ):
        stages[stage] = payload
    overview = stages["overview"]["metrics"]
    coverage = overview["coverage"]
    rank_ic = overview["ic_stats"]["rank_ic"]
    print(
        f"   交易日 {coverage['days']}，股票 {coverage['stocks']}，"
        f"池内覆盖率 {coverage['coverage_universe']:.2%}，"
        f"RankIC 均值 {rank_ic['mean']:.4f}"
    )

    print("2. 运行 MOM_20 Top-20 月频回测")
    data = load_market_data(
        args.start,
        args.end,
        ["MOM_20"],
        benchmark_code=None,
        data_snapshot={"schema": "synthetic_demo_v1"},
    )
    strategy = StrategySpec(
        name="Synthetic MOM_20 Top20",
        signal=StrategySpec().signal,
        portfolio=PortfolioSpec(top_n=20),
        benchmark=BenchmarkSpec(index_code=None),
    )
    result = run_backtest(
        data,
        strategy,
        BacktestParams(start=args.start, end=args.end, init_cash=1_000_000),
        build_snapshot=False,
    )
    print(
        f"   累计收益 {result.metrics['total_return']:.2%}，"
        f"年化 {result.metrics['annual_return']:.2%}，"
        f"最大回撤 {result.metrics['max_drawdown']:.2%}，"
        f"成交 {len(result.orders)} 笔"
    )
    print("演示完成。Web 启动：QUANT_DATA_DIR=" + str(data_dir) + " bash tools/monitor/serve.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
