"""内置策略定义。

内置策略只是预先写进策略表的普通策略版本：固定的策略标识、可复现的配置 JSON，
与后续前端「新建策略」保存出来的策略走同一条存储与运行路径。
"""

from __future__ import annotations

from backtest.specs import PortfolioSpec, SignalSpec, StrategySpec

MOM20_ID = "builtin-mom20-top30"
MULTIFACTOR_ID = "builtin-multifactor-top30"

BUILTINS: tuple[tuple[str, StrategySpec, str], ...] = (
    (
        MOM20_ID,
        StrategySpec(
            name="MOM_20 Top30",
            signal=SignalSpec(factors=("MOM_20",), weights=(1.0,)),
            portfolio=PortfolioSpec(top_n=30, weighting="equal"),
        ),
        "内置基准策略：MOM_20 截面分数取 Top-30，全 A 可交易池，月频等权，"
        "收盘生成信号、次一交易日开盘执行",
    ),
    (
        MULTIFACTOR_ID,
        StrategySpec(
            name="多因子 Top30",
            signal=SignalSpec(factors=("MOM_20", "EP_TTM", "VOL_20"), weights=(1.0, 1.0, -1.0)),
            portfolio=PortfolioSpec(top_n=30, weighting="equal"),
        ),
        "内置多因子策略：MOM_20 + EP_TTM - VOL_20 线性组合，Top-30，全 A 可交易池，"
        "月频等权，收盘生成信号、次一交易日开盘执行",
    ),
)


def builtin_ids() -> tuple[str, ...]:
    """返回内置策略标识。"""
    return tuple(strategy_id for strategy_id, _, _ in BUILTINS)

