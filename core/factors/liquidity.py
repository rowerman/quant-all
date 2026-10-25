"""流动性因子。

成交额、换手率与 Amihud 非流动性共同刻画「这只股票好不好买、好不好卖」。
窗口只覆盖有效交易日，停牌天不参与平均。
"""

from factors.ops import ts_mean
from factors.registry import define

define(
    name="AMIHUD_20",
    family="liquidity",
    formula="mean(|ret| / (amount / 1e8), 20)",
    description="20 日 Amihud 非流动性：每亿元成交额带来的绝对收益，值越大越不流动。",
    steps=(
        "把成交额换算成亿元：amount_cny / 1e8",
        "算每日的「|收益| ÷ 成交额」，即每成交一亿元价格变动多少",
        "对这 20 天的比值取平均，值越大越不流动",
    ),
    layers=ts_mean("abs(ret) / nullif(amount_cny / 100000000.0, 0)", 20),
    direction=1,
    fields=("ret", "amount_cny"),
)

define(
    name="TURN_20",
    family="liquidity",
    formula="mean(turnover_rate, 20)",
    description="20 日平均换手率，高换手在 A 股通常对应未来收益偏弱。",
    steps=(
        "取最近 20 个交易日的换手率 turnover_rate",
        "取平均，代表近一个月的交易活跃度",
    ),
    layers=ts_mean("turnover_rate", 20),
    direction=-1,
    fields=("turnover_rate",),
)

define(
    name="LN_AMT_20",
    family="liquidity",
    formula="ln(mean(amount, 20))",
    description="20 日平均成交额取对数，小盘流动性差的股票长期有溢价。",
    steps=(
        "取最近 20 个交易日的成交额 amount_cny",
        "取平均后取自然对数，压缩量纲",
    ),
    layers=f"ln({ts_mean('amount_cny', 20)})",
    direction=-1,
    fields=("amount_cny",),
)
