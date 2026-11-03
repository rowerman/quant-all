"""A 股特色因子。

涨跌停、停牌、隔夜跳空这些制度性变量在成熟市场没有对应物，
但在 A 股是短周期收益的重要解释项。所有窗口都只覆盖有效交易日。
"""

from factors.ops import ts_mean, ts_sum
from factors.registry import define

define(
    name="LIMIT_UP_CNT_20",
    family="cn_special",
    formula="sum(is_limit_up, 20)",
    description="20 日内涨停天数，衡量资金关注度；涨停越多短期越容易反转。",
    steps=(
        "逐日判断是否涨停（is_limit_up）",
        "统计最近 20 个有效交易日里的涨停天数",
    ),
    layers=ts_sum("case when is_limit_up then 1 else 0 end", 20),
    direction=-1,
    fields=("is_limit_up",),
)

define(
    name="OVERNIGHT_RET_20",
    family="cn_special",
    formula="mean(open / pre_close - 1, 20)",
    description="20 日隔夜跳空均值，隔夜收益在 A 股长期偏负，反映散户开盘追高。",
    steps=(
        "算每日隔夜跳空：open / 前收 - 1",
        "取最近 20 天的平均值",
    ),
    layers=ts_mean("open / nullif(pre_close, 0) - 1", 20),
    direction=-1,
    fields=("open", "pre_close"),
)
