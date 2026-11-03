"""波动与风险因子。

低波动异象在 A 股长期存在；极值收益（彩票效应）与偏度是短端最强的负向因子之一。
"""

from factors.ops import ts_max, ts_skew, ts_std, ts_std_downside
from factors.registry import define

define(
    name="VOL_20",
    family="volatility",
    formula="std(ret, 20)",
    description="20 日收益波动率，低波动异象：波动越低未来收益越好。",
    steps=(
        "取最近 20 个有效交易日的日收益 ret",
        "算这 20 个数的样本标准差",
        "标准差越大说明波动越高",
    ),
    layers=ts_std("ret", 20),
    direction=-1,
    fields=("ret",),
)

define(
    name="DOWNSIDE_VOL_20",
    family="volatility",
    formula="sqrt(mean(min(ret, 0)^2, 20))",
    description="20 日下行波动，只统计亏损方向的波动，比总波动更贴近风险厌恶。",
    steps=(
        "取最近 20 个交易日的日收益 ret",
        "把正收益截成 0，只保留亏损",
        "平方后取平均再开方，得到只反映下跌的下行波动",
    ),
    layers=ts_std_downside("ret", 20),
    direction=-1,
    fields=("ret",),
)

define(
    name="MAX_RET_20",
    family="volatility",
    formula="max(ret, 20)",
    description="20 日最大单日收益，彩票效应：极值收益高的股票未来偏弱。",
    steps=(
        "取最近 20 个交易日的日收益 ret",
        "取其中最大的一个，代表近期单日涨幅最猛的程度",
    ),
    layers=ts_max("ret", 20),
    direction=-1,
    fields=("ret",),
)

define(
    name="SKEW_20",
    family="volatility",
    formula="skewness(ret, 20)",
    description="20 日收益偏度，右偏（博彩型）股票预期收益偏低。",
    steps=(
        "取最近 20 个交易日的日收益 ret",
        "算偏度：右偏为正、左偏为负",
    ),
    layers=ts_skew("ret", 20),
    direction=-1,
    fields=("ret",),
)
