"""量价关系因子。

刻画「价格变动有没有成交量配合」，是短周期里最常用的资金行为代理。
"""

from factors.ops import ts_corr, ts_mean
from factors.registry import define

define(
    name="PV_CORR_20",
    family="volume_price",
    formula="corr(ret, volume, 20)",
    description="20 日收益与成交量的相关系数，量价背离（相关低）往往意味着抛压小。",
    steps=(
        "取最近 20 个交易日的日收益 ret 与成交量 volume_share",
        "算两者的相关系数，衡量「涨跌有没有成交量配合」",
    ),
    layers=ts_corr("ret", "volume_share", 20),
    direction=-1,
    fields=("ret", "volume_share"),
)

define(
    name="VWAP_BIAS_5",
    family="volume_price",
    formula="mean(close / vwap - 1, 5)",
    description="5 日收盘价相对成交均价的平均偏离，衡量近期买盘是否一直在更高价成交。",
    steps=(
        "算每日收盘价相对当日成交均价的偏离：close / vwap - 1",
        "取最近 5 天的平均偏离",
    ),
    layers=ts_mean("close / nullif(vwap, 0) - 1", 5),
    direction=-1,
    fields=("close", "vwap"),
)
