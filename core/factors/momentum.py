"""动量与反转因子。

A 股的经验规律：1 个月以内偏反转，3~12 个月偏动量。
所有动量都用后复权价，避免除权日的假跳空。
"""

from factors.ops import ts_delay
from factors.registry import define

define(
    name="MOM_20",
    family="momentum",
    formula="close_t / close_(t-20) - 1",
    description="20 交易日累计收益，短中期动量；A 股短端更偏反转，方向以 IC 实测为准。",
    steps=(
        "取当日后复权收盘价 close",
        "取 20 个交易日之前的收盘价 lag(close, 20)",
        "相除再减 1，得到这 20 个交易日的累计收益率",
    ),
    layers=f"close / {ts_delay('close', 20)} - 1",
    direction=1,
    fields=("close",),
)

define(
    name="MOM_60",
    family="momentum",
    formula="close_t / close_(t-60) - 1",
    description="60 交易日累计收益，季度动量。",
    steps=(
        "取当日后复权收盘价 close",
        "取 60 个交易日之前的收盘价 lag(close, 60)",
        "相除再减 1，得到这 60 个交易日的累计收益率",
    ),
    layers=f"close / {ts_delay('close', 60)} - 1",
    direction=1,
    fields=("close",),
)

define(
    name="MOM_60_20",
    family="momentum",
    formula="close_(t-20) / close_(t-60) - 1",
    description="跳过最近 20 日的中期动量，剔除短期反转对动量因子的稀释。",
    steps=(
        "取 20 个交易日前的收盘价",
        "取 60 个交易日前的收盘价",
        "相除再减 1，得到「跳过最近 20 日」那 40 个交易日的收益率",
    ),
    layers=f"{ts_delay('close', 20)} / {ts_delay('close', 60)} - 1",
    direction=1,
    fields=("close",),
)

define(
    name="REV_5",
    family="momentum",
    formula="-(close_t / close_(t-5) - 1)",
    description="5 日反转：过去一周跌得多的股票，因子值更大。",
    steps=(
        "先算 5 日累计收益率：close / lag(close, 5) - 1",
        "再取负号，这样「跌得多」的股票因子值更大",
    ),
    layers=f"-(close / {ts_delay('close', 5)} - 1)",
    direction=1,
    fields=("close",),
)

define(
    name="REV_20",
    family="momentum",
    formula="-(close_t / close_(t-20) - 1)",
    description="20 日反转：过去一个月跌幅越大，因子值越大。",
    steps=(
        "先算 20 日累计收益率：close / lag(close, 20) - 1",
        "再取负号，这样「跌得多」的股票因子值更大",
    ),
    layers=f"-(close / {ts_delay('close', 20)} - 1)",
    direction=1,
    fields=("close",),
)
