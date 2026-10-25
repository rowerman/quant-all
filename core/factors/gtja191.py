"""国泰君安 Alpha191 复现（只取不依赖财报的公式）。

191 个公式里大量用到 ``DECAYLINEAR`` / ``SMA`` 这类递归平滑，
本项目用多层写法实现；已完成的部分见下。
"""

from factors.ops import cs_rank, frame, ts_sum
from factors.registry import define

define(
    name="GTJA_191_001",
    family="gtja191",
    formula="-corr(rank(delta(ln(volume),1)), rank((close-open)/open), 6)",
    description="原始 Alpha#1：成交量变化的分位与日内涨跌幅的分位做 6 日相关，再取负。",
    steps=(
        "第 1 层：算每日日内涨跌幅 (close - open) / open",
        "第 2 层：算成交量的对数变化 ln(volume) - lag(ln(volume), 1)",
        "第 3 层：把日内涨跌幅放到当日截面上排名",
        "第 4 层：把成交量变化放到当日截面上排名",
        "第 5 层：算这两个排名最近 6 日的相关系数，再取负",
    ),
    layers=(
        "(close - open) / nullif(open, 0)",
        "ln(volume_share) - lag(ln(volume_share), 1) over w",
        f"{cs_rank('v1')}",
        f"{cs_rank('v2')}",
        f"-corr(v3, v4) over {frame(6)}",
    ),
    lookback=6,
    direction=1,
    fields=("open", "close", "volume_share"),
)

define(
    name="GTJA_191_003",
    family="gtja191",
    formula="sum(close == high ? 1 : (close == low ? -1 : 0), 6)",
    description="原始 Alpha#3：6 日内收在最高价与最低价的天数之差，衡量收盘强弱。",
    steps=(
        "逐日比较：收在最高价记 +1、收在最低价记 −1、其余记 0",
        "把最近 6 天的得分求和，衡量收盘强弱",
    ),
    layers=ts_sum("case when close = high then 1 when close = low then -1 else 0 end", 6),
    direction=1,
    fields=("close", "high", "low"),
)
