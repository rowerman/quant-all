"""WorldQuant Alpha101 复现（只取不依赖财报的公式）。

原始公式里的算子在这里对应：

===================  ==========================================
Alpha101 算子        本项目实现
===================  ==========================================
rank(x)              ``cs_rank``
ts_argmax(x, d)      多层写法：行号层 + ``ts_argmax`` + 相减
delta(x, d)          ``ts_delta``
corr(x, y, d)        ``ts_corr``
stddev(x, d)         ``ts_std``
signedpower(x, a)    ``signed_power``
===================  ==========================================

需要 ``decay_linear`` / ``ts_rank`` 的公式还没实现，写法见
``docs/core/factors/conventions.md`` 的多层示例。
"""

from factors.ops import cs_rank, neg, ts_corr, ts_delta, ts_std
from factors.registry import define

define(
    name="ALPHA101_001",
    family="alpha101",
    formula="rank(5 日内极大值出现在几天前) - 0.5，极大值口径为 power(ret<0 ? std(ret,20) : close, 2)",
    description="原始 Alpha#1：先按 20 日波动放大价格，再看最近 5 日极值出现在几天前，截面排序。",
    steps=(
        "第 1 层：给每行编号 row_number，用来记录极值出现的位置",
        "第 2 层：算 20 日日收益标准差 std(ret, 20)",
        "第 3 层：当日收益为负时取这个标准差、否则取收盘价，再平方，得到放大后的价格序列",
        "第 4 层：在最近 5 日里找出该序列最大的一行，返回它的行号",
        "第 5 层：用「当前行号 − 极值行号」得到极值距今几天，做截面排名后减 0.5",
    ),
    layers=(
        "row_number() over w",
        ts_std("ret", 20),
        "power(case when ret < 0 then v2 else close end, 2)",
        "arg_max(v1, v3) over (partition by stock_code order by trade_date rows between 4 preceding and current row)",
        f"{cs_rank('(v1 - v4)')} - 0.5",
    ),
    lookback=24,
    direction=1,
    fields=("close", "ret"),
)

define(
    name="ALPHA101_006",
    family="alpha101",
    formula="-corr(open, volume, 10)",
    description="原始 Alpha#6：开盘价与成交量的负相关，量价背离的经典写法。",
    steps=(
        "取最近 10 个交易日的开盘价与成交量",
        "算两者的相关系数",
        "取负号，把「量价正相关」变成反向信号",
    ),
    layers=neg(ts_corr("open", "volume_share", 10)),
    direction=1,
    fields=("open", "volume_share"),
)

define(
    name="ALPHA101_012",
    family="alpha101",
    formula="sign(delta(volume, 1)) * (-delta(close, 1))",
    description="原始 Alpha#12：放量则记当日跌幅，缩量则记当日涨幅的取负。",
    steps=(
        "算成交量的一日变动并取符号：放量为正、缩量为负",
        "算收盘价的一日变动并取负",
        "两者相乘",
    ),
    layers=f"sign({ts_delta('volume_share', 1)}) * {neg(ts_delta('close', 1))}",
    direction=1,
    fields=("close", "volume_share"),
)
