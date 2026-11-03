"""技术形态因子。

RSI / BIAS / 布林带位置都是「价格偏离近期中枢多少」的度量，
窗口不足时为空值，不做填充。
"""

from factors.ops import ts_mean, ts_std
from factors.registry import define

_rsi_gain = ts_mean("greatest(ret, 0)", 14)
_rsi_loss = ts_mean("greatest(-ret, 0)", 14)

_boll_mid = ts_mean("close", 20)
_boll_std = ts_std("close", 20)

define(
    name="RSI_14",
    family="technical",
    formula="100 * mean(gain, 14) / (mean(gain, 14) + mean(loss, 14))",
    description="14 日相对强弱指标（SMA 口径，非 Wilder 平滑），高值代表短期超买。",
    steps=(
        "把每日收益拆成上涨部分 greatest(ret, 0) 与下跌部分 greatest(-ret, 0)",
        "两部分各取 14 日均值，得到平均涨幅与平均跌幅",
        "RSI = 100 × 平均涨幅 ÷（平均涨幅 + 平均跌幅）；两边都是 0 时置空",
    ),
    layers=f"100 * ({_rsi_gain}) / nullif(({_rsi_gain}) + ({_rsi_loss}), 0)",
    direction=-1,
    fields=("ret",),
)

define(
    name="BIAS_20",
    family="technical",
    formula="close / mean(close, 20) - 1",
    description="20 日均线乖离率，价格偏离中枢越远，均值回复压力越大。",
    steps=(
        "取最近 20 个交易日的收盘价均值，即 20 日均线",
        "用当日收盘价除以均线再减 1，得到偏离百分比",
    ),
    layers=f"close / {ts_mean('close', 20)} - 1",
    direction=-1,
    fields=("close",),
)

define(
    name="BOLL_POS_20",
    family="technical",
    formula="(close - (mid - 2*sd)) / (4*sd), mid=mean(close,20), sd=std(close,20)",
    description="布林带位置：0 贴着下轨、1 贴着上轨；上轨附近短期偏弱。",
    steps=(
        "算 20 日均线 mid 与 20 日收盘价标准差 sd",
        "上下轨取 mid ± 2sd",
        "(close - 下轨) ÷ (上轨 - 下轨) 得到当日价格在带内的位置：0 贴下轨、1 贴上轨",
    ),
    layers=f"(close - ({_boll_mid} - 2 * {_boll_std})) / nullif(4 * {_boll_std}, 0)",
    direction=-1,
    fields=("close",),
)
