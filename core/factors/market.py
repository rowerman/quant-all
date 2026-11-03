"""市场与相对强度因子。

基准指数取沪深300（2002-01-04 起）；2002 年之前这些因子为空值。
"""

from factors.ops import ts_count, ts_slope, ts_sum
from factors.registry import define

# 回归类因子的最少观测数：60 日窗口里至少要有 40 个有效交易日，
# 否则次新股开板后的两三个点会算出上千的伪贝塔。
_MIN_OBS_60 = ts_count("ret", 60)

define(
    name="RS_20",
    family="market",
    formula="sum(ret, 20) - sum(idx_ret, 20)",
    description="20 日相对沪深300 的超额收益，相对强度因子。",
    steps=(
        "算个股最近 20 个交易日的累计收益",
        "算沪深300 同期累计收益",
        "两者相减，得到 20 日超额收益",
    ),
    layers=f"{ts_sum('ret', 20)} - {ts_sum('idx_ret_300', 20)}",
    direction=1,
    fields=("ret",),
    extra_sources=("index_000300",),
)

define(
    name="BETA_60",
    family="market",
    formula="regr_slope(ret, idx_ret, 60)",
    description="60 日市场贝塔，低贝塔异象：贝塔低的股票风险调整后收益更好；窗口内少于 40 个交易日则置空。",
    steps=(
        "取最近 60 个交易日的个股日收益与沪深300 日收益",
        "对两者做一元回归，取斜率即为贝塔",
        "窗口内有效交易日不足 40 天时置空，避免次新股算出上千的伪贝塔",
    ),
    layers=f"case when {_MIN_OBS_60} >= 40 then {ts_slope('ret', 'idx_ret_300', 60)} else null end",
    direction=-1,
    fields=("ret",),
    extra_sources=("index_000300",),
)
