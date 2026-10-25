"""估值与规模因子。

估值列直接取自 Tushare ``daily_basic``（已入库）：pe_ttm 覆盖约八成行，
pb / ps_ttm 覆盖九成六；亏损股的 pe_ttm 为负，用 case 过滤掉而不是硬除。
"""

from factors.registry import define

define(
    name="EP_TTM",
    family="valuation",
    formula="1 / pe_ttm, 仅当 pe_ttm > 0",
    description="盈利收益率（市盈率倒数），价值因子；亏损股置空值而不是记成负的 EP。",
    steps=(
        "取滚动市盈率 pe_ttm",
        "只在 pe_ttm > 0（盈利）时取倒数 1 / pe_ttm",
        "亏损股置空，不记成负值",
    ),
    layers="case when pe_ttm > 0 then 1.0 / pe_ttm else null end",
    direction=1,
    fields=("pe_ttm",),
)

define(
    name="BP",
    family="valuation",
    formula="1 / pb, 仅当 pb > 0",
    description="账面市值比（市净率倒数），最经典的价值因子。",
    steps=(
        "取市净率 pb",
        "只在 pb > 0 时取倒数 1 / pb，得到账面市值比",
    ),
    layers="case when pb > 0 then 1.0 / pb else null end",
    direction=1,
    fields=("pb",),
)

define(
    name="LN_MV",
    family="valuation",
    formula="ln(total_mv)",
    description="总市值取对数，规模因子；A 股长期小盘溢价，但 2017 年后明显减弱。",
    steps=(
        "取总市值 total_mv（万元）",
        "取自然对数，压缩量纲，方便做截面比较",
    ),
    layers="ln(total_mv)",
    direction=-1,
    fields=("total_mv",),
)
