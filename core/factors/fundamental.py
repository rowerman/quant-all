"""财务与质量因子。

财务数据来自建库阶段的 ``fin.*`` 表，因子表达式通过 ASOF 连接按公告日对齐。
公告日次一交易日起才可用，避免收盘后公告被当天收盘信号提前使用。
"""

from factors.registry import define


def _pct(column: str, limit: float) -> str:
    """把 Tushare 百分数转小数，并过滤明显异常的极值。"""
    return (
        f"CASE WHEN isfinite({column}) AND abs({column}) <= {limit} "
        f"THEN {column} / 100.0 ELSE NULL END"
    )


def _frac(expression: str, low: float, high: float) -> str:
    """只保留闭区间内的有限数值。"""
    return (
        f"CASE WHEN isfinite({expression}) AND ({expression}) BETWEEN {low} AND {high} "
        f"THEN {expression} ELSE NULL END"
    )


_INDICATOR = ("fin_indicator",)

define(
    name="ROE_YEARLY",
    family="fundamental",
    formula="roe_yearly / 100",
    description="年化净资产收益率，衡量股东资金的使用效率。",
    steps=(
        "取财务指标中的年化 ROE",
        "把百分数除以 100 转成小数",
        "绝对值超过 500% 的异常值置空",
    ),
    layers=_pct("fi_roe_yearly", 500),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="ROA_YEARLY",
    family="fundamental",
    formula="roa_yearly / 100",
    description="年化总资产收益率，衡量全部资产的盈利效率。",
    steps=(
        "取财务指标中的年化 ROA",
        "把百分数除以 100 转成小数",
        "绝对值超过 500% 的异常值置空",
    ),
    layers=_pct("fi_roa_yearly", 500),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="ROIC",
    family="fundamental",
    formula="roic / 100",
    description="投入资本回报率，兼顾债权与股权资本的使用效率。",
    steps=(
        "取财务指标中的 ROIC",
        "把百分数除以 100 转成小数",
        "绝对值超过 500% 的异常值置空",
    ),
    layers=_pct("fi_roic", 500),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="GROSS_MARGIN",
    family="fundamental",
    formula="grossprofit_margin / 100",
    description="毛利率，反映产品定价能力与成本控制。",
    steps=(
        "取财务指标中的毛利率",
        "把百分数除以 100 转成小数",
        "绝对值超过 500% 的异常值置空",
    ),
    layers=_pct("fi_grossprofit_margin", 500),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="NET_MARGIN",
    family="fundamental",
    formula="netprofit_margin / 100",
    description="净利率，反映收入最终转化为净利润的比例。",
    steps=(
        "取财务指标中的净利率",
        "把百分数除以 100 转成小数",
        "绝对值超过 500% 的异常值置空",
    ),
    layers=_pct("fi_netprofit_margin", 500),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="REV_GROWTH",
    family="fundamental",
    formula="or_yoy / 100",
    description="营业收入同比增速，衡量业务扩张。",
    steps=(
        "取财务指标中的营业收入同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_or_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="NP_GROWTH",
    family="fundamental",
    formula="netprofit_yoy / 100",
    description="归母净利润同比增速，衡量盈利增长。",
    steps=(
        "取财务指标中的净利润同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_netprofit_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="OP_GROWTH",
    family="fundamental",
    formula="op_yoy / 100",
    description="营业利润同比增速，剔除营业外项目后的增长。",
    steps=(
        "取财务指标中的营业利润同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_op_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="OCF_GROWTH",
    family="fundamental",
    formula="ocf_yoy / 100",
    description="经营现金流同比增速，衡量现金创造能力的改善。",
    steps=(
        "取财务指标中的经营现金流同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_ocf_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="ASSET_GROWTH",
    family="fundamental",
    formula="assets_yoy / 100",
    description="总资产同比增速，衡量扩张速度。",
    steps=(
        "取财务指标中的总资产同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_assets_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="EQUITY_GROWTH",
    family="fundamental",
    formula="eqt_yoy / 100",
    description="净资产同比增速，衡量内生积累与扩张。",
    steps=(
        "取财务指标中的净资产同比增速",
        "把百分数除以 100 转成小数",
        "绝对值超过 1000% 的异常值置空",
    ),
    layers=_pct("fi_eqt_yoy", 1000),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="ACCRUAL",
    family="fundamental",
    formula="(净利润 - 经营现金流) / 总资产",
    description="应计利润占资产比例，越高说明利润的现金含量越差。",
    steps=(
        "取利润表净利润与现金流量表经营现金流净额",
        "两者相减后除以资产负债表总资产",
        "只保留 [-1, 1] 内的数值",
    ),
    layers=(
        "CASE WHEN bs_total_assets > 0 THEN "
        + _frac("(inc_n_income - cf_n_cashflow_act) / bs_total_assets", -1, 1)
        + " ELSE NULL END"
    ),
    direction=-1,
    extra_sources=("fin_income", "fin_balance", "fin_cashflow"),
)

define(
    name="OCF_TO_NP",
    family="fundamental",
    formula="经营现金流净额 / 净利润",
    description="净利润的现金含量，越高说明盈利质量越好。",
    steps=(
        "取现金流量表经营现金流净额与利润表净利润",
        "两者相除，只对正净利润计算",
        "只保留 [-10, 10] 内的数值",
    ),
    layers=(
        "CASE WHEN inc_n_income > 0 THEN "
        + _frac("cf_n_cashflow_act / inc_n_income", -10, 10)
        + " ELSE NULL END"
    ),
    direction=1,
    extra_sources=("fin_income", "fin_cashflow"),
)

define(
    name="ASSET_TURNOVER",
    family="fundamental",
    formula="assets_turn",
    description="总资产周转率，衡量资产产生收入的能力。",
    steps=(
        "取财务指标中的总资产周转率",
        "只保留 [0, 10] 内的数值",
    ),
    layers=_frac("fi_assets_turn", 0, 10),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="CURRENT_RATIO",
    family="fundamental",
    formula="current_ratio",
    description="流动比率，衡量短期偿债能力。",
    steps=(
        "取财务指标中的流动比率",
        "只保留 [0, 20] 内的数值",
    ),
    layers=_frac("fi_current_ratio", 0, 20),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="QUICK_RATIO",
    family="fundamental",
    formula="quick_ratio",
    description="速动比率，剔除存货后的短期偿债能力。",
    steps=(
        "取财务指标中的速动比率",
        "只保留 [0, 20] 内的数值",
    ),
    layers=_frac("fi_quick_ratio", 0, 20),
    direction=1,
    extra_sources=_INDICATOR,
)

define(
    name="DEBT_TO_ASSETS",
    family="fundamental",
    formula="debt_to_assets / 100",
    description="资产负债率，越高通常财务风险越大。",
    steps=(
        "取财务指标中的资产负债率",
        "把百分数除以 100 转成小数",
        "只保留 [0, 2] 内的数值",
    ),
    layers=_frac("fi_debt_to_assets / 100.0", 0, 2),
    direction=-1,
    extra_sources=_INDICATOR,
)

define(
    name="FCF_YIELD",
    family="fundamental",
    formula="(经营现金流 - 资本开支) / 流通市值",
    description="自由现金流收益率，衡量现金回报相对市值的吸引力。",
    steps=(
        "取经营现金流净额与购建长期资产支付的现金",
        "两者相减得到自由现金流",
        "除以流通市值（万元转元），只保留 [-1, 1] 内的数值",
    ),
    layers=(
        "CASE WHEN circ_mv > 0 THEN "
        + _frac(
            "(cf_n_cashflow_act - cf_c_pay_acq_const_fiolta) / (circ_mv * 10000.0)",
            -1,
            1,
        )
        + " ELSE NULL END"
    ),
    direction=1,
    fields=("circ_mv",),
    extra_sources=("fin_cashflow",),
)
