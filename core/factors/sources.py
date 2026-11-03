"""因子依赖的附加数据源。

因子表达式里引用的外部列（指数收益等）在这里定义。构建脚本会把这些 SQL
包成 CTE，并按 ``trade_date`` 左连接到日线基表上；新增数据源只要往
``SOURCES`` 里加一条，不需要改构建脚本。

SQL 里的 ``{prefix}`` 由调用方替换成表前缀：构建时传 ``src.``（挂载的主库），
手工复现时传空串（直接连主库）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    """一个附加数据源。

    参数:
        key: 数据源标识，因子定义里用这个名字引用。
        description: 一句话说明。
        columns: 该数据源产出的列名，因子表达式只能用这些列。
        sql: 产出 ``trade_date`` 与 ``columns`` 的查询，含 ``{prefix}`` 占位符。
        join_kind: ``date`` 按交易日等值连接；``asof`` 按公告日做 ASOF 连接。
        asof_date_column: ASOF 连接使用的右表公告日列。
    """

    key: str
    description: str
    columns: tuple[str, ...]
    sql: str
    join_kind: str = "date"
    asof_date_column: str = "ann_date"


SOURCES: dict[str, SourceSpec] = {
    "index_000300": SourceSpec(
        key="index_000300",
        description="沪深300 指数日收益，2002-01-04 起",
        columns=("idx_ret_300",),
        sql=(
            "SELECT trade_date, pct_chg / 100.0 AS idx_ret_300 "
            "FROM {prefix}md.index_daily WHERE index_code = '000300.SH'"
        ),
    ),
    "index_000905": SourceSpec(
        key="index_000905",
        description="中证500 指数日收益，2004-12-31 起",
        columns=("idx_ret_500",),
        sql=(
            "SELECT trade_date, pct_chg / 100.0 AS idx_ret_500 "
            "FROM {prefix}md.index_daily WHERE index_code = '000905.SH'"
        ),
    ),
    "index_000985": SourceSpec(
        key="index_000985",
        description="中证全指日收益，2004-12-31 起，做市场基准时样本更全",
        columns=("idx_ret_all",),
        sql=(
            "SELECT trade_date, pct_chg / 100.0 AS idx_ret_all "
            "FROM {prefix}md.index_daily WHERE index_code = '000985.CSI'"
        ),
    ),
    "fin_income": SourceSpec(
        key="fin_income",
        description="利润表核心指标，公告日次一交易日起可用",
        columns=(
            "inc_total_revenue", "inc_revenue", "inc_oper_cost", "inc_operate_profit",
            "inc_n_income", "inc_n_income_attr_p", "inc_rd_exp",
        ),
        sql=(
            "SELECT stock_code, available_date AS ann_date, report_date, "
            "inc_total_revenue, inc_revenue, inc_oper_cost, inc_operate_profit, "
            "inc_n_income, inc_n_income_attr_p, inc_rd_exp "
            "FROM (SELECT stock_code, available_date, report_date, "
            "total_revenue AS inc_total_revenue, revenue AS inc_revenue, "
            "oper_cost AS inc_oper_cost, operate_profit AS inc_operate_profit, "
            "n_income AS inc_n_income, n_income_attr_p AS inc_n_income_attr_p, "
            "rd_exp AS inc_rd_exp, "
            "row_number() OVER (PARTITION BY stock_code, available_date "
            "ORDER BY report_date DESC, update_flag DESC) AS rn "
            "FROM {prefix}fin.income_statement WHERE available_date IS NOT NULL) "
            "WHERE rn = 1"
        ),
        join_kind="asof",
    ),
    "fin_balance": SourceSpec(
        key="fin_balance",
        description="资产负债表核心指标，公告日次一交易日起可用",
        columns=(
            "bs_total_assets", "bs_total_liab", "bs_total_hldr_eqy_exc_min_int",
            "bs_total_cur_assets", "bs_total_cur_liab", "bs_money_cap",
            "bs_accounts_receiv", "bs_inventories", "bs_goodwill", "bs_total_share",
        ),
        sql=(
            "SELECT stock_code, available_date AS ann_date, report_date, "
            "bs_total_assets, bs_total_liab, bs_total_hldr_eqy_exc_min_int, "
            "bs_total_cur_assets, bs_total_cur_liab, bs_money_cap, "
            "bs_accounts_receiv, bs_inventories, bs_goodwill, bs_total_share "
            "FROM (SELECT stock_code, available_date, report_date, "
            "total_assets AS bs_total_assets, total_liab AS bs_total_liab, "
            "total_hldr_eqy_exc_min_int AS bs_total_hldr_eqy_exc_min_int, "
            "total_cur_assets AS bs_total_cur_assets, "
            "total_cur_liab AS bs_total_cur_liab, money_cap AS bs_money_cap, "
            "accounts_receiv AS bs_accounts_receiv, inventories AS bs_inventories, "
            "goodwill AS bs_goodwill, total_share AS bs_total_share, "
            "row_number() OVER (PARTITION BY stock_code, available_date "
            "ORDER BY report_date DESC, update_flag DESC) AS rn "
            "FROM {prefix}fin.balance_sheet WHERE available_date IS NOT NULL) "
            "WHERE rn = 1"
        ),
        join_kind="asof",
    ),
    "fin_cashflow": SourceSpec(
        key="fin_cashflow",
        description="现金流量表核心指标，公告日次一交易日起可用",
        columns=(
            "cf_n_cashflow_act", "cf_n_cashflow_inv_act", "cf_n_cash_flows_fnc_act",
            "cf_c_pay_acq_const_fiolta", "cf_free_cashflow", "cf_depr_fa_coga_dpba",
            "cf_n_incr_cash_cash_equ",
        ),
        sql=(
            "SELECT stock_code, available_date AS ann_date, report_date, "
            "cf_n_cashflow_act, cf_n_cashflow_inv_act, cf_n_cash_flows_fnc_act, "
            "cf_c_pay_acq_const_fiolta, cf_free_cashflow, cf_depr_fa_coga_dpba, "
            "cf_n_incr_cash_cash_equ "
            "FROM (SELECT stock_code, available_date, report_date, "
            "n_cashflow_act AS cf_n_cashflow_act, "
            "n_cashflow_inv_act AS cf_n_cashflow_inv_act, "
            "n_cash_flows_fnc_act AS cf_n_cash_flows_fnc_act, "
            "c_pay_acq_const_fiolta AS cf_c_pay_acq_const_fiolta, "
            "free_cashflow AS cf_free_cashflow, "
            "depr_fa_coga_dpba AS cf_depr_fa_coga_dpba, "
            "n_incr_cash_cash_equ AS cf_n_incr_cash_cash_equ, "
            "row_number() OVER (PARTITION BY stock_code, available_date "
            "ORDER BY report_date DESC, update_flag DESC) AS rn "
            "FROM {prefix}fin.cash_flow WHERE available_date IS NOT NULL) "
            "WHERE rn = 1"
        ),
        join_kind="asof",
    ),
    "fin_indicator": SourceSpec(
        key="fin_indicator",
        description="财务指标核心字段，公告日次一交易日起可用",
        columns=(
            "fi_profit_dedt", "fi_roe", "fi_roe_waa", "fi_roe_dt", "fi_roe_yearly",
            "fi_roa", "fi_roa_yearly", "fi_roa2_yearly", "fi_roic",
            "fi_grossprofit_margin", "fi_netprofit_margin", "fi_debt_to_assets",
            "fi_current_ratio", "fi_quick_ratio", "fi_cash_ratio", "fi_assets_turn",
            "fi_or_yoy", "fi_tr_yoy", "fi_netprofit_yoy", "fi_op_yoy", "fi_ocf_yoy",
            "fi_assets_yoy", "fi_eqt_yoy", "fi_q_sales_yoy", "fi_q_op_qoq",
        ),
        sql=(
            "SELECT stock_code, available_date AS ann_date, report_date, "
            "fi_profit_dedt, fi_roe, fi_roe_waa, fi_roe_dt, fi_roe_yearly, "
            "fi_roa, fi_roa_yearly, fi_roa2_yearly, fi_roic, "
            "fi_grossprofit_margin, fi_netprofit_margin, fi_debt_to_assets, "
            "fi_current_ratio, fi_quick_ratio, fi_cash_ratio, fi_assets_turn, "
            "fi_or_yoy, fi_tr_yoy, fi_netprofit_yoy, fi_op_yoy, fi_ocf_yoy, "
            "fi_assets_yoy, fi_eqt_yoy, fi_q_sales_yoy, fi_q_op_qoq "
            "FROM (SELECT stock_code, available_date, report_date, "
            "profit_dedt AS fi_profit_dedt, roe AS fi_roe, roe_waa AS fi_roe_waa, "
            "roe_dt AS fi_roe_dt, roe_yearly AS fi_roe_yearly, roa AS fi_roa, "
            "roa_yearly AS fi_roa_yearly, roa2_yearly AS fi_roa2_yearly, "
            "roic AS fi_roic, grossprofit_margin AS fi_grossprofit_margin, "
            "netprofit_margin AS fi_netprofit_margin, "
            "debt_to_assets AS fi_debt_to_assets, current_ratio AS fi_current_ratio, "
            "quick_ratio AS fi_quick_ratio, cash_ratio AS fi_cash_ratio, "
            "assets_turn AS fi_assets_turn, or_yoy AS fi_or_yoy, "
            "tr_yoy AS fi_tr_yoy, netprofit_yoy AS fi_netprofit_yoy, "
            "op_yoy AS fi_op_yoy, ocf_yoy AS fi_ocf_yoy, "
            "assets_yoy AS fi_assets_yoy, eqt_yoy AS fi_eqt_yoy, "
            "q_sales_yoy AS fi_q_sales_yoy, q_op_qoq AS fi_q_op_qoq, "
            "row_number() OVER (PARTITION BY stock_code, available_date "
            "ORDER BY report_date DESC, update_flag DESC) AS rn "
            "FROM {prefix}fin.fina_indicator WHERE available_date IS NOT NULL) "
            "WHERE rn = 1"
        ),
        join_kind="asof",
    ),
}


def join_clause(key: str, base_alias: str = "base") -> str:
    """生成附加数据源与日线基表的连接 SQL。

    参数:
        key: 数据源标识。
        base_alias: 日线基表别名。
    """
    spec = SOURCES[key]
    if spec.join_kind == "asof":
        return (
            f"ASOF LEFT JOIN src_{key} ON {base_alias}.stock_code = src_{key}.stock_code "
            f"AND {base_alias}.trade_date > src_{key}.{spec.asof_date_column}"
        )
    return f"LEFT JOIN src_{key} ON src_{key}.trade_date = {base_alias}.trade_date"


def source_columns(keys) -> tuple[str, ...]:
    """把数据源标识展开成列名。

    参数:
        keys: 数据源标识序列。
    """
    columns: list[str] = []
    for key in keys:
        if key not in SOURCES:
            raise KeyError(f"未知数据源 {key!r}，可选：{sorted(SOURCES)}")
        columns.extend(SOURCES[key].columns)
    return tuple(columns)
