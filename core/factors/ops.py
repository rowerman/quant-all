"""因子算子库：把常用时序与截面算子拼成 DuckDB SQL 表达式。

约定：

1. 时序窗口一律按 ``stock_code`` 分组、``trade_date`` 排序，且只覆盖有效交易日
   （构建时已先把 ``status='trading'`` 的行筛出来），所以停牌自然不占窗口；
2. 命名窗口 ``w`` 由构建脚本在每个因子 CTE 里定义，取值见 ``WINDOW_DEF``；
3. 窗口长度按行数计，历史不足时结果为空值，不做填充。

算子返回的是 SQL 片段，可以互相嵌套拼接；但不能把两个窗口函数嵌在一起
（DuckDB 不允许），这种情况用因子定义的多层写法。
"""

from __future__ import annotations

WINDOW = "w"
WINDOW_DEF = "(partition by stock_code order by trade_date)"


def frame(n: int) -> str:
    """时序窗口：当前行与之前 n-1 行。

    参数:
        n: 窗口长度（交易日数）。
    """
    if n < 1:
        raise ValueError(f"窗口长度必须大于 0，收到 {n}")
    return f"(partition by stock_code order by trade_date rows between {n - 1} preceding and current row)"


def cs_frame() -> str:
    """截面窗口：同一个交易日的全部股票。"""
    return "(partition by trade_date)"


def ts_delay(col: str, n: int) -> str:
    """n 个交易日前的取值。"""
    if n < 1:
        raise ValueError(f"滞后阶数必须大于 0，收到 {n}")
    return f"lag({col}, {n}) over {WINDOW}"


def ts_delta(col: str, n: int) -> str:
    """与 n 个交易日前的差。"""
    return f"({col} - {ts_delay(col, n)})"


def ts_mean(col: str, n: int) -> str:
    """n 日均值。"""
    return f"avg({col}) over {frame(n)}"


def ts_sum(col: str, n: int) -> str:
    """n 日求和。"""
    return f"sum({col}) over {frame(n)}"


def ts_min(col: str, n: int) -> str:
    """n 日最小值。"""
    return f"min({col}) over {frame(n)}"


def ts_max(col: str, n: int) -> str:
    """n 日最大值。"""
    return f"max({col}) over {frame(n)}"


def ts_median(col: str, n: int) -> str:
    """n 日中位数。"""
    return f"median({col}) over {frame(n)}"


def ts_std(col: str, n: int, sample: bool = True) -> str:
    """n 日标准差。

    参数:
        col: 取值列。
        n: 窗口长度。
        sample: 真用样本标准差，假用总体标准差。
    """
    func = "stddev_samp" if sample else "stddev_pop"
    return f"{func}({col}) over {frame(n)}"


def ts_skew(col: str, n: int) -> str:
    """n 日偏度。"""
    return f"skewness({col}) over {frame(n)}"


def ts_kurt(col: str, n: int) -> str:
    """n 日峰度。"""
    return f"kurtosis({col}) over {frame(n)}"


def ts_product(col: str, n: int) -> str:
    """n 日累积乘积。"""
    return f"product({col}) over {frame(n)}"


def ts_count(col: str, n: int) -> str:
    """n 日内非空取值个数。"""
    return f"count({col}) over {frame(n)}"


def ts_corr(a: str, b: str, n: int) -> str:
    """n 日相关系数。"""
    return f"corr({a}, {b}) over {frame(n)}"


def ts_cov(a: str, b: str, n: int, sample: bool = True) -> str:
    """n 日协方差。"""
    func = "covar_samp" if sample else "covar_pop"
    return f"{func}({a}, {b}) over {frame(n)}"


def ts_slope(y: str, x: str, n: int) -> str:
    """n 日一元回归斜率（y 对 x）。"""
    return f"regr_slope({y}, {x}) over {frame(n)}"


def ts_r2(y: str, x: str, n: int) -> str:
    """n 日一元回归决定系数。"""
    return f"regr_r2({y}, {x}) over {frame(n)}"


def ts_rank(col: str, n: int) -> str:
    """当前取值在最近 n 个交易日中的分位（0~1）。

    DuckDB 的 ``percent_rank`` 按 SQL 标准忽略窗口帧，直接加 ``rows between``
    会退化成「全历史分位」，所以这里用 n-1 次 ``lag`` 比较来数：
    分子是最近 n-1 个交易日里小于等于当前值的个数，并列计在一起，
    前值缺失按不计入处理。

    参数:
        col: 取值列。
        n: 窗口长度，至少为 2。
    """
    if n < 2:
        raise ValueError(f"ts_rank 的窗口长度至少为 2，收到 {n}")
    parts = [
        f"coalesce(cast({col} >= {ts_delay(col, step)} as integer), 0)"
        for step in range(1, n)
    ]
    return "(" + " + ".join(parts) + f") / {n - 1}.0"


def ts_std_downside(col: str, n: int) -> str:
    """下行波动：只把负值计入波动。"""
    return f"sqrt(avg(power(least({col}, 0), 2)) over {frame(n)})"


def ts_row_number() -> str:
    """窗口内的行号，用于需要「第几行」的多层因子。"""
    return f"row_number() over {WINDOW}"


def ts_arg_max(col: str, index_col: str, n: int) -> str:
    """最近 n 日最大值对应的是哪一个行号。

    参数:
        col: 比较大小的取值列。
        index_col: 行号列，由前一层用 ``ts_row_number`` 或 ``row_number() over w`` 产出。
        n: 窗口长度。

    DuckDB 的 ``arg_max(arg, val)`` 是「val 最大处返回 arg」，所以行号写在第一个参数。
    用法：第一层产出行号列，第二层写 ``ts_arg_max("v3", "v1", 5)``，
    第三层用「当前行号 - 该行号」得到最大值距今多少个交易日。
    """
    return f"arg_max({index_col}, {col}) over {frame(n)}"


def cs_rank(col: str) -> str:
    """截面分位（0~1），每个交易日一组。"""
    return f"percent_rank() over (partition by trade_date order by {col})"


def cs_mean(col: str) -> str:
    """截面均值。"""
    return f"avg({col}) over (partition by trade_date)"


def cs_std(col: str, sample: bool = False) -> str:
    """截面标准差。"""
    func = "stddev_samp" if sample else "stddev_pop"
    return f"{func}({col}) over (partition by trade_date)"


def cs_zscore(col: str) -> str:
    """截面标准化，标准差为 0 时返回空值。"""
    return f"({col} - {cs_mean(col)}) / nullif({cs_std(col)}, 0)"


def cs_demean(col: str) -> str:
    """截面去均值。"""
    return f"({col} - {cs_mean(col)})"


def cs_scale(col: str) -> str:
    """截面归一：除以当日绝对值和（Alpha101 里的 scale）。"""
    return f"{col} / nullif(sum(abs({col})) over (partition by trade_date), 0)"


def signed_power(col: str, power: float) -> str:
    """保号幂：sign(x) * |x| ** p。"""
    return f"sign({col}) * power(abs({col}), {power})"


def safe_div(a: str, b: str) -> str:
    """除法，分母为 0 时返回空值。"""
    return f"{a} / nullif({b}, 0)"


def neg(expr: str) -> str:
    """取负，用于把因子方向摊到明面上。"""
    return f"(-({expr}))"
