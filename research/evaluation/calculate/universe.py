"""股票池：全 A 可交易池与指数成分池。

两个池子共用同一套可交易性过滤（可买入收盘 + 非 ST + 上市满 60 个自然日），
口径与 ``scripts/build_factors/validate.py`` 的 ``POOL_FILTER`` 保持一致，
否则评估结论会与既有自检对不上。

指数成分用**上一个已结束月份**的月末权重快照，避免用到当月才披露的权重造成前视。
"""

from __future__ import annotations

from dataclasses import dataclass

from calculate.errors import UniverseError

# 可交易性过滤：与 validate.py 的 POOL_FILTER 同口径，SQL 里日线表别名为 b
MIN_LISTED_DAYS = 60
TRADABLE_FILTER = (
    "b.can_buy_close AND NOT b.is_st "
    f"AND datediff('day', b.list_date, b.trade_date) >= {MIN_LISTED_DAYS}"
)

ALL_A_KEY = "all_a"
INDEX_PREFIX = "index:"

# 库里 ref.index_weight 覆盖的经典指数，键为 Tushare 指数代码
INDEX_UNIVERSES: tuple[tuple[str, str], ...] = (
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
    ("000852.SH", "中证1000"),
    ("000016.SH", "上证50"),
    ("000906.SH", "中证800"),
    ("000903.SH", "中证100"),
    ("000510.SH", "中证A500"),
    ("000688.SH", "科创50"),
    ("399001.SZ", "深证成指"),
    ("399006.SZ", "创业板指"),
    ("399303.SZ", "国证2000"),
    ("899050.BJ", "北证50"),
)


@dataclass(frozen=True)
class UniverseSpec:
    """一个股票池。

    参数:
        key: 标识，``all_a`` 或 ``index:<指数代码>``。
        label: 中文名。
        detail: 一句话口径说明，会展示到前端。
        index_code: 指数代码，全 A 池为空。
    """

    key: str
    label: str
    detail: str
    index_code: str | None = None

    @property
    def is_index(self) -> bool:
        """是否是指数成分池。"""
        return self.index_code is not None


ALL_A = UniverseSpec(
    key=ALL_A_KEY,
    label="全A可交易",
    detail=(
        f"可买入收盘、非 ST、上市满 {MIN_LISTED_DAYS} 个自然日；"
        "与因子自检 validate.py 的股票池口径一致"
    ),
)


def list_universes() -> tuple[UniverseSpec, ...]:
    """返回全部可选股票池，全 A 池排第一。"""
    specs = [ALL_A]
    for code, name in INDEX_UNIVERSES:
        specs.append(
            UniverseSpec(
                key=f"{INDEX_PREFIX}{code}",
                label=name,
                detail=f"{name}（{code}）成分，用上一个已结束月份的月末权重快照，叠加同一套可交易性过滤",
                index_code=code,
            )
        )
    return tuple(specs)


def resolve_universe(key: str) -> UniverseSpec:
    """按标识取股票池，非法标识直接报错并列出可选项。

    参数:
        key: 股票池标识。
    """
    for spec in list_universes():
        if spec.key == key:
            return spec
    raise UniverseError(
        f"未知股票池 {key!r}；可选：{[spec.key for spec in list_universes()]}"
    )


def membership_cte(spec: UniverseSpec, start: str, end: str) -> str:
    """生成指数成分的 CTE（逐交易日的成分集合）。

    参数:
        spec: 股票池定义，必须是指数池。
        start: 评估区间起始日。
        end: 评估区间截止日。

    返回:
        ``member AS (...)`` 的 CTE 文本，产出 ``trade_date`` 与 ``stock_code`` 两列。
    """
    if not spec.is_index:
        raise UniverseError(f"股票池 {spec.key} 不是指数成分池，没有成分数据")
    code = spec.index_code
    return (
        "month_end AS ("
        " SELECT date_trunc('month', trade_date) AS month_start, max(trade_date) AS snapshot_date"
        " FROM src.md.calendar"
        f" WHERE trade_date <= DATE '{end}' GROUP BY 1), "
        "effective AS ("
        " SELECT current.month_start AS month_start, prev.snapshot_date AS snapshot_date"
        " FROM month_end current"
        " JOIN month_end prev"
        " ON prev.month_start = CAST(current.month_start - INTERVAL 1 MONTH AS DATE)), "
        "member AS ("
        " SELECT days.trade_date AS trade_date, weights.stock_code AS stock_code"
        " FROM (SELECT trade_date, date_trunc('month', trade_date) AS month_start"
        f" FROM src.md.calendar WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}') days"
        " JOIN effective ON effective.month_start = days.month_start"
        " JOIN src.ref.index_weight weights"
        " ON weights.trade_date = effective.snapshot_date"
        f" AND weights.index_code = '{code}' AND weights.weight > 0"
        " GROUP BY 1, 2)"
    )
