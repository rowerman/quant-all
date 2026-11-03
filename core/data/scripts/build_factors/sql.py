"""因子 SQL 生成：把因子定义拼成可执行的 DuckDB 查询。

约定：

1. ``spine`` 是输出行集，等于目标区间内 ``md.daily_hfq`` 的全部行（含停牌与退市股）；
2. ``base`` 只取有效交易日（``status = 'trading'``），内容是**截止日的全部历史**。
   所有因子表达式都建在这层上，所以：窗口按有效交易日推进，停牌不占窗口；
   任何日期的横截面都是完整的（截面算子不会被截断）；
   停牌大半年的股票也一定取得到自己的窗口。代价是逐年构建时每年要扫一遍历史，
   实测单族单年 1~3 秒；换来的是「因子值与怎么分块无关」这个性质。
3. 因子结果左连接到 ``spine``，停牌行的因子值因此为空值，与标签口径一致；
4. 一个因子一条 CTE 链，命名窗口 ``w`` 就在链里声明，跟主查询解耦。
"""

from __future__ import annotations

from pathlib import Path

from factors.ops import WINDOW_DEF
from factors.registry import FactorSpec
from factors.sources import SOURCES, join_clause

DEFAULT_DB_ALIAS = "src"


def date_literal(value: str) -> str:
    """把 ``2024-01-02`` 写成 SQL 日期字面量。

    参数:
        value: ISO 格式日期字符串。
    """
    return f"DATE '{value}'"


def family_file(factor_dir: Path | str, family: str, year: int) -> Path:
    """某个族某一年的落盘路径。

    参数:
        factor_dir: 因子数据根目录。
        family: 因子族。
        year: 年份。
    """
    return Path(factor_dir) / family / f"year={year}" / "part.parquet"


def family_glob(factor_dir: Path | str, family: str) -> str:
    """某个族全部年份的 parquet 通配路径。

    参数:
        factor_dir: 因子数据根目录。
        family: 因子族。
    """
    return str(Path(factor_dir) / family / "year=*" / "part.parquet")


def factor_ctes(spec: FactorSpec, source: str = "base") -> tuple[list[str], str]:
    """生成一个因子的 CTE 链。

    参数:
        spec: 因子定义。
        source: 链的输入关系名，默认 ``base``；用到附加数据源时是 ``enriched``。

    返回:
        ``(CTE 列表, 最后一层 CTE 的名字)``。中间层把新列命名为 ``v1``、``v2``…
        并保留上一层全部列，最后一层只输出主键与因子列，并转成 float32。
    """
    ctes: list[str] = []
    previous = source
    last = len(spec.layers)
    final = ""
    for index, layer in enumerate(spec.layers, start=1):
        name = f"f_{spec.name}_{index}"
        if index == last:
            # 常收益序列的 corr 会算出 NaN，除零可能算出 inf，落盘前统一置空
            body = (
                f"SELECT stock_code, trade_date, "
                f"CAST(CASE WHEN isfinite({layer}) THEN {layer} ELSE NULL END AS FLOAT) "
                f"AS {spec.name} "
                f"FROM {previous} WINDOW w AS {WINDOW_DEF}"
            )
            final = name
        else:
            body = f"SELECT *, {layer} AS v{index} FROM {previous} WINDOW w AS {WINDOW_DEF}"
        ctes.append(f"{name} AS ({body})")
        previous = name
    return ctes, final


def base_cte(spine_end: str, db: str = DEFAULT_DB_ALIAS) -> str:
    """构造因子计算的输入层。

    参数:
        spine_end: 目标区间最后一个交易日，也是因子计算的截止日。
        db: 主库在连接里的别名。

    返回:
        ``base AS (...)`` 的 CTE 文本：截止日之前的全部有效交易日。
    """
    # 直接喂「截止日之前的全部有效交易日」：任何日期的横截面都是完整的，
    # 停牌股的窗口也一定取得到，因子值因此与「不分块一次算」完全一致。
    # 代价是每年都要扫一遍历史，实测单族单年 1~3 秒。
    return (
        "base AS ("
        f" SELECT b.* FROM {db}.md.daily_hfq b"
        f" WHERE b.status = 'trading' AND b.trade_date <= {date_literal(spine_end)}"
        ")"
    )


def family_query(
    specs,
    *,
    spine_start: str,
    spine_end: str,
    db: str = DEFAULT_DB_ALIAS,
    include_row_id: bool = False,
) -> str:
    """把一族的因子拼成一条查询。

    参数:
        specs: 该族的因子定义序列。
        spine_start: 输出行集起始日（含）。
        spine_end: 输出行集截止日（含），也是因子计算的截止日。
        db: 主库在连接里的别名。
        include_row_id: 为正式发布保留稳定行号，便于写盘后用可信行情键回填。
    """
    specs = list(specs)
    if not specs:
        raise ValueError("family_query 至少需要一个因子定义")

    source_keys = sorted({key for spec in specs for key in spec.extra_sources})
    source_ctes = [
        f"src_{key} AS ({SOURCES[key].sql.format(prefix=f'{db}.')})" for key in source_keys
    ]
    source_select = "".join(
        f", src_{key}.{column}" for key in source_keys for column in SOURCES[key].columns
    )
    source_join = " ".join(join_clause(key) for key in source_keys)

    row_id = (
        ", row_number() OVER (ORDER BY stock_code, trade_date) AS row_id"
        if include_row_id
        else ""
    )
    ctes = [
        f"spine AS (SELECT stock_code, trade_date{row_id} FROM "
        f"{db}.md.daily_hfq WHERE trade_date BETWEEN {date_literal(spine_start)} "
        f"AND {date_literal(spine_end)})"
    ]
    ctes.extend(source_ctes)
    ctes.append(base_cte(spine_end, db))

    # 用到指数这类附加数据时，先按交易日挂到 base 上，再算因子
    relation = "base"
    if source_keys:
        ctes.append(f"enriched AS (SELECT base.*{source_select} FROM base {source_join})")
        relation = "enriched"

    finals: list[tuple[str, FactorSpec]] = []
    for spec in specs:
        spec_ctes, final = factor_ctes(spec, relation)
        ctes.extend(spec_ctes)
        finals.append((final, spec))

    columns = ["spine.stock_code", "spine.trade_date"]
    if include_row_id:
        columns.append("spine.row_id")
    columns.extend(f"{alias}.{spec.name}" for alias, spec in finals)
    joins = " ".join(f"LEFT JOIN {alias} USING (stock_code, trade_date)" for alias, _ in finals)
    return (
        "WITH " + ",\n".join(ctes) + "\n"
        f"SELECT {', '.join(columns)}\n"
        f"FROM spine\n{joins}\n"
        "ORDER BY spine.trade_date, spine.stock_code"
    )


def coverage_query(specs, factor_dir: Path | str, family: str) -> str:
    """生成一族因子的覆盖统计查询。

    参数:
        specs: 该族的因子定义序列。
        factor_dir: 因子数据根目录。
        family: 因子族。
    """
    glob = family_glob(factor_dir, family)
    parts = []
    for spec in specs:
        column = spec.name
        parts.append(
            f"SELECT '{column}' AS name, CAST(year AS INTEGER) AS year, count(*) AS rows, "
            f"count({column}) AS non_null, min({column}) AS min_value, "
            f"max({column}) AS max_value, avg({column}) AS mean_value, "
            f"stddev_samp({column}) AS std_value, quantile_cont({column}, 0.01) AS p01, "
            f"quantile_cont({column}, 0.99) AS p99 "
            f"FROM read_parquet('{glob}', hive_partitioning = 1) GROUP BY 1, 2"
        )
    return " UNION ALL ".join(parts)
