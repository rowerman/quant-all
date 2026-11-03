"""因子读取层：从 data/factors 的 parquet 与因子目录库取数。

三个接口：

- :func:`load_factor_list`：因子清单与覆盖率，来自 ``data/databases/factors.duckdb``；
- :func:`load_factor_panel`：长表，返回主键 + 指定因子列；
- :func:`load_factor_wide`：单因子宽表（交易日 × 股票），便于截面运算。

读取时直接在 DuckDB 里 join 各族 parquet，不会把全量拉进 pandas；
所有族的行集都与 ``md.daily_hfq`` 一致，所以可以用任一族当主键骨架。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

import factors
from factors.registry import FactorSpec
from scripts.config import FACTOR_DB_FILE, FACTOR_DIR
from scripts.errors import FactorDataError

KEY_COLUMNS = ("stock_code", "trade_date")


def _family_glob(family: str, factor_dir: Path | str) -> str:
    """返回某个族的 parquet 通配路径。

    参数:
        family: 因子族。
        factor_dir: 因子数据根目录。
    """
    return str(Path(factor_dir) / family / "year=*" / "part.parquet")


def built_families(factor_dir: Path | str = FACTOR_DIR) -> tuple[str, ...]:
    """列出磁盘上已经落盘的因子族。

    参数:
        factor_dir: 因子数据根目录。
    """
    root = Path(factor_dir)
    if not root.exists():
        return ()
    return tuple(
        family for family in factors.ALL_FAMILIES
        if list((root / family).glob("year=*/part.parquet"))
    )


def built_names(family: str, factor_dir: Path | str = FACTOR_DIR) -> tuple[str, ...]:
    """列出某个族在磁盘上实际有数据的因子列。

    参数:
        family: 因子族。
        factor_dir: 因子数据根目录。
    """
    files = sorted((Path(factor_dir) / family).glob("year=*/part.parquet"))
    if not files:
        raise FactorDataError(
            f"因子族 {family} 还没有落盘数据；已构建的族：{built_families(factor_dir) or '（无）'}，"
            "先跑 python -m scripts.build_factors.build"
        )
    columns = duckdb.sql(f"SELECT * FROM read_parquet('{files[-1]}') LIMIT 0").columns
    return tuple(name for name in columns if name not in (*KEY_COLUMNS, "year"))


def _resolve(names, families, factor_dir) -> dict[str, tuple[FactorSpec, ...]]:
    """把请求的因子名或族解析成 ``{族: 因子定义}``。

    参数:
        names: 因子名序列；None 表示不按名字筛。
        families: 因子族序列；None 表示不按族筛。
        factor_dir: 因子数据根目录。
    """
    available = built_families(factor_dir)
    if not available:
        raise FactorDataError(
            f"{Path(factor_dir)} 下还没有因子数据；先跑 python -m scripts.build_factors.build"
        )

    if families:
        unknown_family = [
            family for family in families if family not in factors.ALL_FAMILIES
        ]
        if unknown_family:
            raise FactorDataError(
                f"未知因子族 {unknown_family}；可选：{list(factors.FAMILIES)}"
            )
        missing = [family for family in families if family not in available]
        if missing:
            raise FactorDataError(f"因子族 {missing} 还没有落盘数据；已构建：{list(available)}")
        chosen = list(families)
    else:
        chosen = list(available)

    resolved: dict[str, tuple[FactorSpec, ...]] = {}
    if names:
        grouped: dict[str, list[FactorSpec]] = {}
        for name in names:
            spec = factors.resolve(name, factor_dir)
            if spec.family not in chosen:
                if spec.family not in available:
                    raise FactorDataError(
                        f"因子 {name} 属于 {spec.family} 族，但该族还没有落盘数据；"
                        f"已构建：{list(available)}"
                    )
                chosen.append(spec.family)
            grouped.setdefault(spec.family, []).append(spec)
        resolved = {family: tuple(specs) for family, specs in grouped.items()}
    else:
        resolved = {}
        for family in chosen:
            if family == "mined":
                specs = tuple(
                    item
                    for item in factors.available_specs(factor_dir)
                    if item.family == family
                )
            else:
                specs = factors.family_factors(family)
            resolved[family] = specs

    for family, specs in resolved.items():
        present = set(built_names(family, factor_dir))
        missing_columns = [spec.name for spec in specs if spec.name not in present]
        if missing_columns:
            raise FactorDataError(
                f"因子 {missing_columns} 在 {family} 族的落盘文件里不存在；"
                f"该族实际有：{sorted(present)}"
            )
    return resolved


def load_factor_list(family: str | None = None, status: str | None = None,
                     db_path: Path | str | None = None) -> pd.DataFrame:
    """读取因子清单（定义 + 覆盖率）。

    参数:
        family: 只看某个族。
        status: ``implemented`` 只看已实现，``planned`` 只看待补齐。
        db_path: 因子目录库路径，默认 data/databases/factors.duckdb。

    返回:
        因子名、族、状态、公式（formula）、白话步骤（steps，多行用换行分隔）、
        实现表达式（expr）、方向、回看窗口与最新年份覆盖率的 DataFrame。
    """
    target = Path(db_path or FACTOR_DB_FILE)
    if not target.exists():
        raise FileNotFoundError(
            f"因子目录库不存在: {target}；先跑 python -m scripts.build_factors.build"
        )
    conditions = []
    params: list = []
    if family:
        conditions.append("d.family = ?")
        params.append(family)
    if status:
        conditions.append("d.status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
    SELECT d.name, d.family, d.family_label, d.status, d.direction, d.lookback,
           d.formula, d.description, d.steps, d.expr,
           max(c.year) AS last_year,
           avg(c.coverage) AS avg_coverage,
           sum(c.non_null) AS non_null_rows
    FROM meta.factor_definition d
    LEFT JOIN meta.factor_coverage c ON c.name = d.name
    {where}
    GROUP BY d.name, d.family, d.family_label, d.status, d.direction, d.lookback,
             d.formula, d.description, d.steps, d.expr
    ORDER BY d.family, d.status, d.name
    """
    with duckdb.connect(str(target), read_only=True) as conn:
        return conn.execute(sql, params).df()


def load_factor_panel(start: str | None = None, end: str | None = None,
                      names=None, families=None, stocks=None,
                      factor_dir: Path | str | None = None) -> pd.DataFrame:
    """读取因子面板（长表）。

    参数:
        start: 起始交易日，闭区间。
        end: 截止交易日，闭区间。
        names: 需要哪些因子；None 表示按 ``families`` 取整族。
        families: 需要哪些族；``names`` 与 ``families`` 都为空时取全部已构建的族。
        stocks: 股票池；None 表示全部。
        factor_dir: 因子数据根目录，默认 data/factors。

    返回:
        长表，列为主键 ``stock_code``、``trade_date`` 加请求的因子列
        （传了 ``names`` 就按传入顺序，否则按族分组），按 ``trade_date``、``stock_code`` 排序。
    """
    root = Path(factor_dir or FACTOR_DIR)
    resolved = _resolve(names, families, root)

    ordered = [family for family in factors.ALL_FAMILIES if family in resolved]
    spine_family = ordered[0]
    selections = [f"r_{spine_family}.stock_code", f"r_{spine_family}.trade_date"]
    ctes = []
    for family in ordered:
        specs = resolved[family]
        columns = ", ".join(spec.name for spec in specs)
        ctes.append(
            f"r_{family} AS (SELECT stock_code, trade_date, {columns} "
            f"FROM read_parquet('{_family_glob(family, root)}', hive_partitioning = 1))"
        )
        selections.extend(f"r_{family}.{spec.name} AS {spec.name}" for spec in specs)

    conditions = []
    params: list = []
    if start:
        conditions.append(f"r_{spine_family}.trade_date >= DATE '{start}'")
    if end:
        conditions.append(f"r_{spine_family}.trade_date <= DATE '{end}'")
    if stocks:
        placeholders = ", ".join(["?"] * len(stocks))
        conditions.append(f"r_{spine_family}.stock_code IN ({placeholders})")
        params.extend(stocks)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    joins = " ".join(
        f"LEFT JOIN r_{family} USING (stock_code, trade_date)"
        for family in ordered if family != spine_family
    )
    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        f"SELECT {', '.join(selections)}\n"
        f"FROM r_{spine_family}\n{joins}{where}\n"
        "ORDER BY r_" + spine_family + ".trade_date, r_" + spine_family + ".stock_code"
    )
    with duckdb.connect() as conn:
        frame = conn.execute(sql, params).df()
    if names:
        # 按调用方给的顺序出列，而不是按族分组后的顺序
        wanted = list(dict.fromkeys(names))
        frame = frame[["stock_code", "trade_date", *wanted]]
    return frame


def load_factor_wide(name: str, start: str | None = None, end: str | None = None,
                     factor_dir: Path | str | None = None) -> pd.DataFrame:
    """把单个因子透视成「交易日 × 股票」宽表。

    参数:
        name: 因子名。
        start: 起始交易日。
        end: 截止交易日。
        factor_dir: 因子数据根目录，默认 data/factors。
    """
    frame = load_factor_panel(start=start, end=end, names=[name], factor_dir=factor_dir)
    return frame.pivot(index="trade_date", columns="stock_code", values=name)
