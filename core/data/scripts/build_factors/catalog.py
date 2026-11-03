"""因子目录库：定义、覆盖率、自检结果、视图与文档生成。

目录库单独放在 ``data/databases/factors.duckdb``，不往 7.2GB 的主库里反复写入。
表结构：

=====================  ====================================================
表                     内容
=====================  ====================================================
factor_definition      因子名、族、公式、方向、回看、状态（已实现/待补齐）
factor_coverage        因子 × 年份的行数、非空数、覆盖率、极值、分位数
factor_check           自检结果：Rank IC / ICIR、前视检查、行集检查
factor_build           每次构建的运行记录
factor_manifest        因子库版本与规模
=====================  ====================================================
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb

import factors
from factors import ALL_FAMILIES, FAMILIES, FAMILY_LABELS, all_factors
from factors.published import published_specs
from factors.planned import PLANNED
from factors.registry import FactorRegistryError, FactorSpec
from factors.sources import SOURCES, join_clause
from scripts.build_factors.sql import factor_ctes, family_glob
from scripts.config import DOCS_DIR, FACTOR_DB_FILE, FACTOR_DIR
from scripts.logging_conf import get_logger

logger = get_logger("build_factors.catalog")

FACTOR_VERSION = "factors_v1"
CATALOG_DOC = DOCS_DIR / "core" / "factors" / "catalog.md"
FORMULA_DOC = DOCS_DIR / "core" / "factors" / "formulas.md"

READING_GUIDE = """## 怎么读这份文档

| 写法 | 含义 |
|---|---|
| `close_t` | 第 t 个交易日的收盘价（后复权） |
| `close_(t-20)` | 20 个交易日之前的收盘价 |
| `lag(close, 20)` | 同上，SQL 里的写法 |
| `mean(x, 20)` | 最近 20 个有效交易日的平均 |
| `std(x, 20)` | 最近 20 个有效交易日的样本标准差 |
| `corr(x, y, 20)` | 最近 20 日 x 与 y 的相关系数 |
| `rank(x)` | 同一天全部股票里 x 的排名分位（0~1） |
| `w` | 窗口定义：按 `stock_code` 分组、按 `trade_date` 排序 |

三条必须知道的口径：

1. 价格一律是**后复权价**，除权日不会出现假跳空；
2. 停牌日**不占窗口**，「20 个交易日」指 20 个真正有交易的日子，
   停牌行的因子值直接为空；
3. 历史不够时结果为空，不做填充。所以次新股的前 60 个交易日没有 60 日动量。

想自己动手算一遍、和库里的数对照，看 `cookbook.md`。"""


CATALOG_DDL = (
    "CREATE SCHEMA IF NOT EXISTS meta",
    "CREATE SCHEMA IF NOT EXISTS v",
    """
    CREATE TABLE IF NOT EXISTS meta.factor_definition (
        name VARCHAR PRIMARY KEY,
        family VARCHAR,
        family_label VARCHAR,
        status VARCHAR,
        formula VARCHAR,
        description VARCHAR,
        direction INTEGER,
        lookback INTEGER,
        layer_count INTEGER,
        fields VARCHAR,
        extra_sources VARCHAR,
        updated_at TIMESTAMP,
        steps VARCHAR,
        expr VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta.factor_coverage (
        name VARCHAR,
        family VARCHAR,
        year INTEGER,
        rows BIGINT,
        non_null BIGINT,
        coverage DOUBLE,
        min_value DOUBLE,
        max_value DOUBLE,
        mean_value DOUBLE,
        std_value DOUBLE,
        p01 DOUBLE,
        p99 DOUBLE,
        computed_at TIMESTAMP,
        PRIMARY KEY (name, year)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta.factor_check (
        name VARCHAR,
        kind VARCHAR,
        scope VARCHAR,
        value DOUBLE,
        detail VARCHAR,
        checked_at TIMESTAMP,
        PRIMARY KEY (name, kind, scope)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta.factor_build (
        run_id VARCHAR PRIMARY KEY,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        families VARCHAR,
        years VARCHAR,
        factor_count INTEGER,
        rows_written BIGINT,
        status VARCHAR,
        message VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta.factor_manifest (
        factor_version VARCHAR,
        families VARCHAR,
        factor_count INTEGER,
        rows BIGINT,
        built_at TIMESTAMP
    )
    """,
)


def connect(path: Path | str = FACTOR_DB_FILE, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """打开因子目录库。

    参数:
        path: 目录库路径，默认 data/databases/factors.duckdb。
        read_only: 只读打开。
    """
    target = Path(path)
    if read_only and not target.exists():
        raise FileNotFoundError(f"因子目录库不存在: {target}")
    return duckdb.connect(str(target), read_only=read_only)


# 老库升级用：新增列时补一句 ALTER，已有列会跳过
CATALOG_MIGRATIONS = (
    "ALTER TABLE meta.factor_definition ADD COLUMN IF NOT EXISTS steps VARCHAR",
    "ALTER TABLE meta.factor_definition ADD COLUMN IF NOT EXISTS expr VARCHAR",
)


def ensure_catalog(conn: duckdb.DuckDBPyConnection) -> None:
    """建好目录库的 schema 与表（含老库的列迁移）。

    参数:
        conn: 目录库连接。
    """
    for statement in CATALOG_DDL:
        conn.execute(statement)
    for statement in CATALOG_MIGRATIONS:
        conn.execute(statement)


def _spec_row(spec: FactorSpec, status: str, now: datetime) -> tuple:
    """把因子定义转成目录表的一行。

    参数:
        spec: 因子定义。
        status: implemented / planned。
        now: 写入时间。
    """
    return (
        spec.name,
        spec.family,
        FAMILY_LABELS.get(spec.family, spec.family),
        status,
        spec.formula,
        spec.description,
        spec.direction,
        spec.lookback,
        len(spec.layers),
        ",".join(spec.fields),
        ",".join(spec.extra_sources),
        now,
        "\n".join(spec.steps),
        "\n".join(spec.layers),
    )


def refresh_definitions(conn: duckdb.DuckDBPyConnection) -> int:
    """把已实现因子与规划因子一起写进 factor_definition。

    参数:
        conn: 目录库连接。
    """
    now = datetime.now()
    implemented = {spec.name: spec for spec in all_factors()}
    for spec in published_specs():
        if spec.name in implemented:
            raise FactorRegistryError(f"已发布因子与内置因子重名：{spec.name}")
        implemented[spec.name] = spec
    rows = [_spec_row(spec, "implemented", now) for spec in implemented.values()]
    for item in PLANNED:
        if item["name"] in implemented:
            continue
        rows.append((
            item["name"], item["family"], FAMILY_LABELS.get(item["family"], item["family"]),
            "planned", item["formula"], item["note"], 0, 0, 0, "", "", now, "", "",
        ))
    conn.execute("DELETE FROM meta.factor_definition")
    conn.executemany(
        "INSERT INTO meta.factor_definition VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    logger.info("factor_definition 写入 %d 条（已实现 %d）", len(rows), len(implemented))
    return len(rows)


def refresh_coverage(conn: duckdb.DuckDBPyConnection, rows) -> int:
    """写入因子覆盖率统计。

    参数:
        conn: 目录库连接。
        rows: ``(name, family, year, rows, non_null, min, max, mean, std, p01, p99)`` 序列。
    """
    if not rows:
        return 0
    now = datetime.now()
    payload = []
    for row in rows:
        coverage = (row["non_null"] / row["rows"]) if row["rows"] else None
        payload.append((
            row["name"], row["family"], row["year"], row["rows"], row["non_null"], coverage,
            row["min_value"], row["max_value"], row["mean_value"], row["std_value"],
            row["p01"], row["p99"], now,
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO meta.factor_coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        payload,
    )
    logger.info("factor_coverage 写入 %d 行", len(payload))
    return len(payload)


def refresh_checks(conn: duckdb.DuckDBPyConnection, rows) -> int:
    """写入自检结果。

    参数:
        conn: 目录库连接。
        rows: ``(name, kind, scope, value, detail)`` 序列。
    """
    if not rows:
        return 0
    now = datetime.now()
    payload = [(row["name"], row["kind"], row["scope"], row["value"], row["detail"], now) for row in rows]
    conn.executemany(
        "INSERT OR REPLACE INTO meta.factor_check VALUES (?, ?, ?, ?, ?, ?)", payload
    )
    logger.info("factor_check 写入 %d 行", len(payload))
    return len(payload)


def record_build(conn: duckdb.DuckDBPyConnection, record: dict) -> None:
    """记录一次构建运行。

    参数:
        conn: 目录库连接。
        record: 含 run_id / started_at / finished_at / families / years / factor_count /
            rows_written / status / message 的字典。
    """
    conn.execute(
        "INSERT OR REPLACE INTO meta.factor_build VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            record["run_id"], record["started_at"], record["finished_at"], record["families"],
            record["years"], record["factor_count"], record["rows_written"], record["status"],
            record["message"],
        ],
    )


def refresh_manifest(conn: duckdb.DuckDBPyConnection) -> None:
    """根据覆盖率表刷新因子库版本信息。

    参数:
        conn: 目录库连接。
    """
    conn.execute("DELETE FROM meta.factor_manifest")
    conn.execute(
        """
        INSERT INTO meta.factor_manifest
        SELECT
            ?,
            (SELECT string_agg(DISTINCT family, ',') FROM meta.factor_coverage),
            (SELECT count(DISTINCT name) FROM meta.factor_coverage),
            (SELECT sum(rows) FROM (
                SELECT DISTINCT family, year, rows FROM meta.factor_coverage
            ) s),
            now()
        """,
        [FACTOR_VERSION],
    )


def refresh_views(conn: duckdb.DuckDBPyConnection, factor_dir: Path | str = FACTOR_DIR) -> list[str]:
    """按族建视图，把 parquet 挂进 SQL。

    参数:
        conn: 目录库连接。
        factor_dir: 因子数据根目录。
    """
    created = []
    for family in ALL_FAMILIES:
        glob = family_glob(factor_dir, family)
        if not list((Path(factor_dir) / family).glob("year=*/part.parquet")):
            continue
        conn.execute(
            f"CREATE OR REPLACE VIEW v.{family} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = 1)"
        )
        created.append(family)
    logger.info("因子视图已刷新: %s", created)
    return created


def coverage_frame(conn: duckdb.DuckDBPyConnection):
    """读取因子定义与覆盖率的合并表，供文档与外部查看。"""
    return conn.execute(
        """
        SELECT d.name, d.family, d.family_label, d.status, d.formula, d.description,
               d.direction, d.lookback,
               max(c.year) AS last_year, sum(c.non_null) AS non_null_rows,
               avg(c.coverage) AS avg_coverage
        FROM meta.factor_definition d
        LEFT JOIN meta.factor_coverage c ON c.name = d.name
        GROUP BY ALL
        ORDER BY d.family, d.status, d.name
        """
    ).df()


def write_catalog_doc(path: Path | str = CATALOG_DOC) -> Path:
    """从注册表与规划清单生成因子清单文档。

    参数:
        path: 输出路径，默认 docs/core/factors/catalog.md。
    """
    implemented = {spec.name: spec for spec in all_factors()}
    planned = {item["name"]: item for item in PLANNED}
    names = list(dict.fromkeys(list(implemented) + [item["name"] for item in PLANNED]))

    lines = [
        "# 因子清单",
        "",
        "本文件由 `python -m scripts.build_factors.catalog --write-docs` 生成，不要手工编辑；",
        "改因子定义后重新生成即可。因子的唯一来源是 `factors/` 包里的注册表。",
        "",
        f"- 规划条目：{len(names)} 条",
        f"- 已实现：{len(implemented)} 条",
        f"- 覆盖因子族：{len(FAMILIES)} 个",
        "",
        "状态含义：**已实现** 表示已经注册并能参与构建；**待实现** 表示公式已列在这里，",
        "照着写一个 `define(...)` 就能纳入构建。",
        "",
    ]
    for family in FAMILIES:
        rows = []
        for name in names:
            spec = implemented.get(name)
            item = planned.get(name)
            if spec is not None and spec.family == family:
                rows.append((name, "已实现", spec.formula, spec.description, spec.direction))
            elif item is not None and item["family"] == family:
                rows.append((name, "待实现", item["formula"], item["note"], 0))
        if not rows:
            continue
        rows.sort(key=lambda row: (row[1] != "已实现", row[0]))
        lines.append(f"## {FAMILY_LABELS.get(family, family)}（`{family}`）")
        lines.append("")
        lines.append(f"共 {len(rows)} 条，其中已实现 {sum(1 for row in rows if row[1] == '已实现')} 条。")
        lines.append("")
        lines.append("| 因子 | 状态 | 方向 | 公式 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for name, status, formula, note, direction in rows:
            arrow = {1: "越大越看多", -1: "越大越看空", 0: ""}[direction]
            lines.append(
                f"| `{name}` | {status} | {arrow} | `{formula}` | {note.replace('|', '/')} |"
            )
        lines.append("")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("因子清单文档写入 %s（%d 条）", target, len(names))
    return target


def _factor_section(spec: FactorSpec, heading: str = "###") -> list[str]:
    """渲染一个因子的说明段落。

    参数:
        spec: 因子定义。
        heading: Markdown 标题级别。
    """
    arrow = "值越大越看多" if spec.direction > 0 else "值越大越看空"
    sources = f"；附加数据源：{', '.join(spec.extra_sources)}" if spec.extra_sources else ""
    lines = [
        f"{heading} `{spec.name}`",
        "",
        f"**说明**：{spec.description}",
        "",
        f"**方向**：{arrow}　**依赖字段**：`{', '.join(spec.fields)}`"
        f"　**回看窗口**：{spec.lookback} 个交易日　**层数**：{len(spec.layers)}{sources}",
        "",
        f"**公式**：`{spec.formula}`",
        "",
        "**计算步骤**",
        "",
    ]
    for index, step in enumerate(spec.steps, start=1):
        lines.append(f"{index}. {step}")
    lines.extend(["", "**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）", ""])
    if len(spec.layers) == 1:
        lines.extend(["```sql", spec.layers[0], "```"])
    else:
        lines.append("```sql")
        for index, layer in enumerate(spec.layers, start=1):
            lines.append(f"-- 第 {index} 层")
            lines.append(layer)
        lines.append("```")
    lines.append("")
    return lines


def explanation(spec: FactorSpec, stock: str = "SH600000",
                start: str = "2024-01-01", end: str = "2024-03-31") -> str:
    """渲染单个因子的完整讲解，含可以直接跑的手工复现 SQL。

    参数:
        spec: 因子定义。
        stock: 手工复现用的示例股票。
        start: 复现区间的起始日。
        end: 复现区间的截止日。
    """
    lines = _factor_section(spec, heading="#")
    lines.extend([
        "## 手工复现（把下面这段粘进 DuckDB 就能跑）",
        "",
        "```sql",
        reproduction_sql(spec, stock=stock, start=start, end=end),
        "```",
        "",
        "## 怎么跑这段 SQL",
        "",
        "- 命令行或 DBeaver：直接连 `data/databases/quant.duckdb` 执行上面那段；",
        "- Python：`from quant.db import get_conn`，然后 `with get_conn() as conn: conn.execute(sql).df()`；",
        "- 注意窗口是在**全部历史**上算的，最后才过滤出你要看的区间。"
        "把日期过滤写进窗口里面，月初那几行就会因为没有历史而变成空值。",
        "",
        "## 和库里的值对照",
        "",
        "```python",
        "import quant",
        "quant.load_factor_panel(",
        f'    start="{start}", end="{end}", names=["{spec.name}"], stocks=["{stock}"],',
        ")",
        "```",
        "",
    ])
    return "\n".join(lines)


def reproduction_sql(spec: FactorSpec, stock: str = "SH600000",
                     start: str = "2024-01-01", end: str = "2024-03-31") -> str:
    """生成一条可直接执行的手工复现 SQL。

    结构是「先在全部历史上算窗口，再过滤出你要看的区间」：
    如果把日期过滤写在窗口之前，月初那段就没有历史，长窗口因子会凭空变成空值。

    参数:
        spec: 因子定义。
        stock: 示例股票代码。
        start: 起始日（只影响最后过滤出来的行）。
        end: 截止日，同时也是「能看到的历史」的右端点。
    """
    window = "OVER (PARTITION BY stock_code ORDER BY trade_date)"
    expanded = [layer.replace("over w", window) for layer in spec.layers]

    source_keys = sorted(set(spec.extra_sources))
    source_ctes = [f"src_{key} AS ({SOURCES[key].sql.format(prefix='')})" for key in source_keys]
    source_columns = ", ".join(
        f"src_{key}.{column}" for key in source_keys for column in SOURCES[key].columns
    )
    source_join = " ".join(join_clause(key, base_alias="b") for key in source_keys)
    base = (
        "base AS (SELECT b.*" + (f", {source_columns}" if source_columns else "")
        + f" FROM md.daily_hfq b {source_join}".rstrip()
        + f" WHERE b.status = 'trading' AND b.trade_date <= DATE '{end}')"
    )

    if len(expanded) == 1:
        ctes = [
            *source_ctes,
            base,
            f"factor AS (SELECT stock_code, trade_date, {expanded[0]} AS {spec.name} FROM base)",
        ]
        final = "factor"
    else:
        chain, final = factor_ctes(spec, "base")
        ctes = [*source_ctes, base, *chain]

    return (
        "WITH " + ",\n".join(ctes) + "\n"
        f"SELECT stock_code, trade_date, {spec.name}\n"
        f"FROM {final}\n"
        f"WHERE stock_code = '{stock}' AND trade_date >= DATE '{start}'\n"
        "ORDER BY trade_date"
    )


def write_formula_doc(path: Path | str = FORMULA_DOC) -> Path:
    """生成「因子公式与计算步骤」文档。

    参数:
        path: 输出路径，默认 docs/core/factors/formulas.md。
    """
    implemented = {spec.family: [] for spec in all_factors()}
    for spec in all_factors():
        implemented[spec.family].append(spec)
    planned = {item["name"]: item for item in PLANNED}
    done_names = {spec.name for spec in all_factors()}

    lines = [
        "# 因子公式与计算步骤",
        "",
        "本文件由 `python -m scripts.build_factors.catalog --write-docs` 生成，不要手工编辑；",
        "因子的唯一来源是 `factors/` 包里的注册表。",
        "",
        f"已实现并已算出全历史数据的因子 {len(done_names)} 个（每个都给 公式 → 白话步骤 → 真实 SQL），",
        f"另有规划因子 {len(planned) - len(done_names)} 条，公式列在各族末尾的表里。",
        "",
        READING_GUIDE,
        "",
    ]
    for family in FAMILIES:
        specs = implemented.get(family, [])
        pending = [item for item in PLANNED
                   if item["family"] == family and item["name"] not in done_names]
        if not specs and not pending:
            continue
        lines.append(f"## {FAMILY_LABELS.get(family, family)}（`{family}`）")
        lines.append("")
        for spec in specs:
            lines.extend(_factor_section(spec))
        if pending:
            lines.append(f"### 待实现（{len(pending)} 条）")
            lines.append("")
            lines.append("| 因子 | 公式 | 实现要点 |")
            lines.append("|---|---|---|")
            for item in sorted(pending, key=lambda row: row["name"]):
                lines.append(f"| `{item['name']}` | `{item['formula']}` | {item['note']} |")
            lines.append("")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("因子公式文档写入 %s", target)
    return target


def main(argv=None) -> int:
    """目录库命令行入口。

    参数:
        argv: 命令行参数，默认取 sys.argv。
    """
    parser = argparse.ArgumentParser(description="因子目录库：刷新定义/视图，或重新生成清单文档")
    parser.add_argument("--refresh", action="store_true", help="按当前注册表刷新定义与视图")
    parser.add_argument("--write-docs", action="store_true",
                        help="重新生成 docs/core/factors/catalog.md 与 formulas.md")
    parser.add_argument("--explain", default=None, metavar="因子名",
                        help="讲解单个因子：公式、白话步骤、可直接跑的手工复现 SQL")
    parser.add_argument("--stock", default="SH600000", help="--explain 里手工复现用的示例股票")
    parser.add_argument("--start", default="2024-01-01", help="--explain 里手工复现的起始日")
    parser.add_argument("--end", default="2024-03-31", help="--explain 里手工复现的截止日")
    args = parser.parse_args(argv)

    if args.explain:
        try:
            spec = factors.get(args.explain)
        except FactorRegistryError:
            planned = next((item for item in PLANNED if item["name"] == args.explain), None)
            if planned is not None:
                print(
                    f"{args.explain} 还没实现，只有规划好的公式：\n"
                    f"  公式：{planned['formula']}\n"
                    f"  实现要点：{planned['note'] or '（无）'}\n"
                    f"  照 docs/core/factors/README.md 的三步写一条 define(...) 就能纳入构建。"
                )
                return 0
            print(
                f"没有叫 {args.explain!r} 的因子。已实现的因子：\n  "
                + "、".join(spec.name for spec in all_factors())
                + "\n规划中的因子可以用 quant.load_factor_list(status='planned') 查。"
            )
            return 1
        print(explanation(spec, stock=args.stock, start=args.start, end=args.end))
    if args.write_docs:
        print(write_catalog_doc())
        print(write_formula_doc())
    if args.refresh:
        with connect() as conn:
            ensure_catalog(conn)
            refresh_definitions(conn)
            refresh_views(conn)
            refresh_manifest(conn)
        print("目录库已刷新")
    if not args.refresh and not args.write_docs and not args.explain:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
