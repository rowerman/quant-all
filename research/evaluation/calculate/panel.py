"""评估面板：把因子值、标签、池子标记与后续阶段要用的原料对齐成一张表。

一次评估请求只扫一遍库：面板算好之后所有指标都在它上面算，
既不重复读 1795 万行的日线表，也保证各指标用的是同一份样本。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

import factors
from factors.registry import FactorSpec
from calculate.params import MAX_TRADE_DAYS, EvalParams
from calculate.universe import (
    TRADABLE_FILTER,
    UniverseSpec,
    membership_cte,
    resolve_universe,
)
from calculate.config import DB_FILE, FACTOR_DIR
from calculate.errors import (
    DecomposeError,
    EvalFactorError,
    EvalPanelError,
    EvalParamError,
)
from calculate.logging_conf import get_logger

logger = get_logger("calculate.panel")

# 面板列，顺序固定；键是 stock_code + trade_date，label 允许为空（区间末端）
PANEL_COLUMNS: tuple[str, ...] = (
    "stock_code", "trade_date", "factor", "label", "label_1d", "in_universe",
    "is_st", "listed_days", "can_buy_close", "can_sell_close",
    "total_mv", "circ_mv", "turnover_rate", "pe_ttm", "pb",
    "amount_cny", "close", "ret",
)

# 参与计算的主列用 float64，原料列用 float32 省内存
PRIMARY_FLOAT_COLUMNS = ("factor", "label", "label_1d")
RAW_FLOAT_COLUMNS = ("total_mv", "circ_mv", "turnover_rate", "pe_ttm", "pb",
                     "amount_cny", "close", "ret")


def panel_columns(params: EvalParams) -> tuple[str, ...]:
    """面板的完整列清单：固定列 + 多周期标签列（V11-3 加入）。

    参数:
        params: 评估参数，决定除 ``label`` / ``label_1d`` 之外还要带哪几档标签。
    """
    return PANEL_COLUMNS + params.extra_label_columns()


def label_select(params: EvalParams) -> list[str]:
    """标签 CTE 要产出的列（含多周期标签）。

    参数:
        params: 评估参数。
    """
    columns = [
        f"CAST({params.label_column} AS DOUBLE) AS label",
        f"CAST({params.label_column_for(1)} AS DOUBLE) AS label_1d",
    ]
    columns.extend(
        f"CAST({params.label_column_for(horizon)} AS DOUBLE) AS {column}"
        for horizon, column in params.extra_label_specs()
    )
    return columns

# 单次请求的面板上限：超过就拒绝，提示缩小区间
MAX_PANEL_ROWS = 8_000_000


def open_source(db_file: Path | str = DB_FILE) -> duckdb.DuckDBPyConnection:
    """打开内存连接并以只读方式挂载日线主库。

    参数:
        db_file: 主库路径。

    返回:
        连接对象，主库别名固定为 ``src``。
    """
    target = Path(db_file)
    if not target.exists():
        raise FileNotFoundError(f"主库不存在: {target}")
    conn = duckdb.connect()
    conn.execute("SET memory_limit='1GB'")
    conn.execute('SET threads=2')
    conn.execute(f"ATTACH '{target}' AS src (READ_ONLY)")
    return conn


def factor_glob(family: str, factor_dir: Path | str = FACTOR_DIR) -> str:
    """某个族全部年份的 parquet 通配路径。

    参数:
        family: 因子族。
        factor_dir: 因子数据根目录。
    """
    return str(Path(factor_dir) / family / "year=*" / "part.parquet")


def resolve_factor(name: str, factor_dir: Path | str = FACTOR_DIR) -> "FactorView":
    """取得因子定义，并确认它已经有落盘的数值列。

    参数:
        name: 因子名。
        factor_dir: 因子数据根目录。
    """
    # 先在注册表里查名字：这样就不必依赖数据项目的异常类型，未知因子直接报 EVAL-104
    registered = {item.name: item for item in factors.available_specs(factor_dir)}
    spec = registered.get(name)
    if spec is None:
        raise EvalFactorError(
            f"未知因子 {name!r}；数据项目的 factors 注册表里已注册 {len(registered)} 个："
            f"{sorted(registered)}"
        )

    files = sorted(Path(factor_dir).joinpath(spec.family).glob("year=*/part.parquet"))
    if not files:
        raise EvalFactorError(
            f"因子 {name} 属于 {spec.family} 族，但该族还没有落盘数据；"
            "先跑 python -m scripts.build_factors.build"
        )
    with duckdb.connect() as conn:
        columns = [item[0] for item in conn.execute('SELECT * FROM read_parquet(?) LIMIT 0',
                                                  [str(files[-1])]).description]
    if spec.name not in columns:
        available = sorted(col for col in columns if col not in ("stock_code", "trade_date", "year"))
        raise EvalFactorError(
            f"因子 {name} 在 {spec.family} 族的落盘文件里不存在；该族实际有：{available}"
        )
    return FactorView.from_registry(spec, factor_dir)


def count_trade_days(conn: duckdb.DuckDBPyConnection, start: str, end: str) -> int:
    """数区间内的交易日数。

    参数:
        conn: 已挂载主库的连接。
        start: 起始日。
        end: 截止日。
    """
    frame = conn.execute(
        "SELECT count(*) FROM src.md.calendar "
        f"WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'"
    ).fetchone()
    return int(frame[0])


def check_label_columns(conn: duckdb.DuckDBPyConnection, label_columns) -> None:
    """确认标签列存在，避免拼出错误列名后报一堆无关的 SQL 错误。

    参数:
        conn: 已挂载主库的连接。
        label_columns: 期望存在的标签列名序列。
    """
    rows = conn.execute("DESCRIBE SELECT * FROM src.md.fwd_return").fetchall()
    columns = {row[0] for row in rows}
    missing = [name for name in label_columns if name not in columns]
    if missing:
        raise EvalPanelError(
            f"标签列 {missing} 不存在；md.fwd_return 实际有 {sorted(columns)}"
        )


def estimate_panel_rows_from_glob(
    conn: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    start: str,
    end: str,
) -> int:
    """用明确 Parquet 通配路径估算评估区间内的面板行数上界。"""

    first_year, last_year = int(start[:4]), int(end[:4])
    partition_parent = Path(parquet_glob).parent
    files_pattern = f"{partition_parent.name}/{Path(parquet_glob).name}"
    files = [
        path for path in sorted(partition_parent.parent.glob(files_pattern))
        if path.parent.name.startswith("year=")
        and first_year <= int(path.parent.name.split("=")[1]) <= last_year
    ]
    if not files:
        raise EvalFactorError(
            f"因子在 {start} ~ {end} 没有落盘文件：{parquet_glob}"
        )
    total = 0
    for path in files:
        rows = conn.execute(
            "SELECT sum(num_rows) FROM parquet_file_metadata(?)", [str(path)]
        ).fetchone()[0]
        total += int(rows or 0)
    return total


def estimate_panel_rows(conn: duckdb.DuckDBPyConnection, family: str,
                        factor_dir: Path | str, start: str, end: str) -> int:
    """用 parquet 元数据估算区间内的面板行数上界（不扫数据本身）。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        factor_dir: 因子数据根目录。
        start: 起始日。
        end: 截止日。
    """
    return estimate_panel_rows_from_glob(
        conn, factor_glob(family, factor_dir), start, end
    )


def panel_sql(params: EvalParams, spec: "FactorView", universe: UniverseSpec) -> str:
    """生成评估面板的 SQL。

    参数:
        params: 评估参数。
        spec: 因子定义。
        universe: 股票池。
        spec: 正式或候选的明确因子来源。
    """
    cte_tail = ""
    member_join = ""
    if universe.is_index:
        cte_tail = f", {membership_cte(universe, params.start, params.end)}"
        member_join = (
            "LEFT JOIN member m ON m.stock_code = f.stock_code AND m.trade_date = f.trade_date"
        )
        universe_expr = f"({TRADABLE_FILTER} AND m.stock_code IS NOT NULL)"
    else:
        universe_expr = f"({TRADABLE_FILTER})"

    return (
        "WITH "
        "factor_values AS ("
        f" SELECT stock_code, trade_date, CAST({spec.name} AS DOUBLE) AS factor"
        f" FROM read_parquet('{spec.parquet_glob.replace(chr(39), chr(39) * 2)}',"
        " hive_partitioning = 1)"
        f" WHERE trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'), "
        "labels AS ("
        " SELECT stock_code, trade_date, "
        + ", ".join(label_select(params))
        +
        " FROM src.md.fwd_return"
        f" WHERE trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}')"
        f"{cte_tail} "
        "SELECT f.stock_code, f.trade_date, f.factor,"
        f" l.label, l.label_1d,{''.join(f' l.{column},' for column in params.extra_label_columns())}"
        f" {universe_expr} AS in_universe,"
        " b.is_st, datediff('day', b.list_date, b.trade_date) AS listed_days,"
        " b.can_buy_close, b.can_sell_close,"
        " CAST(b.total_mv AS DOUBLE) AS total_mv, CAST(b.circ_mv AS DOUBLE) AS circ_mv,"
        " CAST(b.turnover_rate AS DOUBLE) AS turnover_rate,"
        " CAST(b.pe_ttm AS DOUBLE) AS pe_ttm, CAST(b.pb AS DOUBLE) AS pb,"
        " CAST(b.amount_cny AS DOUBLE) AS amount_cny,"
        " CAST(b.close AS DOUBLE) AS close, CAST(b.ret AS DOUBLE) AS ret"
        " FROM factor_values f"
        " JOIN labels l ON l.stock_code = f.stock_code AND l.trade_date = f.trade_date"
        " JOIN src.md.daily_hfq b ON b.stock_code = f.stock_code AND b.trade_date = f.trade_date"
        f" AND b.trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'"
        f" {member_join}"
        # 不做 ORDER BY：几百万行的排序要 2 秒以上，而下游指标都按交易日分组，不依赖行序
    )


def optimize_panel(frame: pd.DataFrame, params: EvalParams) -> pd.DataFrame:
    """把面板列压成省内存的类型，并固定列顺序。

    参数:
        frame: 查询结果。
        params: 评估参数，决定多周期标签列有哪些。
    """
    columns = panel_columns(params)
    frame = frame.loc[:, list(columns)].copy()
    frame["stock_code"] = frame["stock_code"].astype("category")
    for column in PRIMARY_FLOAT_COLUMNS + params.extra_label_columns():
        frame[column] = frame[column].astype("float64")
    for column in RAW_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float32")
    for column in ("in_universe", "is_st", "can_buy_close", "can_sell_close"):
        frame[column] = frame[column].astype(bool)
    frame["listed_days"] = frame["listed_days"].astype("int32")
    return frame


def summarize_panel(frame: pd.DataFrame, universe: UniverseSpec) -> dict:
    """算一份面板概览，供 meta 事件与覆盖率口径对照。

    参数:
        frame: 评估面板。
        universe: 股票池。
    """
    rows = int(len(frame))
    pool = frame["in_universe"]
    pool_rows = int(pool.sum())
    factor_valid = frame["factor"].notna()
    pair_valid = factor_valid & frame["label"].notna()
    return {
        "rows": rows,
        "days": int(frame["trade_date"].nunique()) if rows else 0,
        "stocks": int(frame["stock_code"].nunique()) if rows else 0,
        "universe": {"key": universe.key, "label": universe.label, "detail": universe.detail},
        "universe_rows": pool_rows,
        "universe_share": (pool_rows / rows) if rows else None,
        "coverage_all": float(factor_valid.mean()) if rows else None,
        "coverage_universe": float(factor_valid[pool].mean()) if pool_rows else None,
        "pair_coverage": float(pair_valid[pool].mean()) if pool_rows else None,
    }


@dataclass(frozen=True)
class FactorView:
    """一次评估使用的因子元数据和明确的数据文件来源。"""

    name: str
    family: str
    formula: str
    description: str
    steps: tuple[str, ...]
    direction: int
    lookback: int
    fields: tuple[str, ...]
    extra_sources: tuple[str, ...]
    parquet_glob: str
    formal: bool = True

    @classmethod
    def from_registry(cls, spec: FactorSpec, factor_dir: Path | str = FACTOR_DIR) -> "FactorView":
        """把正式注册定义包装成带明确文件来源的评估视图。"""

        return cls(
            name=spec.name,
            family=spec.family,
            formula=spec.formula,
            description=spec.description,
            steps=spec.steps,
            direction=spec.direction,
            lookback=spec.lookback,
            fields=spec.fields,
            extra_sources=spec.extra_sources,
            parquet_glob=factor_glob(spec.family, factor_dir),
        )


@dataclass
class EvalContext:
    """一次评估请求的共享上下文。

    参数:
        params: 评估参数。
        spec: 因子定义。
        universe: 股票池。
        panel: 评估面板。
        summary: 面板概览。
        panel_elapsed_ms: 建面板耗时。
        db_file: 日线主库路径，风格因子等按需取数时用。
    """

    params: EvalParams
    spec: FactorView
    universe: UniverseSpec
    panel: pd.DataFrame
    summary: dict
    panel_elapsed_ms: float
    db_file: Path | str = DB_FILE
    _cache: dict = field(default_factory=dict, repr=False)
    factor_dir: Path | str = FACTOR_DIR

    @property
    def label_columns(self) -> tuple[str, ...]:
        """面板上全部标签列，顺序即档位顺序。"""
        return ("label", "label_1d") + self.params.extra_label_columns()

    def horizon_label(self, horizon: int) -> str:
        """取某个标签档位对应的面板列名。

        参数:
            horizon: 标签周期，取值见 ``params.DECAY_HORIZONS``。
        """
        column = self.params.horizon_label_column(horizon)
        if column not in self.panel.columns:
            available = [name for name in self.panel.columns if name.startswith("label")]
            raise DecomposeError(
                f"面板缺少标签列 {column}（档位 {horizon}）；面板现有标签列：{available}"
            )
        return column

    def styles(self) -> pd.DataFrame:
        """风格因子原始值（带缓存）：键 stock_code + trade_date，7 个风格列。

        风格因子需要 260 个交易日的回看，塞进面板要多占几百万行内存，
        所以单独取一次，只有用到拆分阶段才触发；耗时以包 README 的实测为准。
        """
        from calculate.styles import build_styles

        if "styles" not in self._cache:
            self._cache["styles"] = build_styles(
                self.params,
                db_file=self.db_file,
                compact=True,
            )
        return self._cache["styles"]

    def exposures(self, industry_key: str = "off"):
        """风格拆分结果（带缓存）：逐日截面回归的系数、t 值与残差。

        参数:
            industry_key: 行业数据源标识，``off`` 表示不加行业层。
        """
        from calculate.exposure import run_exposure

        key = ("exposure", industry_key or "off")
        if key not in self._cache:
            self._cache[key] = run_exposure(self, industry_key or "off")
        return self._cache[key]

    def daily_ic(self, winsorize: bool | None = None) -> pd.DataFrame:
        """逐日 IC 序列（带缓存），列：ic、rank_ic、n。

        参数:
            winsorize: 是否去极值；留空用参数里的设置。
        """
        from calculate.daily import daily_ic

        flag = self.params.winsorize if winsorize is None else winsorize
        key = ("daily_ic", bool(flag))
        if key not in self._cache:
            self._cache[key] = daily_ic(self.panel, winsorize=flag)
        return self._cache[key]

    def bins(self, quantiles: int | None = None) -> pd.DataFrame:
        """池内样本的分位分组表（带缓存），列见 ``binning.assign_bins``。

        参数:
            quantiles: 分组数；留空用参数里的设置。
        """
        from calculate.binning import assign_bins

        count = self.params.quantiles if quantiles is None else quantiles
        key = ("bins", int(count))
        if key not in self._cache:
            self._cache[key] = assign_bins(
                self.panel, int(count), label_columns=self.label_columns
            )
        return self._cache[key]

    def bin_matrices(self, quantiles: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """（组号宽表, 截面秩宽表）二元组，带缓存：换手与自相关共用同一次透视。

        参数:
            quantiles: 分组数；留空用参数里的设置。
        """
        from calculate.binning import bin_matrix, rank_matrix

        count = self.params.quantiles if quantiles is None else quantiles
        key = ("matrices", int(count))
        if key not in self._cache:
            frame = self.bins(count)
            dates = pd.DatetimeIndex(sorted(self.panel["trade_date"].unique()), name="trade_date")
            self._cache[key] = (bin_matrix(frame).reindex(dates), rank_matrix(frame).reindex(dates))
        return self._cache[key]


def build_panel(params: EvalParams, *, db_file: Path | str = DB_FILE,
                factor_dir: Path | str = FACTOR_DIR,
                factor: FactorView | None = None,
                conn: duckdb.DuckDBPyConnection | None = None) -> EvalContext:
    """构建评估面板。

    参数:
        params: 评估参数。
        db_file: 日线主库路径。
        factor_dir: 正式因子对照根目录。
        factor: 显式因子来源；留空时按名称从正式注册表解析。
        conn: 已有的主库连接；留空则自行打开并在结束时关闭。
    """
    started = time.perf_counter()
    universe = resolve_universe(params.universe)
    spec = factor or resolve_factor(params.factor, factor_dir)
    if spec.name != params.factor:
        raise EvalFactorError(
            f"显式因子名 {spec.name!r} 与评估参数 {params.factor!r} 不一致"
        )

    owns_connection = conn is None
    conn = conn or open_source(db_file)
    try:
        days = count_trade_days(conn, params.start, params.end)
        if days == 0:
            raise EvalParamError(f"区间 {params.start} ~ {params.end} 内没有交易日")
        if days > MAX_TRADE_DAYS:
            raise EvalParamError(
                f"区间含 {days} 个交易日，超过上限 {MAX_TRADE_DAYS}；请缩小起止范围"
            )
        estimate = estimate_panel_rows_from_glob(
            conn, spec.parquet_glob, params.start, params.end
        )
        if estimate > MAX_PANEL_ROWS:
            raise EvalPanelError(
                f"区间内面板约 {estimate:,} 行，超过上限 {MAX_PANEL_ROWS:,} 行；请缩小起止范围"
            )
        labels = [params.label_column, params.label_column_for(1)]
        labels.extend(params.label_column_for(horizon) for horizon, _ in params.extra_label_specs())
        check_label_columns(conn, labels)
        frame = conn.execute(panel_sql(params, spec, universe)).df()
    finally:
        if owns_connection:
            conn.close()

    if frame.empty:
        raise EvalPanelError(
            f"{params.factor} 在 {params.start} ~ {params.end} 没有可用行；"
            "检查因子落盘年份与主库数据区间"
        )
    frame = optimize_panel(frame, params)
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise EvalPanelError("面板主键 stock_code + trade_date 出现重复行")

    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = summarize_panel(frame, universe)
    logger.info(
        "面板就绪：因子=%s 池=%s 行=%d 交易日=%d 股票=%d 池内占比=%.3f 覆盖=%.4f 耗时=%.0fms",
        params.factor, universe.key, summary["rows"], summary["days"], summary["stocks"],
        summary["universe_share"] or 0.0, summary["pair_coverage"] or 0.0, elapsed_ms,
    )
    return EvalContext(
        params=params, spec=spec, universe=universe, panel=frame,
        summary=summary, panel_elapsed_ms=elapsed_ms, db_file=db_file, factor_dir=factor_dir,
    )


def default_params(factor: str, *, db_file: Path | str = DB_FILE, years: int = 3,
                   **overrides) -> EvalParams:
    """按「最近 N 年」构造默认参数。

    参数:
        factor: 因子名。
        db_file: 日线主库路径。
        years: 回看的年数，默认 3。
        overrides: 需要覆盖的其它参数。
    """
    conn = open_source(db_file)
    try:
        earliest, latest = conn.execute(
            "SELECT min(trade_date), max(trade_date) FROM src.md.calendar"
        ).fetchone()
    finally:
        conn.close()
    start = max(earliest, (pd.Timestamp(latest) - pd.DateOffset(years=years)).date())
    overrides.setdefault("start", start.strftime("%Y-%m-%d"))
    overrides.setdefault("end", pd.Timestamp(latest).strftime("%Y-%m-%d"))
    return EvalParams(factor=factor, **overrides)
