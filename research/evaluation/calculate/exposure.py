"""风格暴露与增量 IC 的计算核心（纯计算，不含任何展示结构）。

口径：

- 回归样本 = 池内且因子非空（**不看标签**），这样暴露与标签周期解耦；
- 被评估因子与 7 个风格因子都在**同一批样本**上逐日做「3 倍 MAD 去极值 + z-score」，
  z-score 是逐日单调变换，不影响 RankIC 的可比性；
- 风格算不出来的样本标准化后置 0，属于中性填补假设，不代表真实无暴露；
  整日没有变化的风格不可估计，该系数返回 NaN，其他风格仍可回归；
- 逐日截面 OLS（Fama-MacBeth）：因子 ~ 1 + 7 个风格（可选加行业哑变量），
  系数 = 因子每 1 个标准差的风格暴露，t 值由当日残差方差算出的标准误给出；
- 增量 IC = 残差与标签的逐日 RankIC；对照 raw 在同一残差样本与有效日期上重算。
  差值仅用于比较，不是严格可加的风格贡献。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from calculate.daily import MIN_CROSS_SECTION, daily_corr
from calculate.errors import DecomposeError
from calculate.logging_conf import get_logger
from calculate.normalize import standardize_by_day
from calculate.panel import open_source
from calculate.styles import (
    INDUSTRY_OFF,
    RAW_COLUMNS,
    STYLE_KEYS,
    list_industry_sources,
    resolve_industry_source,
)

logger = get_logger("calculate.exposure")

# 单日回归样本少于这个数就跳过：8 个参数用几十个样本估没有意义
MIN_REGRESSION_ROWS = 30
# 有效回归日少于这个数只报「样本不足」，与 IC 家族口径一致
INSUFFICIENT_DAYS = 20
# 判定「当天显著」的 t 阈值
SIGNIFICANCE_T = 1.96


@dataclass
class ExposureResult:
    """一次风格拆分的结果。

    参数:
        dates: 出结果（没被跳过）的交易日。
        coefficients: 逐日暴露系数，形状 ``(天数, 7)``，列序与 ``STYLE_KEYS`` 一致
            （截距不在这里，截距对标准化后的因子没有解释意义）。
        t_stats: 逐日系数的 t 值，形状同上。
        intercept: 逐日回归截距。
        r2: 逐日回归 R²。
        rows: 逐日回归样本数。
        skipped_days: 因样本太少被跳过的交易日数。
        residual: 回归样本上的表（trade_date / residual / label / factor）；
            factor 为原始因子，供同样本 raw RankIC 对照。
        industry: 行业层信息，供 payload 与页面说明。
    """

    dates: list
    coefficients: np.ndarray
    t_stats: np.ndarray
    intercept: np.ndarray
    r2: np.ndarray
    rows: np.ndarray
    skipped_days: int
    residual: pd.DataFrame
    industry: dict


def _standardized_block(values: np.ndarray, days: np.ndarray) -> np.ndarray:
    """一列取值先按交易日 MAD 去极值，再做截面 z-score。

    参数:
        values: 一维取值。
        days: 与 values 等长的交易日序列。
    """
    return standardize_by_day(days, values)


def _standardized_columns(
    sample: pd.DataFrame,
    columns: tuple[str, ...],
    days: np.ndarray,
) -> pd.DataFrame:
    """按日分组一次后标准化多列，并保持样本原顺序。"""

    from kernels import get_backend
    from calculate.normalize import _grouped_values

    first = sample[columns[0]].to_numpy(dtype="float64")
    grouped = _grouped_values(days, first, finite_dates_only=True)
    if grouped is None:
        return pd.DataFrame({
            column: _standardized_block(
                sample[column].to_numpy(dtype="float64"),
                days,
            )
            for column in columns
        })
    order, offsets, _ = grouped
    backend = get_backend().backend
    standardized: dict[str, np.ndarray] = {}
    for column in columns:
        ordered = np.ascontiguousarray(
            sample[column].to_numpy(dtype="float64")[order]
        )
        computed = backend.daily_winsorize_zscore(ordered, offsets)
        result = np.empty_like(computed)
        result[order] = computed
        standardized[column] = result
    return pd.DataFrame(standardized)


def _combine_value(z_bp: np.ndarray, z_ep: np.ndarray) -> np.ndarray:
    """VALUE = z(1/pb) 与 z(1/pe_ttm) 的均值，缺一个用另一个。

    参数:
        z_bp: 标准化后的账面市值比。
        z_ep: 标准化后的盈利收益率。
    """
    combined = np.full(z_bp.size, np.nan, dtype="float64")
    has_bp = np.isfinite(z_bp)
    has_ep = np.isfinite(z_ep)
    both = has_bp & has_ep
    combined[both] = (z_bp[both] + z_ep[both]) / 2.0
    combined[has_bp & ~has_ep] = z_bp[has_bp & ~has_ep]
    combined[~has_bp & has_ep] = z_ep[~has_bp & has_ep]
    return combined


def _industry_frame(context, industry_key: str) -> tuple[pd.DataFrame | None, dict]:
    """按标识取行业映射，并给出页面要用的说明。

    参数:
        context: 评估上下文。
        industry_key: 行业数据源标识，``off`` 表示关闭。
    """
    source = resolve_industry_source(industry_key)
    available = [
        {"key": item.key, "label": item.label, "note": item.note}
        for item in list_industry_sources()
    ]
    block = {
        "chosen": industry_key or INDUSTRY_OFF,
        "enabled": source is not None,
        "available": available,
        "columns": 0,
        "coverage": None,
        "note": (
            "行业层默认关闭：库里只有静态行业分类（非时点、退市股缺失、与指数行业口径不一致）"
        ),
    }
    if source is None:
        return None, block
    conn = open_source(context.db_file)
    try:
        mapping = source.load(conn)
    finally:
        conn.close()
    block["note"] = source.note
    return mapping, block


def _design_matrix(styles: pd.DataFrame, industry: np.ndarray | None) -> np.ndarray:
    """拼设计矩阵：截距 + 7 个风格（+ 行业哑变量）。

    参数:
        styles: 标准化后的 7 列风格。
        industry: 行业列（字符串类别），None 表示不加行业层。
    """
    parts = [np.ones((len(styles), 1), dtype="float64")]
    parts.append(styles[list(STYLE_KEYS)].to_numpy(dtype="float64"))
    if industry is not None:
        categories = pd.Categorical(industry)
        # 丢掉第一类，避免与截距完全共线
        dummies = pd.get_dummies(categories, drop_first=True, dtype="float64")
        parts.append(dummies.to_numpy(dtype="float64"))
    return np.hstack(parts)


def _daily_regression(
    day_codes: np.ndarray,
    day_count: int,
    design: np.ndarray,
    response: np.ndarray,
    rows: np.ndarray,
    *,
    min_rows: int = MIN_REGRESSION_ROWS,
) -> tuple:
    """按交易日逐日做截面 OLS。

    参数:
        day_codes: 每个样本的交易日编码（0..day_count-1，已排序）。
        day_count: 交易日数。
        design: 设计矩阵（含截距）。
        response: 被解释变量（标准化后的因子）。
        rows: 每个回归样本在残差表里的行号。
        min_rows: 单日最少回归样本数，默认沿用 30。
    """
    order = np.argsort(day_codes, kind="stable")
    starts = np.searchsorted(day_codes[order], np.arange(day_count))
    ends = np.append(starts[1:], day_codes.size)
    n_columns = design.shape[1]
    coefficients, t_stats, r2, sample_rows, used_days = [], [], [], [], []
    residual_rows, residual_values = [], []
    skipped = 0
    for index in range(day_count):
        start, end = int(starts[index]), int(ends[index])
        size = end - start
        if size < min_rows:
            if size:
                skipped += 1
            continue
        positions = order[start:end]
        design_day = design[positions]
        response_day = response[positions]
        if not np.isfinite(response_day).all() or np.ptp(response_day) == 0:
            skipped += 1
            continue
        # 非截距列整日为常数时不可估计，只删除该列，不放弃其他风格。
        active = np.ptp(design_day, axis=0) > 1e-12
        active[0] = True
        reduced = design_day[:, active]
        beta_active, _, rank, _ = np.linalg.lstsq(reduced, response_day, rcond=None)
        if rank < reduced.shape[1] or size <= rank:  # 剩余风格共线或自由度不足，当天不用
            skipped += 1
            continue
        fitted = reduced @ beta_active
        residual_day = response_day - fitted
        # 完全可被风格解释的因子没有剩余信号，不能对数值舍入误差算 RankIC。
        if np.linalg.norm(residual_day) <= 1e-12 * np.linalg.norm(response_day):
            residual_day[:] = 0.0
        degrees = size - rank
        sigma2 = float(residual_day @ residual_day) / degrees
        covariance = sigma2 * np.linalg.inv(reduced.T @ reduced)
        standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        beta = np.full(n_columns, np.nan)
        t_day = np.full(n_columns, np.nan)
        beta[active] = beta_active
        with np.errstate(invalid="ignore", divide="ignore"):
            t_day[active] = beta_active / standard_error
        total = float(response_day @ response_day) - size * float(response_day.mean()) ** 2
        r2_day = 1.0 - float(residual_day @ residual_day) / total if total > 0 else np.nan
        coefficients.append(beta)
        t_stats.append(t_day)
        r2.append(r2_day)
        sample_rows.append(size)
        used_days.append(index)
        residual_rows.append(rows[positions])
        residual_values.append(residual_day)
    if not used_days:
        shape = (0, n_columns)
        return {
            "days": [], "coefficients": np.empty(shape), "t_stats": np.empty(shape),
            "r2": np.empty(0), "rows": np.empty(0, dtype="int64"),
            "residual_rows": np.empty(0, dtype="int64"), "residual_values": np.empty(0),
            "skipped": skipped,
        }
    return {
        "days": used_days,
        "coefficients": np.vstack(coefficients),
        "t_stats": np.vstack(t_stats),
        "r2": np.asarray(r2, dtype="float64"),
        "rows": np.asarray(sample_rows, dtype="int64"),
        "residual_rows": np.concatenate(residual_rows),
        "residual_values": np.concatenate(residual_values),
        "skipped": skipped,
    }


def run_exposure(context, industry_key: str = INDUSTRY_OFF) -> ExposureResult:
    """跑一次风格拆分：逐日截面回归 + 残差。

    参数:
        context: 评估上下文。
        industry_key: 行业数据源标识，默认 ``off``。
    """
    panel = context.panel
    factor = panel["factor"].to_numpy(dtype="float64")
    keep = panel["in_universe"].to_numpy() & np.isfinite(factor)
    if not keep.any():
        raise DecomposeError("池内没有非空因子值，无法做风格拆分")
    sample = pd.DataFrame({
        "stock_code": panel["stock_code"].astype(str).to_numpy()[keep],
        "trade_date": panel["trade_date"].to_numpy()[keep],
        "factor": factor[keep],
        "label": panel["label"].to_numpy(dtype="float64")[keep],
    })
    styles = context.styles()
    sample = sample.merge(styles, on=["stock_code", "trade_date"], how="left")
    if sample.duplicated(["stock_code", "trade_date"]).any():
        raise DecomposeError("风格因子表在 stock_code + trade_date 上有重复行，无法与面板对齐")
    days = sample["trade_date"].to_numpy()
    if sample[list(RAW_COLUMNS)].notna().sum().sum() == 0:
        raise DecomposeError(
            "样本里没有任何风格因子有值：检查 md.daily_hfq 的市值/换手/估值列与 "
            "md.index_daily 是否齐全"
        )

    standardized = _standardized_columns(sample, RAW_COLUMNS, days)
    styles_std = pd.DataFrame({
        "SIZE": standardized["SIZE"],
        "VALUE": _combine_value(
            standardized["BP"].to_numpy(), standardized["EP"].to_numpy()
        ),
        "MOMENTUM": standardized["MOMENTUM"],
        "SHORT_REV": standardized["SHORT_REV"],
        "VOL": standardized["VOL"],
        "LIQUIDITY": standardized["LIQUIDITY"],
        "BETA_60": standardized["BETA_60"],
    })
    # 风格算不出来的样本（次新股没满 250 日、亏损股没有正 pe）标准化后置 0，
    # 等价于「不暴露」，保留该股票参与回归与增量 IC
    styles_std = styles_std.fillna(0.0)
    response = _standardized_columns(sample, ("factor",), days)["factor"].to_numpy(
        dtype="float64"
    )

    mapping, industry_block = _industry_frame(context, industry_key)
    industry_column = None
    if mapping is not None:
        merged = sample[["stock_code"]].merge(mapping, on="stock_code", how="left")
        industry_column = merged["industry"].fillna("未知").to_numpy()
        industry_block["coverage"] = float(np.mean(industry_column != "未知"))
        industry_block["categories"] = int(pd.unique(industry_column).size)

    design = _design_matrix(styles_std, industry_column)
    industry_block["columns"] = int(design.shape[1] - len(STYLE_KEYS) - 1)
    rows = np.arange(len(sample), dtype="int64")
    day_codes, day_levels = pd.factorize(days, sort=True)
    result = _daily_regression(day_codes, day_levels.size, design, response, rows)

    residual = pd.DataFrame({
        "trade_date": days[result["residual_rows"]],
        "residual": result["residual_values"],
        "label": sample["label"].to_numpy(dtype="float64")[result["residual_rows"]],
        "factor": sample["factor"].to_numpy(dtype="float64")[result["residual_rows"]],
    })
    dates = [pd.Timestamp(day_levels[index]) for index in result["days"]]
    logger.info(
        "风格拆分完成：池=%s 样本=%d 回归日=%d 跳过=%d 行业层=%s 设计矩阵列=%d",
        context.params.universe, len(sample), len(dates), result["skipped"],
        industry_block["chosen"], design.shape[1],
    )
    return ExposureResult(
        dates=dates,
        # 设计矩阵第 0 列是截距，风格系数从第 1 列起
        coefficients=result["coefficients"][:, 1:1 + len(STYLE_KEYS)],
        t_stats=result["t_stats"][:, 1:1 + len(STYLE_KEYS)],
        intercept=result["coefficients"][:, 0],
        r2=result["r2"],
        rows=result["rows"],
        skipped_days=result["skipped"],
        residual=residual,
        industry=industry_block,
    )


def residual_rank_ic(result: ExposureResult, *, min_n: int = MIN_CROSS_SECTION) -> pd.DataFrame:
    """残差与标签的逐日 RankIC。

    参数:
        result: 风格拆分结果。
        min_n: 单日最少有效配对数。
    """
    work = result.residual.loc[result.residual["label"].notna()]
    return daily_corr(work, "residual", "label", min_n=min_n, rank=True)


def describe_styles() -> list[dict]:
    """风格因子的展示信息（名称、口径、窗口与最小观测）。"""
    from calculate.styles import STYLE_FACTORS

    return [
        {
            "key": spec.key,
            "label": spec.label,
            "definition": spec.definition,
            "window": spec.window,
            "min_obs": spec.min_obs,
            "note": spec.note,
        }
        for spec in STYLE_FACTORS
    ]


__all__ = [
    "INSUFFICIENT_DAYS",
    "MIN_REGRESSION_ROWS",
    "SIGNIFICANCE_T",
    "ExposureResult",
    "describe_styles",
    "residual_rank_ic",
    "run_exposure",
]
