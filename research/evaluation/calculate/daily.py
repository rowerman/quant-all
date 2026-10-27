"""逐日截面相关：IC、RankIC 与任意两列的每日相关系数。

把「按交易日分组算相关系数」写成一次 bincount 聚合，而不是逐日 apply，
实测全历史单因子（6454 个交易日）约 7 秒；同一份结果被 IC 统计、时序与分布复用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calculate.normalize import winsorize_by_day

if TYPE_CHECKING:
    from kernels import BackendSelection, NativeKernelBackend

# 单日有效配对数少于这个数就不参与统计：样本太小算出来的相关系数没有意义
MIN_CROSS_SECTION = 5


def _kernel_selection(
    backend: "NativeKernelBackend | None" = None,
) -> "BackendSelection":
    """返回本次逐日计算使用的后端。"""

    from kernels import BackendSelection, get_backend

    if backend is None:
        return get_backend()
    if isinstance(backend, BackendSelection):
        return backend
    return BackendSelection(backend, backend.name, None)


def _offsets_from_codes(codes: np.ndarray) -> np.ndarray:
    """把非负组号转换为 C 连续分组边界。"""

    if codes.size == 0:
        return np.array([0], dtype=np.int64)
    sizes = np.bincount(codes)
    return np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(sizes, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)


def _grouped_arrays(
    codes: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """把样本稳定整理为按组连续，返回边界和同序数组。"""

    order = np.argsort(codes, kind="stable")
    ordered_codes = codes[order]
    offsets = _offsets_from_codes(ordered_codes)
    grouped = tuple(np.ascontiguousarray(array[order]) for array in arrays)
    return offsets, grouped


def _daily_pearson(codes: np.ndarray, left: np.ndarray, right: np.ndarray,
                   min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """用分组求和算每个交易日的 Pearson 相关系数。

    参数:
        codes: 每个样本所属交易日的整数编码。
        left: 左变量取值。
        right: 右变量取值。
        min_n: 单日最少有效配对数。

    返回:
        ``(相关系数数组, 有效配对数数组)``，按编码顺序排列；
        样本不足或某一侧方差为 0 的交易日返回 NaN。
    """
    count = np.bincount(codes).astype("float64")
    safe_count = np.where(count > 0, count, np.nan)
    mean_left = np.bincount(codes, weights=left) / safe_count
    mean_right = np.bincount(codes, weights=right) / safe_count
    # 先中心化，避免大偏置或常数秩的 E[x²]-E[x]² 舍入误差制造伪相关。
    left_dev = left - mean_left[codes]
    right_dev = right - mean_right[codes]
    left_dev -= (np.bincount(codes, weights=left_dev) / safe_count)[codes]
    right_dev -= (np.bincount(codes, weights=right_dev) / safe_count)[codes]
    covariance = np.bincount(codes, weights=left_dev * right_dev)
    var_left = np.bincount(codes, weights=left_dev * left_dev)
    var_right = np.bincount(codes, weights=right_dev * right_dev)
    scale = np.sqrt(var_left * var_right)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(scale > 0, covariance / scale, np.nan)
    corr = np.where(count >= min_n, np.clip(corr, -1.0, 1.0), np.nan)
    return corr, count.astype("int64")


def rank_pct_by_day(codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """按交易日分组算截面分位（0~1），与 ``groupby(...).rank(pct=True)`` 同语义。

    参数:
        codes: 每个样本所属交易日的整数编码，取值为 ``0..n-1``。
        values: 一维取值数组，空值（NaN）原样保留为空。

    返回:
        分位数组；并列值取平均名次，分母是该交易日该列的非空样本数。

    说明：pandas 的 ``groupby().rank(pct=True)`` 在多标签上要 8 秒以上，
    这里改成「整体按 (交易日, 取值) 排一次序，再对并列块取平均名次」，
    实测快 3 倍以上，取值与 pandas 完全一致。
    """
    result = np.full(values.size, np.nan, dtype="float64")
    valid = ~np.isnan(values)
    count = int(valid.sum())
    if count == 0:
        return result
    sub_days = codes[valid]
    order = np.lexsort((values[valid], sub_days))
    ordered_days = sub_days[order]
    ordered_values = values[valid][order]
    # 并列块 = 同一交易日内取值相同的连续区间
    same = np.empty(count, dtype=bool)
    same[0] = False
    same[1:] = (ordered_days[1:] == ordered_days[:-1]) & (
        ordered_values[1:] == ordered_values[:-1]
    )
    starts = np.flatnonzero(~same)
    ends = np.append(starts[1:], count)
    # 名次必须按「交易日内的位置」算，不能用整体位置：否则会凭空加上前面交易日的行数
    sizes_by_day = np.bincount(ordered_days)
    day_start = np.concatenate(([0], np.cumsum(sizes_by_day)[:-1]))
    positions = np.arange(count, dtype="float64") - day_start[ordered_days]
    block_average = (positions[starts] + positions[ends - 1]) / 2.0 + 1.0
    block_pct = block_average / sizes_by_day[ordered_days][starts]
    result[np.flatnonzero(valid)[order]] = np.repeat(block_pct, ends - starts)
    return result


def daily_corr(frame: pd.DataFrame, x: str, y: str, *,
               min_n: int = MIN_CROSS_SECTION, rank: bool = False) -> pd.DataFrame:
    """按交易日算两列的截面相关系数。

    参数:
        frame: 至少含 ``trade_date``、``x``、``y`` 三列的数据表。
        x: 左变量列名。
        y: 右变量列名。
        min_n: 单日最少有效配对数。
        rank: 真则先做截面分位再算相关，等价于 Spearman。

    返回:
        以交易日为索引、含 ``corr`` 与 ``n`` 两列的表；``n`` 为当日有效配对数。
    """
    work = frame.loc[:, ["trade_date", x, y]].dropna()
    if work.empty:
        return pd.DataFrame(
            {"corr": [], "n": []},
            index=pd.Index([], name="trade_date", dtype="datetime64[ns]"),
        )
    work = work.loc[np.isfinite(work[x]) & np.isfinite(work[y])]
    if work.empty:
        return pd.DataFrame(
            {"corr": [], "n": []},
            index=pd.Index([], name="trade_date", dtype="datetime64[ns]"),
        )
    codes, levels = pd.factorize(work["trade_date"], sort=True)
    left = np.ascontiguousarray(work[x].to_numpy(dtype="float64"))
    right = np.ascontiguousarray(work[y].to_numpy(dtype="float64"))
    offsets, (left, right) = _grouped_arrays(codes, left, right)
    selection = _kernel_selection()
    if rank:
        left = selection.backend.daily_rank_pct(left, offsets)
        right = selection.backend.daily_rank_pct(right, offsets)
    corr, count = selection.backend.daily_pearson(
        left,
        right,
        offsets,
        min_n,
    )
    return pd.DataFrame({"corr": corr, "n": count}, index=pd.Index(levels, name="trade_date"))


def daily_ic(panel: pd.DataFrame, *, winsorize: bool = True,
             min_n: int = MIN_CROSS_SECTION, pool_only: bool = True) -> pd.DataFrame:
    """算因子与标签的逐日 IC 与 RankIC。

    参数:
        panel: 评估面板，需要 ``trade_date``、``factor``、``label`` 三列。
        winsorize: 是否对因子与标签做截面 3 倍 MAD 去极值（只影响 Pearson IC）。
        min_n: 单日最少有效配对数。
        pool_only: 是否只用池内样本；默认真，与股票池口径保持一致。

    返回:
        以交易日为索引、含 ``ic``、``rank_ic``、``n`` 三列的表。
    """
    subset = panel.loc[panel["in_universe"]] if pool_only else panel
    work = subset.loc[:, ["trade_date", "factor", "label"]].dropna()
    if work.empty:
        return pd.DataFrame(
            {"ic": [], "rank_ic": [], "n": []},
            index=pd.Index([], name="trade_date", dtype="datetime64[ns]"),
        )
    work = work.loc[np.isfinite(work["factor"]) & np.isfinite(work["label"])]
    if work.empty:
        return pd.DataFrame({"ic": [], "rank_ic": [], "n": []},
                            index=pd.DatetimeIndex([], name="trade_date"))
    raw_factor = np.ascontiguousarray(
        work["factor"].to_numpy(dtype="float64")
    )
    raw_label = np.ascontiguousarray(
        work["label"].to_numpy(dtype="float64")
    )
    codes, levels = pd.factorize(work["trade_date"], sort=True)
    selection = _kernel_selection()

    if winsorize:
        days = work["trade_date"].to_numpy()
        factor_values = np.ascontiguousarray(
            winsorize_by_day(days, raw_factor)
        )
        label_values = np.ascontiguousarray(
            winsorize_by_day(days, raw_label)
        )
    else:
        factor_values = raw_factor
        label_values = raw_label
    offsets, (
        factor_values,
        label_values,
        rank_factor_raw,
        rank_label_raw,
    ) = _grouped_arrays(
        codes,
        factor_values,
        label_values,
        raw_factor,
        raw_label,
    )
    ic, count = selection.backend.daily_pearson(
        factor_values,
        label_values,
        offsets,
        min_n,
    )

    # RankIC 直接对原始值排名：排名本身对极值稳健，先去极值会把极端值压成并列，反而改变秩
    rank_factor = selection.backend.daily_rank_pct(rank_factor_raw, offsets)
    rank_label = selection.backend.daily_rank_pct(rank_label_raw, offsets)
    rank_ic, _ = selection.backend.daily_pearson(
        rank_factor,
        rank_label,
        offsets,
        min_n,
    )
    return pd.DataFrame(
        {"ic": ic, "rank_ic": rank_ic, "n": count},
        index=pd.Index(levels, name="trade_date"),
    )


def daily_rank_ic_multi(panel: pd.DataFrame, columns, *,
                        min_n: int = MIN_CROSS_SECTION,
                        pool_only: bool = True) -> dict[str, pd.DataFrame]:
    """对多档标签列各算一条逐日 RankIC 序列（多周期衰减用）。

    参数:
        panel: 评估面板，需要 ``trade_date``、``factor`` 与待算的标签列。
        columns: 标签列名序列，例如 ``("label", "label_h20")``。
        min_n: 单日最少有效配对数。
        pool_only: 是否只用池内样本；默认真，与股票池口径保持一致。

    返回:
        ``{列名: DataFrame(rank_ic, n)}``，行是交易日，索引与 ``daily_ic`` 一致。

    口径与 ``daily_ic`` 的 RankIC 相同（截面平均名次的 Spearman）：
    各档的因子秩与标签秩都在同一批有效配对上计算。
    各档必须先取因子与标签的有限配对，再在配对样本内排名。
    删除截面的一部分股票会改变名次间距，不能复用完整截面的因子秩；
    末端不足 N 个交易日的行只从该档剔除，不做任何填充。
    """
    index = pd.Index([], name="trade_date", dtype="datetime64[ns]")
    empty = pd.DataFrame({"rank_ic": [], "n": []}, index=index)
    if panel.empty:
        return {column: empty for column in columns}
    # 用布尔掩码逐列切样本，别用 ``panel.loc[池内]``：那会把近 20 列整张复制一遍
    keep = panel["in_universe"].to_numpy() if pool_only else None
    if keep is not None and not keep.any():
        return {column: empty for column in columns}
    dates = panel["trade_date"].to_numpy()
    codes, levels = pd.factorize(dates if keep is None else dates[keep], sort=True)

    def column_values(name: str) -> np.ndarray:
        """取一列并按池内掩码切出样本。

        参数:
            name: 列名。
        """
        values = panel[name].to_numpy(dtype="float64")
        return values if keep is None else values[keep]

    factor_values = column_values("factor")
    selection = _kernel_selection()
    index = pd.Index(levels, name="trade_date")
    series: dict[str, pd.DataFrame] = {}
    for column in columns:
        label_values = column_values(column)
        mask = np.isfinite(factor_values) & np.isfinite(label_values)
        corr = np.full(levels.size, np.nan, dtype="float64")
        count = np.zeros(levels.size, dtype="int64")
        if mask.any():
            day_codes = np.ascontiguousarray(codes[mask])
            offsets, (paired_factor, paired_label) = _grouped_arrays(
                day_codes,
                factor_values[mask],
                label_values[mask],
            )
            rank_factor = selection.backend.daily_rank_pct(
                paired_factor,
                offsets,
            )
            rank_label = selection.backend.daily_rank_pct(
                paired_label,
                offsets,
            )
            values, sizes = selection.backend.daily_pearson(
                rank_factor,
                rank_label,
                offsets,
                min_n,
            )
            positions = np.flatnonzero(
                np.bincount(day_codes, minlength=levels.size)
            )
            corr[positions] = values[positions]
            count[positions] = sizes[positions]
        series[column] = pd.DataFrame({"rank_ic": corr, "n": count}, index=index)
    return series
