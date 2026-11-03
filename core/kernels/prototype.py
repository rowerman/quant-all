"""逐日排名和批量 Spearman 的 Numba 验证实现。"""

from __future__ import annotations

import numba
from numba import njit, prange
import numpy as np

from common.logging_conf import get_logger
from common.paths import LOGS_DIR
from kernels.backend import (
    BinReturnsResult,
    BinTurnoverResult,
    DailyOlsResult,
    FloatArray,
    IntArray,
    native_threads,
    validate_float_array,
    validate_minimum,
    validate_offsets,
)
from kernels.fallback import PythonKernelBackend

logger = get_logger("kernels.prototype", LOGS_DIR / "core")


@njit(cache=True, nogil=True)
def _rank_pct(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Numba 内核对每个分组计算并列平均分位。"""

    result = np.full(values.size, np.nan)
    for group in range(offsets.size - 1):
        start = offsets[group]
        end = offsets[group + 1]
        valid = np.empty(end - start, dtype=np.int64)
        valid_count = 0
        for position in range(start, end):
            value = values[position]
            if not np.isnan(value):
                valid[valid_count] = position
                valid_count += 1
        if valid_count == 0:
            continue
        compact = values[valid[:valid_count]]
        order = np.argsort(compact)
        begin = 0
        while begin < valid_count:
            finish = begin + 1
            while finish < valid_count and compact[order[finish]] == compact[order[begin]]:
                finish += 1
            average = (begin + finish - 1) / 2.0 + 1.0
            percentile = average / valid_count
            for item in range(begin, finish):
                result[valid[order[item]]] = percentile
            begin = finish
    return result


@njit(cache=True, nogil=True)
def _paired_ranks(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对一组有限配对计算平均秩。"""

    left_order = np.argsort(left)
    right_order = np.argsort(right)
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    begin = 0
    while begin < left.size:
        finish = begin + 1
        while finish < left.size and left[left_order[finish]] == left[left_order[begin]]:
            finish += 1
        average = (begin + finish - 1) / 2.0 + 1.0
        for item in range(begin, finish):
            left_rank[left_order[item]] = average
        begin = finish
    begin = 0
    while begin < right.size:
        finish = begin + 1
        while finish < right.size and right[right_order[finish]] == right[right_order[begin]]:
            finish += 1
        average = (begin + finish - 1) / 2.0 + 1.0
        for item in range(begin, finish):
            right_rank[right_order[item]] = average
        begin = finish
    return left_rank, right_rank


@njit(cache=True, nogil=True)
def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """计算一组数组中已排除缺失值的 Pearson 相关。"""

    if left.size < 2:
        return np.nan
    left_mean = np.mean(left)
    right_mean = np.mean(right)
    left_dev = left - left_mean
    right_dev = right - right_mean
    left_norm = np.sqrt(np.sum(left_dev * left_dev))
    right_norm = np.sqrt(np.sum(right_dev * right_dev))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return np.nan
    value = np.sum(left_dev * right_dev) / (left_norm * right_norm)
    return min(1.0, max(-1.0, value))


@njit(cache=True, nogil=True)
def _spearman_group(
    values: np.ndarray,
    start: int,
    end: int,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算一个分组内的完整成对 Spearman 矩阵。"""

    columns = values.shape[1]
    correlation = np.full((columns, columns), np.nan)
    count = np.zeros((columns, columns), dtype=np.int64)
    rows = end - start
    if rows == 0:
        return correlation, count
    for left_column in range(columns):
        for right_column in range(left_column, columns):
            finite_count = 0
            for row in range(start, end):
                if np.isfinite(values[row, left_column]) and np.isfinite(
                    values[row, right_column]
                ):
                    finite_count += 1
            count[left_column, right_column] = finite_count
            count[right_column, left_column] = finite_count
            if finite_count < min_n:
                continue
            left = np.empty(finite_count, dtype=np.float64)
            right = np.empty(finite_count, dtype=np.float64)
            position = 0
            for row in range(start, end):
                left_value = values[row, left_column]
                right_value = values[row, right_column]
                if np.isfinite(left_value) and np.isfinite(right_value):
                    left[position] = left_value
                    right[position] = right_value
                    position += 1
            left_rank, right_rank = _paired_ranks(left, right)
            value = _pearson(left_rank, right_rank)
            correlation[left_column, right_column] = value
            correlation[right_column, left_column] = value
    return correlation, count


@njit(cache=True, nogil=True, parallel=True)
def _spearman(
    values: np.ndarray,
    offsets: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按分组并行计算完整 Spearman 矩阵。"""

    group_count = offsets.size - 1
    columns = values.shape[1]
    correlation = np.full((group_count, columns, columns), np.nan)
    count = np.zeros((group_count, columns, columns), dtype=np.int64)
    for group in prange(group_count):
        group_corr, group_count_values = _spearman_group(
            values,
            offsets[group],
            offsets[group + 1],
            min_n,
        )
        correlation[group] = group_corr
        count[group] = group_count_values
    return correlation, count


def configure_threads() -> int:
    """按项目线程预算配置 Numba，并返回实际线程数。"""

    threads = native_threads()
    numba.set_num_threads(threads)
    return threads


class NumbaPrototypeBackend:
    """仅用于验证排名和 Spearman 的实验后端。"""

    name = "numba"

    def __init__(self) -> None:
        self._fallback = PythonKernelBackend()

    def daily_rank_pct(self, values: FloatArray, group_offsets: IntArray) -> FloatArray:
        """调用 Numba 排名原型；参数与协议一致。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        configure_threads()
        return _rank_pct(array, offsets)

    def daily_winsorize_mad(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_winsorize_mad requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_winsorize_mad(values, group_offsets, n_mad)

    def daily_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
    ) -> FloatArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_zscore requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_zscore(values, group_offsets)

    def daily_winsorize_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_winsorize_zscore requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_winsorize_zscore(
            values,
            group_offsets,
            n_mad,
        )

    def daily_winsorize_zscore_columns(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """未验证多列核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_winsorize_zscore_columns "
            "requested=numba backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_winsorize_zscore_columns(
            values,
            group_offsets,
            n_mad,
        )

    def daily_winsorize_quantile(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> FloatArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_winsorize_quantile requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_winsorize_quantile(
            values,
            group_offsets,
            lower_quantile,
            upper_quantile,
        )

    def daily_zscore_population(
        self,
        values: FloatArray,
        group_offsets: IntArray,
    ) -> FloatArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_zscore_population requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_zscore_population(values, group_offsets)

    def daily_pearson(
        self,
        left: FloatArray,
        right: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_pearson requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_pearson(left, right, group_offsets, min_n)

    def pairwise_spearman(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
        target_index: int | None = None,
    ) -> tuple[FloatArray, IntArray]:
        """调用 Numba Spearman 原型；参数与协议一致。"""

        array = validate_float_array(values, "values", ndim=2)
        offsets = validate_offsets(group_offsets, array.shape[0])
        minimum = validate_minimum(min_n, name="min_n", minimum=5)
        if target_index is not None:
            logger.info(
                "内核显式回退 kernel=pairwise_spearman requested=numba "
                "backend=python reason=numba_target_not_implemented"
            )
            return self._fallback.pairwise_spearman(
                array,
                offsets,
                minimum,
                target_index,
            )
        configure_threads()
        return _spearman(array, offsets, minimum)

    def daily_ols(
        self,
        design: FloatArray,
        response: FloatArray,
        group_offsets: IntArray,
        min_rows: int = 30,
    ) -> DailyOlsResult:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_ols requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_ols(design, response, group_offsets, min_rows)

    def daily_quantile_bins(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        quantiles: int,
    ) -> IntArray:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_quantile_bins requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_quantile_bins(
            values,
            group_offsets,
            quantiles,
        )

    def bin_returns_multi(
        self,
        values: FloatArray,
        day_codes: IntArray,
        bin_codes: IntArray,
        weights: FloatArray,
        day_count: int,
        bin_count: int,
    ) -> BinReturnsResult:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=bin_returns_multi requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.bin_returns_multi(
            values,
            day_codes,
            bin_codes,
            weights,
            day_count,
            bin_count,
        )

    def bin_turnover(
        self,
        values: FloatArray,
        quantiles: int,
    ) -> BinTurnoverResult:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=bin_turnover requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.bin_turnover(values, quantiles)

    def daily_row_spearman(
        self,
        left: FloatArray,
        right: FloatArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """未验证核显式委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=daily_row_spearman requested=numba "
            "backend=python reason=numba_kernel_not_implemented"
        )
        return self._fallback.daily_row_spearman(left, right, min_n)
