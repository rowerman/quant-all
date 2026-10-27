"""统一批量核的 Python/NumPy 回退实现。"""

from __future__ import annotations

import numpy as np

from calculate.correlation import _daily_spearman
from calculate.daily import _daily_pearson, rank_pct_by_day
from calculate.exposure import _daily_regression
from calculate.normalize import winsorize_by_day
from kernels.backend import (
    DailyOlsResult,
    FloatArray,
    IntArray,
    group_codes,
    validate_minimum,
    validate_float_array,
    validate_n_mad,
    validate_offsets,
    validate_target_index,
)
from kernels.errors import KernelArrayError


class PythonKernelBackend:
    """直接复用现有评估实现的可靠回退后端。"""

    name = "python"

    def daily_rank_pct(self, values: FloatArray, group_offsets: IntArray) -> FloatArray:
        """按组计算平均名次分位；参数 values 与 group_offsets 已满足接口契约。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        return rank_pct_by_day(group_codes(offsets), array)

    def daily_winsorize_mad(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """按组调用现有 MAD 去极值；参数 n_mad 为裁剪倍数。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        return winsorize_by_day(
            group_codes(offsets),
            array,
            n_mad=validate_n_mad(n_mad),
        )

    def daily_pearson(
        self,
        left: FloatArray,
        right: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """按组过滤非有限配对后计算 Pearson；参数为两列、边界和最小样本。"""

        left_values = validate_float_array(left, "left", ndim=1)
        right_values = validate_float_array(right, "right", ndim=1)
        if left_values.shape != right_values.shape:
            raise KernelArrayError("left 与 right 形状必须一致")
        offsets = validate_offsets(group_offsets, left_values.size)
        minimum = validate_minimum(min_n, name="min_n", minimum=2)
        group_count = offsets.size - 1
        codes = group_codes(offsets)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        finite_codes = codes[finite]
        correlation = np.full(group_count, np.nan, dtype=np.float64)
        count = np.bincount(finite_codes, minlength=group_count).astype(np.int64)
        if finite_codes.size:
            values, _ = _daily_pearson(
                finite_codes,
                left_values[finite],
                right_values[finite],
                minimum,
            )
            positions = np.unique(finite_codes)
            correlation[positions] = values[positions]
        return correlation, count

    def pairwise_spearman(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
        target_index: int | None = None,
    ) -> tuple[FloatArray, IntArray]:
        """按组调用现有 Spearman，并补齐有效配对数。"""

        array = validate_float_array(values, "values", ndim=2)
        offsets = validate_offsets(group_offsets, array.shape[0])
        minimum = validate_minimum(min_n, name="min_n", minimum=5)
        target = validate_target_index(target_index, array.shape[1])
        group_count = offsets.size - 1
        columns = array.shape[1]
        correlation = np.full(
            (group_count, columns, columns),
            np.nan,
            dtype=np.float64,
        )
        count = np.zeros((group_count, columns, columns), dtype=np.int64)
        for group in range(group_count):
            start, end = int(offsets[group]), int(offsets[group + 1])
            finite = np.isfinite(array[start:end])
            group_count_values = finite.T.astype(np.int64) @ finite.astype(np.int64)
            if target is None:
                count[group] = group_count_values
            else:
                count[group, target] = group_count_values[target]
                count[group, :, target] = group_count_values[:, target]
            if end > start:
                matrix = _daily_spearman(
                    np.ascontiguousarray(array[start:end]),
                    target_index=target,
                )
                if minimum > 5:
                    count_mask = count[group] if target is None else count[group, target]
                    if target is None:
                        matrix[count_mask < minimum] = np.nan
                    else:
                        invalid = count_mask < minimum
                        matrix[target, invalid] = np.nan
                        matrix[invalid, target] = np.nan
                correlation[group] = matrix
        return correlation, count

    def daily_ols(
        self,
        design: FloatArray,
        response: FloatArray,
        group_offsets: IntArray,
        min_rows: int = 30,
    ) -> DailyOlsResult:
        """按组调用现有截面 OLS；参数为设计矩阵、响应、边界和最少行数。"""

        design_values = validate_float_array(design, "design", ndim=2)
        response_values = validate_float_array(response, "response", ndim=1)
        if design_values.shape[0] != response_values.size:
            raise KernelArrayError("design 行数必须与 response 长度一致")
        offsets = validate_offsets(group_offsets, response_values.size)
        minimum = validate_minimum(min_rows, name="min_rows", minimum=1)
        codes = group_codes(offsets)
        result = _daily_regression(
            codes,
            offsets.size - 1,
            design_values,
            response_values,
            np.arange(response_values.size, dtype=np.int64),
            min_rows=minimum,
        )
        return {
            "days": np.asarray(result["days"], dtype=np.int64),
            "coefficients": np.asarray(result["coefficients"], dtype=np.float64),
            "t_stats": np.asarray(result["t_stats"], dtype=np.float64),
            "r2": np.asarray(result["r2"], dtype=np.float64),
            "rows": np.asarray(result["rows"], dtype=np.int64),
            "residual_rows": np.asarray(
                result["residual_rows"],
                dtype=np.int64,
            ),
            "residual_values": np.asarray(
                result["residual_values"],
                dtype=np.float64,
            ),
            "skipped": int(result["skipped"]),
        }
