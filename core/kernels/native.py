"""C++ 批量计算核适配层。"""

from __future__ import annotations

import time
from types import ModuleType
from typing import Any

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
    validate_codes,
    validate_float_array,
    validate_minimum,
    validate_n_mad,
    validate_offsets,
    validate_target_index,
)
from kernels.errors import KernelArrayError, KernelBackendError
from kernels.fallback import PythonKernelBackend

logger = get_logger("kernels.native", LOGS_DIR / "core")


def _merge_ols_results(
    native_result: DailyOlsResult,
    fallback_results: list[tuple[int, int, DailyOlsResult]],
) -> DailyOlsResult:
    """合并原生正常日与 Python 回退日的 OLS 结果。"""

    if not fallback_results:
        return native_result
    day_results = []
    for position, day in enumerate(native_result["days"]):
        day_results.append({
            "day": int(day),
            "coefficients": native_result["coefficients"][position],
            "t_stats": native_result["t_stats"][position],
            "r2": native_result["r2"][position],
            "rows": native_result["rows"][position],
            "active": native_result["active"][position],
        })
    residual_rows = [native_result["residual_rows"]]
    residual_values = [native_result["residual_values"]]
    skip_insufficient_rows = native_result["skip_insufficient_rows"]
    skip_invalid_response = native_result["skip_invalid_response"]
    skip_rank_deficient = native_result["skip_rank_deficient"]
    for group, start, result in fallback_results:
        for position, relative_day in enumerate(result["days"]):
            day_results.append({
                "day": group + int(relative_day),
                "coefficients": result["coefficients"][position],
                "t_stats": result["t_stats"][position],
                "r2": result["r2"][position],
                "rows": result["rows"][position],
                "active": result["active"][position],
            })
        residual_rows.append(result["residual_rows"] + start)
        residual_values.append(result["residual_values"])
        skip_insufficient_rows += result["skip_insufficient_rows"]
        skip_invalid_response += result["skip_invalid_response"]
        skip_rank_deficient += result["skip_rank_deficient"]

    day_results.sort(key=lambda item: item["day"])
    rows = np.concatenate(residual_rows)
    values = np.concatenate(residual_values)
    order = np.argsort(rows, kind="stable")
    return {
        "days": np.asarray([item["day"] for item in day_results], dtype=np.int64),
        "coefficients": (
            np.vstack([item["coefficients"] for item in day_results])
            if day_results
            else np.empty((0, native_result["coefficients"].shape[1]))
        ),
        "t_stats": (
            np.vstack([item["t_stats"] for item in day_results])
            if day_results
            else np.empty((0, native_result["t_stats"].shape[1]))
        ),
        "r2": np.asarray([item["r2"] for item in day_results], dtype=np.float64),
        "rows": np.asarray([item["rows"] for item in day_results], dtype=np.int64),
        "residual_rows": rows[order].astype(np.int64, copy=False),
        "residual_values": values[order],
        "skipped": (
            skip_insufficient_rows
            + skip_invalid_response
            + skip_rank_deficient
        ),
        "skip_insufficient_rows": skip_insufficient_rows,
        "skip_invalid_response": skip_invalid_response,
        "skip_rank_deficient": skip_rank_deficient,
        "active": (
            np.vstack([item["active"] for item in day_results])
            if day_results
            else np.empty(
                (0, native_result["active"].shape[1]),
                dtype=np.bool_,
            )
        ),
    }


class CppKernelBackend:
    """显式调用 C++ 扩展的批量核后端。"""

    name = "native"

    def __init__(self) -> None:
        """加载 quant_native；扩展缺失时明确失败。"""

        try:
            import quant_native
        except (ImportError, OSError) as exc:
            raise KernelBackendError("请求的 native 后端不可用") from exc
        self._native: ModuleType = quant_native
        self._fallback = PythonKernelBackend()

    def _fallback_unsupported(self, kernel: str, function: Any, *args, **kwargs):
        """记录未实现核并委托 Python 回退。"""

        logger.info(
            "内核显式回退 kernel=%s requested=native backend=python "
            "reason=cpp_kernel_not_implemented",
            kernel,
        )
        return function(*args, **kwargs)

    def daily_rank_pct(self, values: FloatArray, group_offsets: IntArray) -> FloatArray:
        """调用 C++ 逐日排名核；参数与统一协议一致。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_rank_pct(array, offsets)
        except Exception as exc:
            raise KernelBackendError(f"C++ 逐日排名计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_rank_pct requested=native backend=native "
            "rows=%d groups=%d threads=%d elapsed_ms=%.3f",
            array.size,
            offsets.size - 1,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.float64)

    def daily_winsorize_mad(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """调用 C++ 逐日 MAD 去极值核；参数与统一协议一致。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        multiplier = validate_n_mad(n_mad)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_winsorize_mad(
                array,
                offsets,
                multiplier,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ MAD 去极值计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_winsorize_mad requested=native backend=native "
            "rows=%d groups=%d n_mad=%.6g threads=%d elapsed_ms=%.3f",
            array.size,
            offsets.size - 1,
            multiplier,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.float64)

    def daily_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
    ) -> FloatArray:
        """调用 C++ 逐日 z-score 核；参数与统一协议一致。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_zscore(array, offsets)
        except Exception as exc:
            raise KernelBackendError(f"C++ z-score 计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_zscore requested=native backend=native "
            "rows=%d groups=%d threads=%d elapsed_ms=%.3f",
            array.size,
            offsets.size - 1,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.float64)

    def daily_winsorize_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """调用 C++ 融合标准化核；参数与统一协议一致。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        multiplier = validate_n_mad(n_mad)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_winsorize_zscore(
                array,
                offsets,
                multiplier,
            )
        except Exception as exc:
            raise KernelBackendError(
                f"C++ 融合 MAD/z-score 计算失败：{exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_winsorize_zscore requested=native backend=native "
            "rows=%d groups=%d n_mad=%.6g threads=%d elapsed_ms=%.3f",
            array.size,
            offsets.size - 1,
            multiplier,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.float64)

    def daily_winsorize_zscore_columns(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """调用 C++ 多列融合标准化核；参数与统一协议一致。"""

        matrix = validate_float_array(values, "values", ndim=2)
        offsets = validate_offsets(group_offsets, matrix.shape[0])
        multiplier = validate_n_mad(n_mad)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_winsorize_zscore_columns(
                matrix,
                offsets,
                multiplier,
            )
        except Exception as exc:
            raise KernelBackendError(
                f"C++ 多列融合 MAD/z-score 计算失败：{exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_winsorize_zscore_columns "
            "requested=native backend=native rows=%d groups=%d columns=%d "
            "n_mad=%.6g threads=%d elapsed_ms=%.3f",
            matrix.shape[0],
            offsets.size - 1,
            matrix.shape[1],
            multiplier,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.float64)

    def daily_pearson(
        self,
        left: FloatArray,
        right: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """调用 C++ 逐日 Pearson 核；参数为两列、边界和最小样本。"""

        left_values = validate_float_array(left, "left", ndim=1)
        right_values = validate_float_array(right, "right", ndim=1)
        if left_values.shape != right_values.shape:
            raise KernelArrayError("left 与 right 形状必须一致")
        offsets = validate_offsets(group_offsets, left_values.size)
        minimum = validate_minimum(min_n, name="min_n", minimum=2)
        threads = native_threads()
        started = time.perf_counter()
        try:
            correlation, count = self._native.daily_pearson(
                left_values,
                right_values,
                offsets,
                minimum,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ Pearson 计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_pearson requested=native backend=native "
            "rows=%d groups=%d min_n=%d threads=%d elapsed_ms=%.3f",
            left_values.size,
            offsets.size - 1,
            minimum,
            threads,
            elapsed_ms,
        )
        return (
            np.asarray(correlation, dtype=np.float64),
            np.asarray(count, dtype=np.int64),
        )

    def pairwise_spearman(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
        target_index: int | None = None,
    ) -> tuple[FloatArray, IntArray]:
        """调用 C++ 成对 Spearman 核；参数与统一协议一致。"""

        array = validate_float_array(values, "values", ndim=2)
        offsets = validate_offsets(group_offsets, array.shape[0])
        minimum = validate_minimum(min_n, name="min_n", minimum=5)
        target = validate_target_index(target_index, array.shape[1])
        threads = native_threads()
        started = time.perf_counter()
        try:
            correlation, count = self._native.pairwise_spearman(
                array,
                offsets,
                minimum,
                target,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ Spearman 计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=pairwise_spearman requested=native backend=native "
            "target_index=%s rows=%d groups=%d columns=%d threads=%d elapsed_ms=%.3f",
            "none" if target is None else target,
            array.shape[0],
            offsets.size - 1,
            array.shape[1],
            threads,
            elapsed_ms,
        )
        return (
            np.asarray(correlation, dtype=np.float64),
            np.asarray(count, dtype=np.int64),
        )

    def daily_ols(
        self,
        design: FloatArray,
        response: FloatArray,
        group_offsets: IntArray,
        min_rows: int = 30,
    ) -> DailyOlsResult:
        """调用 C++ 逐日 OLS；病态矩阵按日显式回退 Python。"""

        design_values = validate_float_array(design, "design", ndim=2)
        response_values = validate_float_array(response, "response", ndim=1)
        if design_values.shape[0] != response_values.size:
            raise KernelArrayError("design 行数必须与 response 长度一致")
        offsets = validate_offsets(group_offsets, response_values.size)
        minimum = validate_minimum(min_rows, name="min_rows", minimum=1)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_ols(
                design_values,
                response_values,
                offsets,
                minimum,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ OLS 计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_ols requested=native backend=native "
            "rows=%d groups=%d columns=%d min_rows=%d threads=%d elapsed_ms=%.3f",
            design_values.shape[0],
            offsets.size - 1,
            design_values.shape[1],
            minimum,
            threads,
            elapsed_ms,
        )
        converted: DailyOlsResult = {
            "days": np.asarray(result["days"], dtype=np.int64),
            "coefficients": np.asarray(
                result["coefficients"],
                dtype=np.float64,
            ),
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
            "skip_insufficient_rows": int(
                result["skip_insufficient_rows"]
            ),
            "skip_invalid_response": int(
                result["skip_invalid_response"]
            ),
            "skip_rank_deficient": int(
                result["skip_rank_deficient"]
            ),
            "active": np.asarray(result["active"], dtype=np.bool_),
        }
        fallback_results = []
        for group in np.asarray(result["fallback_days"], dtype=np.int64):
            start = int(offsets[group])
            end = int(offsets[group + 1])
            fallback_results.append((
                int(group),
                start,
                self._fallback.daily_ols(
                    np.ascontiguousarray(design_values[start:end]),
                    np.ascontiguousarray(response_values[start:end]),
                    np.asarray([0, end - start], dtype=np.int64),
                    minimum,
                ),
            ))
        if fallback_results:
            logger.info(
                "内核逐日回退 kernel=daily_ols requested=native backend=python "
                "reason=native_condition_guard groups=%d",
                len(fallback_results),
            )
        return _merge_ols_results(converted, fallback_results)

    def daily_quantile_bins(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        quantiles: int,
    ) -> IntArray:
        """调用 C++ 平均并列名次分位组核。"""

        array = validate_float_array(values, "values", ndim=1)
        offsets = validate_offsets(group_offsets, array.size)
        count = validate_minimum(quantiles, name="quantiles", minimum=2)
        threads = native_threads()
        started = time.perf_counter()
        try:
            result = self._native.daily_quantile_bins(
                array,
                offsets,
                count,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ 分位分组计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_quantile_bins requested=native backend=native "
            "rows=%d groups=%d quantiles=%d threads=%d elapsed_ms=%.3f",
            array.size,
            offsets.size - 1,
            count,
            threads,
            elapsed_ms,
        )
        return np.asarray(result, dtype=np.int64)

    def bin_returns_multi(
        self,
        values: FloatArray,
        day_codes: IntArray,
        bin_codes: IntArray,
        weights: FloatArray,
        day_count: int,
        bin_count: int,
    ) -> BinReturnsResult:
        """调用 C++ 多标签分位收益核。"""

        matrix = validate_float_array(values, "values", ndim=2)
        days = validate_minimum(day_count, name="day_count", minimum=1)
        bins = validate_minimum(bin_count, name="bin_count", minimum=1)
        day_values = validate_codes(
            day_codes,
            "day_codes",
            length=matrix.shape[0],
            maximum=days,
        )
        bin_values = validate_codes(
            bin_codes,
            "bin_codes",
            length=matrix.shape[0],
            maximum=bins,
        )
        weight_values = validate_float_array(weights, "weights", ndim=1)
        if weight_values.size != matrix.shape[0]:
            raise KernelArrayError("weights 长度必须与 values 行数一致")
        threads = native_threads()
        started = time.perf_counter()
        try:
            returns, pools = self._native.bin_returns_multi(
                matrix,
                day_values,
                bin_values,
                weight_values,
                days,
                bins,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ 分位收益计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=bin_returns_multi requested=native backend=native "
            "rows=%d days=%d bins=%d labels=%d threads=%d elapsed_ms=%.3f",
            matrix.shape[0],
            days,
            bins,
            matrix.shape[1],
            threads,
            elapsed_ms,
        )
        return {
            "returns": np.asarray(returns, dtype=np.float64),
            "pools": np.asarray(pools, dtype=np.float64),
        }

    def bin_turnover(
        self,
        values: FloatArray,
        quantiles: int,
    ) -> BinTurnoverResult:
        """调用 C++ 分位换手核。"""

        matrix = validate_float_array(values, "values", ndim=2)
        count = validate_minimum(quantiles, name="quantiles", minimum=1)
        threads = native_threads()
        started = time.perf_counter()
        try:
            per_bin, overall = self._native.bin_turnover(matrix, count)
        except Exception as exc:
            raise KernelBackendError(f"C++ 分位换手计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=bin_turnover requested=native backend=native "
            "days=%d stocks=%d quantiles=%d threads=%d elapsed_ms=%.3f",
            matrix.shape[0],
            matrix.shape[1],
            count,
            threads,
            elapsed_ms,
        )
        return {
            "per_bin": np.asarray(per_bin, dtype=np.float64),
            "overall": np.asarray(overall, dtype=np.float64),
        }

    def daily_row_spearman(
        self,
        left: FloatArray,
        right: FloatArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """调用 C++ 逐行 Spearman 核。"""

        left_values = validate_float_array(left, "left", ndim=2)
        right_values = validate_float_array(right, "right", ndim=2)
        if left_values.shape != right_values.shape:
            raise KernelArrayError("left 与 right 形状必须一致")
        minimum = validate_minimum(min_n, name="min_n", minimum=2)
        threads = native_threads()
        started = time.perf_counter()
        try:
            correlation, count = self._native.daily_row_spearman(
                left_values,
                right_values,
                minimum,
            )
        except Exception as exc:
            raise KernelBackendError(f"C++ 逐行 Spearman 计算失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "内核执行 kernel=daily_row_spearman requested=native backend=native "
            "days=%d stocks=%d min_n=%d threads=%d elapsed_ms=%.3f",
            left_values.shape[0],
            left_values.shape[1],
            minimum,
            threads,
            elapsed_ms,
        )
        return (
            np.asarray(correlation, dtype=np.float64),
            np.asarray(count, dtype=np.int64),
        )
