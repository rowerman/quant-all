"""C++ 批量计算核适配层。"""

from __future__ import annotations

import time
from types import ModuleType
from typing import Any

import numpy as np

from common.logging_conf import get_logger
from common.paths import LOGS_DIR
from kernels.backend import (
    DailyOlsResult,
    FloatArray,
    IntArray,
    native_threads,
    validate_float_array,
    validate_minimum,
    validate_offsets,
    validate_target_index,
)
from kernels.errors import KernelArrayError, KernelBackendError
from kernels.fallback import PythonKernelBackend

logger = get_logger("kernels.native", LOGS_DIR / "core")


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
        """MAD 去极值暂未使用 C++，显式委托 Python 回退。"""

        return self._fallback_unsupported(
            "daily_winsorize_mad",
            self._fallback.daily_winsorize_mad,
            values,
            group_offsets,
            n_mad,
        )

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
        """OLS 暂未使用 C++，显式委托 Python 回退。"""

        return self._fallback_unsupported(
            "daily_ols",
            self._fallback.daily_ols,
            design,
            response,
            group_offsets,
            min_rows,
        )
