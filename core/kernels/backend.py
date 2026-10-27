"""与具体加速实现无关的批量计算核契约和后端选择。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from common.logging_conf import get_logger
from common.paths import LOGS_DIR
from kernels.errors import (
    KernelArrayError,
    KernelBackendError,
    KernelOffsetError,
    KernelParameterError,
)

logger = get_logger("kernels.backend", LOGS_DIR / "core")

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class DailyOlsResult(TypedDict):
    """逐日 OLS 核的固定结果结构。"""

    days: IntArray
    coefficients: FloatArray
    t_stats: FloatArray
    r2: FloatArray
    rows: IntArray
    residual_rows: IntArray
    residual_values: FloatArray
    skipped: int
    skip_insufficient_rows: int
    skip_invalid_response: int
    skip_rank_deficient: int
    active: BoolArray


class BinReturnsResult(TypedDict):
    """多标签分位收益核的固定结果结构。"""

    returns: FloatArray
    pools: FloatArray


class BinTurnoverResult(TypedDict):
    """分位换手核的固定结果结构。"""

    per_bin: FloatArray
    overall: FloatArray


class NativeKernelBackend(Protocol):
    """批量数值核的统一协议。"""

    name: str

    def daily_rank_pct(self, values: FloatArray, group_offsets: IntArray) -> FloatArray:
        """按组计算分位排名，并列取平均名次。"""

    def daily_winsorize_mad(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """按组使用中位数和 MAD 裁剪极值。"""

    def daily_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
    ) -> FloatArray:
        """按组计算样本标准差口径的 z-score。"""

    def daily_winsorize_zscore(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """按组融合 MAD 去极值和样本标准差 z-score。"""

    def daily_winsorize_zscore_columns(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        n_mad: float = 3.0,
    ) -> FloatArray:
        """按组对多列融合 MAD 去极值和样本标准差 z-score。"""

    def daily_pearson(
        self,
        left: FloatArray,
        right: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """按组计算 Pearson 相关系数和有效配对数。"""

    def pairwise_spearman(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        min_n: int = 5,
        target_index: int | None = None,
    ) -> tuple[FloatArray, IntArray]:
        """按组计算完整相关矩阵或单目标行及有效配对数。"""

    def daily_ols(
        self,
        design: FloatArray,
        response: FloatArray,
        group_offsets: IntArray,
        min_rows: int = 30,
    ) -> DailyOlsResult:
        """按组计算截面 OLS、t 值、残差和跳过原因。"""

    def daily_quantile_bins(
        self,
        values: FloatArray,
        group_offsets: IntArray,
        quantiles: int,
    ) -> IntArray:
        """按组计算 0 起始的分位组号，无效样本返回 -1。"""

    def bin_returns_multi(
        self,
        values: FloatArray,
        day_codes: IntArray,
        bin_codes: IntArray,
        weights: FloatArray,
        day_count: int,
        bin_count: int,
    ) -> BinReturnsResult:
        """一次计算多标签分位收益与池内均值。"""

    def bin_turnover(
        self,
        values: FloatArray,
        quantiles: int,
    ) -> BinTurnoverResult:
        """按完整交易日矩阵计算全池与各分位换手。"""

    def daily_row_spearman(
        self,
        left: FloatArray,
        right: FloatArray,
        min_n: int = 5,
    ) -> tuple[FloatArray, IntArray]:
        """按行重新排名后计算 Spearman 和有效配对数。"""


@dataclass(frozen=True)
class BackendSelection:
    """一次后端选择结果及显式回退原因。"""

    backend: NativeKernelBackend
    requested: str
    fallback_reason: str | None

    @property
    def name(self) -> str:
        """返回实际后端名称。"""

        return self.backend.name


def validate_float_array(values: object, name: str, *, ndim: int) -> FloatArray:
    """校验 C 连续 float64 数组；参数 values 为待检查输入。"""

    if not isinstance(values, np.ndarray):
        raise KernelArrayError(f"{name} 必须是 numpy.ndarray")
    if values.dtype != np.dtype("float64"):
        raise KernelArrayError(f"{name} 必须是 float64，实际为 {values.dtype}")
    if values.ndim != ndim:
        raise KernelArrayError(f"{name} 必须是 {ndim} 维，实际为 {values.ndim} 维")
    if not values.flags.c_contiguous:
        raise KernelArrayError(f"{name} 必须是 C 连续数组")
    if values.size == 0:
        raise KernelArrayError(f"{name} 不能为空")
    return values


def validate_offsets(group_offsets: object, sample_count: int) -> IntArray:
    """校验 C 连续 int64 分组边界；参数 sample_count 为第一维样本数。"""

    if not isinstance(group_offsets, np.ndarray):
        raise KernelOffsetError("group_offsets 必须是 numpy.ndarray")
    if group_offsets.dtype != np.dtype("int64"):
        raise KernelOffsetError(
            f"group_offsets 必须是 int64，实际为 {group_offsets.dtype}"
        )
    if group_offsets.ndim != 1:
        raise KernelOffsetError("group_offsets 必须是一维数组")
    if not group_offsets.flags.c_contiguous:
        raise KernelOffsetError("group_offsets 必须是 C 连续数组")
    if group_offsets.size < 2:
        raise KernelOffsetError("group_offsets 至少包含起始和结束边界")
    if int(group_offsets[0]) != 0 or int(group_offsets[-1]) != sample_count:
        raise KernelOffsetError("group_offsets 必须从 0 开始并以样本数结束")
    if np.any(np.diff(group_offsets) < 0):
        raise KernelOffsetError("group_offsets 必须单调不减")
    return group_offsets


def validate_minimum(min_n: object, *, name: str, minimum: int) -> int:
    """校验最小样本参数；参数 name 用于错误信息。"""

    if type(min_n) is not int or min_n < minimum:
        raise KernelParameterError(f"{name} 必须是不小于 {minimum} 的整数")
    return min_n


def validate_target_index(target_index: object, columns: int) -> int | None:
    """校验相关性目标列；参数 columns 为输入矩阵列数。"""

    if target_index is None:
        return None
    if type(target_index) is not int or not 0 <= target_index < columns:
        raise KernelParameterError("target_index 必须是矩阵列范围内的整数")
    return target_index


def validate_codes(
    values: object,
    name: str,
    *,
    length: int,
    maximum: int | None = None,
) -> IntArray:
    """校验一维 C 连续 int64 编码；参数 maximum 为开区间上界。"""

    if not isinstance(values, np.ndarray):
        raise KernelArrayError(f"{name} 必须是 numpy.ndarray")
    if values.dtype != np.dtype("int64"):
        raise KernelArrayError(f"{name} 必须是 int64，实际为 {values.dtype}")
    if values.ndim != 1:
        raise KernelArrayError(f"{name} 必须是一维数组")
    if not values.flags.c_contiguous:
        raise KernelArrayError(f"{name} 必须是 C 连续数组")
    if values.size != length:
        raise KernelArrayError(f"{name} 长度必须为 {length}")
    if values.size and int(values.min()) < 0:
        raise KernelArrayError(f"{name} 不能包含负编码")
    if maximum is not None and values.size and int(values.max()) >= maximum:
        raise KernelArrayError(f"{name} 编码必须小于 {maximum}")
    return values


def validate_n_mad(n_mad: object) -> float:
    """校验 MAD 倍数；参数 n_mad 必须为有限正数。"""

    if isinstance(n_mad, bool) or not isinstance(n_mad, (int, float)):
        raise KernelParameterError("n_mad 必须是有限正数")
    value = float(n_mad)
    if not np.isfinite(value) or value <= 0:
        raise KernelParameterError("n_mad 必须是有限正数")
    return value


def group_codes(group_offsets: IntArray) -> IntArray:
    """把分组边界转换为逐样本组号；参数 group_offsets 已通过校验。"""

    lengths = np.diff(group_offsets)
    return np.repeat(
        np.arange(group_offsets.size - 1, dtype=np.int64),
        lengths,
    )


def native_threads() -> int:
    """返回 `QUANT_NATIVE_THREADS`，默认单线程。"""

    raw = os.environ.get("QUANT_NATIVE_THREADS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise KernelBackendError("QUANT_NATIVE_THREADS 必须是正整数") from exc
    if value < 1:
        raise KernelBackendError("QUANT_NATIVE_THREADS 必须是正整数")
    return value


def get_backend(requested: str | None = None) -> BackendSelection:
    """按显式参数或环境变量选择后端；默认优先 native。"""

    from kernels.fallback import PythonKernelBackend

    if requested is None:
        requested = os.environ.get("QUANT_KERNEL_BACKEND", "auto")
        fallback_reason = None
    else:
        fallback_reason = None
    name = str(requested).strip().lower()
    if name == "python":
        selection = BackendSelection(PythonKernelBackend(), name, fallback_reason)
    elif name == "auto":
        try:
            from kernels.native import CppKernelBackend

            selection = BackendSelection(CppKernelBackend(), "auto", None)
        except KernelBackendError:
            selection = BackendSelection(
                PythonKernelBackend(),
                name,
                "cpp_extension_unavailable",
            )
    elif name == "numba":
        try:
            from kernels.prototype import NumbaPrototypeBackend
        except ImportError as exc:
            raise KernelBackendError("请求的 Numba 后端不可用") from exc
        selection = BackendSelection(NumbaPrototypeBackend(), name, None)
    elif name == "native":
        try:
            from kernels.native import CppKernelBackend
        except ImportError as exc:
            raise KernelBackendError("请求的 native 后端不可用") from exc
        selection = BackendSelection(CppKernelBackend(), name, None)
    else:
        raise KernelBackendError(
            f"未知内核后端 {requested!r}，仅支持 auto、python、numba 或 native"
        )
    logger.info(
        "内核后端选择 requested=%s backend=%s fallback_reason=%s",
        requested,
        selection.name,
        selection.fallback_reason,
    )
    return selection
