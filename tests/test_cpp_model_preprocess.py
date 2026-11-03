"""模型预处理分位裁剪和总体标准差核测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import threading

import numpy as np
import pandas as pd
import pytest

from kernels import KernelParameterError, get_backend
from kernels.backend import group_codes
from kernels.native import CppKernelBackend

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


def _edge_values() -> tuple[np.ndarray, np.ndarray]:
    """构造空组、单样本、缺失、无穷、常数和并列数据。"""

    values = np.array(
        [
            np.nan,
            3.0,
            3.0,
            3.0,
            1.0,
            2.0,
            np.nan,
            100.0,
            -5.0,
            4.0,
            np.inf,
            -np.inf,
            7.0,
            8.0,
            9.0,
            10.0,
        ],
        dtype=np.float64,
    )
    offsets = np.array([0, 0, 1, 4, 7, 12, 16], dtype=np.int64)
    return values, offsets


def _pandas_quantile_gold(
    values: np.ndarray,
    offsets: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """独立构造 pandas 分位裁剪金标准。"""

    frame = pd.DataFrame(
        {
            "group": group_codes(offsets),
            "value": np.where(np.isfinite(values), values, np.nan),
        }
    )
    grouped = frame.groupby("group", sort=False)["value"]
    lower = grouped.transform(lambda item: item.quantile(lower_quantile))
    upper = grouped.transform(lambda item: item.quantile(upper_quantile))
    return frame["value"].clip(lower=lower, upper=upper).to_numpy()


def _pandas_zscore_gold(
    values: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """独立构造 pandas 总体标准差金标准。"""

    frame = pd.DataFrame(
        {
            "group": group_codes(offsets),
            "value": np.where(np.isfinite(values), values, np.nan),
        }
    )
    grouped = frame.groupby("group", sort=False)["value"]
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform("std", ddof=0)
    return ((frame["value"] - mean) / standard_deviation.replace(0.0, np.nan)).to_numpy()


@requires_native
def test_cpp_quantile_winsorize_matches_pandas_gold():
    """分位裁剪保持缺失、无穷、空组、常数和插值语义。"""

    values, offsets = _edge_values()
    backend = CppKernelBackend()

    for lower, upper in ((0.01, 0.99), (0.1, 0.9), (0.0, 1.0)):
        actual = backend.daily_winsorize_quantile(
            values,
            offsets,
            lower,
            upper,
        )
        expected = _pandas_quantile_gold(values, offsets, lower, upper)
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


@requires_native
def test_cpp_population_zscore_matches_pandas_gold():
    """总体 z-score 与 pandas ddof=0 金标准一致。"""

    values, offsets = _edge_values()
    actual = CppKernelBackend().daily_zscore_population(values, offsets)
    expected = _pandas_zscore_gold(values, offsets)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-10,
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


@requires_native
def test_cpp_population_zscore_matches_pandas_groupby_float64():
    """总体 z-score 对大样本与 pandas groupby 输出逐值一致。"""

    rng = np.random.default_rng(20261029)
    values = rng.normal(size=3_000).astype(np.float64)
    values[::37] = np.nan
    offsets = np.array([0, 0, 1_000, 3_000], dtype=np.int64)
    actual = CppKernelBackend().daily_zscore_population(values, offsets)
    expected = _pandas_zscore_gold(values, offsets)

    np.testing.assert_array_equal(actual, expected)


@requires_native
def test_cpp_preprocess_kernels_are_isolated_under_concurrency():
    """并发调用不共享跨请求状态。"""

    rng = np.random.default_rng(20261027)
    values = rng.normal(size=20_000).astype(np.float64)
    values[::97] = np.nan
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.full(80, 250, dtype=np.int64)),
        )
    )
    backend = CppKernelBackend()
    expected_cut = backend.daily_winsorize_quantile(values, offsets)
    expected_z = backend.daily_zscore_population(values, offsets)

    with ThreadPoolExecutor(max_workers=4) as executor:
        calls = [
            lambda: backend.daily_winsorize_quantile(values.copy(), offsets.copy())
            for _ in range(8)
        ] + [
            lambda: backend.daily_zscore_population(values.copy(), offsets.copy())
            for _ in range(8)
        ]
        results = list(executor.map(lambda call: call(), calls))

    for result in results[:8]:
        np.testing.assert_array_equal(result, expected_cut)
    for result in results[8:]:
        np.testing.assert_array_equal(result, expected_z)


@requires_native
def test_cpp_preprocess_kernels_release_gil():
    """较长核运行期间 Python 线程可以继续执行。"""

    rng = np.random.default_rng(20261028)
    values = rng.normal(size=1_000_000).astype(np.float64)
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.full(100, 10_000, dtype=np.int64)),
        )
    )
    backend = CppKernelBackend()
    finished = threading.Event()

    def invoke() -> None:
        backend.daily_winsorize_quantile(values, offsets)
        backend.daily_zscore_population(values, offsets)
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    counter = 0
    while not finished.is_set():
        counter += 1
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert counter > 1000


@requires_native
@pytest.mark.parametrize(
    ("lower", "upper"),
    [(-0.1, 0.9), (0.1, 1.1), (0.8, 0.2), (0.5, 0.5), (np.nan, 0.9)],
)
def test_cpp_quantile_rejects_invalid_bounds(lower, upper):
    """直接调用分位接口时拒绝非法边界。"""

    import quant_native

    values = np.arange(5, dtype=np.float64)
    offsets = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        quant_native.daily_winsorize_quantile(
            values,
            offsets,
            lower,
            upper,
        )


def test_python_backend_rejects_invalid_quantiles():
    """Python 回退与原生接口使用相同分位校验。"""

    values = np.arange(5, dtype=np.float64)
    offsets = np.array([0, 5], dtype=np.int64)
    with pytest.raises(KernelParameterError):
        get_backend("python").backend.daily_winsorize_quantile(
            values,
            offsets,
            0.9,
            0.1,
        )
