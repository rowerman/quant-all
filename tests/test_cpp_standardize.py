"""C++ 逐日 MAD、z-score 与融合标准化核测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from calculate.normalize import (
    _winsorize_by_day_numpy,
    _winsorize_zscore_by_day_numpy,
    _zscore_by_day_numpy,
    standardize_by_day,
    winsorize_by_day,
    zscore_by_day,
)
from kernels import KernelBackendError
from kernels.backend import group_codes
from kernels.native import CppKernelBackend

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


def _edge_values() -> tuple[np.ndarray, np.ndarray]:
    """构造空组、单样本、无穷、NaN、常数和小 MAD 覆盖。"""

    values = np.array(
        [
            np.nan,
            np.inf,
            1.0,
            2.0,
            1.000000000001,
            1.000000000002,
            3.0,
            4.0,
            5.0,
            5.0,
            5.0,
            100.0,
            -4.0,
            -5.0,
            2.0,
            9.0,
            9.0,
            9.0,
            -np.inf,
        ],
        dtype=np.float64,
    )
    offsets = np.array([0, 0, 1, 3, 6, 11, 19], dtype=np.int64)
    return values, offsets


@requires_native
def test_cpp_standardize_kernels_match_pandas_gold():
    """三个原生核覆盖缺失、无穷、常数、空组和极小 MAD。"""

    values, offsets = _edge_values()
    codes = group_codes(offsets)
    backend = CppKernelBackend()
    expected = {
        "daily_winsorize_mad": _winsorize_by_day_numpy(codes, values),
        "daily_zscore": _zscore_by_day_numpy(codes, values),
        "daily_winsorize_zscore": _winsorize_zscore_by_day_numpy(codes, values),
    }

    for name, gold in expected.items():
        actual = getattr(backend, name)(values, offsets)
        np.testing.assert_allclose(
            actual,
            gold,
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        assert np.array_equal(np.isnan(actual), np.isnan(gold))
        infinite = np.isinf(actual)
        assert np.array_equal(actual[infinite], gold[infinite])


@requires_native
def test_cpp_multi_column_standardize_matches_single_column():
    """多列融合核逐列复用原融合标准化语义。"""

    values, offsets = _edge_values()
    matrix = np.column_stack([values, -values, np.full(values.size, 3.0)])
    backend = CppKernelBackend()

    actual = backend.daily_winsorize_zscore_columns(matrix, offsets)

    for column in range(matrix.shape[1]):
        expected = backend.daily_winsorize_zscore(
            np.ascontiguousarray(matrix[:, column]),
            offsets,
        )
        np.testing.assert_allclose(
            actual[:, column],
            expected,
            atol=1e-12,
            rtol=1e-12,
            equal_nan=True,
        )


@requires_native
def test_public_standardize_routes_restore_original_order(monkeypatch):
    """公开函数经 native 后端计算并恢复打乱后的输入顺序。"""

    import quant_native  # noqa: F401

    rng = np.random.default_rng(1701)
    days = np.repeat(np.arange(7), [1, 2, 5, 9, 13, 21, 34])
    values = rng.normal(size=days.size)
    values[::13] = np.nan
    values[2] = np.inf
    order = rng.permutation(values.size)
    shuffled_days = days[order]
    shuffled_values = values[order]

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = {
        "winsorize": winsorize_by_day(shuffled_days, shuffled_values),
        "zscore": zscore_by_day(shuffled_days, shuffled_values),
        "standardize": standardize_by_day(shuffled_days, shuffled_values),
    }
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual = {
        "winsorize": winsorize_by_day(shuffled_days, shuffled_values),
        "zscore": zscore_by_day(shuffled_days, shuffled_values),
        "standardize": standardize_by_day(shuffled_days, shuffled_values),
    }

    for key in expected:
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )


@requires_native
@pytest.mark.parametrize(
    ("values", "error"),
    [
        (np.ones(5, dtype=np.float32), TypeError),
        (np.ones(6, dtype=np.float64)[::2], ValueError),
        (np.ones((5, 1), dtype=np.float64), ValueError),
        (np.array([], dtype=np.float64), ValueError),
    ],
)
def test_cpp_standardize_direct_array_errors(values, error):
    """直接调用扩展仍检查 dtype、连续性、维度和空输入。"""

    import quant_native

    offsets = np.array([0, values.shape[0]], dtype=np.int64)
    for name in (
        "daily_winsorize_mad",
        "daily_zscore",
        "daily_winsorize_zscore",
    ):
        with pytest.raises(error):
            getattr(quant_native, name)(values, offsets)


@requires_native
@pytest.mark.parametrize("n_mad", [0.0, -1.0, np.inf, np.nan])
def test_cpp_standardize_rejects_invalid_n_mad(n_mad):
    """原生融合与 MAD 接口拒绝非有限正数倍数。"""

    import quant_native

    values = np.arange(5, dtype=np.float64)
    offsets = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        quant_native.daily_winsorize_mad(values, offsets, n_mad)
    with pytest.raises(ValueError):
        quant_native.daily_winsorize_zscore(values, offsets, n_mad)


@requires_native
def test_cpp_standardize_releases_gil():
    """长核运行期间 Python 线程可以继续执行。"""

    rng = np.random.default_rng(1801)
    values = rng.normal(size=2_000_000).astype(np.float64)
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.full(80, 25_000, dtype=np.int64)),
        )
    )
    backend = CppKernelBackend()
    started = threading.Event()
    finished = threading.Event()

    def invoke() -> None:
        started.set()
        backend.daily_winsorize_zscore(values, offsets)
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    assert started.wait(timeout=2)
    counter = 0
    while not finished.is_set():
        counter += 1
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert counter > 1000


@requires_native
def test_cpp_standardize_concurrent_calls_are_isolated():
    """并发重复调用没有跨请求状态污染。"""

    rng = np.random.default_rng(1811)
    values = rng.integers(0, 9, 1_500).astype(np.float64)
    values[::23] = np.nan
    offsets = np.array([0, 400, 1_500], dtype=np.int64)
    backend = CppKernelBackend()
    expected = backend.daily_winsorize_zscore(values, offsets)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: backend.daily_winsorize_zscore(values.copy(), offsets.copy()),
                range(16),
            )
        )

    for result in results:
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )


@requires_native
def test_cpp_standardize_subprocess_stress():
    """独立子进程反复调用三个标准化核不崩溃。"""

    script = """
import numpy as np
import quant_native

rng = np.random.default_rng(1901)
values = rng.normal(size=600).astype(np.float64)
values[::29] = np.nan
values[7] = np.inf
offsets = np.array([0, 300, 600], dtype=np.int64)
for _ in range(50):
    assert quant_native.daily_winsorize_mad(values, offsets).shape == (600,)
    assert quant_native.daily_zscore(values, offsets).shape == (600,)
    assert quant_native.daily_winsorize_zscore(values, offsets).shape == (600,)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@requires_native
def test_cpp_standardize_runtime_errors_are_converted(monkeypatch):
    """已加载扩展的运行期异常转换为统一 KERNEL-004。"""

    backend = CppKernelBackend()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    monkeypatch.setattr(backend._native, "daily_winsorize_zscore", fail)
    with pytest.raises(KernelBackendError) as error:
        backend.daily_winsorize_zscore(
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 3], dtype=np.int64),
        )
    assert error.value.code == "KERNEL-004"
