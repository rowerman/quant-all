"""C++ 逐日排名与 Pearson 核的契约、边界和并发测试。"""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from kernels import KernelBackendError, get_backend
from kernels.native import CppKernelBackend

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


@requires_native
def test_cpp_daily_rank_matches_python_edge_cases():
    """排名核保持并列、缺失、无穷、空组和原位置语义。"""

    values = np.array(
        [
            3.0,
            1.0,
            1.0,
            np.nan,
            2.0,
            np.inf,
            7.0,
            -np.inf,
            4.0,
            2.0,
            2.0,
        ],
        dtype=np.float64,
    )
    offsets = np.array([0, 0, 7, 11], dtype=np.int64)

    actual = CppKernelBackend().daily_rank_pct(values, offsets)
    expected = get_backend("python").backend.daily_rank_pct(values, offsets)

    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(actual[3])


@requires_native
def test_cpp_daily_pearson_matches_python_and_counts():
    """Pearson 核保持有限配对、不足样本、常数和自定义门槛语义。"""

    rng = np.random.default_rng(601)
    sizes = np.array([0, 8, 31, 5, 17], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    rows = int(offsets[-1])
    left = rng.normal(size=rows).astype(np.float64)
    right = (0.7 * left + rng.normal(scale=0.4, size=rows)).astype(np.float64)
    left[[2, 13, 42]] = np.nan
    right[[5, 13, 48]] = np.inf
    left[37:42] = 3.0
    right[37:42] = 4.0

    actual, actual_count = CppKernelBackend().daily_pearson(
        left,
        right,
        offsets,
        7,
    )
    expected, expected_count = get_backend("python").backend.daily_pearson(
        left,
        right,
        offsets,
        7,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-10,
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual_count, expected_count)
    assert actual_count[0] == 0
    assert np.isnan(actual[0])


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
def test_cpp_daily_rank_direct_contract_errors(values, error):
    """直接调用排名扩展仍检查 dtype、连续性、维度和空输入。"""

    import quant_native

    with pytest.raises(error):
        quant_native.daily_rank_pct(
            values,
            np.array([0, values.shape[0]], dtype=np.int64),
        )


@requires_native
def test_cpp_daily_pearson_direct_contract_errors():
    """直接调用 Pearson 扩展仍检查长度、offset 和最小样本。"""

    import quant_native

    values = np.ones(5, dtype=np.float64)
    offsets = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        quant_native.daily_pearson(values, np.ones(4), offsets, 5)
    with pytest.raises(ValueError):
        quant_native.daily_pearson(
            values,
            values,
            np.array([1, 5], dtype=np.int64),
            5,
        )
    with pytest.raises(ValueError):
        quant_native.daily_pearson(values, values, offsets, 1)


@requires_native
def test_cpp_daily_kernels_release_gil():
    """两个长核运行期间 Python 线程都能继续执行。"""

    rng = np.random.default_rng(701)
    values = rng.normal(size=5_000_000).astype(np.float64)
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.full(100, 50_000, dtype=np.int64)),
        )
    )
    backend = CppKernelBackend()

    for call in (
        lambda: backend.daily_rank_pct(values, offsets),
        lambda: backend.daily_pearson(values, values * 2.0, offsets),
    ):
        started = threading.Event()
        finished = threading.Event()

        def invoke() -> None:
            started.set()
            call()
            finished.set()

        worker = threading.Thread(target=invoke)
        worker.start()
        started.wait(timeout=2)
        counter = 0
        while not finished.is_set():
            counter += 1
        worker.join(timeout=15)
        assert not worker.is_alive()
        assert counter > 1000


@requires_native
def test_cpp_daily_kernels_concurrent_calls_are_isolated():
    """重复并发调用没有跨请求状态污染。"""

    rng = np.random.default_rng(811)
    values = rng.normal(size=1_000).astype(np.float64)
    values[::17] = np.nan
    offsets = np.array([0, 400, 1_000], dtype=np.int64)
    backend = CppKernelBackend()
    expected_rank = backend.daily_rank_pct(values, offsets)
    expected_pearson = backend.daily_pearson(values, values + 0.3, offsets)

    def invoke(_: int):
        return (
            backend.daily_rank_pct(values.copy(), offsets.copy()),
            backend.daily_pearson(values.copy(), (values + 0.3).copy(), offsets.copy()),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(invoke, range(16)))

    for rank, pearson in results:
        np.testing.assert_array_equal(rank, expected_rank)
        np.testing.assert_allclose(
            pearson[0],
            expected_pearson[0],
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(pearson[1], expected_pearson[1])


@requires_native
def test_cpp_daily_kernels_subprocess_stress():
    """独立子进程反复调用两个核不崩溃。"""

    script = """
import numpy as np
import quant_native

rng = np.random.default_rng(907)
values = rng.integers(0, 8, 120).astype(np.float64)
values[::11] = np.nan
offsets = np.array([0, 40, 120], dtype=np.int64)
for _ in range(80):
    rank = quant_native.daily_rank_pct(values, offsets)
    correlation, count = quant_native.daily_pearson(values, values, offsets, 5)
    assert rank.shape == (120,)
    assert correlation.shape == (2,)
    assert count.shape == (2,)
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
def test_cpp_daily_kernel_runtime_errors_are_converted(monkeypatch):
    """已加载扩展的运行期异常转换为统一 KERNEL-004。"""

    backend = CppKernelBackend()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    monkeypatch.setattr(backend._native, "daily_rank_pct", fail)
    with pytest.raises(KernelBackendError) as error:
        backend.daily_rank_pct(
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 3], dtype=np.int64),
        )
    assert error.value.code == "KERNEL-004"
