"""C++ Spearman 核的正确性、契约、并发和进程隔离测试。"""

from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from scipy.stats import spearmanr

import kernels.native as native_module
from calculate.correlation import daily_spearman_matrix
from kernels import KernelBackendError, get_backend
from kernels.native import CppKernelBackend

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


@requires_native
def test_cpp_pairwise_spearman_matches_python_edge_cases():
    """C++ 核覆盖分组、缺失、无穷、常数和并列平均名次。"""

    values = np.array(
        [
            [1.0, 2.0, np.nan, 7.0],
            [2.0, 1.0, np.nan, 7.0],
            [2.0, 1.0, 1.0, 7.0],
            [4.0, 3.0, 2.0, 7.0],
            [5.0, 6.0, 3.0, np.inf],
            [6.0, 5.0, 4.0, 8.0],
            [9.0, 8.0, 5.0, 9.0],
            [1.0, 4.0, 8.0, -np.inf],
        ],
        dtype=np.float64,
    )
    offsets = np.array([0, 0, 6, 8], dtype=np.int64)

    actual, count = CppKernelBackend().pairwise_spearman(values, offsets)
    expected, expected_count = get_backend("python").backend.pairwise_spearman(
        values,
        offsets,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(count, expected_count)
    assert np.isnan(actual[0]).all()


@requires_native
def test_cpp_pairwise_spearman_matches_scipy_with_ties():
    """随机并列截面逐对匹配 SciPy 和现有 Python 金标准。"""

    rng = np.random.default_rng(719)
    values = rng.integers(0, 9, (37, 5)).astype(np.float64)
    values[::7, 0] = np.nan
    values[::11, 2] = np.inf
    offsets = np.array([0, 37], dtype=np.int64)

    actual, count = CppKernelBackend().pairwise_spearman(values, offsets)
    expected = daily_spearman_matrix(values)

    np.testing.assert_allclose(actual[0], expected, atol=1e-10, rtol=1e-10, equal_nan=True)
    for left in range(values.shape[1]):
        for right in range(left, values.shape[1]):
            pair = values[:, [left, right]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            assert count[0, left, right] == len(pair)
            if len(pair) < 5 or np.ptp(pair[:, 0]) == 0 or np.ptp(pair[:, 1]) == 0:
                assert np.isnan(actual[0, left, right])
            else:
                expected_value = spearmanr(pair[:, 0], pair[:, 1]).statistic
                assert actual[0, left, right] == pytest.approx(
                    expected_value,
                    abs=1e-10,
                )


@requires_native
def test_cpp_custom_minimum_and_group_parallelism(monkeypatch):
    """自定义门槛和多组计算均保持独立组边界。"""

    import quant_native

    monkeypatch.setenv("QUANT_NATIVE_THREADS", "2")
    rng = np.random.default_rng(83)
    sizes = np.array([5, 9, 14], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    values = rng.integers(0, 5, (int(offsets[-1]), 4)).astype(np.float64)

    actual, count = quant_native.pairwise_spearman(values, offsets, 9)
    expected, expected_count = get_backend("python").backend.pairwise_spearman(
        values,
        offsets,
        min_n=9,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(count, expected_count)
    assert np.isnan(actual[0]).all()


@requires_native
def test_cpp_target_row_matches_python_with_empty_groups():
    """目标模式覆盖多组、空组、缺失、无穷、并列和常数语义。"""

    rng = np.random.default_rng(211)
    values = rng.integers(0, 7, (43, 6)).astype(np.float64)
    values[::5, 0] = np.nan
    values[::7, 2] = np.inf
    values[::11, 4] = -np.inf
    values[:, 5] = 3.0
    offsets = np.array([0, 0, 17, 43, 43], dtype=np.int64)

    for target in range(values.shape[1]):
        actual, count = CppKernelBackend().pairwise_spearman(
            values,
            offsets,
            5,
            target,
        )
        expected, expected_count = get_backend("python").backend.pairwise_spearman(
            values,
            offsets,
            5,
            target,
        )
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(count, expected_count)
        off_target = np.ones(values.shape[1], dtype=bool)
        off_target[target] = False
        assert np.isnan(actual[:, off_target][:, :, off_target]).all()


@requires_native
def test_cpp_target_minimum_matches_python():
    """目标模式的非默认样本门槛与 Python 回退一致。"""

    rng = np.random.default_rng(223)
    values = rng.integers(0, 9, (31, 5)).astype(np.float64)
    values[::4, 1] = np.nan
    offsets = np.array([0, 12, 31], dtype=np.int64)

    actual, count = CppKernelBackend().pairwise_spearman(
        values,
        offsets,
        11,
        2,
    )
    expected, expected_count = get_backend("python").backend.pairwise_spearman(
        values,
        offsets,
        11,
        2,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(count, expected_count)


@requires_native
@pytest.mark.parametrize(
    ("values", "offsets", "min_n", "error"),
    [
        (
            np.ones((5, 2), dtype=np.float32),
            np.array([0, 5], dtype=np.int64),
            5,
            TypeError,
        ),
        (
            np.ones((5, 4), dtype=np.float64)[:, ::2],
            np.array([0, 5], dtype=np.int64),
            5,
            ValueError,
        ),
        (
            np.ones(5, dtype=np.float64),
            np.array([0, 5], dtype=np.int64),
            5,
            ValueError,
        ),
        (
            np.ones((5, 2), dtype=np.float64),
            np.array([1, 5], dtype=np.int64),
            5,
            ValueError,
        ),
        (
            np.ones((5, 2), dtype=np.float64),
            np.array([0, 4], dtype=np.int64),
            5,
            ValueError,
        ),
        (
            np.ones((5, 2), dtype=np.float64),
            np.array([0, 5], dtype=np.int64),
            4,
            ValueError,
        ),
    ],
)
def test_cpp_direct_contract_errors(values, offsets, min_n, error):
    """直接调用扩展时也不能绕过形状、dtype、offset 和门槛契约。"""

    import quant_native

    with pytest.raises(error):
        quant_native.pairwise_spearman(values, offsets, min_n)


@requires_native
def test_cpp_thread_environment_errors(monkeypatch):
    """C++ 层拒绝非法线程预算，不启动无界线程。"""

    import quant_native

    monkeypatch.setenv("QUANT_NATIVE_THREADS", "0")
    with pytest.raises(ValueError, match="QUANT_NATIVE_THREADS"):
        quant_native.pairwise_spearman(
            np.ones((5, 2), dtype=np.float64),
            np.array([0, 5], dtype=np.int64),
            5,
        )


@requires_native
@pytest.mark.parametrize(
    ("target_index", "error"),
    [(True, TypeError), ("1", TypeError), (-1, ValueError), (5, ValueError)],
)
def test_cpp_target_index_contract_errors(target_index, error):
    """直接调用扩展时目标列必须是非负整数且在列范围内。"""

    import quant_native

    with pytest.raises(error):
        quant_native.pairwise_spearman(
            np.ones((5, 5), dtype=np.float64),
            np.array([0, 5], dtype=np.int64),
            5,
            target_index,
        )


@requires_native
def test_cpp_call_releases_gil():
    """长核运行期间 Python 主线程仍能获得 GIL。"""

    rng = np.random.default_rng(991)
    values = rng.normal(size=(30000, 24)).astype(np.float64)
    offsets = np.array([0, values.shape[0]], dtype=np.int64)
    backend = CppKernelBackend()
    started = threading.Event()
    finished = threading.Event()
    counter = 0

    def invoke() -> None:
        started.set()
        backend.pairwise_spearman(values, offsets)
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    started.wait(timeout=2)
    while not finished.is_set():
        counter += 1
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert counter > 1000


@requires_native
def test_cpp_repeated_concurrent_calls_are_isolated():
    """并发重复调用结果稳定且没有共享状态污染。"""

    rng = np.random.default_rng(177)
    values = rng.integers(0, 7, (500, 12)).astype(np.float64)
    values[::13, 0] = np.nan
    offsets = np.array([0, 240, 500], dtype=np.int64)
    backend = CppKernelBackend()
    expected = backend.pairwise_spearman(values, offsets)

    def call():
        return backend.pairwise_spearman(values.copy(), offsets.copy())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: call(), range(16)))

    for correlation, count in results:
        np.testing.assert_allclose(
            correlation,
            expected[0],
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(count, expected[1])


@requires_native
def test_cpp_subprocess_stress():
    """独立子进程反复调用扩展不崩溃。"""

    script = """
import numpy as np
import quant_native

rng = np.random.default_rng(203)
values = rng.integers(0, 8, (80, 8)).astype(np.float64)
values[::9, 1] = np.nan
offsets = np.array([0, 35, 80], dtype=np.int64)
for _ in range(60):
    correlation, count = quant_native.pairwise_spearman(values, offsets, 5)
    assert correlation.shape == (2, 8, 8)
    assert count.shape == (2, 8, 8)
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


def test_native_backend_missing_extension_is_explicit(monkeypatch):
    """扩展缺失时显式 native 请求失败，不静默切换后端。"""

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "quant_native":
            raise ModuleNotFoundError("simulated missing quant_native")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(KernelBackendError) as error:
        get_backend("native")
    assert error.value.code == "KERNEL-004"


def test_auto_backend_falls_back_only_when_extension_is_missing(monkeypatch):
    """auto 在扩展不可导入时回退 Python，并保留明确原因。"""

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "quant_native":
            raise ModuleNotFoundError("simulated missing quant_native")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    selected = get_backend("auto")
    assert selected.name == "python"
    assert selected.requested == "auto"
    assert selected.fallback_reason == "cpp_extension_unavailable"


@requires_native
def test_native_backend_ols_uses_cpp_with_1e5_contract(monkeypatch):
    """OLS 走 C++，结果按 1e-5 业务门槛与 Python 一致。"""

    messages = []
    monkeypatch.setattr(
        native_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    design = np.column_stack(
        (
            np.ones(4, dtype=np.float64),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )
    )
    response = np.array([2.0, 4.0, 6.0, 8.2])
    offsets = np.array([0, 4], dtype=np.int64)
    actual = CppKernelBackend().daily_ols(design, response, offsets, min_rows=1)
    expected = get_backend("python").backend.daily_ols(
        design,
        response,
        offsets,
        min_rows=1,
    )
    np.testing.assert_array_equal(actual["days"], expected["days"])
    np.testing.assert_allclose(
        actual["coefficients"],
        expected["coefficients"],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["t_stats"],
        expected["t_stats"],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["r2"],
        expected["r2"],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_array_equal(actual["residual_rows"], expected["residual_rows"])
    np.testing.assert_allclose(
        actual["residual_values"],
        expected["residual_values"],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_array_equal(actual["active"], expected["active"])
    assert messages[0].startswith(
        "内核执行 kernel=daily_ols requested=native backend=native"
    )
