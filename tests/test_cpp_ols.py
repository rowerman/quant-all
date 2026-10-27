"""C++ 逐日 OLS 的 1e-5 金标准、边界和并发测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util

import numpy as np
import pytest

from calculate.exposure import _daily_regression
from kernels import KernelArrayError, KernelOffsetError, KernelParameterError
from kernels.backend import group_codes
from kernels.native import CppKernelBackend

requires_native = pytest.mark.skipif(
    importlib.util.find_spec("quant_native") is None,
    reason="quant_native 扩展未构建",
)


def _golden(design, response, offsets, min_rows=30):
    """调用现有 Python 实现作为金标准。"""

    return _daily_regression(
        group_codes(offsets),
        offsets.size - 1,
        design,
        response,
        np.arange(response.size, dtype=np.int64),
        min_rows=min_rows,
    )


def _assert_close(actual, expected, tolerance=1e-5):
    """按用户确认的 OLS 业务门槛比较结果。"""

    for key in ("days", "rows", "residual_rows"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("coefficients", "t_stats", "r2", "residual_values"):
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
    np.testing.assert_array_equal(actual["active"], expected["active"])
    for key in (
        "skipped",
        "skip_insufficient_rows",
        "skip_invalid_response",
        "skip_rank_deficient",
    ):
        assert actual[key] == expected[key]


@requires_native
def test_cpp_ols_matches_python_for_random_groups():
    """随机三日回归的系数、t、R2、残差和 active mask 在 1e-5 内一致。"""

    rng = np.random.default_rng(617)
    sizes = np.array([41, 37, 33], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    count = int(offsets[-1])
    design = np.column_stack([
        np.ones(count),
        rng.normal(size=count),
        rng.normal(size=count),
        rng.normal(size=count),
    ])
    response = (
        0.7 + 1.3 * design[:, 1] - 0.4 * design[:, 2]
        + 0.2 * design[:, 3] + rng.normal(scale=0.15, size=count)
    )

    actual = CppKernelBackend().daily_ols(design, response, offsets)
    expected = _golden(design, response, offsets)

    _assert_close(actual, expected)
    assert actual["days"].tolist() == [0, 1, 2]
    assert actual["active"].all()


@requires_native
def test_cpp_ols_skip_reasons_and_constant_columns():
    """少样本、无效响应、常数列和秩亏分别按原口径处理。"""

    rng = np.random.default_rng(619)
    sizes = np.array([36, 35, 34, 6, 35], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    count = int(offsets[-1])
    design = np.column_stack([
        np.ones(count),
        rng.normal(size=count),
        np.ones(count),
        rng.normal(size=count),
    ])
    response = 1.0 + design[:, 1] + 2.0 * design[:, 3] + rng.normal(
        scale=0.1, size=count
    )
    response[int(offsets[1]) - 1] = np.nan
    response[int(offsets[1]):int(offsets[2])] = 2.0
    design[int(offsets[2]):int(offsets[3]), 3] = design[
        int(offsets[2]):int(offsets[3]), 1
    ]

    actual = CppKernelBackend().daily_ols(design, response, offsets)
    expected = _golden(design, response, offsets)

    _assert_close(actual, expected)
    assert actual["skip_insufficient_rows"] == 1
    assert actual["skip_invalid_response"] == 2
    assert actual["skip_rank_deficient"] == 1
    assert actual["skipped"] == 4
    assert not actual["active"][:, 2].any()


@requires_native
def test_cpp_ols_near_collinearity_uses_python_day_fallback():
    """病态设计矩阵按日回退 Python，近共线结果仍与金标准一致。"""

    rng = np.random.default_rng(631)
    sizes = np.array([32, 40], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    count = int(offsets[-1])
    left = rng.normal(size=count)
    design = np.column_stack([
        np.ones(count),
        left,
        left + 1e-10 * rng.normal(size=count),
    ])
    response = (
        0.5 + 1.2 * design[:, 1] - 0.3 * design[:, 2]
        + rng.normal(scale=0.2, size=count)
    )

    actual = CppKernelBackend().daily_ols(
        design, response, offsets, min_rows=35
    )
    expected = _golden(design, response, offsets, min_rows=35)

    _assert_close(actual, expected, tolerance=1e-10)
    assert actual["days"].tolist() == [1]
    assert actual["skip_insufficient_rows"] == 1


@requires_native
def test_cpp_ols_empty_groups_and_gil_concurrency():
    """空组和并发调用没有状态污染。"""

    rng = np.random.default_rng(641)
    sizes = np.array([30, 0, 30], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    count = int(offsets[-1])
    design = np.column_stack([np.ones(count), rng.normal(size=count)])
    design[int(offsets[2]):int(offsets[3]), 1] = 4.0
    response = rng.normal(size=count)
    backend = CppKernelBackend()

    expected = backend.daily_ols(design, response, offsets)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: backend.daily_ols(design, response, offsets),
                range(8),
            )
        )

    assert expected["days"].tolist() == [0, 2]
    assert expected["active"].tolist() == [[True, True], [True, False]]
    for actual in results:
        _assert_close(actual, expected)


@pytest.mark.parametrize(
    ("design", "response", "offsets", "min_rows", "error"),
    [
        (
            np.ones((4, 2), dtype=np.float32),
            np.ones(4),
            np.array([0, 4], dtype=np.int64),
            1,
            KernelArrayError,
        ),
        (
            np.ones((4, 2), dtype=np.float64),
            np.ones(3),
            np.array([0, 4], dtype=np.int64),
            1,
            KernelArrayError,
        ),
        (
            np.ones((4, 2), dtype=np.float64),
            np.ones(4),
            np.array([0, 3], dtype=np.int64),
            1,
            KernelOffsetError,
        ),
        (
            np.ones((4, 2), dtype=np.float64),
            np.ones(4),
            np.array([0, 4], dtype=np.int64),
            0,
            KernelParameterError,
        ),
    ],
)
def test_ols_contract_errors(design, response, offsets, min_rows, error):
    """OLS 适配层对形状、offset 和参数错误明确报错。"""

    with pytest.raises(error):
        CppKernelBackend().daily_ols(design, response, offsets, min_rows)
