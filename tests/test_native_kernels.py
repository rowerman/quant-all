"""统一批量计算核的契约、回退和 Numba 原型测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

import kernels.prototype as prototype_module
from calculate.correlation import daily_spearman_matrix
from calculate.daily import rank_pct_by_day
from calculate.exposure import _daily_regression
from calculate.normalize import winsorize_by_day
from kernels import (
    KernelArrayError,
    KernelBackendError,
    KernelOffsetError,
    KernelParameterError,
    PythonKernelBackend,
    get_backend,
    native_threads,
)
from kernels.backend import group_codes
from kernels.prototype import NumbaPrototypeBackend


def test_rank_matches_pandas_with_ties_missing_and_inf():
    """排名核保持并列平均名次、原位置和缺失语义。"""

    values = np.array(
        [3.0, 1.0, 1.0, np.nan, 2.0, np.inf, 7.0, -np.inf, 4.0, 2.0, 2.0]
    )
    offsets = np.array([0, 7, 11], dtype=np.int64)
    expected = pd.DataFrame(
        {"day": group_codes(offsets), "value": values}
    ).groupby("day")["value"].rank(pct=True).to_numpy(dtype=np.float64)

    backend = PythonKernelBackend()
    actual = backend.daily_rank_pct(values, offsets)

    np.testing.assert_array_equal(actual, expected)


def test_rank_matches_existing_numpy_and_numba():
    """Python 回退与 Numba 排名和现有 NumPy 实现逐值一致。"""

    rng = np.random.default_rng(311)
    lengths = np.array([7, 0, 31, 45, 12], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    values = rng.integers(0, 9, size=offsets[-1]).astype(np.float64)
    values[[0, 12, 40, 70]] = np.nan

    expected = rank_pct_by_day(group_codes(offsets), values)
    python_actual = PythonKernelBackend().daily_rank_pct(values, offsets)
    numba_actual = NumbaPrototypeBackend().daily_rank_pct(values, offsets)

    np.testing.assert_array_equal(python_actual, expected)
    np.testing.assert_array_equal(numba_actual, expected)


def test_winsorize_matches_current_daily_implementation():
    """Python 回退的逐组 MAD 裁剪与当前日期分组实现一致。"""

    values = np.array(
        [1., 2., 3., 100., np.nan, -4., -5., 2., 9., 9., 9.]
    )
    offsets = np.array([0, 5, 11], dtype=np.int64)
    expected = winsorize_by_day(group_codes(offsets), values, n_mad=3.0)
    actual = PythonKernelBackend().daily_winsorize_mad(values, offsets, n_mad=3.0)

    np.testing.assert_array_equal(actual, expected)


def test_pearson_matches_scipy_and_reports_pair_counts():
    """Pearson 核按有效配对过滤，样本不足和常数日返回 NaN。"""

    left = np.array([1., 2., 3., 4., 5., np.nan, 2., 3., 4., 5., 9.])
    right = np.array([2., 4., 6., 8., 10., 7., np.inf, 4., 5., 6., 8.])
    offsets = np.array([0, 6, 11], dtype=np.int64)
    correlation, count = PythonKernelBackend().daily_pearson(
        left, right, offsets, min_n=5
    )

    first = np.isfinite(left[:6]) & np.isfinite(right[:6])
    expected = pearsonr(left[:6][first], right[:6][first]).statistic
    assert correlation[0] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(count, np.array([5, 4]))
    assert np.isnan(correlation[1])


def test_pearson_constant_group_returns_nan():
    """常数截面没有 Pearson 相关，但有效计数照实返回。"""

    values = np.ones(6, dtype=np.float64)
    offsets = np.array([0, 6], dtype=np.int64)
    correlation, count = PythonKernelBackend().daily_pearson(
        values, values, offsets, min_n=5
    )

    assert np.isnan(correlation[0])
    assert count[0] == 6


def test_pairwise_spearman_matches_current_and_scipy():
    """批量 Spearman 保持逐日逐对重排名、缺失和常数语义。"""

    rng = np.random.default_rng(107)
    values = rng.integers(0, 7, (43, 6)).astype(np.float64)
    values[::5, 0] = np.nan
    values[::7, 1] = np.inf
    values[::11, 2] = -np.inf
    values[:, 4] = 2.0
    offsets = np.array([0, 20, 43], dtype=np.int64)
    backend = PythonKernelBackend()

    correlation, count = backend.pairwise_spearman(values, offsets, min_n=5)

    assert correlation.shape == (2, 6, 6)
    assert count.shape == (2, 6, 6)
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        expected = daily_spearman_matrix(values[start:end])
        np.testing.assert_allclose(
            correlation[group], expected, atol=1e-12, equal_nan=True
        )
        finite = np.isfinite(values[start:end])
        np.testing.assert_array_equal(
            count[group],
            finite.T.astype(np.int64) @ finite.astype(np.int64),
        )


def test_numba_pairwise_spearman_matches_numpy_for_edge_cases():
    """Numba 原型与现有实现覆盖缺失、并列、常数和空组。"""

    values = np.array(
        [
            [1., 2., np.nan, 7.],
            [2., 1., np.nan, 7.],
            [2., 1., 1., 7.],
            [4., 3., 2., 7.],
            [5., 6., 3., 7.],
            [6., 5., 4., 7.],
            [9., 8., 5., 7.],
        ]
    )
    offsets = np.array([0, 0, 7], dtype=np.int64)

    expected, expected_count = PythonKernelBackend().pairwise_spearman(
        values, offsets, min_n=5
    )
    actual, actual_count = NumbaPrototypeBackend().pairwise_spearman(
        values, offsets, min_n=5
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(actual_count, expected_count)
    assert np.isnan(actual[0]).all()


def test_numba_unsupported_kernel_logs_explicit_fallback(monkeypatch):
    """Numba 未实现核必须记录具体名称和回退原因。"""

    messages = []
    monkeypatch.setattr(
        prototype_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    values = np.array([1., 2., 3., 100.])
    backend = NumbaPrototypeBackend()
    backend.daily_winsorize_mad(
        values,
        np.array([0, values.size], dtype=np.int64),
    )

    assert messages == [
        "内核显式回退 kernel=daily_winsorize_mad requested=numba "
        "backend=python reason=numba_kernel_not_implemented"
    ]


def test_pairwise_spearman_scipy_reference():
    """Numba Spearman 对随机并列值逐对匹配 SciPy。"""

    rng = np.random.default_rng(409)
    values = rng.integers(0, 8, (35, 5)).astype(np.float64)
    values[::6, 2] = np.nan
    offsets = np.array([0, values.shape[0]], dtype=np.int64)
    actual, count = NumbaPrototypeBackend().pairwise_spearman(
        values, offsets, min_n=5
    )

    for left in range(values.shape[1]):
        for right in range(left, values.shape[1]):
            pair = values[:, [left, right]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            assert count[0, left, right] == len(pair)
            if len(pair) < 5 or np.ptp(pair[:, 0]) == 0 or np.ptp(pair[:, 1]) == 0:
                assert np.isnan(actual[0, left, right])
            else:
                expected = spearmanr(pair[:, 0], pair[:, 1]).statistic
                assert actual[0, left, right] == pytest.approx(
                    expected, abs=1e-12
                )


def test_daily_ols_matches_existing_regression_and_residual_rows():
    """OLS 回退直接复用现有逐日回归并返回同样本残差。"""

    rng = np.random.default_rng(29)
    group_sizes = np.array([40, 35, 3], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(group_sizes))).astype(np.int64)
    sample_count = int(offsets[-1])
    design = np.column_stack(
        [
            np.ones(sample_count),
            rng.normal(size=sample_count),
            rng.normal(size=sample_count),
        ]
    ).astype(np.float64)
    response = (
        1.5 + 2.0 * design[:, 1] - 0.5 * design[:, 2] + rng.normal(scale=.2, size=sample_count)
    ).astype(np.float64)
    expected = _daily_regression(
        group_codes(offsets),
        offsets.size - 1,
        design,
        response,
        np.arange(sample_count, dtype=np.int64),
    )

    actual = PythonKernelBackend().daily_ols(design, response, offsets, min_rows=30)

    np.testing.assert_array_equal(actual["days"], expected["days"])
    np.testing.assert_allclose(actual["coefficients"], expected["coefficients"])
    np.testing.assert_allclose(actual["t_stats"], expected["t_stats"])
    np.testing.assert_allclose(actual["r2"], expected["r2"])
    np.testing.assert_array_equal(actual["rows"], expected["rows"])
    np.testing.assert_array_equal(actual["residual_rows"], expected["residual_rows"])
    np.testing.assert_allclose(actual["residual_values"], expected["residual_values"])
    assert actual["skipped"] == 1


def test_daily_ols_honours_custom_minimum_rows():
    """OLS 回退按协议使用调用方给出的最少样本数。"""

    rng = np.random.default_rng(31)
    sizes = np.array([40, 35], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    count = int(offsets[-1])
    design = np.column_stack(
        [np.ones(count), rng.normal(size=count), rng.normal(size=count)]
    ).astype(np.float64)
    response = (design @ np.array([1., 2., -1.]) + rng.normal(scale=.1, size=count))

    actual = PythonKernelBackend().daily_ols(
        design,
        response.astype(np.float64),
        offsets,
        min_rows=40,
    )

    np.testing.assert_array_equal(actual["days"], np.array([0]))
    np.testing.assert_array_equal(actual["rows"], np.array([40]))
    assert actual["skipped"] == 1


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (np.arange(6, dtype=np.float32), KernelArrayError),
        (np.arange(12, dtype=np.float64)[::2], KernelArrayError),
        (np.arange(6, dtype=np.float64).reshape(2, 3), KernelArrayError),
        (np.array([], dtype=np.float64), KernelArrayError),
    ],
)
def test_rank_array_contract_errors(values, error):
    """错误 dtype、非连续、维度或空输入必须明确失败。"""

    offsets = np.array([0, 6], dtype=np.int64)
    with pytest.raises(error):
        PythonKernelBackend().daily_rank_pct(values, offsets)


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([0, 3], dtype=np.int32),
        np.array([1, 3], dtype=np.int64),
        np.array([0, 2, 1], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ],
)
def test_offset_contract_errors(offsets):
    """offset dtype、首尾边界、单调性和长度错误必须使用 KERNEL-002。"""

    with pytest.raises(KernelOffsetError):
        PythonKernelBackend().daily_rank_pct(
            np.array([1., 2., 3.]),
            offsets,
        )


def test_pairwise_and_shape_contract_errors():
    """二维核拒绝错误形状，Pearson 两列长度必须一致。"""

    backend = PythonKernelBackend()
    with pytest.raises(KernelArrayError):
        backend.pairwise_spearman(
            np.array([1., 2., 3.]),
            np.array([0, 3], dtype=np.int64),
        )
    with pytest.raises(KernelArrayError):
        backend.daily_pearson(
            np.ones(3),
            np.ones(4),
            np.array([0, 3], dtype=np.int64),
        )


@pytest.mark.parametrize("n_mad", [0, -1, np.inf, "3", True])
def test_winsorize_parameter_errors(n_mad):
    """MAD 倍数必须是有限正数。"""

    with pytest.raises(KernelParameterError):
        PythonKernelBackend().daily_winsorize_mad(
            np.array([1., 2., 3.]),
            np.array([0, 3], dtype=np.int64),
            n_mad=n_mad,
        )


def test_environment_backend_selection_and_thread_budget(monkeypatch):
    """默认 Python、显式 Numba 和线程预算均可稳定选择。"""

    monkeypatch.delenv("QUANT_KERNEL_BACKEND", raising=False)
    monkeypatch.setenv("QUANT_NATIVE_THREADS", "2")
    default = get_backend()
    assert default.name == "python"
    assert default.fallback_reason == "default_python"
    assert native_threads() == 2

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "numba")
    selected = get_backend()
    assert selected.name == "numba"
    assert selected.fallback_reason is None

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "auto")
    with pytest.raises(KernelBackendError) as error:
        get_backend()
    assert error.value.code == "KERNEL-004"

    monkeypatch.setenv("QUANT_NATIVE_THREADS", "0")
    with pytest.raises(KernelBackendError):
        native_threads()
