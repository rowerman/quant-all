"""C++ 分位分组、收益、换手和逐行 Spearman 的边界与路由测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest

from calculate.binning import (
    assign_bins,
    bin_daily_returns,
    bin_matrix,
    bin_returns_multi,
    bin_turnover,
    factor_autocorrelation,
    rank_matrix,
)
from kernels import KernelArrayError, KernelParameterError, get_backend
from kernels.native import CppKernelBackend

requires_native = pytest.mark.skipif(
    importlib.util.find_spec("quant_native") is None,
    reason="quant_native 扩展未构建",
)


def _frame(days: int = 8, stocks: int = 20) -> pd.DataFrame:
    """构造含并列、缺失和非有限值的分位输入。"""

    rng = np.random.default_rng(701)
    dates = pd.bdate_range("2024-01-02", periods=days)
    codes = [f"SH600{index:03d}" for index in range(stocks)]
    frame = pd.MultiIndex.from_product(
        [dates, codes],
        names=["trade_date", "stock_code"],
    ).to_frame(index=False)
    frame["factor"] = rng.integers(0, 8, size=len(frame)).astype("float64")
    frame["label_1d"] = rng.normal(scale=.01, size=len(frame))
    for horizon in (3, 5, 10, 20):
        frame[f"label_h{horizon}"] = rng.normal(
            scale=.02 * np.sqrt(horizon / 5),
            size=len(frame),
        )
    frame["label"] = frame["label_h5"]
    frame["total_mv"] = rng.uniform(1e5, 1e6, size=len(frame))
    frame["in_universe"] = True
    frame.loc[frame.index[::29], "factor"] = np.nan
    frame.loc[frame.index[::31], "factor"] = np.inf
    frame.loc[frame.index[::37], "label"] = np.nan
    frame.loc[frame.index[::41], "total_mv"] = np.nan
    return frame.sample(frac=1.0, random_state=811).reset_index(drop=True)


@requires_native
def test_quantile_bins_match_python_with_ties_missing_inf_and_empty_group():
    """分位组保持平均名次、空组、NaN 和正负无穷语义。"""

    values = np.array(
        [4., np.nan, 0., 4., 2., 3., 3., np.inf, -np.inf, 7.],
        dtype=np.float64,
    )
    offsets = np.array([0, 0, 7, 10], dtype=np.int64)
    python = get_backend("python").backend
    native = get_backend("native").backend

    expected = python.daily_quantile_bins(values, offsets, 5)
    actual = native.daily_quantile_bins(values, offsets, 5)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == actual[3]
    assert actual[1] == -1
    assert actual[8] < actual[9] < actual[7]


@requires_native
def test_bin_returns_match_python_for_equal_and_market_cap():
    """多标签等权与市值加权同时保持池内均值。"""

    rng = np.random.default_rng(719)
    day_codes = np.repeat(np.arange(5, dtype=np.int64), 9)
    bin_codes = rng.integers(0, 4, size=day_codes.size).astype(np.int64)
    values = rng.normal(size=(day_codes.size, 3))
    values[::13, 1] = np.nan
    values[::17, 2] = np.inf
    weights = rng.uniform(0, 2, size=day_codes.size)
    weights[::11] = np.nan
    python = get_backend("python").backend
    native = get_backend("native").backend

    for selected in (np.ones_like(weights), weights):
        expected = python.bin_returns_multi(
            values,
            day_codes,
            bin_codes,
            selected,
            5,
            4,
        )
        actual = native.bin_returns_multi(
            values,
            day_codes,
            bin_codes,
            selected,
            5,
            4,
        )
        np.testing.assert_allclose(
            actual["returns"],
            expected["returns"],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual["pools"],
            expected["pools"],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


@requires_native
def test_turnover_matches_python_with_empty_day_and_pool_changes():
    """换手保留空交易日、首日空值和股票进出池口径。"""

    values = np.array(
        [
            [0., 1., 2., np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [2., 0., 2., 1.],
            [0., 2., np.nan, 1.],
        ],
        dtype=np.float64,
    )
    python = get_backend("python").backend.bin_turnover(values, 3)
    native = get_backend("native").backend.bin_turnover(values, 3)

    np.testing.assert_allclose(
        native["per_bin"],
        python["per_bin"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        native["overall"],
        python["overall"],
        equal_nan=True,
    )


@requires_native
def test_row_spearman_reranks_common_finite_pairs():
    """逐行相关只对共同有限股票重新排名。"""

    left = np.array(
        [
            [1., 2., 3., 4., 5., np.nan],
            [1., 2., 3., 4., 5., np.inf],
            [2., 2., 2., 2., 2., 2.],
        ]
    )
    right = np.array(
        [
            [5., 4., 3., 2., 1., 9.],
            [1., 3., 2., 5., 4., 8.],
            [1., 2., 3., 4., 5., 6.],
        ]
    )
    python = get_backend("python").backend
    native = get_backend("native").backend

    expected = python.daily_row_spearman(left, right, 5)
    actual = native.daily_row_spearman(left, right, 5)

    np.testing.assert_allclose(
        actual[0],
        expected[0],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0][0] == pytest.approx(-1.0)
    assert np.isnan(actual[0][2])


@requires_native
def test_binning_business_functions_match_python(monkeypatch):
    """分箱、收益、换手、秩宽表和自相关的公开结果与 Python 一致。"""

    panel = _frame()
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected_bins = assign_bins(panel, 5)
    expected_equal = bin_daily_returns(expected_bins, 5)
    expected_weighted = bin_daily_returns(
        expected_bins,
        5,
        weight_column="total_mv",
    )
    expected_multi, expected_pools = bin_returns_multi(
        expected_bins,
        5,
        ["label_1d", "label"],
        "total_mv",
    )
    expected_matrix = rank_matrix(expected_bins)
    expected_per_bin, expected_overall = bin_turnover(
        bin_matrix(expected_bins),
        5,
    )
    expected_auto = factor_autocorrelation(expected_matrix, (1, 3))

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual_bins = assign_bins(panel, 5)
    actual_equal = bin_daily_returns(actual_bins, 5)
    actual_weighted = bin_daily_returns(
        actual_bins,
        5,
        weight_column="total_mv",
    )
    actual_multi, actual_pools = bin_returns_multi(
        actual_bins,
        5,
        ["label_1d", "label"],
        "total_mv",
    )
    actual_matrix = rank_matrix(actual_bins)
    actual_per_bin, actual_overall = bin_turnover(
        bin_matrix(actual_bins),
        5,
    )

    pd.testing.assert_frame_equal(actual_bins, expected_bins)
    pd.testing.assert_frame_equal(actual_equal, expected_equal, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(
        actual_weighted,
        expected_weighted,
        rtol=1e-12,
        atol=1e-12,
    )
    for column in expected_multi:
        pd.testing.assert_frame_equal(
            actual_multi[column],
            expected_multi[column],
            rtol=1e-12,
            atol=1e-12,
        )
        pd.testing.assert_series_equal(
            actual_pools[column],
            expected_pools[column],
            rtol=1e-12,
            atol=1e-12,
        )
    pd.testing.assert_frame_equal(
        actual_matrix,
        expected_matrix,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        actual_per_bin,
        expected_per_bin,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        actual_overall,
        expected_overall,
        rtol=1e-12,
        atol=1e-12,
    )
    actual_auto = factor_autocorrelation(actual_matrix, (1, 3))
    pd.testing.assert_frame_equal(
        actual_auto,
        expected_auto,
        rtol=1e-12,
        atol=1e-12,
    )


@requires_native
def test_binning_contract_errors():
    """新增核拒绝错误编码、形状和参数。"""

    backend = CppKernelBackend()
    with pytest.raises(KernelArrayError):
        backend.bin_returns_multi(
            np.ones((3, 1)),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int64),
            np.ones(3),
            3,
            2,
        )
    with pytest.raises(KernelArrayError):
        backend.bin_returns_multi(
            np.ones((3, 1)),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([0, 0, 0], dtype=np.int64),
            np.ones(3),
            3,
            2,
        )
    with pytest.raises(KernelParameterError):
        backend.daily_quantile_bins(
            np.ones(3),
            np.array([0, 3], dtype=np.int64),
            1,
        )
    with pytest.raises(KernelArrayError):
        backend.daily_row_spearman(
            np.ones((3, 2)),
            np.ones((2, 3)),
        )


@requires_native
def test_binning_kernels_release_gil_and_are_concurrent():
    """长核释放 GIL，重复并发调用没有状态污染。"""

    rng = np.random.default_rng(821)
    values = rng.normal(size=(800, 300)).astype(np.float64)
    values[::17, ::11] = np.nan
    reversed_values = np.ascontiguousarray(values[::-1])
    native = get_backend("native").backend
    expected = native.daily_row_spearman(values, reversed_values, 5)

    started = threading.Event()
    finished = threading.Event()

    def invoke() -> None:
        started.set()
        native.daily_row_spearman(values, reversed_values, 5)
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    started.wait(timeout=2)
    counter = 0
    while not finished.is_set():
        counter += 1
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert counter > 1000

    def repeated(_: int):
        return native.daily_row_spearman(
            values.copy(),
            reversed_values.copy(),
            5,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(repeated, range(8)))
    for correlation, count in results:
        np.testing.assert_array_equal(correlation, expected[0])
        np.testing.assert_array_equal(count, expected[1])


@requires_native
def test_binning_kernels_subprocess_stress():
    """独立子进程重复调用新核不崩溃。"""

    script = """
import numpy as np
import quant_native

rng = np.random.default_rng(907)
rows, columns = 80, 40
values = rng.integers(0, 9, size=rows).astype(np.float64)
offsets = np.array([0, 30, 80], dtype=np.int64)
for _ in range(50):
    bins = quant_native.daily_quantile_bins(values, offsets, 5)
    matrix = rng.integers(0, 5, size=(4, columns)).astype(np.float64)
    returns, pools = quant_native.bin_returns_multi(
        matrix.T[:, :1].repeat(4, axis=0),
        np.repeat(np.arange(4, dtype=np.int64), columns),
        matrix.ravel().astype(np.int64) % 5,
        np.ones(4 * columns),
        4,
        5,
    )
    turnover, overall = quant_native.bin_turnover(matrix, 5)
    corr, count = quant_native.daily_row_spearman(matrix, matrix, 5)
    assert bins.shape == (80,)
    assert returns.shape == (4, 5, 1)
    assert pools.shape == (4, 1)
    assert turnover.shape == (4, 5)
    assert overall.shape == corr.shape == count.shape == (4,)
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
