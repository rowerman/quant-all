"""相关性独立验证：缺失、并列、常数与分块累计。"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from calculate.correlation import (
    _accumulate_target_correlations,
    _daily_spearman,
    accumulate_correlations,
    available_factors,
    correlation_result,
    daily_spearman_matrix,
)
from calculate.errors import MetricError


def test_pairwise_scipy_and_chunks():
    rng = np.random.default_rng(43)
    names = ['a', 'b', 'c', 'constant']
    frame = pd.DataFrame(rng.integers(0, 7, (240, 3)), columns=names[:3]).astype(float)
    frame['constant'] = 1.
    frame['trade_date'] = np.repeat(pd.date_range('2024-01-18', periods=30), 8)
    frame.loc[::7, 'a'] = np.nan
    frame.loc[::11, 'b'] = np.inf
    sums = np.zeros((4, 4)); days = np.zeros((4, 4), dtype=int)
    for _, month in frame.groupby(frame.trade_date.dt.to_period('M')):
        accumulate_correlations(month, names, sums, days)
    result = correlation_result(names, sums, days, 'a', 2)
    for i in range(3):
        for j in range(3):
            expected = []
            for _, day in frame.groupby('trade_date'):
                pair = day[[names[i], names[j]]].to_numpy()
                pair = pair[np.isfinite(pair).all(axis=1)]
                if len(pair) >= 5:
                    corr = spearmanr(pair[:, 0], pair[:, 1]).statistic
                    if np.isfinite(corr): expected.append(corr)
            assert days[i, j] == len(expected)
            assert result['matrix'][i, j] == pytest.approx(np.mean(expected), abs=1e-10)
    np.testing.assert_allclose(result['matrix'], result['matrix'].T, equal_nan=True)
    assert np.isnan(result['matrix'][3]).all()
    assert all(row['factor'] != 'a' for row in result['top'])


def test_target_row_matches_full_matrix_with_missing_ties_and_constants():
    rng = np.random.default_rng(20261027)
    values = rng.integers(0, 6, (40, 7)).astype(float)
    values[:, 3] = 2.0
    values[::5, 0] = np.nan
    values[::7, 1] = np.inf
    values[::11, 2] = -np.inf
    full = daily_spearman_matrix(values)

    for target in range(values.shape[1]):
        actual = _daily_spearman(values, target_index=target)
        np.testing.assert_allclose(actual[target], full[target], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(actual[:, target], full[:, target], equal_nan=True, atol=1e-12)
        off_target = np.ones(values.shape[1], dtype=bool)
        off_target[target] = False
        assert np.isnan(actual[np.ix_(off_target, off_target)]).all()


def test_target_accumulation_matches_full_matrix_target_row():
    rng = np.random.default_rng(81)
    names = ['a', 'b', 'c', 'd']
    frame = pd.DataFrame(rng.integers(0, 8, (15 * 10, 4)), columns=names).astype(float)
    frame['trade_date'] = np.repeat(pd.date_range('2024-01-02', periods=15), 10)
    frame.loc[::6, 'a'] = np.nan
    frame.loc[::9, 'c'] = np.inf
    full_sums = np.zeros((4, 4))
    full_days = np.zeros((4, 4), dtype=np.int64)
    target_sums = np.zeros((4, 4))
    target_days = np.zeros((4, 4), dtype=np.int64)

    accumulate_correlations(frame, names, full_sums, full_days)
    _accumulate_target_correlations(frame, names, 'b', target_sums, target_days)

    position = names.index('b')
    np.testing.assert_allclose(target_sums[position], full_sums[position])
    np.testing.assert_allclose(target_sums[:, position], full_sums[:, position])
    np.testing.assert_array_equal(target_days[position], full_days[position])
    np.testing.assert_array_equal(target_days[:, position], full_days[:, position])
    other = np.arange(len(names)) != position
    assert np.count_nonzero(target_sums[np.ix_(other, other)]) == 0
    assert np.count_nonzero(target_days[np.ix_(other, other)]) == 0


def test_target_correlation_input_validation():
    with pytest.raises(MetricError):
        _daily_spearman(np.ones(3), target_index=0)
    with pytest.raises(MetricError):
        _daily_spearman(np.ones((3, 2)), target_index=2)


def test_insufficient_and_top_validation():
    result = correlation_result(['a'], np.array([[19.]]), np.array([[19]]), 'a')
    assert np.isnan(result['matrix']).all() and result['insufficient']
    with pytest.raises(MetricError):
        correlation_result(['a'], np.zeros((1, 1)), np.zeros((1, 1)), 'a', 0)


def test_available_empty(tmp_path):
    assert available_factors(tmp_path) == {}


def test_factor_resolution_owns_connection():
    from concurrent.futures import ThreadPoolExecutor
    from calculate import resolve_factor
    names = ['MOM_20', 'RSI_14'] * 20
    with ThreadPoolExecutor(max_workers=2) as executor:
        resolved = list(executor.map(resolve_factor, names))
    assert [spec.name for spec in resolved] == names


def test_pool_filter_and_missing_labels_do_not_change_correlation(tmp_path):
    from types import SimpleNamespace
    from calculate import EvalParams
    from calculate.correlation import compute_correlation
    dates = np.repeat(pd.date_range('2024-01-18', periods=25), 10)
    x = np.tile(np.arange(10), 25)
    frame = pd.DataFrame({'trade_date': dates, 'stock_code': np.tile([f'SH{i:06d}' for i in range(10)], 25),
                          'MOM_20': x.astype(float), 'MOM_60': np.where(x < 5, x, -x).astype(float)})
    target = tmp_path / 'momentum/year=2024/part.parquet'
    target.parent.mkdir(parents=True)
    frame.to_parquet(target, index=False)
    pool = frame[['stock_code', 'trade_date']].assign(in_universe=x < 5, label=np.nan)
    context = SimpleNamespace(panel=pool, params=EvalParams('MOM_20', '2024-01-18', '2024-02-11'),
                              factor_dir=tmp_path)
    result = compute_correlation(context, {'top_n': 10})
    assert result['top'] == [{'factor': 'MOM_60', 'correlation': pytest.approx(1.), 'days': 25}]


def test_candidate_correlation_only_fills_target_row(tmp_path):
    from types import SimpleNamespace
    from calculate import EvalParams
    from calculate.correlation import compute_target_correlation
    dates = np.repeat(pd.date_range('2024-01-18', periods=25), 10)
    x = np.tile(np.arange(10), 25)
    frame = pd.DataFrame({
        'trade_date': dates,
        'stock_code': np.tile([f'SH{i:06d}' for i in range(10)], 25),
        'MOM_20': x.astype(float),
        'MOM_60': np.where(x < 5, x, -x).astype(float),
        'MOM_60_20': (x % 3).astype(float),
    })
    target = tmp_path / 'momentum/year=2024/part.parquet'
    target.parent.mkdir(parents=True)
    frame.to_parquet(target, index=False)
    pool = frame[['stock_code', 'trade_date']].assign(in_universe=True)
    context = SimpleNamespace(
        panel=pool,
        params=EvalParams('MOM_20', '2024-01-18', '2024-02-11'),
        factor_dir=tmp_path,
        spec=SimpleNamespace(
            parquet_glob=str(tmp_path / 'momentum/year=*/part.parquet'),
            formal=False,
        ),
    )

    result = compute_target_correlation(context, {'top_n': 10})

    position = result['names'].index('MOM_20')
    other = np.arange(len(result['names'])) != position
    assert np.isfinite(result['matrix'][position, other]).all()
    assert np.isnan(result['matrix'][np.ix_(other, other)]).all()
    assert np.count_nonzero(result['days'][np.ix_(other, other)]) == 0
    assert {row['factor'] for row in result['top']} == {'MOM_60', 'MOM_60_20'}


def test_candidate_correlation_without_formal_peers(tmp_path):
    from types import SimpleNamespace
    from calculate import EvalParams
    from calculate.correlation import compute_target_correlation
    dates = np.repeat(pd.date_range('2024-01-18', periods=25), 10)
    frame = pd.DataFrame({
        'trade_date': dates,
        'stock_code': np.tile([f'SH{i:06d}' for i in range(10)], 25),
        'MOM_20': np.tile(np.arange(10, dtype=float), 25),
    })
    target = tmp_path / 'momentum/year=2024/part.parquet'
    target.parent.mkdir(parents=True)
    frame.to_parquet(target, index=False)
    pool = frame[['stock_code', 'trade_date']].assign(in_universe=True)
    context = SimpleNamespace(
        panel=pool,
        params=EvalParams('MOM_20', '2024-01-18', '2024-02-11'),
        factor_dir=tmp_path,
        spec=SimpleNamespace(
            parquet_glob=str(tmp_path / 'momentum/year=*/part.parquet'),
            formal=False,
        ),
    )

    result = compute_target_correlation(context, {'top_n': 10})

    assert result['names'] == ['MOM_20']
    assert result['matrix'].shape == (1, 1)
    assert result['matrix'][0, 0] == pytest.approx(1.0)
    assert result['days'][0, 0] == 25
    assert result['top'] == []
    assert result['insufficient']


def test_real_correlation_custom_root(momentum_dir):
    from calculate import EvalParams, evaluate
    events = dict(evaluate('MOM_20', EvalParams('MOM_20', '2024-01-01', '2024-02-29'),
                           factor_dir=momentum_dir, stages=['correlation']))
    result = events['correlation']['metrics']['correlation']
    assert 'MOM_20' in result['names']
    assert result['days'][result['names'].index('MOM_20')][result['names'].index('MOM_20')] >= 20
