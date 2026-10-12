"""V11 审查回归：独立参考值覆盖局部缺失、退化截面和扩展契约。"""

import importlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from calculate import EvalParams, evaluate
from calculate.daily import daily_ic, daily_rank_ic_multi
from calculate.errors import EvalParamError, MetricError
from calculate.metrics import Metric, metrics_by_stage, register_metric, unregister_metric
from calculate.metrics.decay import compute_decay
from calculate.metrics.exposure import compute_exposure, compute_incremental_ic
from calculate.stats import compound, max_drawdown
from calculate.tests.test_eval_decay import make_frame, make_context as decay_context
from calculate.tests.test_eval_exposure import make_data, make_context as exposure_context


def test_decay_reranks_each_pairwise_sample():
    """各档先取有效配对，再排名，局部因子/标签缺失都与 scipy 一致。"""
    frame = make_frame(days=25, stocks=20)
    frame.loc[frame.index[::7], 'label_h20'] = np.nan
    frame.loc[frame.index[::11], 'factor'] = np.nan
    frame['label_h3'] = np.round(frame['label_h3'], 2)
    series = daily_rank_ic_multi(frame, ['label', 'label_h20', 'label_h3'])
    for column, result in series.items():
        expected = {
            day: spearmanr(group['factor'], group[column]).statistic
            for day, group in frame.dropna(subset=['factor', column]).groupby('trade_date')
        }
        assert np.allclose(result['rank_ic'], pd.Series(expected).reindex(result.index), atol=1e-12)
    assert np.allclose(series['label']['rank_ic'], daily_ic(frame)['rank_ic'], atol=1e-12)


def test_drawdown_includes_initial_capital():
    """初始净值 1 属于历史峰值，首日亏损不能漏掉。"""
    assert max_drawdown(compound([-.1, 0.])) == pytest.approx(-.1)
    assert max_drawdown(compound([-.1, -.1])) == pytest.approx(-.19)
    assert max_drawdown(compound([.1, -.1])) == pytest.approx(-.1)


def test_constant_rank_ic_has_no_ir_but_does_not_crash():
    """零波动 IC 序列的 ICIR 无定义，载荷与结论应明确返回空值。"""
    frame = make_frame(days=25)
    frame['label'] = frame['factor']
    result = compute_decay(decay_context(frame), {})
    primary = next(item for item in result['curve'] if item['horizon'] == 5)
    assert primary['rank_ic_mean'] == pytest.approx(1.)
    assert primary['rank_ic_ir'] is None
    assert '无定义' in result['hint']


def test_constant_response_day_is_not_a_valid_regression():
    """因子常数日没有可估计响应，应跳过并准确统计有效回归日。"""
    panel, styles = make_data(days=25, stocks=40)
    day = panel.trade_date.iloc[0]
    panel.loc[panel.trade_date == day, 'factor'] = 1.
    result = compute_exposure(exposure_context(panel, styles), {})
    assert result['days'] == 24
    assert result['skipped_days'] == 1
    assert str(day.date()) not in result['series']['dates']
    assert all(np.isfinite(values).all() for values in result['series']['coefficient'].values())


def test_whole_missing_style_does_not_discard_other_estimable_styles():
    """某风格整日缺失只使该系数不可估，其他风格和残差仍有结果。"""
    panel, styles = make_data(days=25, stocks=40)
    day = panel.trade_date.iloc[0]
    styles.loc[styles.trade_date == day, 'VOL'] = np.nan
    result = compute_exposure(exposure_context(panel, styles), {})
    assert result['days'] == 25
    assert result['skipped_days'] == 0
    assert np.isnan(result['series']['coefficient']['VOL'][0])
    assert np.isfinite(result['series']['coefficient']['SIZE'][0])


def test_raw_incremental_ic_is_computed_on_residual_rows():
    """回归样本过滤后，raw 必须在残差的同一批有效配对行上重算。"""
    panel, styles = make_data(days=25, stocks=40)
    # 有限因子但风格行响应退化的日子会被排除；检验日度配对一致而非全池复用。
    panel.loc[panel.trade_date == panel.trade_date.iloc[0], 'factor'] = 1.
    context = exposure_context(panel, styles)
    result = compute_incremental_ic(context, {})
    assert result['raw']['days'] == result['residual']['days'] == 24
    assert len(result['series']['raw']) == len(result['series']['residual']) == 24


def test_metric_order_preserves_registration_within_stage():
    """卡片顺序应由注册顺序决定，不由指标 key 字母顺序决定。"""
    try:
        register_metric(Metric('zzz_regression', '先注册', 'correlation', lambda c, o: {}))
        register_metric(Metric('aaa_regression', '后注册', 'correlation', lambda c, o: {}))
        assert [m.key for m in metrics_by_stage('correlation')] == ['zzz_regression', 'aaa_regression']
    finally:
        unregister_metric('zzz_regression')
        unregister_metric('aaa_regression')


def test_invalid_metric_stage_cannot_be_silently_ignored():
    """注册非法 stage 必须报错，避免指标注册了却永远不执行。"""
    try:
        with pytest.raises(MetricError):
            register_metric(Metric('invalid_stage_regression', '非法阶段', 'oops', lambda c, o: {}))
    finally:
        unregister_metric('invalid_stage_regression')


def test_evaluate_passes_validated_metric_options(monkeypatch):
    """统一入口把指标参数传到计算函数，不再强制默认行业 off。"""
    module = importlib.import_module('calculate.evaluate')
    panel, styles = make_data(days=25, stocks=40)
    context = exposure_context(panel, styles)
    monkeypatch.setattr(module, 'build_panel', lambda *a, **kw: context)
    from calculate.metrics.exposure import INDUSTRY_PARAM
    original = metrics_by_stage('exposure')[0]
    seen = []
    register_metric(Metric('exposure', original.label, 'exposure',
                          lambda c, o: seen.append(o) or {'industry': o['industry']},
                          params=(INDUSTRY_PARAM,)), overwrite=True)
    try:
        events = dict(evaluate('MOM_20', context.params, stages=['exposure'],
                               metric_options={'exposure': {'industry': 'stock_basic'}}))
        assert seen == [{'industry': 'stock_basic'}]
        assert events['exposure']['metrics']['exposure']['industry'] == 'stock_basic'
        assert events['meta']['stages'] == ['exposure']
    finally:
        register_metric(original, overwrite=True)


@pytest.mark.parametrize('kwargs', [
    {'stages': ['not_a_stage']},
    {'metric_options': {'not_a_metric': {}}},
    {'metric_options': {'exposure': {'industry': 'not_a_source'}}},
    {'metric_options': {'exposure': {'unknown': True}}},
])
def test_invalid_evaluation_options_rejected_before_reading(monkeypatch, kwargs):
    """非法 stage 或指标参数在读库之前给明确错误。"""
    module = importlib.import_module('calculate.evaluate')
    monkeypatch.setattr(module, 'build_panel', lambda *a, **kw: pytest.fail('不应读取数据库'))
    panel, styles = make_data(days=25, stocks=40)
    context = exposure_context(panel, styles)
    with pytest.raises((EvalParamError, MetricError)):
        list(evaluate('MOM_20', context.params, **kwargs))


def test_metric_module_is_discovered_without_editing_imports(tmp_path, monkeypatch):
    """新增模块自动导入并注册，不需要编辑 metrics/__init__.py。"""
    import calculate.metrics as metrics
    module_name = 'v11_discovered_regression'
    (tmp_path / f'{module_name}.py').write_text(
        'from calculate.metrics.base import Metric, register_metric\n'
        'register_metric(Metric("discovered_regression", "自动发现", "correlation", '
        'lambda c, o: {"ok": True}))\n'
    )
    monkeypatch.setattr(metrics, '__path__', [*metrics.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    try:
        metrics.discover_metrics()
        assert metrics.metrics_by_stage('correlation')[0].key == 'discovered_regression'
        metrics.discover_metrics()  # 重复发现不重复注册
    finally:
        unregister_metric('discovered_regression')
        import sys
        sys.modules.pop(f'calculate.metrics.{module_name}', None)


def test_fully_explained_factor_has_zero_residual_not_roundoff_signal():
    """完全由一个风格产生的因子，其残差为零，残差 RankIC 无定义。"""
    panel, styles = make_data(days=25, stocks=40)
    panel['factor'] = styles['SIZE'].to_numpy()
    context = exposure_context(panel, styles)
    result = context.exposures()
    assert np.equal(result.residual['residual'].to_numpy(), 0.).all()
    payload = compute_incremental_ic(context, {})
    assert payload['insufficient'] is True
    assert payload['days'] == 0
    assert payload['residual']['mean'] is None


def test_autocorrelation_reranks_shared_stocks_after_pool_changes():
    """池子变化时先取两期共有股票，再分别排名，结果与 scipy 相同。"""
    from calculate.binning import factor_autocorrelation
    ranks = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8],
                          [8, 1, np.nan, 4, np.nan, 2, 7, 3]],
                         index=pd.bdate_range('2024-01-02', periods=2))
    keep = ranks.iloc[1].notna()
    expected = spearmanr(ranks.iloc[0][keep], ranks.iloc[1][keep]).statistic
    result = factor_autocorrelation(ranks, (1,))
    assert result['lag1'].iloc[1] == pytest.approx(expected, abs=1e-12)


def test_matrix_lags_do_not_compress_empty_pool_dates():
    """股票池整日为空时仍保留该交易日，lag 1 不能跳过空日。"""
    frame = make_frame(days=3, stocks=20)
    dates = sorted(frame.trade_date.unique())
    frame.loc[frame.trade_date == dates[1], 'in_universe'] = False
    context = decay_context(frame)
    matrix, ranks = context.bin_matrices()
    assert len(matrix.index) == len(ranks.index) == 3
    assert matrix.loc[dates[1]].isna().all()
    from calculate.binning import factor_autocorrelation
    assert factor_autocorrelation(ranks, (1,)).loc[dates[2], 'lag1'] != 1.


def test_style_lookback_preserves_history_after_long_suspension():
    """260 个市场交易日不足个股的 250 次交易时，应继续取更早历史。"""
    import duckdb
    from calculate.styles import build_styles
    dates = pd.bdate_range('2020-01-02', periods=600)
    valid = np.ones(len(dates), dtype=bool)
    valid[200:380] = False
    bars = pd.DataFrame({'trade_date': dates[valid], 'stock_code': 'SH600000',
                         'status': 'trading', 'close': np.arange(1., valid.sum()+1),
                         'ret': .01, 'total_mv': 100., 'turnover_rate': 1.,
                         'pe_ttm': 10., 'pb': 2.})
    cal = pd.DataFrame({'trade_date': dates})
    index = pd.DataFrame({'trade_date': dates, 'index_code': '000300.SH',
                          'pct_chg': np.sin(np.arange(len(dates)))})
    conn = duckdb.connect()
    conn.execute("ATTACH ':memory:' AS src")
    conn.execute('CREATE SCHEMA src.md')
    conn.register('bars', bars)
    conn.register('cal', cal)
    conn.register('index_rows', index)
    conn.execute('CREATE TABLE src.md.daily_hfq AS SELECT * FROM bars')
    conn.execute('CREATE TABLE src.md.calendar AS SELECT * FROM cal')
    conn.execute('CREATE TABLE src.md.index_daily AS SELECT * FROM index_rows')
    try:
        end = str(dates[520].date())
        short = build_styles(EvalParams(factor='MOM_20', start=end, end=end), conn=conn)
        full = build_styles(EvalParams(factor='MOM_20', start=str(dates[0].date()), end=end), conn=conn)
        expected = full.loc[full.trade_date == dates[520], 'MOMENTUM'].iloc[0]
        assert np.isfinite(expected)
        assert short['MOMENTUM'].iloc[0] == pytest.approx(expected, abs=1e-12)
    finally:
        conn.close()


@pytest.mark.parametrize('overrides', [
    {'start': '2024-01-01\' OR 1=1 --'}, {'start': '2024-02-30'},
    {'end': '2024-1-2'}, {'horizon': True}, {'horizon': 5.0},
    {'quantiles': 5.0}, {'winsorize': 'false'},
])
def test_parameters_have_strict_dates_and_types(overrides):
    """输入参数在拼 SQL 前校验日期格式与类型。"""
    from calculate import EvalParams
    values = dict(factor='MOM_20', start='2024-01-02', end='2024-12-31') | overrides
    with pytest.raises(EvalParamError):
        EvalParams(**values)
