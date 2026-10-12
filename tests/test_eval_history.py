"""轻量 IC 路径与面板逐日手算、自检脚本的口径一致。"""

import numpy as np
import pytest
from calculate import EvalParams, build_panel, load_daily_ic
from calculate.config import DB_FILE

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason='需要真实主库')


@pytest.mark.parametrize('winsorize', [False, True])
@pytest.mark.parametrize('universe', ['all_a', 'index:000300.SH'])
def test_history_ic_matches_panel_on_same_dates(momentum_dir, winsorize, universe):
    """同样本下 SQL 平均并列名次与 numpy/pandas 结果相等，包含局部缺失。"""
    params = EvalParams(factor='MOM_20', start='2024-05-06', end='2024-05-31',
                        winsorize=winsorize, universe=universe)
    expected = build_panel(params, factor_dir=momentum_dir).daily_ic()
    actual = load_daily_ic(params, factor_dir=momentum_dir)
    assert actual.index.equals(expected.index)
    assert np.array_equal(actual['n'], expected['n'])
    assert np.allclose(actual['ic'], expected['ic'], atol=1e-10, rtol=0., equal_nan=True)
    assert np.allclose(actual['rank_ic'], expected['rank_ic'], atol=1e-10, rtol=0., equal_nan=True)
    assert actual.attrs['elapsed_ms'] > 0


def test_sql_average_ties_and_invalid_days_match_scipy(tmp_path, monkeypatch):
    """有并列、局部缺失、常数与小截面时，自检与轻量 IC 都与 scipy 一致。"""
    import duckdb
    import pandas as pd
    import factors
    from scipy.stats import spearmanr
    from calculate.panel import open_source
    from scripts.build_factors import validate
    dates = pd.bdate_range('2024-01-02', periods=25)
    codes = [f'SH60000{i}' for i in range(8)]
    rows = pd.MultiIndex.from_product([dates,codes],names=['trade_date','stock_code']).to_frame(index=False)
    values = np.tile([1.,2.,2.,4.,5.,6.,6.,8.],len(dates))
    labels = np.tile([8.,1.,3.,4.,5.,2.,7.,3.],len(dates))
    labels[2::8] = np.nan
    values[:8] = 1.  # 常数日不算有效日
    labels[8:12] = np.nan  # 第二天不足 5 个配对
    factor_rows = rows.assign(MOM_20=values)
    label_rows = rows.assign(fwd_ret_close_5=labels)
    bars = rows.assign(can_buy_close=True,is_st=False,list_date=pd.Timestamp('2000-01-01'))
    calendar = pd.DataFrame({'trade_date':dates})
    db = tmp_path/'market.duckdb'
    conn=duckdb.connect(str(db))
    conn.execute('CREATE SCHEMA md')
    for name,frame in [('daily_hfq',bars),('fwd_return',label_rows),('calendar',calendar)]:
        conn.register('frame',frame)
        conn.execute(f'CREATE TABLE md.{name} AS SELECT * FROM frame')
    conn.close()
    target=tmp_path/'factors'/'momentum'/'year=2024'
    target.mkdir(parents=True)
    factor_rows.to_parquet(target/'part.parquet',index=False)
    params=EvalParams(factor='MOM_20',start=str(dates[0].date()),end=str(dates[-1].date()),winsorize=False)
    actual=load_daily_ic(params,db_file=db,factor_dir=target.parents[1])
    combined=rows.assign(factor=values,label=labels).dropna()
    expected={day:spearmanr(g.factor,g.label).statistic for day,g in combined.groupby('trade_date')
              if len(g)>=5 and g.factor.nunique()>1 and g.label.nunique()>1}
    assert len(expected)==23
    assert np.allclose(actual.rank_ic.dropna(),list(expected.values()),atol=1e-12,rtol=0.)
    monkeypatch.setattr(validate,'family_factors',lambda family:[factors.get('MOM_20')])
    conn=open_source(db)
    try:
        check=validate.check_direction(conn,'momentum',params.start,params.end,target.parents[1])[0]
    finally:
        conn.close()
    assert check['value']==pytest.approx(np.mean(list(expected.values())),abs=1e-12)
    assert '23 个交易日' in check['detail']
