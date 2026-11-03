"""IC 家族的单元测试：与 scipy / numpy 手算逐一对齐，边界分支单独覆盖。"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import factors
from calculate.daily import daily_corr, daily_ic, rank_pct_by_day
from calculate.metrics import coverage as coverage_metric
from calculate.metrics import ic as ic_metric
from calculate.normalize import mad_winsorize
from calculate.panel import EvalContext, summarize_panel
from calculate.params import EvalParams
from calculate.universe import resolve_universe

START, END = "2024-01-02", "2024-02-29"


def make_panel(days: int = 40, stocks: int = 30, seed: int = 7) -> pd.DataFrame:
    """构造一份合成面板：标签与因子正相关，便于核对相关系数。

    参数:
        days: 交易日数。
        stocks: 股票数。
        seed: 随机种子。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    codes = [f"SH600{i:03d}" for i in range(stocks)]
    frame = pd.MultiIndex.from_product(
        [dates, codes], names=["trade_date", "stock_code"]
    ).to_frame(index=False)
    factor_values = rng.normal(size=len(frame))
    frame["factor"] = factor_values
    frame["label"] = 0.5 * factor_values + rng.normal(size=len(frame))
    frame["in_universe"] = True
    return frame


def make_context(panel: pd.DataFrame, *, winsorize: bool = True,
                 factor: str = "MOM_20", universe: str = "all_a") -> EvalContext:
    """按合成面板拼一个评估上下文，绕开数据库。"""
    spec = factors.get(factor)
    pool = resolve_universe(universe)
    params = EvalParams(
        factor=factor, start=START, end=END, universe=universe, winsorize=winsorize
    )
    return EvalContext(
        params=params, spec=spec, universe=pool, panel=panel,
        summary=summarize_panel(panel, pool), panel_elapsed_ms=0.0,
    )


def test_daily_ic_matches_scipy_per_day():
    """逐日 IC / RankIC 与 scipy 的 Pearson / Spearman 完全一致。"""
    panel = make_panel()
    series = daily_ic(panel, winsorize=False)
    subset = panel[["trade_date", "factor", "label"]]
    expected_ic = pd.Series({
        day: stats.pearsonr(group["factor"], group["label"]).statistic
        for day, group in subset.groupby("trade_date")
    })
    expected_rank = pd.Series({
        day: stats.spearmanr(group["factor"], group["label"]).statistic
        for day, group in subset.groupby("trade_date")
    })
    assert np.allclose(series["ic"], expected_ic.reindex(series.index), atol=1e-12)
    assert np.allclose(series["rank_ic"], expected_rank.reindex(series.index), atol=1e-12)
    assert (series["n"] == 30).all()


def test_perfect_rank_factor_gives_unit_rank_ic():
    """单调因子（含并列）的 RankIC 应为 1，Pearson IC 允许略低。"""
    panel = make_panel(days=25, stocks=20)
    panel["label"] = panel.groupby("trade_date")["factor"].rank()
    series = daily_ic(panel, winsorize=False)
    assert np.allclose(series["rank_ic"].to_numpy(), 1.0, atol=1e-12)


def test_rank_pct_by_day_matches_pandas_with_ties_and_missing():
    """截面分位与 pandas 同语义：并列取平均名次，空值不参与也留空。"""
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(5), 12)
    values = rng.integers(0, 4, size=days.size).astype("float64")  # 大量并列
    values[[3, 17, 40]] = np.nan
    mine = rank_pct_by_day(days, values)
    expected = pd.DataFrame({"day": days, "value": values}).groupby("day")["value"].rank(
        pct=True
    ).to_numpy()
    assert np.allclose(mine, expected, atol=1e-15, equal_nan=True)


def test_daily_corr_returns_empty_after_filtering_non_finite_pairs():
    """有限配对过滤后为空时返回空表，不把空数组送入批量核。"""

    panel = make_panel(days=2, stocks=6)
    panel["factor"] = np.inf

    result = daily_corr(panel, "factor", "label", rank=True)

    assert result.empty
    assert list(result.columns) == ["corr", "n"]


def test_series_stats_matches_numpy():
    """四件套（均值、标准差、IR / t、胜率）与 numpy 手算一致。"""
    values = np.array([0.1, -0.2, 0.05, 0.3, np.nan])
    result = ic_metric.series_stats(values)
    valid = np.array([0.1, -0.2, 0.05, 0.3])
    mean, std = valid.mean(), valid.std(ddof=1)
    assert result["mean"] == pytest.approx(mean)
    assert result["std"] == pytest.approx(std)
    assert result["ir"] == pytest.approx(mean / std)
    assert result["t"] == pytest.approx(mean / std * np.sqrt(valid.size))
    assert result["win_rate"] == pytest.approx(3 / 4)
    assert result["days"] == 4


def test_insufficient_sample_masks_numbers():
    """有效交易日不足时只给标记，不给数值。"""
    context = make_context(make_panel(days=10))
    result = ic_metric.compute_stats(context, {})
    assert result["insufficient"] is True
    assert result["valid_days"] == 10
    assert result["ic"]["mean"] is None
    assert result["rank_ic"]["mean"] is None
    assert result["direction_aligned"] is None
    assert "有效交易日" in result["hint"]


def test_small_cross_section_days_excluded():
    """单日有效配对少于 5 只的截面不参与统计，但样本数照实报出来。"""
    panel = make_panel(days=30, stocks=30)
    days = panel["trade_date"].unique()[:3]
    dropped = panel["stock_code"].unique()[:26]
    keep = ~(panel["trade_date"].isin(days) & panel["stock_code"].isin(dropped))
    series = daily_ic(panel[keep], winsorize=False)
    small = series.loc[series.index[:3]]
    assert small["ic"].isna().all()
    assert (small["n"] == 4).all()
    assert series["ic"].notna().sum() == 27


def test_winsorize_uses_mad_clip():
    """去极值口径：按截面 3 倍 MAD 裁剪后再算 Pearson。"""
    panel = make_panel(days=25, stocks=20)
    panel.loc[panel.index[::20], "factor"] = 1e6
    with_clip = daily_ic(panel, winsorize=True)
    without_clip = daily_ic(panel, winsorize=False)
    assert not np.allclose(with_clip["ic"], without_clip["ic"])

    day = panel["trade_date"].unique()[0]
    group = panel[panel["trade_date"] == day]
    expected = stats.pearsonr(
        mad_winsorize(group["factor"].to_numpy()),
        mad_winsorize(group["label"].to_numpy()),
    ).statistic
    assert with_clip.loc[day, "ic"] == pytest.approx(expected, abs=1e-12)


def test_series_and_dist_shapes():
    """时序长度与直方图分箱要和交易日数对得上。"""
    context = make_context(make_panel(days=40))
    series = ic_metric.compute_series(context, {})
    assert len(series["dates"]) == 40
    np.testing.assert_allclose(
        series["cumulative"]["ic"], np.cumsum(series["ic"]), atol=1e-12
    )
    np.testing.assert_allclose(
        series["cumulative"]["rank_ic"], np.cumsum(series["rank_ic"]), atol=1e-12
    )
    assert len(series["rolling"]["ic"]) == 40
    assert series["rolling"]["window"] == ic_metric.ROLLING_WINDOW

    distribution = ic_metric.compute_dist(context, {})
    assert len(distribution["ic"]["edges"]) == distribution["bins"] + 1
    assert sum(distribution["ic"]["counts"]) == 40


def test_cumulative_series_skips_invalid_days_without_contamination():
    """累计 IC 遇到无效日保留空值，后续有效日从此前累计值继续。"""
    panel = make_panel(days=25, stocks=30)
    days = panel["trade_date"].unique()[:2]
    dropped = panel["stock_code"].unique()[:26]
    panel = panel.loc[
        ~(panel["trade_date"].isin(days) & panel["stock_code"].isin(dropped))
    ]
    result = ic_metric.compute_series(make_context(panel), {})
    cumulative = result["cumulative"]["ic"]
    raw = result["ic"]
    assert np.isnan(cumulative[:2]).all()
    assert cumulative[2] == pytest.approx(raw[2])


def test_coverage_metric_counts_missing_and_infinite():
    """覆盖率、缺失与非有限值分别统计，互不混淆。"""
    panel = make_panel(days=30, stocks=20)
    panel.loc[panel.index[:5], "factor"] = np.nan
    panel.loc[panel.index[5:7], "factor"] = np.inf
    panel.loc[panel.index[7:9], "label"] = np.nan
    panel.loc[panel.index[10:12], "in_universe"] = False
    context = make_context(panel)
    result = coverage_metric.compute(context, {})

    pool = panel["in_universe"]
    assert result["rows"] == len(panel) == 600
    assert result["pool_rows"] == int(pool.sum()) == 598
    assert result["missing"] == 5
    assert result["non_finite"] == 2
    assert result["coverage_all"] == pytest.approx(panel["factor"].notna().mean())
    assert result["coverage_universe"] == pytest.approx(
        panel.loc[pool, "factor"].notna().mean()
    )
    expected_pairs = (pool & panel["factor"].notna() & panel["label"].notna()).sum()
    assert result["pair_coverage"] == pytest.approx(expected_pairs / pool.sum())
