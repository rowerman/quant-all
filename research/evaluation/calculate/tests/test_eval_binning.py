"""分位分层与 binplot 的单元测试：分组、收益、换手、自相关都对照手算。"""

import json

import numpy as np
import pandas as pd
import pytest

import factors
from calculate.binning import (
    assign_bins,
    bin_daily_returns,
    bin_matrix,
    bin_turnover,
    factor_autocorrelation,
    rank_matrix,
    row_correlation,
)
from calculate.jsonutil import json_safe
from calculate.metrics import bins as bins_metric
from calculate.metrics import turnover as turnover_metric
from calculate.panel import EvalContext, summarize_panel
from calculate.params import EvalParams
from calculate.stats import annualised_sharpe, compound, max_drawdown, series_stats
from calculate.universe import resolve_universe

START = "2024-01-02"
QUANTILES = 5


def make_frame(days: int = 10, stocks: int = 20, *, factor=None, label: float = 0.0,
               label_1d: float = 0.0, market_cap: float = 1.0,
               seed: int = 3) -> pd.DataFrame:
    """构造一份评估面板。

    参数:
        days: 交易日数。
        stocks: 股票数。
        factor: 因子取值；传序列时每个交易日复用同一套排列，传 None 用随机值。
        label: 期收益常量。
        label_1d: 1 日收益常量。
        market_cap: 市值常量。
        seed: 随机种子。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=days)
    codes = [f"SH600{i:03d}" for i in range(stocks)]
    frame = pd.MultiIndex.from_product(
        [dates, codes], names=["trade_date", "stock_code"]
    ).to_frame(index=False)
    if factor is None:
        frame["factor"] = rng.normal(size=len(frame))
    else:
        values = np.asarray(factor, dtype="float64")
        assert values.size == stocks
        frame["factor"] = np.tile(values, days)
    frame["label"] = float(label)
    frame["label_1d"] = float(label_1d)
    frame["total_mv"] = float(market_cap)
    frame["in_universe"] = True
    return frame


def make_context(frame: pd.DataFrame, *, quantiles: int = QUANTILES,
                 winsorize: bool = True) -> EvalContext:
    """按合成面板拼一个评估上下文。

    参数:
        frame: 合成面板。
        quantiles: 分组数。
        winsorize: 是否去极值。
    """
    spec = factors.get("MOM_20")
    pool = resolve_universe("all_a")
    params = EvalParams(
        factor="MOM_20", start=START, end="2024-12-31",
        quantiles=quantiles, winsorize=winsorize,
    )
    return EvalContext(
        params=params, spec=spec, universe=pool, panel=frame,
        summary=summarize_panel(frame, pool), panel_elapsed_ms=0.0,
    )


def test_assign_bins_splits_into_equal_groups():
    """20 只股票分 5 组，每组 4 只，组号 1 是最低因子值。"""
    frame = make_frame(days=1, stocks=20, factor=np.arange(1, 21))
    binned = assign_bins(frame, QUANTILES)
    ordered = binned.sort_values("factor")
    assert ordered["bin"].tolist() == [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
    assert len(binned) == 20


def test_assign_bins_keeps_ties_together_and_skips_pool_outside_rows():
    """并列值落在同一组；池外与因子为空的行不参与分箱。"""
    frame = make_frame(days=1, stocks=20, factor=np.arange(1, 21))
    frame.loc[frame["stock_code"] == "SH600004", "factor"] = np.nan
    frame.loc[frame["stock_code"] == "SH600019", "in_universe"] = False
    binned = assign_bins(frame, QUANTILES)
    assert len(binned) == 18
    assert set(binned["stock_code"]) == set(frame["stock_code"]) - {"SH600004", "SH600019"}


def test_bin_daily_returns_matches_hand_computation():
    """等权是组内均值；市值加权是组内按 total_mv 的加权平均。"""
    frame = make_frame(days=3, stocks=20)
    rng = np.random.default_rng(11)
    frame["label"] = rng.normal(scale=0.02, size=len(frame))
    frame["total_mv"] = rng.uniform(1e5, 1e6, size=len(frame))
    binned = assign_bins(frame, QUANTILES)

    equal = bin_daily_returns(binned, QUANTILES)
    weighted = bin_daily_returns(binned, QUANTILES, weight_column="total_mv")
    day = binned["trade_date"].unique()[1]
    group = binned[(binned["trade_date"] == day) & (binned["bin"] == 3)]

    assert equal.loc[day, 3] == pytest.approx(group["label"].mean())
    expected = (group["label"] * group["total_mv"]).sum() / group["total_mv"].sum()
    assert weighted.loc[day, 3] == pytest.approx(expected)


def test_weighted_returns_skip_missing_market_cap():
    """缺市值的样本不进加权口径，但等权口径照常算。"""
    frame = make_frame(days=2, stocks=20)
    frame["label"] = 0.01
    frame.loc[frame.index[:4], "total_mv"] = np.nan
    binned = assign_bins(frame, QUANTILES)
    equal = bin_daily_returns(binned, QUANTILES)
    weighted = bin_daily_returns(binned, QUANTILES, weight_column="total_mv")
    day = binned["trade_date"].unique()[0]
    assert equal.loc[day].notna().all()
    assert weighted.loc[day].notna().all()
    assert equal.loc[day, 1] == pytest.approx(weighted.loc[day, 1])


def test_bin_turnover_matches_hand_computation():
    """换手 = 今天在组里、昨天在池内但不在同组的比例。"""
    frame = make_frame(days=3, stocks=4, factor=[1, 2, 3, 4])
    # 第二天整体反序：两组互换，换手 100%；第三天与第二天相同，换手 0
    frame.loc[frame["trade_date"] == frame["trade_date"].unique()[1], "factor"] = [3, 4, 1, 2]
    frame.loc[frame["trade_date"] == frame["trade_date"].unique()[2], "factor"] = [3, 4, 1, 2]
    binned = assign_bins(frame, 2)
    per_bin, overall = bin_turnover(bin_matrix(binned), 2)
    days = frame["trade_date"].unique()
    assert np.isnan(per_bin.loc[days[0], 1])          # 首日没有上一交易日可比
    assert per_bin.loc[days[1], 1] == pytest.approx(1.0)
    assert per_bin.loc[days[1], 2] == pytest.approx(1.0)
    assert per_bin.loc[days[2], 1] == pytest.approx(0.0)
    assert overall.loc[days[1]] == pytest.approx(1.0)
    assert overall.loc[days[2]] == pytest.approx(0.0)


def test_factor_autocorrelation_perfect_and_reversed():
    """同一批股票的排列不变时自相关为 1，整体反序时为 -1。"""
    same = make_frame(days=2, stocks=10, factor=np.arange(1, 11))
    assert factor_autocorrelation(rank_matrix(same), (1,))["lag1"].dropna().iloc[-1] == pytest.approx(1.0)

    reversed_frame = make_frame(days=2, stocks=10, factor=np.arange(1, 11))
    reversed_frame.loc[
        reversed_frame["trade_date"] == reversed_frame["trade_date"].unique()[1], "factor"
    ] = np.arange(10, 0, -1)
    assert factor_autocorrelation(
        rank_matrix(reversed_frame), (1,)
    )["lag1"].dropna().iloc[-1] == pytest.approx(-1.0)


def test_row_correlation_aligns_by_position():
    """两表行索引标签不同时也必须按位置配对，不能按标签取并集。"""
    ascending = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    current = pd.DataFrame([ascending, ascending[::-1]], index=[1, 2])
    previous = pd.DataFrame([ascending, ascending], index=[0, 1])
    corr = row_correlation(current, previous)
    assert list(corr.index) == [1, 2]
    assert corr.iloc[0] == pytest.approx(1.0)
    assert corr.iloc[1] == pytest.approx(-1.0)


def test_bin_returns_payload_monotonic_when_label_follows_factor():
    """标签与因子同向时，分位平均收益递增，单调性秩相关为 1。"""
    frame = make_frame(days=30, stocks=20)
    frame["label"] = frame["factor"]
    context = make_context(frame)
    payload = bins_metric.compute_bin_returns(context, {})

    assert payload["quantiles"] == QUANTILES
    assert len(payload["equal"]["mean"]) == QUANTILES
    assert all_present(payload["equal"]["mean"])
    assert payload["monotonicity"]["equal"]["spearman"] == pytest.approx(1.0)
    assert payload["monotonicity"]["market_cap"]["spearman"] == pytest.approx(1.0)
    assert all(value is not None for value in payload["equal"]["t"])
    assert json.dumps(payload)


def test_curves_use_one_day_label():
    """净值曲线用 1 日收益累乘，不会被期收益的重复重叠带偏。"""
    frame = make_frame(days=10, stocks=20, label=0.5, label_1d=0.01)
    context = make_context(frame)
    payload = bins_metric.compute_bin_curves(context, {})
    assert len(payload["dates"]) == 10
    expected = 1.01 ** 10
    assert all(final == pytest.approx(expected, rel=1e-5) for final in payload["equal"]["final"])
    assert payload["basis"] == bins_metric.CURVE_BASIS


def test_spread_uses_top_minus_bottom_bin():
    """多空 = 最高分位 − 最低分位；用 1 日收益时净值就是它的累乘。"""
    factor = np.arange(1, 21)
    frame = make_frame(days=10, stocks=20, factor=factor, label_1d=0.0)
    days = frame["trade_date"].unique()
    top = frame["factor"] > 16
    frame.loc[top & frame["trade_date"].isin(days[:5]), "label_1d"] = 0.01
    frame.loc[top & ~frame["trade_date"].isin(days[:5]), "label_1d"] = 0.02
    context = make_context(frame)
    payload = bins_metric.compute_spread(context, {})
    assert payload["long_bin"] == QUANTILES and payload["short_bin"] == 1
    assert payload["equal"]["mean"] == pytest.approx(0.015)
    assert payload["equal"]["total_return"] == pytest.approx(1.01 ** 5 * 1.02 ** 5 - 1, rel=1e-4)
    assert payload["equal"]["annual_sharpe"] is not None
    assert payload["equal"]["max_drawdown"] == pytest.approx(0.0)


def test_bin_dist_and_turnover_payloads():
    """分布与换手载荷的键、长度与样本数要对得上。"""
    frame = make_frame(days=12, stocks=20)
    frame["label"] = frame["factor"] * 0.01
    context = make_context(frame)

    distribution = bins_metric.compute_bin_dist(context, {})
    assert len(distribution["daily"]["equal"]) == QUANTILES
    assert distribution["daily"]["equal"][0]["count"] == 12
    assert distribution["stocks"][0]["count"] == 12 * 4
    assert distribution["daily"]["equal"][0]["p25"] <= distribution["daily"]["equal"][0]["p75"]

    turnover = turnover_metric.compute_turnover(context, {})
    assert len(turnover["bin_turnover"]["mean"]) == QUANTILES
    assert set(turnover["autocorrelation"]) == {"lag1", "lag5", "lag20"}
    assert turnover["autocorrelation"]["lag1"]["days"] == 11
    # 指标可以返回 numpy 数组，评估入口会统一过一遍 json_safe
    assert json.dumps(json_safe(turnover))


def test_empty_pool_returns_hint():
    """池内没有非空因子值时给出提示而不是报错。"""
    frame = make_frame(days=5, stocks=20)
    frame["in_universe"] = False
    context = make_context(frame)
    for payload in (
        bins_metric.compute_bin_returns(context, {}),
        bins_metric.compute_bin_curves(context, {}),
        bins_metric.compute_spread(context, {}),
        bins_metric.compute_bin_dist(context, {}),
        turnover_metric.compute_turnover(context, {}),
    ):
        assert payload["empty"] is True
        assert payload["hint"]


def test_stats_helpers_match_hand_computation():
    """净值累乘、最大回撤、年化夏普与手算一致。"""
    curve = compound([0.01, -0.02, 0.03])
    assert curve[0] == pytest.approx(1.01)
    assert curve[2] == pytest.approx(1.01 * 0.98 * 1.03)
    assert max_drawdown(np.array([1.0, 1.2, 0.9, 1.1])) == pytest.approx(0.9 / 1.2 - 1)
    returns = np.array([0.01, -0.005, 0.002, 0.004])
    assert annualised_sharpe(returns) == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    )
    stats = series_stats([0.01, np.nan, 0.03])
    assert stats["days"] == 2
    assert stats["stderr"] == pytest.approx(np.std([0.01, 0.03], ddof=1) / np.sqrt(2))


def all_present(values) -> bool:
    """判断一组取值是否都已算出（没有空值）。

    参数:
        values: 取值序列。
    """
    return all(value is not None for value in values)
