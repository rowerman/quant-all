"""多周期衰减的单元测试：各档 RankIC、分位收益与末端缺失都对照手算。"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import factors
from calculate.binning import assign_bins
from calculate.daily import daily_rank_ic_multi
from calculate.errors import DecomposeError
from calculate.jsonutil import json_safe
from calculate.metrics import decay as decay_metric
from calculate.metrics import ic as ic_metric
from calculate.panel import EvalContext, summarize_panel
from calculate.params import DECAY_HORIZONS, EvalParams
from calculate.universe import resolve_universe

START = "2024-01-02"
QUANTILES = 5

# 档位与列名的对应关系（默认 5 日档下）
COLUMNS = {1: "label_1d", 3: "label_h3", 5: "label", 10: "label_h10", 20: "label_h20"}


def make_frame(days: int = 30, stocks: int = 20, *, seed: int = 7,
               horizon: int = 5) -> pd.DataFrame:
    """构造一份带多周期标签的评估面板。

    参数:
        days: 交易日数。
        stocks: 股票数。
        seed: 随机种子。
        horizon: 评估参数里的标签周期。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=days)
    codes = [f"SH600{i:03d}" for i in range(stocks)]
    frame = pd.MultiIndex.from_product(
        [dates, codes], names=["trade_date", "stock_code"]
    ).to_frame(index=False)
    frame["factor"] = np.tile(rng.normal(size=stocks), days)
    frame["label"] = rng.normal(scale=0.02, size=len(frame))
    frame["label_1d"] = rng.normal(scale=0.01, size=len(frame))
    for column in ("label_h3", "label_h10", "label_h20"):
        if column not in frame:
            frame[column] = rng.normal(scale=0.03, size=len(frame))
    frame["total_mv"] = rng.uniform(1e5, 1e6, size=len(frame))
    frame["in_universe"] = True
    if horizon != 5:
        frame = frame.drop(columns=["label", "label_h3", "label_h10", "label_h20"])
        frame["label"] = rng.normal(scale=0.02, size=len(frame))
        for value in DECAY_HORIZONS:
            name = f"label_h{value}"
            if value not in (1, horizon):
                frame[name] = rng.normal(scale=0.02, size=len(frame))
    return frame


def make_context(frame: pd.DataFrame, *, horizon: int = 5,
                 quantiles: int = QUANTILES) -> EvalContext:
    """按合成面板拼一个评估上下文。

    参数:
        frame: 合成面板。
        horizon: 标签周期。
        quantiles: 分组数。
    """
    spec = factors.get("MOM_20")
    pool = resolve_universe("all_a")
    params = EvalParams(
        factor="MOM_20", start=START, end="2024-12-31",
        horizon=horizon, quantiles=quantiles,
    )
    return EvalContext(
        params=params, spec=spec, universe=pool, panel=frame,
        summary=summarize_panel(frame, pool), panel_elapsed_ms=0.0,
    )


def hand_rank_ic(frame: pd.DataFrame, column: str) -> dict:
    """按交易日逐日手算 Spearman（scipy），返回 ``{交易日: 相关}``。

    参数:
        frame: 面板。
        column: 标签列。
    """
    result = {}
    for day, group in frame.groupby("trade_date"):
        if group["factor"].notna().sum() < 2 or group[column].notna().sum() < 2:
            continue
        result[day] = stats.spearmanr(group["factor"], group[column]).statistic
    return result


def test_decay_rank_ic_matches_scipy_hand_computation():
    """各档 RankIC 与 scipy 逐日手算一致，天数与样本数对得上。"""
    frame = make_frame()
    context = make_context(frame)
    payload = decay_metric.compute_decay(context, {})

    assert [item["horizon"] for item in payload["curve"]] == list(DECAY_HORIZONS)
    for item in payload["curve"]:
        expected = hand_rank_ic(frame, COLUMNS[item["horizon"]])
        values = np.array(list(expected.values()), dtype="float64")
        assert item["days"] == len(values)
        assert item["samples"] == len(frame)
        assert item["rank_ic_mean"] == pytest.approx(values.mean(), rel=1e-9)
        assert item["insufficient"] is False


def test_multi_rank_ic_matches_daily_ic_on_primary_horizon():
    """多档 RankIC 在主档上必须与 IC 家族的 RankIC 完全一致。"""
    frame = make_frame()
    context = make_context(frame)
    mine = ic_metric.compute_stats(context, {})["rank_ic"]["mean"]
    series = daily_rank_ic_multi(frame.loc[frame["in_universe"]], ["label", "label_h20"])
    assert series["label"]["rank_ic"].mean() == pytest.approx(mine, abs=1e-12)
    # 因子秩只算一次也不能改口径：与逐列单独算的结果一致
    alone = daily_rank_ic_multi(frame.loc[frame["in_universe"]], ["label_h20"])
    assert np.allclose(
        series["label_h20"]["rank_ic"].to_numpy(),
        alone["label_h20"]["rank_ic"].to_numpy(),
        atol=1e-12,
        equal_nan=True,
    )


def test_tail_missing_label_only_affects_that_horizon():
    """末端标签缺失只从该档剔除样本，不填充、不影响其它档。"""
    frame = make_frame(days=25)
    dates = sorted(frame["trade_date"].unique())
    tail = dates[-4:]
    frame.loc[frame["trade_date"].isin(tail), "label_h20"] = np.nan
    payload = decay_metric.compute_decay(make_context(frame), {})
    curve = {item["horizon"]: item for item in payload["curve"]}

    assert curve[20]["days"] == 21
    assert curve[20]["samples"] == 21 * 20
    for horizon in (1, 3, 5, 10):
        assert curve[horizon]["days"] == 25
        assert curve[horizon]["samples"] == 25 * 20
    # 缺标签的那几天不能变成长周期档里的 0 或上一个值：直接不在样本里
    assert curve[20]["rank_ic_mean"] == pytest.approx(
        np.mean([value for day, value in hand_rank_ic(frame, "label_h20").items()
                 if day not in tail]),
        rel=1e-9,
    )


def test_long_excess_and_long_short_match_hand_computation():
    """多头超额 = 最高分位 − 池内均值；多空 = 最高 − 最低（两套权重都核）。"""
    frame = make_frame()
    context = make_context(frame)
    payload = decay_metric.compute_decay(context, {})
    binned = assign_bins(frame, QUANTILES, label_columns=context.label_columns)
    column = COLUMNS[10]

    def weighted_mean(sub: pd.DataFrame, weight_column: str | None) -> pd.Series:
        """按天算组内均值（可加权）。

        参数:
            sub: 某个分位的行。
            weight_column: 加权列；为空表示等权。
        """
        if weight_column is None:
            return sub.groupby("trade_date")[column].mean()
        work = sub.loc[sub[weight_column] > 0]
        numerator = work.assign(_x=work[column] * work[weight_column])
        grouped = numerator.groupby("trade_date")
        return grouped["_x"].sum() / grouped[weight_column].sum()

    for weight, weight_column in (("equal", None), ("market_cap", "total_mv")):
        top = weighted_mean(binned[binned["bin"] == QUANTILES], weight_column)
        bottom = weighted_mean(binned[binned["bin"] == 1], weight_column)
        pool = weighted_mean(binned, weight_column)
        excess = top - pool.reindex(top.index)
        spread = top - bottom

        item = next(
            entry for entry in payload["long_excess"][weight] if entry["horizon"] == 10
        )
        assert item["mean"] == pytest.approx(excess.mean(), rel=1e-9)
        assert item["t"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(
            excess.notna().sum()
        ), rel=1e-9)
        assert item["days"] == int(excess.notna().sum())

        spread_item = next(
            entry for entry in payload["long_short"][weight] if entry["horizon"] == 10
        )
        assert spread_item["mean"] == pytest.approx(spread.mean(), rel=1e-9)
        assert spread_item["win_rate"] == pytest.approx((spread > 0).mean(), rel=1e-9)


def test_decay_weighting_labels_and_basis():
    """两套权重的口径说明与键齐备，弱化期收益不可比这件事写清楚。"""
    payload = decay_metric.compute_decay(make_context(make_frame()), {})
    assert payload["weightings"] == ["equal", "market_cap"]
    assert payload["weight_labels"]["equal"] == "等权"
    assert payload["quantiles"] == QUANTILES
    assert payload["long_bin"] == QUANTILES
    assert "池内均值" in payload["long_excess_basis"]
    assert "不做填充" in payload["basis"]
    assert "不可直接比较" in payload["note"]
    assert set(payload["long_excess"]) == {"equal", "market_cap"}
    assert json.dumps(json_safe(payload))


def test_short_window_reports_insufficient():
    """有效交易日不足 20 天时只报天数，数值置空并给出提示。"""
    payload = decay_metric.compute_decay(make_context(make_frame(days=10)), {})
    assert all(item["insufficient"] for item in payload["curve"])
    assert all(
        item["rank_ic_mean"] is None and item["rank_ic_ir"] is None and item["t"] is None
        for item in payload["curve"]
    )
    assert all(item["days"] == 10 for item in payload["curve"])
    assert payload["min_days"] == 20
    assert "少于 20 天" in payload["hint"]


def test_empty_pool_returns_hint():
    """池内没有样本时给提示而不是报错。"""
    frame = make_frame()
    frame["in_universe"] = False
    payload = decay_metric.compute_decay(make_context(frame), {})
    assert payload["empty"] is True
    assert payload["long_excess"] == {}
    assert payload["hint"]
    assert [item["horizon"] for item in payload["curve"]] == list(DECAY_HORIZONS)


def test_horizon_label_mapping_and_missing_column():
    """档位 → 面板列名的映射，以及面板缺列时的报错。"""
    context = make_context(make_frame(horizon=20), horizon=20)
    assert context.horizon_label(20) == "label"
    assert context.horizon_label(10) == "label_h10"
    assert context.horizon_label(1) == "label_1d"

    frame = make_frame(horizon=20).drop(columns=["label_h10"])
    context = make_context(frame, horizon=20)
    with pytest.raises(DecomposeError) as error:
        context.horizon_label(10)
    assert "label_h10" in str(error.value)
