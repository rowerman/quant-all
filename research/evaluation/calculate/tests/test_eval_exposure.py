"""风格拆分的单元测试：暴露系数、残差 IC、缺失处理与行业开关都对照手算。"""

import json
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import factors
from calculate.daily import MIN_CROSS_SECTION
from calculate.errors import DecomposeError, IndustryError
from calculate.jsonutil import json_safe
from calculate.metrics import exposure as exposure_metric
from calculate.normalize import zscore_by_day
from calculate.panel import EvalContext, summarize_panel
from calculate.params import EvalParams
from calculate.styles import INDUSTRY_SOURCES, RAW_COLUMNS, STYLE_KEYS
from calculate.universe import resolve_universe
from tools.monitor.baseline_contract import compare_golden

START = "2024-01-02"
DAYS, STOCKS = 40, 30
MAD_SCALE = 1.4826
MIN_ROWS = 30

requires_native = pytest.mark.skipif(
    importlib.util.find_spec("quant_native") is None,
    reason="quant_native 扩展未构建",
)


def make_data(days: int = DAYS, stocks: int = STOCKS, *, seed: int = 5
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造合成面板与风格表：因子 = 0.6·SIZE + 0.4·MOMENTUM + 噪声。

    参数:
        days: 交易日数。
        stocks: 股票数。
        seed: 随机种子。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=days)
    codes = [f"SH600{i:03d}" for i in range(stocks)]
    panel = pd.MultiIndex.from_product(
        [dates, codes], names=["trade_date", "stock_code"]
    ).to_frame(index=False)
    panel["in_universe"] = True
    styles = panel[["trade_date", "stock_code"]].copy()
    for column in RAW_COLUMNS:
        styles[column] = rng.normal(size=len(styles))
    signal = 0.6 * styles["SIZE"] + 0.4 * styles["MOMENTUM"]
    panel["factor"] = signal.to_numpy() + rng.normal(scale=0.5, size=len(panel))
    panel["label"] = signal.to_numpy() + rng.normal(scale=0.5, size=len(panel))
    return panel, styles


def make_context(panel: pd.DataFrame, styles: pd.DataFrame, *, horizon: int = 5) -> EvalContext:
    """拼一个上下文，并把风格表直接塞进缓存（绕开数据库）。

    参数:
        panel: 合成面板。
        styles: 合成风格表。
        horizon: 标签周期。
    """
    spec = factors.get("MOM_20")
    pool = resolve_universe("all_a")
    params = EvalParams(
        factor="MOM_20", start=START, end="2024-12-31", horizon=horizon, quantiles=5
    )
    context = EvalContext(
        params=params, spec=spec, universe=pool, panel=panel,
        summary=summarize_panel(panel, pool), panel_elapsed_ms=0.0,
    )
    context._cache["styles"] = styles
    return context


def hand_standardize(frame: pd.DataFrame, column: str) -> pd.Series:
    """pandas 手算：按交易日做 3 倍 MAD 裁剪 + z-score。

    参数:
        frame: 含 ``trade_date`` 与待处理列的数据表。
        column: 列名。
    """
    def clip(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.size < 3:
            return series
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if mad <= 0:
            return series
        limit = 3 * MAD_SCALE * mad
        return series.clip(lower=median - limit, upper=median + limit)

    def standardize(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.size < 2:
            return series * np.nan
        std = float(valid.std(ddof=1))
        if std <= 0:
            return series * np.nan
        return (series - float(valid.mean())) / std

    clipped = frame.groupby("trade_date")[column].transform(clip)
    work = frame.assign(_clipped=clipped)
    return work.groupby("trade_date")["_clipped"].transform(standardize)


def hand_inputs(panel: pd.DataFrame, styles: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """按 pandas 手算拼出回归样本、标准化风格与被解释变量。

    参数:
        panel: 合成面板。
        styles: 合成风格表。
    """
    sample = panel.merge(styles, on=["stock_code", "trade_date"], how="left").reset_index(
        drop=True
    )
    z = {column: hand_standardize(sample, column) for column in RAW_COLUMNS}
    bp = z["BP"].to_numpy(dtype="float64")
    ep = z["EP"].to_numpy(dtype="float64")
    value = np.where(
        np.isfinite(bp) & np.isfinite(ep), (bp + ep) / 2,
        np.where(np.isfinite(bp), bp, ep),
    )
    design = pd.DataFrame({
        "SIZE": z["SIZE"].to_numpy(dtype="float64"),
        "VALUE": value,
        "MOMENTUM": z["MOMENTUM"].to_numpy(dtype="float64"),
        "SHORT_REV": z["SHORT_REV"].to_numpy(dtype="float64"),
        "VOL": z["VOL"].to_numpy(dtype="float64"),
        "LIQUIDITY": z["LIQUIDITY"].to_numpy(dtype="float64"),
        "BETA_60": z["BETA_60"].to_numpy(dtype="float64"),
    }).fillna(0.0)
    response = hand_standardize(sample, "factor").to_numpy(dtype="float64")
    return sample, design, response


def hand_daily_ols(sample: pd.DataFrame, design: pd.DataFrame, response: np.ndarray
                   ) -> dict:
    """逐日 OLS（正规方程）手算风格系数、t 值与残差。

    参数:
        sample: 回归样本（含 trade_date）。
        design: 标准化后的 7 列风格。
        response: 标准化后的因子。
    """
    matrix = np.column_stack(
        [np.ones(len(sample))] + [design[key].to_numpy() for key in STYLE_KEYS]
    )
    residual = np.full(len(sample), np.nan, dtype="float64")
    coefficients, t_stats, intercepts, used = [], [], [], []
    for day, index in sample.groupby("trade_date").groups.items():
        positions = np.asarray(index)
        if positions.size < MIN_ROWS:
            continue
        design_day = matrix[positions]
        response_day = response[positions]
        # 整日缺失的风格不可估计，其余风格仍用独立正规方程验证。
        active = np.ptp(design_day, axis=0) > 1e-12
        active[0] = True
        reduced = design_day[:, active]
        if np.linalg.matrix_rank(reduced) < reduced.shape[1]:
            continue
        beta_active = np.linalg.solve(reduced.T @ reduced, reduced.T @ response_day)
        error = response_day - reduced @ beta_active
        sigma2 = float(error @ error) / (positions.size - reduced.shape[1])
        se_active = np.sqrt(np.diag(sigma2 * np.linalg.inv(reduced.T @ reduced)))
        beta = np.full(design_day.shape[1], np.nan)
        standard_error = np.full(design_day.shape[1], np.nan)
        beta[active] = beta_active
        standard_error[active] = se_active
        residual[positions] = error
        # 设计矩阵第 0 列是截距，对外只报风格系数（与实现同口径）
        coefficients.append(beta[1:])
        t_stats.append((beta / standard_error)[1:])
        intercepts.append(beta[0])
        used.append(day)
    return {
        "dates": used,
        "coefficients": np.vstack(coefficients),
        "t_stats": np.vstack(t_stats),
        "intercept": np.asarray(intercepts, dtype="float64"),
        "residual": residual,
    }


def test_zscore_by_day_is_standard_and_keeps_missing():
    """截面标准化：每日均值 0、标准差 1，空值原样保留。"""
    frame = pd.DataFrame({
        "day": np.repeat(pd.bdate_range(START, periods=3), 4),
        "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, np.nan, 40.0, 5.0, 5.0, 5.0, 5.0],
    })
    result = zscore_by_day(frame["day"], frame["value"])
    work = frame.assign(z=result)
    for _, group in work.groupby("day"):
        valid = group["z"].dropna()
        if valid.size > 1:
            assert valid.mean() == pytest.approx(0.0, abs=1e-12)
            assert valid.std(ddof=1) == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(work["z"].iloc[6])
    # 取值全相同的交易日标准差为 0，整日置空
    assert work.loc[work["day"] == frame["day"].unique()[2], "z"].isna().all()


@requires_native
def test_exposure_payload_matches_between_python_and_native(monkeypatch):
    """风格暴露经融合标准化核后与 Python 回退 payload 一致。"""

    import quant_native  # noqa: F401

    panel, styles = make_data()
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = exposure_metric.compute_exposure(make_context(panel, styles), {})
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual = exposure_metric.compute_exposure(make_context(panel, styles), {})

    compare_golden(
        json_safe(expected),
        json_safe(actual),
        abs_tol=1e-10,
        rel_tol=1e-10,
    )


def test_exposure_matches_hand_daily_ols():
    """暴露系数、t 值与逐日系数序列与手算逐日 OLS 完全一致。"""
    panel, styles = make_data()
    context = make_context(panel, styles)
    payload = exposure_metric.compute_exposure(context, {})

    sample, design, response = hand_inputs(panel, styles)
    expected = hand_daily_ols(sample, design, response)

    assert payload["insufficient"] is False
    assert payload["days"] == len(expected["dates"])
    assert payload["industry"]["enabled"] is False
    assert payload["industry"]["columns"] == 0
    for index, key in enumerate(STYLE_KEYS):
        mine = np.asarray(payload["series"]["coefficient"][key], dtype="float64")
        assert np.allclose(mine, expected["coefficients"][:, index], atol=1e-9)
        daily_t = expected["t_stats"][:, index]
        daily_coefficient = expected["coefficients"][:, index]
        stats_row = next(item for item in payload["stats"] if item["key"] == key)
        assert stats_row["mean"] == pytest.approx(daily_coefficient.mean(), abs=1e-9)
        # Fama-MacBeth t：日度系数序列的均值 / 标准误
        assert stats_row["t"] == pytest.approx(
            daily_coefficient.mean() / daily_coefficient.std(ddof=1)
            * np.sqrt(daily_coefficient.size),
            abs=1e-9,
        )
        assert stats_row["significant_share"] == pytest.approx(
            (np.abs(daily_t) > 1.96).mean(), abs=1e-12
        )
        assert stats_row["positive_share"] == pytest.approx((daily_t > 0).mean(), abs=1e-12)
        assert stats_row["days"] == daily_t.size
    # 因子由 SIZE 与 MOMENTUM 合成，这两个风格上的暴露应显著大于其它风格
    by_style = {item["key"]: item["mean"] for item in payload["stats"]}
    assert by_style["SIZE"] > 0.3 and by_style["MOMENTUM"] > 0.2
    assert abs(by_style["LIQUIDITY"]) < 0.15
    assert json.dumps(json_safe(payload))


def test_missing_style_is_treated_as_neutral():
    """风格缺失标准化后置 0：样本不被剔除，且与手算「缺失记 0」一致。"""
    panel, styles = make_data(seed=9)
    missing = styles.index[: len(styles) // 5]
    styles.loc[missing, "VOL"] = np.nan
    styles.loc[missing, "BP"] = np.nan
    context = make_context(panel, styles)
    payload = exposure_metric.compute_exposure(context, {})

    sample, design, response = hand_inputs(panel, styles)
    assert design["VOL"].to_numpy()[missing] == pytest.approx(0.0)
    assert design["VALUE"].notna().all()
    expected = hand_daily_ols(sample, design, response)
    key_index = STYLE_KEYS.index("VOL")
    assert payload["days"] == len(expected["dates"])
    assert np.allclose(
        payload["series"]["coefficient"]["VOL"], expected["coefficients"][:, key_index],
        atol=1e-9, equal_nan=True,
    )


def test_incremental_ic_matches_hand_residual():
    """残差 RankIC 与手算残差的 scipy Spearman 一致，且低于原始 IC。"""
    panel, styles = make_data(seed=13)
    # 标签主要由 SIZE 驱动：因子里的 SIZE 暴露被剔掉后，剩下的是动量信号
    panel["label"] = (
        0.8 * styles["SIZE"].to_numpy() + 0.15 * styles["MOMENTUM"].to_numpy()
    )
    context = make_context(panel, styles)
    payload = exposure_metric.compute_incremental_ic(context, {})

    sample, design, response = hand_inputs(panel, styles)
    expected = hand_daily_ols(sample, design, response)
    hand = sample.assign(residual=expected["residual"]).dropna(subset=["residual", "label"])
    per_day = [
        stats.spearmanr(group["residual"], group["label"]).statistic
        for _, group in hand.groupby("trade_date")
        if len(group) >= MIN_CROSS_SECTION
    ]
    assert payload["residual"]["mean"] == pytest.approx(np.mean(per_day), abs=1e-9)
    assert payload["residual"]["days"] == len(per_day)
    assert payload["raw"]["mean"] > payload["residual"]["mean"]
    assert payload["delta"]["mean"] == pytest.approx(
        payload["raw"]["mean"] - payload["residual"]["mean"]
    )
    assert payload["survivor_share"] < 1.0
    assert json.dumps(json_safe(payload))


class FakeIndustry:
    """测试用的假行业源：把股票按 3 个行业平均分。"""

    key = "fake"
    label = "假行业"
    note = "测试用行业源"

    def load(self, conn) -> pd.DataFrame:
        """按股票代码生成行业映射。

        参数:
            conn: 未使用（保持协议签名）。
        """
        codes = sorted({f"SH600{i:03d}" for i in range(STOCKS)})
        return pd.DataFrame({
            "stock_code": codes,
            "industry": [f"行业{i % 3}" for i in range(len(codes))],
        })


def test_industry_switch_adds_dummies(monkeypatch):
    """行业层默认关闭；打开后设计矩阵加上行业哑变量（丢第一类）。"""
    panel, styles = make_data()
    context = make_context(panel, styles)
    off = exposure_metric.compute_exposure(context, {"industry": "off"})
    assert off["industry"]["enabled"] is False
    assert off["industry"]["columns"] == 0
    assert off["industry"]["chosen"] == "off"

    monkeypatch.setitem(INDUSTRY_SOURCES, FakeIndustry.key, FakeIndustry())
    on = exposure_metric.compute_exposure(context, {"industry": FakeIndustry.key})
    assert on["industry"]["enabled"] is True
    assert on["industry"]["columns"] == 2          # 3 个行业丢 1 类
    assert on["industry"]["coverage"] == pytest.approx(1.0)
    assert on["industry"]["categories"] == 3
    assert on["industry"]["note"] == FakeIndustry.note
    # 系数仍然逐日给出，且与关闭行业层时的取值不同
    assert len(on["series"]["dates"]) == len(off["series"]["dates"])
    assert not np.allclose(
        on["series"]["coefficient"]["SIZE"], off["series"]["coefficient"]["SIZE"]
    )
    # 缓存按行业开关分开存，两次调用互不影响
    again = exposure_metric.compute_exposure(context, {"industry": "off"})
    assert np.allclose(
        again["series"]["coefficient"]["SIZE"], off["series"]["coefficient"]["SIZE"]
    )


def test_unknown_industry_source_raises():
    """未知行业源报 EVAL-303 并列出可选项。"""
    from calculate.styles import resolve_industry_source

    with pytest.raises(IndustryError) as error:
        resolve_industry_source("not_exist")
    assert error.value.code == "EVAL-303"
    assert "可选" in error.value.message


def test_insufficient_days_and_empty_pool():
    """有效日不足只给标记；池内没有样本时直接报错。"""
    panel, styles = make_data(days=10)
    context = make_context(panel, styles)
    payload = exposure_metric.compute_exposure(context, {})
    assert payload["insufficient"] is True
    assert payload["stats"] == []
    assert payload["hint"] and "少于 20 天" in payload["hint"]
    incremental = exposure_metric.compute_incremental_ic(context, {})
    assert incremental["insufficient"] is True
    assert incremental["residual"]["days"] == 10

    empty = panel.copy()
    empty["in_universe"] = False
    with pytest.raises(DecomposeError) as error:
        exposure_metric.compute_exposure(make_context(empty, styles), {})
    assert error.value.code == "EVAL-301"
