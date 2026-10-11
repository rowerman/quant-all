"""分位分层：分箱、分位收益、分位换手与因子自相关。

几个口径决定：

1. 分箱只看**因子值**：每个交易日把池内非空因子值按平均名次均分成 Q 组，
   组号 1 = 因子值最小；并列值落在同一组（用平均名次算组号，结果可复现，
   不依赖面板行序，也不会把并列值随机拆到两组去）。
2. 缺标签的行照常参与分箱（分箱是因子自身的属性），只是不算进组收益。
3. 市值加权只在 `total_mv` 非空且大于 0 的样本上算，缺市值的行不进加权口径。
4. 换手 = 「上一交易日在池内、今天换到别的组（或昨天不在本组）」的股票数
   占「今天在池内且在上一交易日也有值」的股票数的比例，与 Alphalens 的分位换手同义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate.daily import MIN_CROSS_SECTION

# 因子自相关的滞后阶数（交易日）
AUTOCORR_LAGS: tuple[int, ...] = (1, 5, 20)


def assign_bins(panel: pd.DataFrame, quantiles: int, *,
                label_columns=("label", "label_1d")) -> pd.DataFrame:
    """给池内非空因子值分配分位组号。

    参数:
        panel: 评估面板。
        quantiles: 分组数。
        label_columns: 一并带进分位表的标签列，默认所选周期与 1 日档；
            多周期衰减要把几档标签都带出来（列名见 ``EvalParams.horizon_label_column``）。

    返回:
        列含 ``trade_date``、``stock_code``、``factor``、``label_columns``、
        ``total_mv``、``bin`` 的数据表；池外或因子为空的行不出现。
    """
    keep = [column for column in label_columns if column in panel.columns]
    subset = panel.loc[
        panel["in_universe"] & panel["factor"].notna(),
        ["trade_date", "stock_code", "factor", *keep, "total_mv"],
    ].copy()
    if subset.empty:
        subset["bin"] = pd.Series(dtype="int16")
        return subset
    grouped = subset.groupby("trade_date", sort=True)["factor"]
    average_rank = grouped.rank(method="average").to_numpy(dtype="float64")
    size = grouped.transform("size").to_numpy(dtype="float64")
    position = np.ceil(average_rank / size * quantiles)
    subset["bin"] = np.clip(position, 1, quantiles).astype("int16")
    return subset.reset_index(drop=True)


def bin_daily_returns(frame: pd.DataFrame, quantiles: int, *,
                      label_column: str = "label",
                      weight_column: str | None = None) -> pd.DataFrame:
    """算每个交易日的分位组合收益。

    参数:
        frame: ``assign_bins`` 的输出。
        quantiles: 分组数。
        label_column: 用哪一列算收益，默认所选周期的标签。
        weight_column: 加权列；为空表示等权（组内取均值）。

    返回:
        以交易日为索引、列为 1~Q 的收益表，当天没有有效样本的组为空值。
    """
    work = frame.loc[frame[label_column].notna()]
    if weight_column:
        work = work.loc[work[weight_column].notna() & (work[weight_column] > 0)]
        scaled = work.assign(_weighted=work[label_column] * work[weight_column])
        grouped = scaled.groupby(["trade_date", "bin"])
        daily = grouped["_weighted"].sum() / grouped[weight_column].sum()
    else:
        daily = work.groupby(["trade_date", "bin"])[label_column].mean()
    return daily.unstack("bin").reindex(columns=range(1, quantiles + 1))


def wide_matrix(frame: pd.DataFrame, values) -> pd.DataFrame:
    """把长表摊成「交易日 × 股票」宽表，缺值表示当天该股票不在池内。

    用 factorize 加 numpy 直接填充，而不是 pandas 的 pivot：几百万行的透视
    实测要 1.5 秒一次，这样写只要零点几秒。

    参数:
        frame: ``assign_bins`` 的输出。
        values: 与 frame 等长的取值序列。
    """
    day_codes, days = pd.factorize(frame["trade_date"], sort=True)
    stock_codes, stocks = pd.factorize(frame["stock_code"], sort=True)
    matrix = np.full((days.size, stocks.size), np.nan, dtype="float32")
    matrix[day_codes, stock_codes] = np.asarray(values, dtype="float32")
    return pd.DataFrame(
        matrix,
        index=pd.Index(days, name="trade_date"),
        columns=pd.Index(stocks, name="stock_code"),
    )


def bin_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """把分位表摊成「交易日 × 股票」的组号宽表。

    参数:
        frame: ``assign_bins`` 的输出。
    """
    return wide_matrix(frame, frame["bin"].to_numpy())


def rank_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """把分位表摊成「交易日 × 股票」的截面秩宽表。

    参数:
        frame: ``assign_bins`` 的输出。
    """
    ranks = frame.groupby("trade_date", sort=False)["factor"].rank(pct=True).to_numpy()
    return wide_matrix(frame, ranks)


def bin_turnover(matrix: pd.DataFrame,
                 quantiles: int) -> tuple[pd.DataFrame, pd.Series]:
    """算每个分位、每个交易日的换手。

    参数:
        matrix: ``bin_matrix`` 产出的组号宽表。
        quantiles: 分组数。

    返回:
        ``(每个分位的逐日换手表, 全池逐日换手序列)``；
        分母是「今天在池内且上一交易日也有值」的股票数，首日为空。
    """
    previous = matrix.shift(1)
    base = matrix.notna() & previous.notna()
    overall = (
        (base & (matrix != previous)).sum(axis=1) / base.sum(axis=1).replace(0, np.nan)
    )
    columns: dict[int, pd.Series] = {}
    for bin_id in range(1, quantiles + 1):
        member = matrix == bin_id
        stayed_base = base & member
        changed = stayed_base & (previous != bin_id)
        columns[bin_id] = changed.sum(axis=1) / stayed_base.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(columns), overall


def row_correlation(current: pd.DataFrame, previous: pd.DataFrame) -> pd.Series:
    """逐行（逐交易日）算两个宽表的 Pearson 相关，缺失按配对剔除。

    参数:
        current: 当期取值（交易日 × 股票）。
        previous: 上一期取值，行索引与 current 对齐。
    """
    # 两侧的行索引标签不同（当期是 t、上期是 t-lag），必须按位置做逐元素运算，
    # 交给 pandas 会按标签对齐成并集，凭空多出行来
    left = current.to_numpy(dtype="float64")
    right = previous.to_numpy(dtype="float64")
    mask = np.isfinite(left) & np.isfinite(right)
    count = mask.sum(axis=1)
    safe_count = np.where(count > 0, count, np.nan)
    left = np.where(mask, left, np.nan)
    right = np.where(mask, right, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        left_dev = left - np.nanmean(left, axis=1, keepdims=True)
        right_dev = right - np.nanmean(right, axis=1, keepdims=True)
        covariance = np.nansum(left_dev * right_dev, axis=1) / safe_count
        scale = np.sqrt(
            np.nansum(left_dev ** 2, axis=1) / safe_count
            * np.nansum(right_dev ** 2, axis=1) / safe_count
        )
        corr = np.where((scale > 0) & (count >= MIN_CROSS_SECTION), covariance / scale, np.nan)
    return pd.Series(corr, index=current.index)


def factor_autocorrelation(ranks: pd.DataFrame,
                           lags: tuple[int, ...] = AUTOCORR_LAGS) -> pd.DataFrame:
    """算因子自相关：当日与 lag 个交易日前在同一批股票上的截面秩相关。

    参数:
        ranks: ``rank_matrix`` 产出的秩宽表。
        lags: 滞后阶数（交易日）。

    返回:
        以交易日为索引、每阶一列的相关系数表。
    """
    series = {
        f"lag{lag}": row_correlation(ranks.iloc[lag:], ranks.iloc[:-lag])
        for lag in lags if len(ranks) > lag
    }
    return pd.DataFrame(series, index=ranks.index)
