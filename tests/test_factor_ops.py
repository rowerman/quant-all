"""因子算子的单元测试：内存 DuckDB 小面板与 pandas 对照。

这里覆盖的是「算子语义」这一层：窗口从哪一天开始、空值怎么处理、
截面算子按什么分组。真实数据的端到端验证在 test_factor_build.py。
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from factors import ops

WINDOW = "(PARTITION BY stock_code ORDER BY trade_date)"
CODES = ("SH600000", "SZ000001")


def _close(code_index: int, index: int) -> float:
    """构造非单调、无重复的价格序列。"""
    return 10.0 + 2.0 * code_index + 0.3 * ((index * 7 + code_index * 3) % 11)


def _panel() -> pd.DataFrame:
    """两只股票、30 个交易日的确定性面板。"""
    rows = []
    for code_index, code in enumerate(CODES):
        for index in range(30):
            close = _close(code_index, index)
            rows.append({
                "stock_code": code,
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "close": close,
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "ret": np.nan if index == 0 else close / _close(code_index, index - 1) - 1,
                "volume_share": 1000.0 + 37.0 * ((index * 5 + code_index) % 13),
            })
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].dt.date
    return frame.set_index(["stock_code", "trade_date"]).sort_index()


@pytest.fixture()
def conn():
    """内存连接，注册好测试面板。"""
    connection = duckdb.connect()
    # 面板本身是 MultiIndex，注册给 DuckDB 时要摊平成列
    connection.register("panel", _panel().reset_index())
    yield connection
    connection.close()


def _evaluate(conn, expr: str) -> pd.Series:
    """执行算子表达式，返回以键为索引的序列。"""
    frame = conn.execute(
        f"SELECT stock_code, trade_date, {expr} AS v FROM panel "
        f"WINDOW w AS {WINDOW} ORDER BY stock_code, trade_date"
    ).df()
    return frame.set_index(["stock_code", "trade_date"])["v"]


def _align(series: pd.Series) -> pd.Series:
    """把索引第二层统一成 pandas Timestamp。

    DuckDB 与 pandas 对日期分别给出 datetime64[us] 与 datetime.date，
    直接比对会因为索引类型不同而失败，这里统一到 datetime64[ns]。
    """
    index = pd.MultiIndex.from_arrays(
        [
            series.index.get_level_values(0),
            pd.to_datetime(series.index.get_level_values(1)).as_unit("ns"),
        ],
        names=["stock_code", "trade_date"],
    )
    return series.set_axis(index)


def _check(conn, expr: str, expected: pd.Series, drop_first: int = 0) -> None:
    """把 SQL 结果与 pandas 结果逐行比对。

    参数:
        conn: 内存连接。
        expr: 待测算子表达式。
        expected: pandas 侧结果，索引为 (stock_code, trade_date)。
        drop_first: 每只股票跳过前几行（窗口未满时两边语义不同）。
    """
    panel = _panel()
    got = _align(_evaluate(conn, expr)).sort_index()
    want = _align(expected.astype("float64")).reindex(_align_index(panel.index)).sort_index()
    if drop_first:
        mask = panel.groupby(level=0).cumcount() >= drop_first
        got, want = got[mask.values], want[mask.values]
    pd.testing.assert_series_equal(got, want, check_names=False, rtol=1e-9, atol=1e-12)


def _align_index(index: pd.MultiIndex) -> pd.MultiIndex:
    """把面板索引的第二层统一成 datetime64[ns]。"""
    return pd.MultiIndex.from_arrays(
        [index.get_level_values(0), pd.to_datetime(index.get_level_values(1)).as_unit("ns")],
        names=["stock_code", "trade_date"],
    )


def _rolling(panel: pd.DataFrame, col: str, n: int, how: str) -> pd.Series:
    """按股票分组做滚动统计。"""
    parts = []
    for _, group in panel.groupby("stock_code", sort=False):
        parts.append(getattr(group[col].rolling(n, min_periods=1), how)())
    return pd.concat(parts)


def _rolling_corr(panel: pd.DataFrame, a: str, b: str, n: int) -> pd.Series:
    """按股票分组做滚动相关。"""
    parts = []
    for _, group in panel.groupby("stock_code", sort=False):
        parts.append(group[a].rolling(n, min_periods=2).corr(group[b]))
    return pd.concat(parts)


def test_frame_rejects_non_positive_window():
    """窗口长度必须为正，否则早点报错而不是生成坏 SQL。"""
    with pytest.raises(ValueError):
        ops.frame(0)


def test_ts_delay_and_delta(conn):
    """滞后与差分与 pandas 的 shift 一致。"""
    panel = _panel()
    shifted = panel["close"].groupby(level=0).shift(1)
    _check(conn, ops.ts_delay("close", 1), shifted)
    _check(conn, ops.ts_delta("close", 1), panel["close"] - shifted, drop_first=1)


def test_ts_mean_and_sum(conn):
    """均值与求和只覆盖窗口内非空值。"""
    panel = _panel()
    _check(conn, ops.ts_mean("close", 5), _rolling(panel, "close", 5, "mean"))
    _check(conn, ops.ts_sum("close", 5), _rolling(panel, "close", 5, "sum"))


def test_ts_max_and_min(conn):
    """极值算子与 pandas 的滚动极值一致。"""
    panel = _panel()
    _check(conn, ops.ts_max("close", 5), _rolling(panel, "close", 5, "max"))
    _check(conn, ops.ts_min("close", 5), _rolling(panel, "close", 5, "min"))


def test_ts_std_skips_nulls(conn):
    """样本标准差与 pandas rolling(std) 一致，首行空值不参与。"""
    panel = _panel()
    expected = pd.concat([
        group["ret"].rolling(5, min_periods=2).std()
        for _, group in panel.groupby("stock_code", sort=False)
    ])
    _check(conn, ops.ts_std("ret", 5), expected)


def test_ts_corr(conn):
    """滚动相关与 pandas 一致。"""
    panel = _panel()
    _check(conn, ops.ts_corr("ret", "volume_share", 5), _rolling_corr(panel, "ret", "volume_share", 5))


def test_ts_rank_counts_previous_values(conn):
    """ts_rank 数的是最近 n-1 个交易日里「不超过当前值」的占比。"""
    panel = _panel()
    parts = []
    for _, group in panel.groupby("stock_code", sort=False):
        series = group["close"]
        values = []
        for index in range(len(series)):
            previous = series.iloc[max(0, index - 4):index]
            hit = int((previous <= series.iloc[index]).sum())
            values.append(hit / 4.0)
        parts.append(pd.Series(values, index=series.index))
    _check(conn, ops.ts_rank("close", 5), pd.concat(parts))


def test_ts_rank_rejects_short_window():
    """窗口长度不足 2 时无法定义分位，直接报错。"""
    with pytest.raises(ValueError):
        ops.ts_rank("close", 1)


def test_cs_rank_and_zscore(conn):
    """截面算子在每个交易日内部排序与标准化。"""
    panel = _panel()
    by_date = panel.groupby("trade_date")["close"]
    rank = by_date.rank().sub(1).div(by_date.transform("size") - 1)
    _check(conn, ops.cs_rank("close"), rank)

    mean = by_date.transform("mean")
    std = by_date.transform(lambda values: values.std(ddof=0))
    _check(conn, ops.cs_zscore("close"), (panel["close"] - mean) / std)


def test_safe_div_returns_null_on_zero(conn):
    """分母为 0 时返回空值，而不是 inf 或报错。"""
    got = _evaluate(conn, ops.safe_div("close", "0"))
    assert got.isna().all()


def test_multi_layer_arg_max_pattern(conn):
    """多层写法：先用行号层，再用 arg_max 找到极值所在行号。"""
    sql = f"""
    WITH a AS (SELECT *, {ops.ts_row_number()} AS v1 FROM panel WINDOW w AS {WINDOW}),
         b AS (SELECT *, {ops.ts_arg_max('close', 'v1', 5)} AS v2 FROM a WINDOW w AS {WINDOW})
    SELECT stock_code, trade_date, v2 FROM b ORDER BY stock_code, trade_date
    """
    got = _align(conn.execute(sql).df().set_index(["stock_code", "trade_date"])["v2"])

    panel = _panel()
    parts = []
    for _, group in panel.groupby("stock_code", sort=False):
        values = []
        for index in range(len(group)):
            start = max(0, index - 4)
            window = group.iloc[start:index + 1]
            values.append(float(start + window["close"].values.argmax() + 1))
        parts.append(pd.Series(values, index=group.index))
    want = _align(pd.concat(parts)).reindex(_align_index(panel.index)).sort_index()
    got = got.sort_index()
    mask = panel.groupby(level=0).cumcount() >= 4
    pd.testing.assert_series_equal(
        got[mask.values].astype("float64"), want[mask.values].astype("float64"),
        check_names=False, rtol=1e-9,
    )
