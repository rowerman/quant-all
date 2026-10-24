"""评估面板与股票池的单元测试（需要真实主库与因子产物）。"""

import numpy as np
import pandas as pd
import pytest

import quant
from calculate.panel import (
    PANEL_COLUMNS,
    build_panel,
    default_params,
    open_source,
    panel_columns,
    resolve_factor,
)
from calculate.params import DECAY_HORIZONS, EvalParams
from calculate.universe import (
    UniverseError,
    list_universes,
    membership_cte,
    resolve_universe,
)
from scripts.build_factors import validate
from calculate.config import DB_FILE
from calculate.errors import EvalFactorError, EvalPanelError, EvalParamError

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb")

START, END = "2024-05-06", "2024-05-10"


def window_params(**overrides) -> EvalParams:
    """构造一个短窗口的评估参数。

    参数:
        overrides: 需要覆盖的字段。
    """
    base = {"factor": "MOM_20", "start": START, "end": END}
    base.update(overrides)
    return EvalParams(**base)


def test_panel_columns_keys_and_dates(momentum_dir):
    """面板列固定、主键唯一、交易日落在区间内。"""
    params = window_params()
    context = build_panel(params, factor_dir=momentum_dir)
    panel = context.panel
    assert list(panel.columns) == list(panel_columns(params))
    # 固定列必须仍是前缀，多周期标签列只往后加
    assert list(panel.columns[: len(PANEL_COLUMNS)]) == list(PANEL_COLUMNS)
    assert not panel.duplicated(["stock_code", "trade_date"]).any()
    assert context.summary["days"] == 5
    assert panel["trade_date"].min() >= pd.Timestamp(START)
    assert panel["trade_date"].max() <= pd.Timestamp(END)
    assert panel["stock_code"].dtype.name == "category"


def test_panel_multi_horizon_labels_match_label_table(momentum_dir):
    """面板多周期标签列与 md.fwd_return 的取值逐行一致（V11-3）。"""
    params = window_params()
    context = build_panel(params, factor_dir=momentum_dir)
    panel = context.panel
    assert params.horizon == 5
    assert context.label_columns == ("label", "label_1d", "label_h3", "label_h10", "label_h20")
    assert context.horizon_label(5) == "label"
    assert context.horizon_label(1) == "label_1d"
    assert context.horizon_label(20) == "label_h20"

    horizons = [horizon for horizon in DECAY_HORIZONS]
    labels = quant.load_label(start=START, end=END, horizons=horizons, price="close")
    left = panel[["stock_code", "trade_date"] + list(context.label_columns[2:])].astype(
        {"stock_code": "str"}
    )
    right = labels.rename(
        columns={f"fwd_ret_close_{horizon}": f"label_h{horizon}" for horizon in horizons}
    )
    right = right[["stock_code", "trade_date", "label_h3", "label_h10", "label_h20"]]
    joined = left.merge(right, on=["stock_code", "trade_date"], suffixes=("_panel", "_src"))
    assert len(joined) == len(left)
    for column in ("label_h3", "label_h10", "label_h20"):
        assert np.allclose(
            joined[f"{column}_panel"], joined[f"{column}_src"], atol=1e-12, equal_nan=True
        )


def test_panel_label_columns_follow_horizon(momentum_dir):
    """换成 20 日档时，面板只带 3/5/10 三档，20 日档复用 label 列。"""
    params = window_params(horizon=20)
    context = build_panel(params, factor_dir=momentum_dir)
    assert context.label_columns == ("label", "label_1d", "label_h3", "label_h5", "label_h10")
    assert context.horizon_label(20) == "label"


def test_panel_rowset_and_values_match_quant_readers(momentum_dir):
    """面板的行集与取值要和 quant 读取层拼出来的完全一致。"""
    context = build_panel(window_params(), factor_dir=momentum_dir)
    factor_frame = quant.load_factor_panel(
        start=START, end=END, names=["MOM_20"], factor_dir=momentum_dir
    )
    label_frame = quant.load_label(start=START, end=END, horizons=[5], price="close")
    merged = factor_frame.merge(label_frame, on=["stock_code", "trade_date"], how="inner")

    panel_keys = set(zip(context.panel["stock_code"].astype(str), context.panel["trade_date"]))
    merged_keys = set(zip(merged["stock_code"], merged["trade_date"]))
    assert panel_keys == merged_keys

    left = context.panel[["stock_code", "trade_date", "factor", "label"]].astype(
        {"stock_code": "str"}
    )
    right = merged.rename(columns={"MOM_20": "factor", "fwd_ret_close_5": "label"})
    joined = left.merge(right, on=["stock_code", "trade_date"], suffixes=("_panel", "_src"))
    assert len(joined) == len(left) == len(right)
    assert np.allclose(joined["factor_panel"], joined["factor_src"], atol=1e-12, equal_nan=True)
    assert np.allclose(joined["label_panel"], joined["label_src"], atol=1e-12, equal_nan=True)


def test_label_present_inside_window(momentum_dir):
    """标签来自全历史口径，区间中间的日子不会因为切块而缺标签。"""
    context = build_panel(window_params(), factor_dir=momentum_dir)
    # 停牌行的未来价为空，标签因此在个别行上缺失，但比例必须很小
    assert context.panel["label"].notna().mean() > 0.98


def test_all_a_pool_matches_validate_filter(momentum_dir):
    """全 A 池的样本数与 validate.py 的 POOL_FILTER 口径一致。"""
    context = build_panel(window_params(), factor_dir=momentum_dir)
    conn = open_source()
    try:
        expected = conn.execute(
            "SELECT count(*) FROM src.md.daily_hfq b"
            " JOIN src.md.fwd_return l"
            " ON l.stock_code = b.stock_code AND l.trade_date = b.trade_date"
            f" WHERE b.trade_date BETWEEN DATE '{START}' AND DATE '{END}'"
            f" AND {validate.POOL_FILTER}"
        ).fetchone()[0]
    finally:
        conn.close()
    assert context.summary["universe_rows"] == int(expected)


def test_index_pool_is_subset_of_month_end_snapshot(momentum_dir):
    """指数成分池用的是上一个已结束月份的权重快照，且叠加了可交易性过滤。"""
    context = build_panel(
        window_params(universe="index:000300.SH"), factor_dir=momentum_dir
    )
    conn = open_source()
    try:
        members = {
            row[0] for row in conn.execute(
                "SELECT stock_code FROM src.ref.index_weight"
                " WHERE index_code = '000300.SH' AND trade_date = DATE '2024-04-30'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert members, "2024-04-30 的沪深300 快照为空，测试前提不成立"
    pool = context.panel.loc[context.panel["in_universe"]]
    assert set(pool["stock_code"].astype(str)) <= members
    daily = pool["trade_date"].value_counts()
    assert daily.min() >= 250, "成分池单日样本太少，可交易性过滤可能写错"


def test_universe_registry_and_errors():
    """股票池清单与非法标识的报错。"""
    universes = list_universes()
    assert universes[0].key == "all_a"
    assert len(universes) == 13
    assert len({spec.key for spec in universes}) == len(universes)
    try:
        resolve_universe("index:999999.SH")
    except UniverseError as error:
        assert error.code == "EVAL-103"
    else:  # pragma: no cover - 走到这里说明校验失效
        raise AssertionError("非法股票池没有报错")
    with pytest.raises(UniverseError):
        membership_cte(resolve_universe("all_a"), START, END)


def test_unknown_factor_raises(momentum_dir):
    """未注册的因子名与未落盘的因子都要给出可执行的提示。"""
    with pytest.raises(EvalFactorError) as error:
        resolve_factor("NOT_A_FACTOR", momentum_dir)
    assert error.value.code == "EVAL-104"
    with pytest.raises(EvalFactorError) as built_error:
        resolve_factor("EP_TTM", momentum_dir)
    assert built_error.value.code == "EVAL-104"
    assert "build_factors" in built_error.value.message


def test_empty_window_raises(momentum_dir):
    """区间里没有交易日时直接报错。"""
    with pytest.raises(EvalParamError) as error:
        build_panel(window_params(start="2024-05-04", end="2024-05-05"), factor_dir=momentum_dir)
    assert error.value.code == "EVAL-101"
    assert "没有交易日" in error.value.message


def test_too_long_window_raises():
    """超过交易日上限的区间报 EVAL-101。"""
    with pytest.raises(EvalParamError) as error:
        build_panel(window_params(start="2000-01-04", end="2026-09-28"))
    assert error.value.code == "EVAL-101"


def test_too_many_rows_raises():
    """交易日没超上限但面板过大的区间报 EVAL-102。"""
    with pytest.raises(EvalPanelError) as error:
        build_panel(window_params(start="2017-01-04", end="2026-09-28"))
    assert error.value.code == "EVAL-102"
    assert "上限" in error.value.message


def test_default_params_window():
    """默认参数是「最近 3 年 + 全A池 + close 5 日」。"""
    params = default_params("MOM_20")
    conn = open_source()
    try:
        latest = conn.execute("SELECT max(trade_date) FROM src.md.calendar").fetchone()[0]
    finally:
        conn.close()
    assert params.end == latest.strftime("%Y-%m-%d")
    assert params.start == (pd.Timestamp(latest) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    assert (params.universe, params.horizon, params.price, params.quantiles) == (
        "all_a", 5, "close", 10
    )
