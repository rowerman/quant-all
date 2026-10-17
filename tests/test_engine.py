"""VectorBT 执行器的集成测试：取整、规则拒单、费用与持仓口径。"""

import numpy as np
import pandas as pd
import pytest

from backtest.data import load_market_data
from backtest.engine import _cash_event_signature, _same_masked_matrix, run_backtest
from backtest.errors import DataAlignmentError, ExecutionError
from backtest.specs import (
    BacktestParams,
    CostSpec,
    ExecutionSpec,
    PortfolioSpec,
    RebalanceSpec,
    RiskSpec,
    SignalSpec,
    StrategySpec,
)

DATES = pd.date_range("2024-01-29", periods=6, freq="B")
STOCKS = ["SH600000", "SZ000001", "SZ000002"]


def test_fixed_point_compares_original_targets_but_ignores_retry_cells():
    """原始调仓股数必须稳定，补回重试日的派生股数允许随当天预算变化。"""
    first = pd.DataFrame([[1_000.0], [500.0]], index=DATES[:2], columns=["A"])
    second = pd.DataFrame([[1_000.0], [600.0]], index=DATES[:2], columns=["A"])
    assert _same_masked_matrix(first, second, np.array([[True], [False]]))
    assert not _same_masked_matrix(first, second, np.array([[False], [True]]))


def test_cash_event_signature_ignores_retry_shrink_rows():
    """同一补回重试中的现金缩水不会被重复登记成新事件。"""
    allocation = pd.DataFrame(
        {
            "trade_date": [DATES[0], DATES[1]],
            "stock_code": ["A", "A"],
            "cash_limited": [True, False],
            "cash_shrunk_shares": [500.0, 300.0],
        }
    )
    assert _cash_event_signature(allocation) == ((str(DATES[0]), "A"),)


def _frames():
    """构造 6 个交易日、3 只股票、价格恒定的输入面板。"""
    score = pd.DataFrame(
        {
            "SH600000": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "SZ000001": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "SZ000002": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=DATES,
    )
    price = pd.DataFrame(10.0, index=DATES, columns=STOCKS)
    return score, price


def _strategy(**overrides):
    """默认日频调仓、两只股票等权的测试策略。"""
    fields = {
        "signal": SignalSpec(factors=("MOM_20",), weights=(1.0,), standardize=False),
        "portfolio": PortfolioSpec(top_n=2),
        "rebalance": RebalanceSpec(frequency="daily"),
    }
    fields.update(overrides)
    return StrategySpec(**fields)


def test_engine_produces_lot_sized_orders_and_exact_fees(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    stages: list[str] = []
    result = run_backtest(data, _strategy(), params, progress=lambda stage, _: stages.append(stage))

    assert stages == ["signals", "probe", "baseline", "orders", "allocation", "execute"]
    assert not result.orders.empty
    assert (result.orders["Size"] % 100 == 0).all()
    assert result.metadata["calibration"]["fee_matrix_max_diff"] == pytest.approx(0.0)
    assert result.metadata["calibration"]["fallback_fee_orders"] == 0
    # 2% 现金缓冲摊到 2 只等权上是每格 1%，其余偏差来自 100 股取整与盘中价格差
    deviation_p95 = result.metadata["calibration"]["weight_deviation_p95"]
    assert deviation_p95 <= 0.02 / 2 + 0.005
    assert result.metadata["calibration"]["target_cells"] > 0
    assert result.daily.index.equals(DATES)
    assert result.daily["portfolio_value"].iloc[0] == pytest.approx(100_000.0)
    assert {"cost", "turnover", "drawdown", "nav"} <= set(result.daily.columns)
    assert result.metrics["trading_days"] == len(DATES)


def test_engine_applies_single_cap_and_turnover_before_allocation(make_data):
    score = pd.DataFrame(
        {
            "SH600000": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
            "SZ000001": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            "SZ000002": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=DATES,
    )
    price = pd.DataFrame(10.0, index=DATES, columns=STOCKS)
    data = make_data(
        dates=DATES,
        stocks=STOCKS,
        open=price.astype("float32"),
        close=price.astype("float32"),
        factors={"MOM_20": score.astype("float32")},
    )
    strategy = _strategy(
        portfolio=PortfolioSpec(top_n=2),
        risk=RiskSpec(max_single_weight=0.4, max_one_way_turnover=0.25),
    )
    result = run_backtest(
        data, strategy, BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    )
    calibration = result.metadata["calibration"]
    assert calibration["single_cap_max_weight"] <= 0.4 + 1e-9
    assert calibration["turnover_limited_days"] > 0
    assert calibration["turnover_limit_error_max"] == pytest.approx(0.0)
    assert calibration["cash_clipped_cells"] == 0


def test_engine_applies_industry_cap_on_signal_day(make_data):
    score, price = _frames()
    industry = pd.DataFrame(
        [[1, 1, 2]] * len(DATES), index=DATES, columns=STOCKS, dtype="int16"
    )
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"), close=price.astype("float32"),
        factors={"MOM_20": score.astype("float32")}, industry_l1=industry,
        industry_labels={1: "A 行业", 2: "B 行业"},
    )
    strategy = _strategy(
        portfolio=PortfolioSpec(top_n=3),
        risk=RiskSpec(max_industry_weight=0.5),
    )
    result = run_backtest(
        data, strategy, BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    )
    calibration = result.metadata["calibration"]
    assert calibration["industry_cap_target_max_weight"] <= 0.5 + 1e-9
    assert calibration["industry_cap_days"] > 0
    assert calibration["cash_clipped_cells"] == 0


def test_engine_holdings_match_target_weights(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    held = result.holdings.groupby("trade_date")["stock_code"].nunique()
    assert held.max() == 2
    weights = result.holdings.groupby("trade_date")["weight"].sum()
    # 满仓 + 100 股取整 + 费用，实际权重贴近目标但不会超过 1
    assert weights.max() <= 1.0 + 1e-9
    assert weights.max() > 0.95
    last = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert set(last["stock_code"]) == {"SH600000", "SZ000001"}


def test_engine_records_limit_up_rejection(make_data):
    score, price = _frames()
    can_buy = pd.DataFrame(True, index=DATES, columns=STOCKS)
    can_buy.loc[DATES[0], "SH600000"] = False  # 次一交易日开盘涨停
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), can_buy_open_next=can_buy,
        factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    assert "limit_up_or_not_buyable" in set(result.rejections["reason"])
    assert result.metadata["calibration"]["rejected_orders"] == len(result.rejections)


def test_engine_retries_blocked_buy_on_next_trading_day(make_data):
    """月频调仓被涨停挡下的买入，下一个交易日补单，费率估计不退回兜底口径。"""
    score, price = _frames()
    can_buy = pd.DataFrame(True, index=DATES, columns=STOCKS)
    can_buy.loc[DATES[2], "SH600000"] = False  # 1 月 31 日信号对应的 2 月 1 日执行日涨停
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), can_buy_open_next=can_buy,
        factors={"MOM_20": score.astype("float32")},
    )
    strategy = _strategy(rebalance=RebalanceSpec(frequency="monthly"))
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 1_000_000.0)
    result = run_backtest(data, strategy, params)
    calibration = result.metadata["calibration"]

    assert len(result.rejections) == 1
    assert result.rejections.iloc[0]["trade_date"] == DATES[3]
    assert calibration["rejected_orders"] == 1
    # 2 月 2 日没有新目标，补单只能来自每日重试
    filled = result.orders[result.orders["stock_code"] == "SH600000"]
    assert set(filled["trade_date"]) == {DATES[4]}
    assert calibration["retry_attempts"] == 1
    assert calibration["retried_filled_cells"] == 1
    assert calibration["pending_at_end_cells"] == 0
    # 补单那一笔的费率估计来自股数差，不落到兜底口径
    assert calibration["fallback_fee_orders"] == 0
    assert calibration["fee_matrix_max_diff"] == pytest.approx(0.0)


def test_engine_reports_missing_membership(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    with pytest.raises(ExecutionError, match="成分矩阵"):
        run_backtest(data, _strategy(universe="index:000300.SH"), params)


def test_engine_rejects_range_without_trading_days(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    with pytest.raises(DataAlignmentError, match="没有交易日"):
        run_backtest(
            data, _strategy(), BacktestParams("2024-03-01", "2024-03-05", 100_000.0)
        )


def test_minimum_commission_is_charged_per_order(make_data):
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    costs = CostSpec(commission_rate=0.0003, minimum_commission=5.0, stamp_tax_rate=0.0005)
    strategy = _strategy(costs=costs)
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 1_000_000.0)
    result = run_backtest(data, strategy, params)
    fees = result.orders["Fees"].astype(float)
    value = result.orders["Size"].abs() * result.orders["Price"].astype(float)
    sells = result.orders["Side"].astype(str).str.lower().eq("sell")
    expected = np.maximum(value * costs.commission_rate, costs.minimum_commission) + np.where(
        sells, value * costs.stamp_tax_rate, 0.0
    )
    assert np.allclose(fees.to_numpy(), expected, atol=1e-6)


def test_execution_switches_off_keeps_orders(make_data):
    score, price = _frames()
    suspended = pd.DataFrame(False, index=DATES, columns=STOCKS)
    suspended.loc[DATES[1], "SH600000"] = True
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), is_suspended=suspended,
        factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(execution=ExecutionSpec(reject_suspended=False)), params)
    assert result.rejections.empty


def _delist_inputs():
    """SZ000001 在 DATES[2] 退市：之后没有行情、也没有因子值。"""
    score, price = _frames()
    open_panel = price.copy()
    close_panel = (price + 0.5).copy()
    for date in DATES[3:]:
        open_panel.loc[date, "SZ000001"] = np.nan
        close_panel.loc[date, "SZ000001"] = np.nan
    score.loc[DATES[2]:, "SZ000001"] = np.nan
    delist = pd.Series(pd.NaT, index=STOCKS)
    delist["SZ000001"] = DATES[2]
    return score, open_panel, close_panel, delist


def test_engine_liquidates_delisted_holding(make_data):
    """持有中的股票退市后必须被清掉，不能靠 ffill 永远留在净值里。"""
    score, open_panel, close_panel, delist = _delist_inputs()
    data = make_data(
        dates=DATES, stocks=STOCKS,
        open=open_panel.astype("float32"), close=close_panel.astype("float32"),
        factors={"MOM_20": score.astype("float32")}, delist_date=delist,
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)

    assert result.metadata["calibration"]["forced_liquidations"] == 1
    record = result.liquidations.iloc[0]
    assert record["stock_code"] == "SZ000001"
    assert record["trade_date"] == DATES[2]
    assert record["size"] > 0
    assert record["value"] == pytest.approx(record["size"] * record["price"])
    sell = result.orders.loc[
        (result.orders["stock_code"] == "SZ000001") & (result.orders["trade_date"] == DATES[2])
    ]
    assert len(sell) == 1
    assert str(sell.iloc[0]["Side"]).lower() == "sell"
    last = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert "SZ000001" not in set(last["stock_code"])


def test_engine_is_reproducible():
    """同一份输入与同一份策略跑两次，订单、净值、指标与自检逐位一致。"""
    data = load_market_data(
        "2026-06-01", "2026-09-28", ["MOM_20"], universe="index:000300.SH"
    )
    params = BacktestParams("2026-06-01", "2026-09-28", 1_000_000.0)
    first = run_backtest(data, StrategySpec(), params)
    second = run_backtest(data, StrategySpec(), params)

    assert len(first.orders) > 0
    pd.testing.assert_frame_equal(first.orders, second.orders, check_exact=True)
    pd.testing.assert_frame_equal(first.daily, second.daily, check_exact=True)
    pd.testing.assert_frame_equal(first.holdings, second.holdings, check_exact=True)
    pd.testing.assert_frame_equal(first.rejections, second.rejections, check_exact=True)
    pd.testing.assert_frame_equal(
        first.target_weights, second.target_weights, check_exact=True
    )
    assert first.metrics == second.metrics
    assert first.metadata["calibration"] == second.metadata["calibration"]


def test_engine_records_benchmark_metadata(make_data):
    """运行结果必须能区分价格指数与全收益指数。"""
    score, price = _frames()
    benchmark = pd.Series(np.arange(len(DATES), dtype=float) + 1000.0, index=DATES)
    info = {
        "code": "H00300.CSI", "name": "300收益", "return_type": "total_return",
        "start": str(DATES[0].date()), "end": str(DATES[-1].date()),
    }
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
        benchmark=benchmark, benchmark_info=info,
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    assert result.metadata["benchmark"] == info


def _allocation_panel(make_data, prices, dates=None, cash=100_000.0):
    """构造「单价分布拉得很开」的面板：部分股票一手金额就超过目标仓位。"""
    dates = list(DATES if dates is None else dates)
    stocks = [f"SH6000{position:02d}" for position in range(len(prices))]
    open_price = pd.DataFrame([prices] * len(dates), index=dates, columns=stocks)
    score = pd.DataFrame(
        [list(range(len(prices), 0, -1))] * len(dates), index=dates, columns=stocks
    )
    data = make_data(
        dates=dates, stocks=stocks, open=open_price.astype("float32"),
        close=(open_price * 1.001).astype("float32"),
        factors={"MOM_20": score.astype("float32")},
    )
    strategy = _strategy(
        portfolio=PortfolioSpec(top_n=len(prices)), rebalance=RebalanceSpec(frequency="daily")
    )
    params = BacktestParams(str(dates[0].date()), str(dates[-1].date()), cash)
    return data, strategy, params


def test_engine_bumps_one_lot_when_target_is_below_a_single_lot(make_data):
    """目标金额小于一手时用取整余量补到一手，不再整只缺席。"""
    prices = [130.0, 130.0, 130.0, 130.0, 250.0]
    data, strategy, params = _allocation_panel(make_data, prices)
    result = run_backtest(data, strategy, params)
    calibration = result.metadata["calibration"]

    assert calibration["bumped_one_lot_cells"] >= 1
    assert calibration["unaffordable_lot_cells"] == 0
    assert calibration["cash_clipped_cells"] == 0
    # 250 元那只一手 25000 元，超过 0.2 × 98000 = 19600 元的理想仓位，靠余量补上
    last_day = result.holdings.loc[result.holdings["trade_date"] == DATES[-1]]
    assert "SH600004" in set(last_day["stock_code"])
    bumped = result.allocations.loc[result.allocations["status"] == "bumped"]
    assert set(bumped["stock_code"]) == {"SH600004"}


def test_engine_keeps_lot_boundary_cells_instead_of_dropping_them(make_data):
    """目标金额恰好压在一手以下时不能整只丢掉（单纯缩放权重会踩这个坑）。"""
    prices = [333.33] * 30
    data, strategy, params = _allocation_panel(make_data, prices, cash=1_000_000.0)
    result = run_backtest(data, strategy, params)
    calibration = result.metadata["calibration"]

    # 理想仓位 32667 元、一手 33333 元：29 只用取整余量补到一手，最后 1 只放不下；
    # 日频调仓下每个执行日都重复这个结论
    execution_days = len(DATES) - 1
    assert calibration["bumped_one_lot_cells"] == 29 * execution_days
    assert calibration["unaffordable_lot_cells"] == execution_days
    assert calibration["cash_clipped_cells"] == 0
    assert calibration["allocation_gap_max_shares"] == 0.0
    held = result.holdings.loc[result.holdings["trade_date"] == DATES[-1], "weight"].sum()
    assert 0.95 < held <= 1.0


def test_engine_marks_allocation_table_for_every_target_cell(make_data):
    """逐格分配明细要能对上目标格数，状态只取约定的几种。"""
    score, price = _frames()
    data = make_data(
        dates=DATES, stocks=STOCKS, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
    )
    params = BacktestParams(str(DATES[0].date()), str(DATES[-1].date()), 100_000.0)
    result = run_backtest(data, _strategy(), params)
    assert len(result.allocations) == result.metadata["calibration"]["planned_orders"]
    assert set(result.allocations["status"]) <= {"allocated", "bumped", "unaffordable", "no_price", "flat"}
    assert result.metadata["execution"] == {
        "sizing": "target_amount", "cash_buffer": 0.02, "lot_size": 100,
        "allocation_iterations": 2, "allocation_converged": True,
        "carry_iterations": 0, "carry_converged": True,
    }



# 周频调仓的现金补回用例：13 个交易日跨三个自然周，信号落在每个自然周最后一个交易日
WEEK_DATES = pd.date_range("2024-01-29", periods=13, freq="B")


def _weekly_panel(make_data, *, sell_blocked_rows=()):
    """构造周频面板：第一周选 SH600000+SZ000001，第二周换成 SH600000+SZ000002。"""
    stocks = list(STOCKS)
    score = pd.DataFrame(
        {"SH600000": 3.0, "SZ000001": 2.0, "SZ000002": 1.0}, index=WEEK_DATES
    )
    score.loc[WEEK_DATES[5]:, ["SZ000001", "SZ000002"]] = [1.0, 2.0]
    can_sell = pd.DataFrame(True, index=WEEK_DATES, columns=stocks)
    for row in sell_blocked_rows:
        can_sell.loc[WEEK_DATES[row]] = False
    price = pd.DataFrame(10.0, index=WEEK_DATES, columns=stocks)
    data = make_data(
        dates=WEEK_DATES, stocks=stocks, open=price.astype("float32"),
        close=(price + 0.5).astype("float32"), factors={"MOM_20": score.astype("float32")},
        can_sell_open_next=can_sell,
    )
    params = BacktestParams(
        str(WEEK_DATES[0].date()), str(WEEK_DATES[-1].date()), 100_000.0
    )
    strategy = _strategy(rebalance=RebalanceSpec(frequency="weekly"))
    return data, strategy, params


def test_engine_refills_cash_limited_buy_after_cash_arrives(make_data):
    """调仓日卖不掉导致现金不够时买单被缩小，次日卖出成交后按日补回。"""
    # WEEK_DATES[9] 是第二周的信号日：次一交易日 WEEK_DATES[10] 执行，当天卖出被拦下
    data, strategy, params = _weekly_panel(make_data, sell_blocked_rows=[9])
    result = run_backtest(data, strategy, params)
    calibration = result.metadata["calibration"]

    assert calibration["cash_limited_cells"] >= 1
    assert calibration["cash_retry_attempts"] >= 1
    assert calibration["cash_retry_filled_cells"] == 1
    assert calibration["cash_pending_at_end_cells"] == 0
    assert calibration["carry_converged"] is True
    assert calibration["cash_clipped_cells"] == 0

    # 被缩小的是替换进来的 SZ000002：调仓日只买到零头，次日补回到目标
    execution_day, retry_day = WEEK_DATES[10], WEEK_DATES[11]
    scaled = result.allocations.loc[
        (result.allocations["trade_date"] == execution_day)
        & (result.allocations["stock_code"] == "SZ000002")
    ].iloc[0]
    assert bool(scaled["cash_limited"]) and scaled["cash_shrunk_shares"] > 0
    buys = result.orders.loc[
        (result.orders["trade_date"] == retry_day)
        & result.orders["Side"].astype(str).str.lower().eq("buy")
    ]
    assert set(buys["stock_code"]) == {"SZ000002"}
    # 卖不掉的旧持仓在次日正常卖出，补回靠的就是这笔现金
    sells = result.orders.loc[
        (result.orders["trade_date"] == retry_day)
        & result.orders["Side"].astype(str).str.lower().eq("sell")
    ]
    assert set(sells["stock_code"]) == {"SZ000001"}


def test_engine_keeps_cash_limited_cell_until_the_end(make_data):
    """现金一直没到账时补回目标留在待重试状态，并记入区间末遗留。"""
    data, strategy, params = _weekly_panel(make_data, sell_blocked_rows=[9, 10, 11])
    result = run_backtest(data, strategy, params)
    calibration = result.metadata["calibration"]

    assert calibration["cash_limited_cells"] >= 1
    assert calibration["cash_retry_attempts"] >= 1
    assert calibration["cash_retry_filled_cells"] == 0
    assert calibration["cash_pending_at_end_cells"] >= 1
    assert calibration["carry_converged"] is True


def test_engine_allows_stable_cash_carry_at_segment_boundary(make_data):
    """中间调仓段允许把稳定的现金补回目标交给下一段。"""
    data, strategy, params = _weekly_panel(make_data, sell_blocked_rows=[9, 10, 11])
    result = run_backtest(data, strategy, params, chunk_boundary=True)
    calibration = result.metadata["calibration"]

    assert calibration["allocation_converged"] is True
    assert calibration["cash_carry_over_cells"] >= 1
    assert result.final_state is not None
    assert result.final_state.carry.notna().any()
