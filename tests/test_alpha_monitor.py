"""P5-2 模型监控的单元、因果性和正式产物测试。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha.errors import AlphaMonitorError
from alpha.monitor import (
    MonitorPolicy,
    evaluate_monitor_frames,
    run_model_monitor,
    simulate_score_decay,
)


def _inputs(days: int = 16):
    """构造可覆盖监控全部维度的最小周频输入。"""

    dates = pd.date_range("2023-01-06", periods=days, freq="W-FRI")
    stocks = ["A", "B", "C", "D", "E"]
    prediction_rows = []
    data_rows = []
    style_rows = []
    holding_rows = []
    ledger_rows = []
    for day_index, day in enumerate(dates):
        for stock_index, stock in enumerate(stocks):
            prediction_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "score": float(len(stocks) - stock_index)
                    + day_index * 0.01,
                }
            )
            data_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "eligible": True,
                    "label_raw": float(len(stocks) - stock_index) / 100.0,
                    "industry_l1": "I1" if stock_index < 2 else "I2",
                }
            )
            holding_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "weight": 0.2,
                }
            )
        for style_index, style in enumerate(
            ("SIZE", "VALUE", "MOMENTUM", "SHORT_REV", "VOL", "LIQUIDITY", "BETA_60")
        ):
            style_rows.append(
                {
                    "trade_date": day,
                    "style": style,
                    "exposure": 0.05 * (style_index + 1),
                    "coverage": 1.0,
                    "holdings": 5,
                }
            )
        ledger_rows.append(
            {
                "trade_date": day,
                "return": 0.01,
                "benchmark_return": 0.002,
                "turnover": 0.1,
                "drawdown": -0.01 * min(day_index, 10),
            }
        )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(data_rows),
        pd.DataFrame(ledger_rows),
        dates,
        pd.DataFrame(style_rows),
        pd.DataFrame(holding_rows),
    )


def _policy(**overrides) -> MonitorPolicy:
    """返回适合短序列测试的滚动策略。"""

    values = {
        "rolling_weeks": 3,
        "min_periods": 2,
    }
    values.update(overrides)
    return MonitorPolicy(**values)


def test_evaluate_monitor_frames_covers_all_metrics_and_is_causal():
    """监控输出覆盖分数、IC、账本、暴露，且未来变化不回写历史。"""

    predictions, data, ledger, dates, styles, holdings = _inputs()
    policy = _policy()
    first = evaluate_monitor_frames(
        predictions,
        data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=40.0,
        policy=policy,
    )
    assert len(first.timeseries) == len(dates)
    assert first.summary["signal_days"] == len(dates)
    assert first.summary["latest_missing_rate"] == 0.0
    assert first.summary["latest_rolling_deployed_rank_ic"] > 0
    assert {
        "raw_score_std",
        "deployed_score_std",
        "rolling_raw_rank_ic",
        "rolling_excess",
        "rolling_turnover_two_way_annual",
        "max_style_abs_exposure",
        "max_abs_industry_active_weight",
    } <= set(first.timeseries.columns)

    changed_predictions = predictions.copy()
    changed_data = data.copy()
    late = dates[-4:]
    changed_predictions.loc[
        changed_predictions["trade_date"].isin(late), "score"
    ] *= -1
    changed_data.loc[changed_data["trade_date"].isin(late), "label_raw"] *= -1
    second = evaluate_monitor_frames(
        changed_predictions,
        changed_data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=40.0,
        policy=policy,
    )
    pd.testing.assert_frame_equal(
        first.timeseries.iloc[:-4],
        second.timeseries.iloc[:-4],
    )


def test_monitor_uses_candidate_smoothing_weeks():
    """监控部署分数使用候选自身平滑周数。"""

    predictions, data, ledger, dates, styles, holdings = _inputs()
    early = predictions["trade_date"].isin(dates[:4])
    predictions.loc[early, "score"] *= -1.0
    raw = evaluate_monitor_frames(
        predictions,
        data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=40.0,
        policy=_policy(),
        smoothing_weeks=1,
    )
    smoothed = evaluate_monitor_frames(
        predictions,
        data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=40.0,
        policy=_policy(),
        smoothing_weeks=4,
    )
    assert not raw.timeseries["deployed_rank_ic"].equals(
        smoothed.timeseries["deployed_rank_ic"]
    )


def test_monitor_accepts_precomputed_rolling_style_exposure():
    """通用评估来源的滚动风格值不会被重复平滑。"""

    predictions, data, ledger, dates, styles, holdings = _inputs()
    styles = styles.copy()
    styles["exposure"] = 2.0
    result = evaluate_monitor_frames(
        predictions,
        data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=40.0,
        policy=_policy(),
        smoothing_weeks=1,
        style_already_rolling=True,
    )
    assert result.timeseries["max_style_abs_exposure"].iloc[0] == 2.0


def test_monitor_alerts_on_missing_exposure_and_drawdown():
    """缺失、风格、行业和回撤阈值能产生对应告警。"""

    predictions, data, ledger, dates, styles, holdings = _inputs()
    predictions = predictions.loc[
        ~(
            predictions["trade_date"].eq(dates[-1])
            & predictions["stock_code"].eq("A")
        )
    ]
    ledger.loc[ledger["trade_date"].eq(dates[-1]), "drawdown"] = -0.36
    styles.loc[:, "exposure"] = 2.1
    holdings.loc[
        holdings["trade_date"].eq(dates[5])
        & holdings["stock_code"].isin(["A", "B"]),
        "weight",
    ] = 0.5
    result = evaluate_monitor_frames(
        predictions,
        data,
        ledger,
        dates,
        styles,
        holdings,
        reference_turnover=10.0,
        policy=_policy(),
    )
    last_alerts = result.alerts.loc[result.alerts["trade_date"].eq(dates[-1])]
    assert set(last_alerts["metric"]) >= {
        "drawdown",
        "max_style_abs_exposure",
        "missing_rate",
    }
    assert "max_abs_industry_active_weight" in set(result.alerts["metric"])
    assert "alert" in set(result.alerts["severity"])


def test_score_decay_reversal_triggers_rank_ic_alert():
    """反向最后 13 周截面排名后，部署分数滚动 RankIC 必须告警。"""

    predictions, data, _, dates, _, _ = _inputs()
    result = simulate_score_decay(
        predictions,
        data,
        dates,
        policy=_policy(),
    )
    assert result["weeks"] == 13
    assert result["alert_triggered"] is True
    assert result["status"] == "alert"
    assert result["latest_decayed_rolling_deployed_rank_ic"] < 0


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"rolling_weeks": 0}, "rolling_weeks"),
        ({"rolling_weeks": 3, "min_periods": 4}, "min_periods"),
        ({"rank_ic_alert_consecutive": 1}, "不能小于"),
        (
            {"turnover_warning_multiple": 2.0, "turnover_alert_multiple": 2.0},
            "alert 倍数",
        ),
        ({"missing_warning": 0.1, "missing_alert": 0.05}, "缺失率"),
    ],
)
def test_monitor_policy_rejects_invalid_thresholds(changes, match):
    """非法窗口和阈值给出明确错误。"""

    with pytest.raises(AlphaMonitorError, match=match):
        MonitorPolicy(**changes)


def test_monitor_rejects_bad_rebalance_dates():
    """预测日期与回测信号日不一致时失败关闭。"""

    predictions, data, ledger, dates, styles, holdings = _inputs()
    with pytest.raises(AlphaMonitorError, match="预测日期"):
        evaluate_monitor_frames(
            predictions,
            data,
            ledger,
            dates[:-1],
            styles,
            holdings,
            reference_turnover=40.0,
            policy=_policy(),
        )


def test_formal_monitor_package_is_complete_and_reusable():
    """正式产物生成可复用监控包，并包含衰减告警证据。"""

    first = run_model_monitor()
    second = run_model_monitor()
    assert first["monitor_id"] == second["monitor_id"]
    assert first["summary"]["start"] == "2018-01-12"
    assert first["summary"]["end"] == "2023-12-29"
    assert first["summary"]["signal_days"] == 300
    assert first["decay_simulation"]["alert_triggered"] is True
    from alpha.artifacts import model_dir

    folder = (
        model_dir(first["sources"]["model_id"])
        / "monitoring"
        / first["monitor_id"]
    )
    assert folder.is_dir()
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    assert complete["complete"] is True
    assert set(complete["files"]) == {
        "alerts.csv",
        "decay_simulation.csv",
        "manifest.json",
        "report.json",
        "report.md",
        "timeseries.csv",
    }
    assert len(pd.read_csv(folder / "timeseries.csv")) == 300
