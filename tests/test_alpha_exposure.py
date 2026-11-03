"""Top-N 组合风格与行业暴露测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha.exposure import compute_portfolio_exposure
from calculate.styles import RAW_COLUMNS


def test_portfolio_exposure_uses_top_holdings_and_industry_weights():
    """组合暴露按实际 Top-2 等权持仓计算。"""

    dates = pd.to_datetime(["2020-01-03", "2020-01-10"])
    stocks = ["A", "B", "C", "D"]
    data_rows = []
    prediction_rows = []
    style_rows = []
    for day_index, day in enumerate(dates):
        for stock_index, stock in enumerate(stocks):
            data_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "eligible": True,
                    "label_raw": 0.01,
                    "industry_l1": "I1" if stock in ("A", "B") else "I2",
                }
            )
            score = 4.0 - stock_index if day_index == 0 else float(stock_index + 1)
            prediction_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "score": score,
                }
            )
            values = {
                "SIZE": float(stock_index),
                "BP": float(stock_index + 1),
                "EP": float(stock_index + 2),
                "MOMENTUM": float(stock_index - 1),
                "SHORT_REV": float(1 - stock_index),
                "VOL": float(stock_index + 1),
                "LIQUIDITY": float(stock_index + 2),
                "BETA_60": float(1.0 + stock_index / 10.0),
            }
            style_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    **values,
                }
            )
    result = compute_portfolio_exposure(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(data_rows),
        top_n=2,
        styles=pd.DataFrame(style_rows).loc[
            :, ["trade_date", "stock_code", *RAW_COLUMNS]
        ],
    )
    assert result.summary["days"] == 2
    assert result.summary["holdings_mean"] == pytest.approx(2.0)
    assert result.summary["hhi_mean"] == pytest.approx(0.5)
    assert result.summary["unknown_industry_mean_weight"] == 0.0
    assert len(result.style_series) == 14
    assert set(result.style_series["style"]) == set(
        ["SIZE", "VALUE", "MOMENTUM", "SHORT_REV", "VOL", "LIQUIDITY", "BETA_60"]
    )
    assert (result.style_series["coverage"] == 1.0).all()
    by_day = result.industry_series.groupby("trade_date")["weight"].sum()
    assert by_day.to_numpy() == pytest.approx([1.0, 1.0])
    first_day = result.industry_series.loc[
        result.industry_series["trade_date"] == dates[0]
    ]
    assert first_day.iloc[0]["industry"] == "I1"
    assert first_day.iloc[0]["weight"] == pytest.approx(1.0)
