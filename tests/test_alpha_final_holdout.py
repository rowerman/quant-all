"""最终留出集日期、门槛和失败关闭测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha.artifacts import validate_dev_range
from alpha.errors import AlphaConfigError
from alpha.final_holdout_data import FinalHoldoutSpec, complete_signal_dates
from alpha.final_holdout_eval import evaluate_final_holdout_gate
from alpha.holdout import (
    FUTURE_HOLDOUT_WINDOW_ID,
    LEGACY_HOLDOUT_WINDOW_ID,
    get_holdout_window,
)
from common.paths import CONFIG_DIR


def _metrics(
    *,
    rank_ic: float,
    icir: float,
    annual: float,
    drawdown: float,
    turnover: float,
) -> dict:
    """构造决策所需模型层和组合层指标。"""

    return {
        "rank_ic_mean": rank_ic,
        "icir": icir,
        "annual_return": annual,
        "max_drawdown": drawdown,
        "turnover_two_way_annual": turnover,
        "yearly": [
            {"year": year, "rank_ic_mean": rank_ic}
            for year in (2024, 2025, 2026)
        ],
    }


def _yearly(excess: dict[int, float]) -> pd.DataFrame:
    """构造候选分年完整账本结果。"""

    return pd.DataFrame(
        [
            {
                "candidate": "LIGHTGBM_SMOOTH_4W",
                "year": year,
                "excess_total_return": value,
            }
            for year, value in excess.items()
        ]
    )


def test_final_holdout_spec_and_range_are_explicit():
    """新窗口由注册表提供，旧窗口已封存，普通接口仍拒绝留出区间。"""

    spec = FinalHoldoutSpec(
        window_id=FUTURE_HOLDOUT_WINDOW_ID,
        plan_path=str(CONFIG_DIR / "alpha" / "final_holdout_2024_2026.json"),
    )
    assert spec.start == "2026-10-16"
    assert spec.end == "2027-10-15"
    assert get_holdout_window(LEGACY_HOLDOUT_WINDOW_ID).sealed is True
    assert get_holdout_window(FUTURE_HOLDOUT_WINDOW_ID).sealed is False
    with pytest.raises(AlphaConfigError, match="最终测试集未开放"):
        validate_dev_range("2023-01-01", "2024-01-02")
    validate_dev_range(
        "2024-01-01",
        "2026-10-15",
        allow_holdout=True,
    )
    validate_dev_range(
        "2026-10-16",
        "2027-10-15",
        allow_holdout=True,
        holdout_window=spec.window,
    )


def test_complete_signal_dates_drop_incomplete_tail():
    """末尾没有未来标签的周频日期不进入模型评估。"""

    rows = []
    for day in pd.to_datetime(["2026-09-24", "2026-09-30", "2026-10-09"]):
        for index in range(120):
            rows.append(
                {
                    "trade_date": day,
                    "eligible": True,
                    "label_raw": (
                        float(index) if day != pd.Timestamp("2026-10-09") else None
                    ),
                }
            )
    dates = complete_signal_dates(
        pd.DataFrame(rows),
        FinalHoldoutSpec(min_label_coverage=0.95, min_signal_days=2),
    )
    assert dates.tolist() == [
        pd.Timestamp("2026-09-24"),
        pd.Timestamp("2026-09-30"),
    ]


def test_gate_passes_only_when_all_registered_conditions_hold():
    """主候选满足两项改善、换手和分年稳定时通过。"""

    model_metrics = {
        "LIGHTGBM_SMOOTH_4W": _metrics(
            rank_ic=0.08,
            icir=0.70,
            annual=0.20,
            drawdown=-0.10,
            turnover=40.0,
        ),
        "RIDGE_RAW": _metrics(
            rank_ic=0.09,
            icir=0.75,
            annual=0.10,
            drawdown=-0.20,
            turnover=50.0,
        ),
    }
    ledger = {
        "LIGHTGBM_SMOOTH_4W": model_metrics["LIGHTGBM_SMOOTH_4W"],
        "RIDGE_RAW": model_metrics["RIDGE_RAW"],
    }
    result = evaluate_final_holdout_gate(
        model_metrics,
        ledger,
        _yearly({2024: 0.1, 2025: 0.2, 2026: 0.03}),
        selected_name="LIGHTGBM_SMOOTH_4W",
        reference_name="RIDGE_RAW",
    )
    assert result["passed"] is True
    assert set(result["gate"]["improvements"]) == {
        "annual_return",
        "max_drawdown",
    }

    failed = evaluate_final_holdout_gate(
        model_metrics,
        ledger,
        _yearly({2024: 0.1, 2025: -0.01, 2026: 0.03}),
        selected_name="LIGHTGBM_SMOOTH_4W",
        reference_name="RIDGE_RAW",
    )
    assert failed["passed"] is False
    assert failed["gate"]["negative_excess_years"] == [2025]


def test_gate_rejects_turnover_above_two_times_ridge():
    """换手超过参照两倍时即使收益更好也不通过。"""

    selected = _metrics(
        rank_ic=0.10,
        icir=0.80,
        annual=0.20,
        drawdown=-0.10,
        turnover=110.0,
    )
    ridge = _metrics(
        rank_ic=0.09,
        icir=0.75,
        annual=0.10,
        drawdown=-0.20,
        turnover=50.0,
    )
    result = evaluate_final_holdout_gate(
        {"LIGHTGBM_SMOOTH_4W": selected, "RIDGE_RAW": ridge},
        {"LIGHTGBM_SMOOTH_4W": selected, "RIDGE_RAW": ridge},
        _yearly({2024: 0.1, 2025: 0.2, 2026: 0.03}),
        selected_name="LIGHTGBM_SMOOTH_4W",
        reference_name="RIDGE_RAW",
    )
    assert result["passed"] is False
    assert result["gate"]["turnover_ok"] is False
