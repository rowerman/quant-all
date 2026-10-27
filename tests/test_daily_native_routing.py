"""逐日排名与 Pearson 的业务路由和 Python/native 一致性测试。"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from calculate.daily import daily_corr, daily_ic, daily_rank_ic_multi

requires_native = pytest.mark.skipif(
    importlib.util.find_spec("quant_native") is None,
    reason="quant_native 扩展未构建",
)


def make_panel(days: int = 24, stocks: int = 18, seed: int = 91) -> pd.DataFrame:
    """构造带多标签缺失和并列值的合成面板。"""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    stock_codes = [f"SH600{index:03d}" for index in range(stocks)]
    frame = pd.MultiIndex.from_product(
        [dates, stock_codes],
        names=["trade_date", "stock_code"],
    ).to_frame(index=False)
    factor = rng.integers(0, 7, size=len(frame)).astype(np.float64)
    frame["factor"] = factor
    frame["label"] = 0.4 * factor + rng.normal(size=len(frame))
    frame["label_h3"] = 0.2 * factor + rng.normal(size=len(frame))
    frame["label_h10"] = rng.normal(size=len(frame))
    frame["in_universe"] = True
    frame.loc[frame.index[::29], "factor"] = np.nan
    frame.loc[frame.index[::37], "label"] = np.inf
    frame.loc[frame.index[::41], "label_h3"] = np.nan
    frame.loc[frame.index[5], "in_universe"] = False
    return frame


def _daily_outputs(panel: pd.DataFrame):
    """计算三条业务路径的稳定输出。"""

    work = panel.sample(frac=1.0, random_state=113).reset_index(drop=True)
    return {
        "ic": daily_ic(work, winsorize=True),
        "rank_decay": daily_rank_ic_multi(
            work,
            ["label", "label_h3", "label_h10"],
        ),
        "rank_corr": daily_corr(work, "factor", "label", rank=True),
        "pearson_corr": daily_corr(work, "factor", "label", rank=False),
    }


@requires_native
def test_daily_paths_match_between_python_and_native(monkeypatch):
    """IC、衰减和增量 IC 的日度路径与 Python 金标准一致。"""

    import quant_native  # noqa: F401

    panel = make_panel()
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = _daily_outputs(panel)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual = _daily_outputs(panel)

    pd.testing.assert_frame_equal(
        actual["ic"],
        expected["ic"],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    pd.testing.assert_frame_equal(
        actual["rank_corr"],
        expected["rank_corr"],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    pd.testing.assert_frame_equal(
        actual["pearson_corr"],
        expected["pearson_corr"],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    for column in expected["rank_decay"]:
        pd.testing.assert_frame_equal(
            actual["rank_decay"][column],
            expected["rank_decay"][column],
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )


@requires_native
def test_native_daily_paths_preserve_truncation_without_future_data(monkeypatch):
    """缩短评估终点时，既有日期的结果必须是完整结果的前缀。"""

    import quant_native  # noqa: F401

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    panel = make_panel()
    dates = sorted(panel["trade_date"].unique())
    cutoff = dates[-6]
    full = daily_ic(panel, winsorize=False)
    truncated = daily_ic(
        panel.loc[panel["trade_date"] <= cutoff],
        winsorize=False,
    )

    pd.testing.assert_frame_equal(
        truncated,
        full.loc[full.index <= cutoff],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
