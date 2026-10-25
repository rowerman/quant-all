"""特征审计纯函数的单元测试。"""

import numpy as np
import pandas as pd

from scripts.build_alpha.audit import (
    audit_year,
    correlation_clusters,
    correlation_matrix,
    feature_summary,
    summarize_ic,
)


def _frame() -> pd.DataFrame:
    """构造两个日期、四只股票的小面板。"""

    rows = []
    for day in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for index, stock in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "eligible": True,
                    "X": float(index + 1),
                    "Y": float(4 - index),
                    "Z": float(index % 2),
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[0, "X"] = np.nan
    return frame


def test_audit_year_reports_coverage_and_missing_concentration():
    """覆盖率和缺失集中度可按手工结果核对。"""

    result = audit_year(_frame(), ("X", "Y"), 2024)
    x = result.set_index("factor").loc["X"]
    assert x["coverage"] == 7 / 8
    assert x["missing_rows"] == 1
    assert x["missing_max_stock_share"] == 1.0
    assert x["nonfinite_rows"] == 0


def test_correlation_and_clusters_detect_opposite_factors():
    """绝对值高相关的因子进入同一簇。"""

    frame = _frame()
    corr = correlation_matrix(
        frame,
        ("X", "Y", "Z"),
        pd.DatetimeIndex(frame["trade_date"].unique()),
        min_stocks=3,
    )
    assert corr.loc["X", "Y"] < -0.99
    clusters = correlation_clusters(corr, threshold=0.9)
    assert clusters["X"] == clusters["Y"]


def test_feature_summary_excludes_only_very_sparse_recent_factor():
    """覆盖规则只剔除近期极稀疏因子。"""

    coverage = pd.DataFrame(
        [
            {"factor": "A", "year": year, "coverage": 0.95}
            for year in range(2021, 2024)
        ]
        + [
            {"factor": "B", "year": year, "coverage": 0.4}
            for year in range(2021, 2024)
        ]
    )
    corr = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
    summary, payload = feature_summary(coverage, corr, ("A", "B"))
    assert summary.loc[summary["factor"] == "A", "selected"].iloc[0]
    assert not summary.loc[summary["factor"] == "B", "selected"].iloc[0]
    assert payload["excluded_factors"] == ["B"]


def test_summarize_ic_reports_ir_and_win_rate():
    """IC 汇总与手算均值和胜率一致。"""

    series = pd.Series([0.1, 0.2, -0.1, 0.3])
    stats = summarize_ic(series)
    assert stats["mean"] == 0.125
    assert stats["win_rate"] == 0.75
    assert stats["days"] == 4
