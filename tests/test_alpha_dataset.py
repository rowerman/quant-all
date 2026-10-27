"""周频数据集组装的单元测试。"""

import importlib.util

import pandas as pd
import pytest

from alpha.dataset import DatasetSpec, assemble_dataset, validate_feature_columns
from alpha.errors import AlphaConfigError

QUANT_NATIVE_AVAILABLE = importlib.util.find_spec("quant_native") is not None
requires_native = pytest.mark.skipif(
    not QUANT_NATIVE_AVAILABLE,
    reason="quant_native 扩展未构建",
)


def _inputs():
    """构造两周五日、三只股票的合成输入。"""

    calendar = pd.bdate_range("2024-01-01", "2024-01-12")
    rows = []
    for day in calendar:
        for index, stock in enumerate(["A", "B", "C"]):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "X": float(index + 1),
                    "Y": float(3 - index),
                    "can_buy_close": True,
                    "is_suspended": False,
                    "is_st": False,
                    "is_delisting": False,
                    "list_date": pd.Timestamp("2020-01-01"),
                    "total_mv": 1000.0 + index,
                    "turnover_rate": 1.0 + index,
                    "fwd_ret_open_5": [0.1, 0.2, 0.3][index],
                }
            )
    frame = pd.DataFrame(rows)
    factors = frame[["trade_date", "stock_code", "X", "Y"]]
    daily = frame[
        [
            "trade_date",
            "stock_code",
            "can_buy_close",
            "is_suspended",
            "is_st",
            "is_delisting",
            "list_date",
            "total_mv",
            "turnover_rate",
        ]
    ]
    labels = frame[["trade_date", "stock_code", "fwd_ret_open_5"]]
    industry = frame[["trade_date", "stock_code"]].assign(
        industry_l1=["I1", "I2", "I2"] * len(calendar)
    )
    return factors, daily, labels, industry, calendar


def test_assemble_dataset_has_unique_weekly_keys_and_rank_labels():
    """只保留周最后交易日，并生成截面标签排名。"""

    factors, daily, labels, industry, calendar = _inputs()
    spec = DatasetSpec(start="2024-01-01", end="2024-01-12", allow_holdout=True)
    data = assemble_dataset(
        factors, daily, labels, industry, calendar, spec, ("X", "Y")
    )
    assert len(data) == 6
    assert not data.duplicated(["trade_date", "stock_code"]).any()
    assert set(data["trade_date"]) == {
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
    }
    first = data.loc[data["trade_date"] == pd.Timestamp("2024-01-05")].set_index("stock_code")
    assert first.loc["A", "label_rank"] == pytest.approx(1 / 3)
    assert first.loc["C", "label_rank"] == pytest.approx(1.0)
    assert first["sample_weight"].tolist() == [1.0, 1.0, 1.0]


def test_assemble_dataset_marks_ineligible_rows_without_label_weight():
    """不可交易行保留，但不参与训练权重。"""

    factors, daily, labels, industry, calendar = _inputs()
    daily.loc[
        (daily["trade_date"] == pd.Timestamp("2024-01-05"))
        & (daily["stock_code"] == "A"),
        "is_st",
    ] = True
    spec = DatasetSpec(start="2024-01-01", end="2024-01-12", allow_holdout=True)
    data = assemble_dataset(
        factors, daily, labels, industry, calendar, spec, ("X", "Y")
    )
    row = data.iloc[0]
    assert not bool(row["eligible"])
    assert row["sample_weight"] == 0.0


def test_feature_validation_rejects_future_and_context_duplicates():
    """未来列、保留列和上下文重复列都不能作为特征。"""

    with pytest.raises(AlphaConfigError, match="未来"):
        validate_feature_columns(("MOM_20", "fwd_ret_open_5"))
    with pytest.raises(AlphaConfigError, match="保留列"):
        validate_feature_columns(("MOM_20", "label_rank"))
    with pytest.raises(AlphaConfigError, match="上下文"):
        validate_feature_columns(("MOM_20", "log_mv"))


def test_dataset_spec_rejects_holdout_by_default():
    """默认数据集不能跨入最终测试集。"""

    with pytest.raises(AlphaConfigError, match="开发期"):
        DatasetSpec(start="2012-01-01", end="2024-01-01")


@requires_native
def test_label_rank_matches_python_and_native(monkeypatch):
    """标签排名经统一 rank 核后仍保持并列和权重语义。"""

    factors, daily, labels, industry, calendar = _inputs()
    daily.loc[
        (daily["trade_date"] == pd.Timestamp("2024-01-05"))
        & (daily["stock_code"] == "B"),
        "is_st",
    ] = True
    labels.loc[
        (labels["trade_date"] == pd.Timestamp("2024-01-05"))
        & (labels["stock_code"] == "C"),
        "fwd_ret_open_5",
    ] = 0.1
    spec = DatasetSpec(start="2024-01-01", end="2024-01-12", allow_holdout=True)
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = assemble_dataset(
        factors,
        daily,
        labels,
        industry,
        calendar,
        spec,
        ("X", "Y"),
    )
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "native")
    actual = assemble_dataset(
        factors,
        daily,
        labels,
        industry,
        calendar,
        spec,
        ("X", "Y"),
    )

    pd.testing.assert_frame_equal(actual, expected)
