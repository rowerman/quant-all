"""数据集、预处理和切分的联动测试。"""

import pandas as pd

from alpha.dataset import DatasetSpec, assemble_dataset
from alpha.preprocess import PreprocessConfig, transform
from alpha.split import SplitConfig, build_folds, fold_masks


def test_dataset_preprocess_split_pipeline_is_causal():
    """小面板联动后，每折训练/验证/测试均非空且互不重叠。"""

    calendar = pd.bdate_range("2012-01-02", "2023-12-29")
    rows = []
    for day in calendar:
        for index, stock in enumerate(["A", "B", "C"]):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "MOM_20": float(index + 1),
                    "can_buy_close": True,
                    "is_suspended": False,
                    "is_st": False,
                    "is_delisting": False,
                    "list_date": pd.Timestamp("2010-01-01"),
                    "total_mv": 1000.0 + index,
                    "turnover_rate": 1.0 + index,
                    "fwd_ret_open_5": 0.001 * (index + 1),
                }
            )
    frame = pd.DataFrame(rows)
    factors = frame[["trade_date", "stock_code", "MOM_20"]]
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
    industry = frame[["trade_date", "stock_code"]].assign(industry_l1="I1")
    spec = DatasetSpec(start="2012-01-01", end="2023-12-31")
    dataset = assemble_dataset(
        factors, daily, labels, industry, calendar, spec, ("MOM_20",)
    )
    preprocessed = transform(dataset, PreprocessConfig(method="rank"))
    folds = build_folds(calendar, SplitConfig())
    for fold in folds:
        masks = fold_masks(preprocessed.data, fold)
        assert masks["train"].sum() > 0
        assert masks["valid"].sum() > 0
        assert masks["test"].sum() > 0
        assert not (masks["train"] & masks["valid"]).any()
        assert not (masks["valid"] & masks["test"]).any()
