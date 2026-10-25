"""Purged walk-forward 时间切分。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from alpha.config import DEFAULT_CONTRACT
from alpha.errors import AlphaSplitError


@dataclass(frozen=True)
class SplitConfig:
    """扩展或滚动窗口的切分配置。"""

    dev_start: str = DEFAULT_CONTRACT.dev_start
    dev_end: str = DEFAULT_CONTRACT.dev_end
    holdout_start: str = DEFAULT_CONTRACT.holdout_start
    first_test_year: int = 2018
    last_test_year: int = 2023
    valid_years: int = 1
    purge_days: int = DEFAULT_CONTRACT.purge_days
    embargo_days: int = DEFAULT_CONTRACT.embargo_days
    mode: str = "expanding"
    train_window_years: int | None = None
    horizon: int = DEFAULT_CONTRACT.horizon

    def __post_init__(self) -> None:
        if self.mode not in ("expanding", "rolling"):
            raise AlphaSplitError("mode 只支持 expanding/rolling")
        if type(self.first_test_year) is not int or type(self.last_test_year) is not int:
            raise AlphaSplitError("测试年份必须是整数")
        if self.first_test_year > self.last_test_year:
            raise AlphaSplitError("首个测试年不能晚于最后一个测试年")
        if type(self.valid_years) is not int or self.valid_years <= 0:
            raise AlphaSplitError("valid_years 必须是正整数")
        minimum_gap = self.horizon + 1
        if self.purge_days < minimum_gap or self.embargo_days < minimum_gap:
            raise AlphaSplitError(f"purge/embargo 不能小于 h+1={minimum_gap}")
        if self.dev_start > self.dev_end:
            raise AlphaSplitError("开发期起始日晚于截止日")
        if self.holdout_start <= self.dev_end:
            raise AlphaSplitError("最终测试集必须晚于开发期")
        if self.mode == "rolling":
            if type(self.train_window_years) is not int or self.train_window_years <= 0:
                raise AlphaSplitError("rolling 模式必须提供正整数 train_window_years")
        elif self.train_window_years is not None:
            raise AlphaSplitError("expanding 模式不能设置 train_window_years")


@dataclass(frozen=True)
class Fold:
    """一折训练、验证和测试区间。"""

    fold_id: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    purge_days: int
    embargo_days: int

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return asdict(self)


def _first_on_or_after(dates: pd.DatetimeIndex, stamp: pd.Timestamp) -> pd.Timestamp:
    """返回不早于给定日期的第一个交易日。"""

    position = int(dates.searchsorted(stamp, side="left"))
    if position >= len(dates):
        raise AlphaSplitError(f"{stamp.date()} 之后没有交易日")
    return pd.Timestamp(dates[position])


def _last_on_or_before(dates: pd.DatetimeIndex, stamp: pd.Timestamp) -> pd.Timestamp:
    """返回不晚于给定日期的最后一个交易日。"""

    position = int(dates.searchsorted(stamp, side="right")) - 1
    if position < 0:
        raise AlphaSplitError(f"{stamp.date()} 之前没有交易日")
    return pd.Timestamp(dates[position])


def build_folds(
    calendar: pd.DatetimeIndex,
    config: SplitConfig | None = None,
) -> tuple[Fold, ...]:
    """构造 2018-2023 的 purged walk-forward 折。"""

    settings = config or SplitConfig()
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    dates = pd.DatetimeIndex(dates)
    if pd.Timestamp(f"{settings.last_test_year}-12-31") >= pd.Timestamp(
        settings.holdout_start
    ):
        raise AlphaSplitError("切分进入最终测试集")
    dev = dates[
        (dates >= pd.Timestamp(settings.dev_start))
        & (dates <= pd.Timestamp(settings.dev_end))
    ]
    if len(dev) == 0:
        raise AlphaSplitError("开发期没有交易日")
    folds: list[Fold] = []
    for test_year in range(settings.first_test_year, settings.last_test_year + 1):
        valid_start_year = test_year - settings.valid_years
        valid_start = _first_on_or_after(dev, pd.Timestamp(f"{valid_start_year}-01-01"))
        valid_end = _last_on_or_before(dev, pd.Timestamp(f"{test_year - 1}-12-31"))
        if valid_start > valid_end:
            raise AlphaSplitError(f"{test_year} 折没有验证期")
        test_year_start = _first_on_or_after(dev, pd.Timestamp(f"{test_year}-01-01"))
        test_index = int(dev.get_indexer([test_year_start])[0])
        embargo_index = test_index + settings.embargo_days
        if embargo_index >= len(dev):
            raise AlphaSplitError(f"{test_year} 折 embargo 后没有测试交易日")
        test_start = pd.Timestamp(dev[embargo_index])
        test_end = _last_on_or_before(dev, pd.Timestamp(f"{test_year}-12-31"))
        if test_start > test_end:
            raise AlphaSplitError(f"{test_year} 折测试期为空")
        valid_index = int(dev.get_indexer([valid_start])[0])
        train_end_index = valid_index - settings.purge_days - 1
        if train_end_index < 0:
            raise AlphaSplitError(f"{test_year} 折训练期不足 purge")
        train_end = pd.Timestamp(dev[train_end_index])
        if settings.mode == "expanding":
            train_start = pd.Timestamp(dev[0])
        else:
            cutoff = train_end - pd.DateOffset(years=int(settings.train_window_years))
            train_start = _first_on_or_after(dev, cutoff)
        if train_start > train_end:
            raise AlphaSplitError(f"{test_year} 折训练期为空")
        if train_end >= valid_start or valid_end >= test_start:
            raise AlphaSplitError(f"{test_year} 折 purge/embargo 未形成间隔")
        if test_end >= pd.Timestamp(settings.holdout_start):
            raise AlphaSplitError(f"{test_year} 折进入最终测试集")
        folds.append(
            Fold(
                fold_id=f"wf_{test_year}",
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                valid_start=str(valid_start.date()),
                valid_end=str(valid_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                purge_days=settings.purge_days,
                embargo_days=settings.embargo_days,
            )
        )
    return tuple(folds)


def fold_masks(data: pd.DataFrame, fold: Fold) -> dict[str, pd.Series]:
    """返回数据集上的训练、验证和测试日期掩码。"""

    dates = pd.to_datetime(data["trade_date"])
    return {
        "train": (dates >= pd.Timestamp(fold.train_start))
        & (dates <= pd.Timestamp(fold.train_end)),
        "valid": (dates >= pd.Timestamp(fold.valid_start))
        & (dates <= pd.Timestamp(fold.valid_end)),
        "test": (dates >= pd.Timestamp(fold.test_start))
        & (dates <= pd.Timestamp(fold.test_end)),
    }


def validate_label_gap(
    calendar: pd.DatetimeIndex,
    fold: Fold,
    *,
    horizon: int = DEFAULT_CONTRACT.horizon,
) -> None:
    """检查训练和验证标签终点没有越过下一段起点。"""

    dates = pd.DatetimeIndex(calendar).sort_values()
    for end, next_start in (
        (fold.train_end, fold.valid_start),
        (fold.valid_end, fold.test_start),
    ):
        end_index = int(dates.get_indexer([pd.Timestamp(end)])[0])
        next_index = int(dates.get_indexer([pd.Timestamp(next_start)])[0])
        if end_index < 0 or next_index < 0:
            raise AlphaSplitError(f"切分日期 {end} 或 {next_start} 不在交易日历中")
        if end_index + horizon + 1 >= next_index:
            raise AlphaSplitError(
                f"标签终点越过切分边界：{end} + {horizon + 1} 日 >= {next_start}"
            )
