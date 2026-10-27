"""截面预处理：rank/z-score、缺失填充、缺失指示和可选中性化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.dataset import RESERVED_COLUMNS
from alpha.errors import AlphaPreprocessError
from alpha.kernel_ops import (
    date_groups,
    get_model_backend,
    rank_pct_by_date,
    winsorize_quantile_by_date,
    zscore_population_by_date,
)


@dataclass(frozen=True)
class PreprocessConfig:
    """一次预处理配置。"""

    method: str = "rank"
    missing_indicator: bool = True
    winsorize: bool = False
    neutralize: tuple[str, ...] = ()
    clip_z: float = 5.0

    def __post_init__(self) -> None:
        if self.method not in ("rank", "zscore"):
            raise AlphaPreprocessError("method 只支持 rank/zscore")
        unknown = set(self.neutralize) - {"industry", "log_mv"}
        if unknown:
            raise AlphaPreprocessError(f"未知中性化项：{sorted(unknown)}")
        if isinstance(self.clip_z, bool) or not isinstance(self.clip_z, (int, float)):
            raise AlphaPreprocessError("clip_z 必须是正数")
        if float(self.clip_z) <= 0:
            raise AlphaPreprocessError("clip_z 必须是正数")


@dataclass(frozen=True)
class PreprocessedFrame:
    """预处理后的数据与诊断信息。"""

    data: pd.DataFrame
    feature_names: tuple[str, ...]
    missing_columns: tuple[str, ...]
    report: dict


def _residualize_log_mv(
    values: pd.Series,
    log_mv: pd.Series,
    dates: pd.Series,
    eligible: pd.Series,
) -> pd.Series:
    """按日对 log_mv 做一元回归并返回残差。"""

    x = log_mv.where(eligible).astype("float64")
    y = values.where(eligible).astype("float64")
    x_mean = x.groupby(dates).transform("mean")
    y_mean = y.groupby(dates).transform("mean")
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = (x_centered * x_centered).groupby(dates).transform("sum")
    numerator = (x_centered * y_centered).groupby(dates).transform("sum")
    beta = numerator.div(denominator.replace(0.0, np.nan))
    return y_centered - beta * x_centered


def _neutralize(
    values: pd.Series,
    frame: pd.DataFrame,
    config: PreprocessConfig,
) -> pd.Series:
    """按配置依次做行业和市值中性化。"""

    dates = frame["trade_date"]
    eligible = frame["eligible"].astype(bool)
    result = values.astype("float64")
    if "industry" in config.neutralize:
        industry = frame["industry_l1"].fillna("UNKNOWN").astype(str)
        group_mean = result.groupby([dates, industry]).transform("mean")
        result = result - group_mean
    if "log_mv" in config.neutralize:
        result = _residualize_log_mv(
            result, frame["log_mv"], dates, eligible
        )
    return result


def _feature_names(frame: pd.DataFrame, supplied: tuple[str, ...] | None) -> tuple[str, ...]:
    """确定参与预处理的原始特征列。"""

    if supplied is not None:
        return tuple(supplied)
    excluded = set(RESERVED_COLUMNS) | {"industry_l1"}
    names = tuple(
        name
        for name in frame.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    )
    if not names:
        raise AlphaPreprocessError("没有可用的数值特征")
    return names


def transform(
    frame: pd.DataFrame,
    config: PreprocessConfig | None = None,
    *,
    feature_columns: tuple[str, ...] | None = None,
) -> PreprocessedFrame:
    """按日截面预处理数据集。"""

    settings = config or PreprocessConfig()
    required = {"trade_date", "stock_code", "eligible", "sample_weight", "label_raw", "label_rank"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlphaPreprocessError(f"数据集缺少列：{missing}")
    names = _feature_names(frame, feature_columns)
    reserved = sorted(set(names) & set(RESERVED_COLUMNS))
    if reserved:
        raise AlphaPreprocessError(f"预处理特征包含保留列：{reserved}")
    missing_features = sorted(set(names) - set(frame.columns))
    if missing_features:
        raise AlphaPreprocessError(f"数据集缺少特征：{missing_features}")
    if "log_mv" not in frame.columns and "log_mv" in settings.neutralize:
        raise AlphaPreprocessError("市值中性化需要 log_mv 列")
    dates = frame["trade_date"]
    backend = get_model_backend()
    groups = date_groups(dates, len(frame))
    eligible = frame["eligible"].astype(bool)
    output: dict[str, pd.Series] = {
        "trade_date": dates,
        "stock_code": frame["stock_code"],
        "eligible": eligible,
        "sample_weight": frame["sample_weight"].astype("float32"),
        "label_raw": frame["label_raw"].astype("float32"),
        "label_rank": frame["label_rank"].astype("float32"),
        "industry_l1": frame.get("industry_l1", pd.Series(pd.NA, index=frame.index)),
    }
    missing_columns: list[str] = []
    missing_rates: dict[str, float] = {}
    for name in names:
        raw = pd.to_numeric(frame[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        raw = raw.where(eligible)
        missing_mask = eligible & raw.isna()
        eligible_count = int(eligible.sum())
        missing_rates[name] = (
            float(missing_mask.sum() / eligible_count) if eligible_count else 0.0
        )
        raw_values = raw.to_numpy(dtype=np.float64, na_value=np.nan)
        work = (
            winsorize_quantile_by_date(
                raw_values,
                dates,
                backend=backend,
                groups=groups,
            )
            if settings.winsorize
            else raw_values
        )
        if settings.method == "rank":
            computed = rank_pct_by_date(
                work,
                dates,
                backend=backend,
                groups=groups,
            )
            fill = 0.5
        else:
            computed = zscore_population_by_date(
                work,
                dates,
                backend=backend,
                groups=groups,
            )
            fill = 0.0
        values = pd.Series(computed, index=frame.index, dtype=np.float64)
        values = values.fillna(fill)
        if settings.neutralize:
            values = _neutralize(values, frame, settings)
        if settings.method == "zscore":
            values = values.clip(-float(settings.clip_z), float(settings.clip_z))
        values = values.astype("float32")
        output[name] = values
        if settings.missing_indicator:
            indicator = f"{name}__missing"
            output[indicator] = missing_mask.astype("int8")
            missing_columns.append(indicator)
    ordered = [
        "trade_date",
        "stock_code",
        "eligible",
        "sample_weight",
        "label_raw",
        "label_rank",
        "industry_l1",
        *names,
        *missing_columns,
    ]
    result = pd.DataFrame(output)[ordered]
    transformed = [name for name in names]
    if settings.missing_indicator:
        transformed.extend(missing_columns)
    report = {
        "method": settings.method,
        "neutralize": list(settings.neutralize),
        "winsorize": settings.winsorize,
        "rows": int(len(result)),
        "dates": int(dates.nunique()),
        "eligible_rows": int(eligible.sum()),
        "missing_rates": missing_rates,
        "feature_count": len(transformed),
    }
    return PreprocessedFrame(
        data=result,
        feature_names=tuple(transformed),
        missing_columns=tuple(missing_columns),
        report=report,
    )
