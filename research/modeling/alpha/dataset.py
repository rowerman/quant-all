"""周频训练数据集的读取、组装、缓存与 manifest。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import quant
from alpha.calendar import load_calendar, weekly_signal_dates
from alpha.config import (
    DEFAULT_CONTRACT,
    MODEL_DATA_DIR,
    dataset_versions,
)
from alpha.errors import AlphaArtifactError, AlphaConfigError, AlphaDataError
from alpha.kernel_ops import date_groups, get_model_backend, rank_pct_by_date
from alpha.logging_conf import get_logger

logger = get_logger("alpha.dataset")

SCHEMA_VERSION = "alpha_dataset_v1"
KEY_COLUMNS = ("stock_code", "trade_date")
DAILY_FIELDS = (
    "can_buy_close",
    "is_suspended",
    "is_st",
    "is_delisting",
    "list_date",
    "total_mv",
    "turnover_rate",
)
CONTEXT_FEATURES = ("log_mv", "turnover_rate", "listed_days")
RESERVED_COLUMNS = {
    "trade_date",
    "stock_code",
    "eligible",
    "sample_weight",
    "label_raw",
    "label_rank",
    "industry_l1",
}


@dataclass(frozen=True)
class DatasetSpec:
    """一次数据集构建的输入。"""

    name: str = "weekly_fwd_open_5_v1"
    start: str = DEFAULT_CONTRACT.dev_start
    end: str = DEFAULT_CONTRACT.dev_end
    features: tuple[str, ...] | None = None
    label: str = DEFAULT_CONTRACT.label
    universe: str = DEFAULT_CONTRACT.universe
    min_listed_days: int = DEFAULT_CONTRACT.min_listed_days
    allow_holdout: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AlphaConfigError("DatasetSpec.name 必须是非空字符串")
        if self.label != DEFAULT_CONTRACT.label:
            raise AlphaConfigError(
                f"第一版只支持标签 {DEFAULT_CONTRACT.label}，收到 {self.label}"
            )
        if self.universe != DEFAULT_CONTRACT.universe:
            raise AlphaConfigError("第一版股票池固定为 all_a")
        if self.start > self.end:
            raise AlphaConfigError("数据集起始日晚于截止日")
        if not self.allow_holdout and self.end > DEFAULT_CONTRACT.dev_end:
            raise AlphaConfigError(
                f"数据集截止日 {self.end} 超过开发期 {DEFAULT_CONTRACT.dev_end}"
            )
        if type(self.min_listed_days) is not int or self.min_listed_days < 0:
            raise AlphaConfigError("min_listed_days 必须是非负整数")
        if self.features is not None:
            validate_feature_columns(self.features)

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        payload = asdict(self)
        payload["features"] = None if self.features is None else list(self.features)
        return payload


@dataclass(frozen=True)
class DatasetArtifact:
    """已落盘数据集的索引信息。"""

    dataset_id: str
    path: Path
    manifest: dict
    spec: DatasetSpec
    row_count: int
    date_count: int
    stock_count: int


def validate_feature_columns(features: tuple[str, ...] | list[str]) -> None:
    """拒绝保留列和带未来语义的列。"""

    names = [str(name) for name in features]
    if not names:
        raise AlphaConfigError("特征清单不能为空")
    if len(set(names)) != len(names):
        raise AlphaConfigError("特征清单存在重复列")
    reserved = sorted(set(names) & RESERVED_COLUMNS)
    if reserved:
        raise AlphaConfigError(f"特征清单包含保留列：{reserved}")
    context = sorted(set(names) & set(CONTEXT_FEATURES))
    if context:
        raise AlphaConfigError(f"特征清单不能重复包含上下文特征：{context}")
    future = [
        name
        for name in names
        if name.startswith("fwd_") or name.startswith("label") or name == "ret"
    ]
    if future:
        raise AlphaConfigError(f"特征清单包含未来或收益列：{future}")


def implemented_features() -> tuple[str, ...]:
    """返回因子目录中的全部已实现因子。"""

    frame = quant.load_factor_list(status="implemented")
    if frame.empty:
        raise AlphaDataError("因子目录没有 implemented 因子")
    return tuple(frame["name"].astype(str))


def _canonical(payload: dict) -> str:
    """生成稳定 JSON 文本。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dataset_id(
    spec: DatasetSpec,
    features: tuple[str, ...],
    versions: dict,
) -> str:
    """由规格、特征和版本生成确定性数据集 ID。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "features": list(features),
        "versions": versions,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


def _eligible(frame: pd.DataFrame, min_listed_days: int) -> pd.Series:
    """按回测口径计算可交易标记。"""

    listed_days = (
        pd.to_datetime(frame["trade_date"]) - pd.to_datetime(frame["list_date"])
    ).dt.days
    return (
        frame["can_buy_close"].fillna(False).astype(bool)
        & ~frame["is_suspended"].fillna(True).astype(bool)
        & ~frame["is_st"].fillna(True).astype(bool)
        & ~frame["is_delisting"].fillna(True).astype(bool)
        & (listed_days >= min_listed_days)
    )


def _industry_panel(
    stock_codes: pd.Series,
    dates: pd.Series,
    start: str,
    end: str,
) -> pd.DataFrame:
    """按周频日期读取时点申万一级行业。"""

    pairs = pd.DataFrame(
        {
            "stock_code": stock_codes.astype(str),
            "trade_date": pd.to_datetime(dates),
        }
    ).drop_duplicates()
    if pairs.empty:
        return pd.DataFrame(columns=["stock_code", "trade_date", "industry_l1"])
    connection = quant.db.get_conn()
    try:
        connection.register("weekly_pairs", pairs)
        frame = connection.execute(
            """
            SELECT w.stock_code, w.trade_date, min(i.l1_code) AS industry_l1
            FROM weekly_pairs w
            LEFT JOIN ref.sw_industry_member i
              ON i.stock_code = w.stock_code
             AND i.in_date <= CAST(w.trade_date AS DATE)
             AND (i.out_date IS NULL OR i.out_date >= CAST(w.trade_date AS DATE))
            WHERE i.in_date <= CAST(? AS DATE)
              AND (i.out_date IS NULL OR i.out_date >= CAST(? AS DATE))
            GROUP BY 1, 2
            """,
            [end, start],
        ).df()
    finally:
        connection.close()
    return frame


def assemble_dataset(
    factor_panel: pd.DataFrame,
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    industry: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    spec: DatasetSpec,
    features: tuple[str, ...],
    signal_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """把因子、行情状态、标签和行业组装成周频数据集。"""

    validate_feature_columns(features)
    required_factors = set(KEY_COLUMNS) | set(features)
    missing = sorted(required_factors - set(factor_panel.columns))
    if missing:
        raise AlphaDataError(f"因子面板缺少列：{missing}")
    dates = (
        weekly_signal_dates(calendar)
        if signal_dates is None
        else pd.DatetimeIndex(signal_dates)
    )
    frame = factor_panel.loc[
        pd.to_datetime(factor_panel["trade_date"]).isin(dates)
    ].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    daily = daily[list(KEY_COLUMNS) + list(DAILY_FIELDS)].copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    label_column = spec.label
    labels = labels[list(KEY_COLUMNS) + [label_column]].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    frame = frame.merge(daily, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    frame = frame.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if industry is not None and not industry.empty:
        industry = industry.copy()
        industry["trade_date"] = pd.to_datetime(industry["trade_date"])
        frame = frame.merge(
            industry, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
        )
    else:
        frame["industry_l1"] = pd.NA
    frame["eligible"] = _eligible(frame, spec.min_listed_days)
    frame["listed_days"] = (
        frame["trade_date"] - pd.to_datetime(frame["list_date"])
    ).dt.days.astype("float32")
    frame["log_mv"] = np.log(
        pd.to_numeric(frame["total_mv"], errors="coerce").where(
            lambda value: value > 0
        )
    ).astype("float32")
    frame["turnover_rate"] = pd.to_numeric(
        frame["turnover_rate"], errors="coerce"
    ).astype("float32")
    frame["label_raw"] = pd.to_numeric(frame[label_column], errors="coerce").astype(
        "float32"
    )
    valid_label = frame["eligible"] & frame["label_raw"].notna()
    label_rank = rank_pct_by_date(
        frame["label_raw"].where(valid_label).to_numpy(dtype=np.float64),
        frame["trade_date"],
        backend=get_model_backend(),
        groups=date_groups(frame["trade_date"], len(frame)),
    )
    frame["label_rank"] = label_rank.astype("float32")
    frame["sample_weight"] = (
        (frame["eligible"] & frame["label_rank"].notna()).astype("float32")
    )
    for name in (*features, *CONTEXT_FEATURES):
        values = pd.to_numeric(frame[name], errors="coerce").astype("float32")
        frame[name] = values.replace([np.inf, -np.inf], np.nan)
    frame["industry_l1"] = frame["industry_l1"].astype("string")
    columns = [
        "trade_date",
        "stock_code",
        "eligible",
        "sample_weight",
        "label_raw",
        "label_rank",
        "industry_l1",
        *CONTEXT_FEATURES,
        *features,
    ]
    result = frame[columns].sort_values(["trade_date", "stock_code"]).reset_index(
        drop=True
    )
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise AlphaDataError("数据集键存在重复")
    return result


def _year_bounds(spec: DatasetSpec) -> list[tuple[int, str, str]]:
    """把数据集区间拆成自然年。"""

    first = pd.Timestamp(spec.start).year
    last = pd.Timestamp(spec.end).year
    bounds = []
    for year in range(first, last + 1):
        start = max(pd.Timestamp(spec.start), pd.Timestamp(f"{year}-01-01"))
        end = min(pd.Timestamp(spec.end), pd.Timestamp(f"{year}-12-31"))
        bounds.append((year, str(start.date()), str(end.date())))
    return bounds


def _load_year(
    spec: DatasetSpec,
    features: tuple[str, ...],
    start: str,
    end: str,
    signal_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int]:
    """读取并组装一年数据。"""

    factor_panel = quant.load_factor_panel(
        start=start, end=end, names=list(features)
    )
    daily = quant.load_panel(start=start, end=end, fields=list(DAILY_FIELDS))
    labels = quant.load_label(
        start=start,
        end=end,
        horizons=[DEFAULT_CONTRACT.horizon],
        price=DEFAULT_CONTRACT.price,
    )
    calendar = load_calendar(start, end)
    weekly = signal_dates
    reference = factor_panel.loc[
        pd.to_datetime(factor_panel["trade_date"]).isin(weekly),
        ["stock_code", "trade_date"],
    ]
    industry = _industry_panel(
        reference["stock_code"], reference["trade_date"], start, end
    )
    frame = assemble_dataset(
        factor_panel,
        daily,
        labels,
        industry,
        calendar,
        spec,
        features,
        signal_dates=weekly,
    )
    return frame, len(frame)


def build_dataset(spec: DatasetSpec | None = None, *, force: bool = False) -> DatasetArtifact:
    """构建或复用周频数据集。"""

    settings = spec or DatasetSpec()
    features = settings.features or implemented_features()
    validate_feature_columns(features)
    versions = dataset_versions()
    dataset_id = _dataset_id(settings, features, versions)
    folder = MODEL_DATA_DIR / "datasets" / dataset_id
    data_path = folder / "data.parquet"
    manifest_path = folder / "manifest.json"
    if data_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return DatasetArtifact(
            dataset_id=dataset_id,
            path=data_path,
            manifest=manifest,
            spec=settings,
            row_count=int(manifest["row_count"]),
            date_count=int(manifest["date_count"]),
            stock_count=int(manifest["stock_count"]),
        )
    frames = []
    yearly_rows = []
    full_calendar = load_calendar(settings.start, settings.end)
    full_weekly = weekly_signal_dates(full_calendar)
    for year, start, end in _year_bounds(settings):
        logger.info("构建数据集 %s 年：%s ~ %s", year, start, end)
        year_weekly = full_weekly[
            (full_weekly >= pd.Timestamp(start)) & (full_weekly <= pd.Timestamp(end))
        ]
        frame, count = _load_year(settings, features, start, end, year_weekly)
        frames.append(frame)
        yearly_rows.append({"year": year, "rows": count})
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    if data.duplicated(list(KEY_COLUMNS)).any():
        raise AlphaDataError("数据集键存在重复")
    eligible = data["eligible"].astype(bool)
    missing_rates = {
        name: float(data.loc[eligible, name].isna().mean())
        for name in (*CONTEXT_FEATURES, *features)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "spec": settings.to_dict(),
        "features": list(features),
        "context_features": list(CONTEXT_FEATURES),
        "versions": versions,
        "row_count": int(len(data)),
        "date_count": int(data["trade_date"].nunique()),
        "stock_count": int(data["stock_code"].nunique()),
        "date_min": str(data["trade_date"].min().date()),
        "date_max": str(data["trade_date"].max().date()),
        "eligible_rows": int(data["eligible"].sum()),
        "label_coverage": float(
            data.loc[eligible, "label_rank"].notna().mean()
        ),
        "missing_rates": missing_rates,
        "yearly_rows": yearly_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "data.parquet.tmp"
    data.to_parquet(temporary, index=False)
    os.replace(temporary, data_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "数据集就绪：%s 行=%d 日期=%d 股票=%d",
        dataset_id,
        len(data),
        manifest["date_count"],
        manifest["stock_count"],
    )
    return DatasetArtifact(
        dataset_id=dataset_id,
        path=data_path,
        manifest=manifest,
        spec=settings,
        row_count=len(data),
        date_count=manifest["date_count"],
        stock_count=manifest["stock_count"],
    )


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    """按数据集 ID 读取数据和 manifest。"""

    folder = MODEL_DATA_DIR / "datasets" / dataset_id
    data_path = folder / "data.parquet"
    manifest_path = folder / "manifest.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise AlphaArtifactError(f"数据集不存在：{dataset_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id:
        raise AlphaArtifactError(
            f"数据集 manifest ID 不匹配：目录 {dataset_id}，manifest {manifest.get('dataset_id')}"
        )
    return pd.read_parquet(data_path), manifest
