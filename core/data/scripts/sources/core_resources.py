"""限时 Tushare token 的核心研究资源下载器。

财务和事件数据先按原始口径分片落盘，本模块不负责清洗或正式入库。
所有分片写入前先完整取回并校验，成功后再原子替换目标文件；
``core_resources.json`` 记录完成与失败状态，重跑时只补未完成分片。
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable
import warnings

import pandas as pd

from scripts.config import EXTRA_DIR
from scripts.errors import SourceError
from scripts.logging_conf import get_logger
from scripts.sources import tushare_src

logger = get_logger("sources.core_resources")

TUSHARE_DIR = EXTRA_DIR / "tushare"
FINANCIAL_DIR = TUSHARE_DIR / "raw_financial"
EVENT_DIR = TUSHARE_DIR / "raw_events"
INDUSTRY_ROOT = TUSHARE_DIR / "raw_industry"
INDUSTRY_DIRS = {
    "SW2014": INDUSTRY_ROOT / "sw2014",
    "SW2021": INDUSTRY_ROOT / "sw2021",
}
# 兼容旧测试和外部脚本：默认仍指向 SW2021。
INDUSTRY_DIR = INDUSTRY_DIRS["SW2021"]
INDUSTRY_HISTORY_DIR = INDUSTRY_DIR / "history_members"
INDUSTRY_REPAIR_DIR = INDUSTRY_ROOT / "repair"
MANIFEST_PATH = TUSHARE_DIR / "manifests" / "core_resources.json"

DEFAULT_START_YEAR = 1990
DIVIDEND_BUCKETS = 256
MAX_PAGES = 100
DISCLOSURE_HARD_CAP = 6000
ALWAYS_NONEMPTY_APIS = frozenset(
    {"income_vip", "balancesheet_vip", "cashflow_vip", "fina_indicator_vip"}
)


@dataclass(frozen=True)
class PeriodResource:
    """按报告期全市场拉取的一类资源。"""

    api: str
    parameter: str
    directory: Path
    page_size: int
    annual_only: bool = False
    min_period: str = "19900101"


PERIOD_RESOURCES = (
    PeriodResource("income_vip", "period", FINANCIAL_DIR / "income", 5000),
    PeriodResource("balancesheet_vip", "period", FINANCIAL_DIR / "balancesheet", 5000),
    PeriodResource("cashflow_vip", "period", FINANCIAL_DIR / "cashflow", 5000),
    PeriodResource("fina_indicator_vip", "period", FINANCIAL_DIR / "fina_indicator", 5000),
    PeriodResource("forecast_vip", "period", EVENT_DIR / "forecast", 3000),
    PeriodResource("express_vip", "period", EVENT_DIR / "express", 3000),
    PeriodResource("disclosure_date", "end_date", EVENT_DIR / "disclosure_date", 5000),
    PeriodResource(
        "fina_audit", "period", EVENT_DIR / "fina_audit", 3000,
        annual_only=True, min_period="19971231",
    ),
)

DATE_COLUMNS = (
    "ann_date", "f_ann_date", "end_date", "period", "in_date", "out_date",
    "record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date",
)


class ResourceManifest:
    """核心资源下载状态清单。"""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        if path.exists():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {"version": 1, "updated_at": None, "items": {}}

    def is_complete(self, key: str, path: Path | None = None) -> bool:
        """判断分片是否已经完成；有产物的分片还必须确认文件存在。"""
        item = self.payload["items"].get(key, {})
        if item.get("status") != "complete":
            return False
        return path is None or int(item.get("rows", 0)) == 0 or path.exists()

    def update(self, key: str, **values) -> None:
        """更新一条状态并原子写回清单。"""
        item = dict(self.payload["items"].get(key, {}))
        item.update(values)
        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.payload["items"][key] = item
        self.payload["updated_at"] = item["updated_at"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def summary(self) -> dict:
        """返回完成、失败和总行数摘要。"""
        items = list(self.payload["items"].values())
        return {
            "complete": sum(item.get("status") == "complete" for item in items),
            "failed": sum(item.get("status") == "failed" for item in items),
            "rows": sum(int(item.get("rows", 0)) for item in items if item.get("status") == "complete"),
        }


def quarter_periods(start_year: int, cutoff: str) -> list[str]:
    """生成不晚于截止日的季度末列表。"""
    last = date.fromisoformat(f"{cutoff[:4]}-{cutoff[4:6]}-{cutoff[6:8]}")
    periods = []
    for year in range(start_year, last.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if period <= last:
                periods.append(period.strftime("%Y%m%d"))
    return periods


def _page_signature(frame: pd.DataFrame) -> tuple:
    """生成分页响应的轻量签名，用于识别服务端忽略 offset。"""
    if frame.empty:
        return (0, ())
    columns = [name for name in ("ts_code", "ann_date", "end_date", "period") if name in frame]
    sample = frame[columns].head(3).astype(str).to_records(index=False).tolist() if columns else []
    return (len(frame), tuple(map(tuple, sample)))


def fetch_paged(api: str, params: dict, page_size: int) -> pd.DataFrame:
    """用 limit/offset 拉完整接口响应，并拒绝重复页。"""
    pages: list[pd.DataFrame] = []
    signatures: set[tuple] = set()
    for page in range(MAX_PAGES):
        offset = page * page_size
        frame = tushare_src.call(api, **params, limit=page_size, offset=offset)
        # 中转服务偶尔会把有数据的 VIP 查询返回为空；第一页只做一次低成本复核。
        if page == 0 and (frame is None or frame.empty):
            time.sleep(1.0)
            frame = tushare_src.call(api, **params, limit=page_size, offset=offset)
        if frame is None or frame.empty:
            break
        signature = _page_signature(frame)
        if page and signature in signatures:
            raise SourceError(f"{api} 分页重复，服务端可能忽略 offset：params={params}")
        signatures.add(signature)
        pages.append(frame)
        if len(frame) < page_size:
            break
    else:
        raise SourceError(f"{api} 超过最大分页数 {MAX_PAGES}：params={params}")
    if not pages:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        merged = pd.concat(pages, ignore_index=True)
    return merged.drop_duplicates().reset_index(drop=True)


def fetch_disclosure(period: str) -> pd.DataFrame:
    """拉取单期披露日。

    ``disclosure_date`` 实测忽略 offset，但把 limit 提高到 10000 可返回完整一期；
    若触及服务端 6000 行上限则拒绝落盘，避免静默截断。
    """
    frame = tushare_src.call("disclosure_date", end_date=period, limit=10000)
    if frame is None:
        return pd.DataFrame()
    if len(frame) >= DISCLOSURE_HARD_CAP:
        raise SourceError(f"disclosure_date {period} 命中 {DISCLOSURE_HARD_CAP} 行上限")
    return frame.drop_duplicates().reset_index(drop=True)


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """从常见日期列中提取产物覆盖范围。"""
    values = []
    for column in DATE_COLUMNS:
        if column in frame:
            values.extend(
                value for value in frame[column].dropna().astype(str).tolist()
                if len(value) >= 8 and value[:8].isdigit()
            )
    return (min(values)[:8], max(values)[:8]) if values else (None, None)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """先写临时文件，再原子替换正式分片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _manifest_path(path: Path) -> str:
    """正式数据目录记相对路径，测试或外部目录保留完整路径。"""
    return str(path.relative_to(TUSHARE_DIR)) if path.is_relative_to(TUSHARE_DIR) else str(path)


def fetch_period_resources(
    periods: Iterable[str], manifest: ResourceManifest | None = None
) -> dict:
    """下载所有按报告期分片的财务与事件资源。"""
    state = manifest or ResourceManifest()
    failures: list[str] = []
    for resource in PERIOD_RESOURCES:
        pending = []
        for period in periods:
            if period < resource.min_period:
                continue
            if resource.annual_only and not period.endswith("1231"):
                continue
            path = resource.directory / f"period={period}.parquet"
            key = f"{resource.api}:{period}"
            if state.is_complete(key, path):
                continue
            pending.append(period)

        def worker(period: str):
            try:
                params = {resource.parameter: period}
                frame = (
                    fetch_disclosure(period)
                    if resource.api == "disclosure_date"
                    else fetch_paged(resource.api, params, resource.page_size)
                )
                return period, frame, ""
            except Exception as exc:
                return period, None, str(exc)

        with cf.ThreadPoolExecutor(max_workers=tushare_src.CONCURRENCY) as pool:
            futures = [pool.submit(worker, period) for period in pending]
            for future in cf.as_completed(futures):
                period, frame, error = future.result()
                path = resource.directory / f"period={period}.parquet"
                key = f"{resource.api}:{period}"
                if error:
                    state.update(
                        key, status="failed", api=resource.api, partition=period,
                        path=_manifest_path(path), rows=0, error=error[:500],
                    )
                    failures.append(key)
                    logger.error("%s %s 失败：%s", resource.api, period, error)
                    continue
                assert frame is not None
                if (
                    resource.api in ALWAYS_NONEMPTY_APIS
                    and period >= "20050101"
                    and frame.empty
                ):
                    error = f"{resource.api} {period} 异常返回空数据"
                    state.update(
                        key, status="failed", api=resource.api, partition=period,
                        path=_manifest_path(path), rows=0, error=error,
                    )
                    failures.append(key)
                    logger.error(error)
                    continue
                if not frame.empty:
                    _write_parquet_atomic(path, frame)
                low, high = _date_range(frame)
                state.update(
                    key, status="complete", api=resource.api, partition=period,
                    path=_manifest_path(path) if not frame.empty else None,
                    rows=len(frame), min_date=low, max_date=high, error=None,
                )
                logger.info("%s %s 完成：%d 行", resource.api, period, len(frame))
    return {"failures": failures, **state.summary()}


def _industry_dir(classification: str) -> Path:
    """返回某个申万分类版本的原始数据目录。"""
    try:
        return INDUSTRY_DIRS[classification]
    except KeyError as exc:
        raise SourceError(f"不支持的申万分类版本：{classification}") from exc


def _sw_catalog(
    classification: str, manifest: ResourceManifest
) -> pd.DataFrame:
    """读取或下载一个申万分类版本的 L1/L2/L3 目录。"""
    directory = _industry_dir(classification)
    path = directory / "classify.parquet"
    key = f"index_classify:{classification}"
    if manifest.is_complete(key, path):
        return pd.read_parquet(path)

    frames = []
    for level in ("L1", "L2", "L3"):
        frame = tushare_src.call("index_classify", level=level, src=classification)
        if frame is None or frame.empty:
            raise SourceError(f"index_classify {level}/{classification} 返回空数据")
        frame = frame.copy()
        frame["src"] = classification
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["index_code", "src"]
    )
    required = {"index_code", "industry_name", "parent_code", "level", "src"}
    missing = required - set(catalog.columns)
    if missing:
        raise SourceError(f"index_classify {classification} 缺少字段 {sorted(missing)}")
    _write_parquet_atomic(path, catalog)
    manifest.update(
        key, status="complete", api="index_classify", partition=classification,
        path=_manifest_path(path), rows=len(catalog),
        min_date=None, max_date=None, error=None,
    )
    return catalog


def _sw_history_key(classification: str, code: str) -> str:
    """保留 SW2021 的旧 key，SW2014 使用独立前缀。"""
    return f"index_member:{code}" if classification == "SW2021" else (
        f"index_member_sw2014:{code}"
    )


def fetch_sw_history(
    manifest: ResourceManifest | None = None,
    classification: str = "SW2021",
) -> dict:
    """下载一个申万分类版本的三级行业目录及历史成员。"""
    state = manifest or ResourceManifest()
    catalog = _sw_catalog(classification, state)
    directory = _industry_dir(classification)
    history_dir = directory / "history_members"
    level3 = catalog[catalog["level"].astype(str) == "L3"].copy()
    if classification == "SW2021":
        level3 = level3[level3["is_pub"].astype(str).isin({"1", "True", "true"})]
    pending = []
    for code in level3["index_code"].astype(str):
        path = history_dir / f"l3_code={code}.parquet"
        key = _sw_history_key(classification, code)
        if state.is_complete(key, path):
            continue
        pending.append(code)

    def worker(code: str):
        try:
            frame = tushare_src.call("index_member", index_code=code)
            if frame is None:
                frame = pd.DataFrame()
            if len(frame) >= 5000:
                raise SourceError(f"index_member {code} 命中 5000 行上限")
            required = {"index_code", "con_code", "in_date", "out_date", "is_new"}
            missing = required - set(frame.columns)
            if not frame.empty and missing:
                raise SourceError(f"index_member {code} 缺少字段 {sorted(missing)}")
            if not frame.empty and set(frame["index_code"].astype(str)) != {code}:
                raise SourceError(f"index_member {code} 返回了其它行业代码")
            if not frame.empty and frame["in_date"].isna().any():
                raise SourceError(f"index_member {code} 存在空 in_date")
            return code, frame, ""
        except Exception as exc:
            return code, None, str(exc)

    failures: list[str] = []
    empty: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=tushare_src.CONCURRENCY) as pool:
        futures = [pool.submit(worker, code) for code in pending]
        for future in cf.as_completed(futures):
            code, frame, error = future.result()
            path = history_dir / f"l3_code={code}.parquet"
            key = _sw_history_key(classification, code)
            if error:
                state.update(
                    key, status="failed", api="index_member", partition=code,
                    path=_manifest_path(path), rows=0, error=error[:500],
                )
                failures.append(key)
                logger.error("index_member %s %s 失败：%s", classification, code, error)
                continue
            assert frame is not None
            if frame.empty:
                # 空响应不是可用的历史分片，不能伪装成 complete 后静默跳过。
                state.update(
                    key, status="empty", api="index_member", partition=code,
                    path=None, rows=0, min_date=None, max_date=None,
                    error="服务端返回空成员",
                )
                empty.append(key)
                logger.warning("index_member %s %s 返回空成员", classification, code)
                continue
            _write_parquet_atomic(path, frame)
            low, high = _date_range(frame)
            state.update(
                key, status="complete", api="index_member", partition=code,
                path=_manifest_path(path), rows=len(frame),
                min_date=low, max_date=high, error=None,
            )
    return {
        "classification": classification,
        "failures": failures,
        "empty": empty,
        **state.summary(),
    }


def fetch_sw_all(manifest: ResourceManifest | None = None) -> dict:
    """下载 SW2014 与 SW2021 两套申万历史成员。"""
    state = manifest or ResourceManifest()
    results = {
        classification: fetch_sw_history(state, classification)
        for classification in ("SW2014", "SW2021")
    }
    return {
        "failures": [
            key for result in results.values() for key in result["failures"]
        ],
        "empty": [
            key for result in results.values() for key in result["empty"]
        ],
        "details": results,
        **state.summary(),
    }


def fetch_sw_repair(
    codes: Iterable[str], manifest: ResourceManifest | None = None
) -> dict:
    """按股票下载跨版本行业记录，用于补齐分类分片中的缺口。"""
    state = manifest or ResourceManifest()
    pending = []
    for code in sorted({str(code) for code in codes}):
        path = INDUSTRY_REPAIR_DIR / f"ts_code={code}.parquet"
        key = f"index_member_ts_code:{code}"
        if state.is_complete(key, path):
            continue
        pending.append(code)

    def worker(code: str):
        try:
            frame = tushare_src.call("index_member", ts_code=code)
            if frame is None:
                frame = pd.DataFrame()
            if len(frame) >= 5000:
                raise SourceError(f"index_member {code} 命中 5000 行上限")
            required = {"index_code", "con_code", "in_date", "out_date", "is_new"}
            missing = required - set(frame.columns)
            if not frame.empty and missing:
                raise SourceError(f"index_member {code} 缺少字段 {sorted(missing)}")
            if not frame.empty and set(frame["con_code"].astype(str)) != {code}:
                raise SourceError(f"index_member {code} 返回了其它股票代码")
            return code, frame, ""
        except Exception as exc:
            return code, None, str(exc)

    failures: list[str] = []
    empty: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=tushare_src.CONCURRENCY) as pool:
        futures = [pool.submit(worker, code) for code in pending]
        for future in cf.as_completed(futures):
            code, frame, error = future.result()
            path = INDUSTRY_REPAIR_DIR / f"ts_code={code}.parquet"
            key = f"index_member_ts_code:{code}"
            if error:
                state.update(
                    key, status="failed", api="index_member", partition=code,
                    path=_manifest_path(path), rows=0, error=error[:500],
                )
                failures.append(key)
                logger.error("index_member 逐股修复 %s 失败：%s", code, error)
                continue
            assert frame is not None
            if frame.empty:
                state.update(
                    key, status="empty", api="index_member", partition=code,
                    path=None, rows=0, min_date=None, max_date=None,
                    error="服务端返回空成员",
                )
                empty.append(key)
                continue
            _write_parquet_atomic(path, frame)
            low, high = _date_range(frame)
            state.update(
                key, status="complete", api="index_member", partition=code,
                path=_manifest_path(path), rows=len(frame),
                min_date=low, max_date=high, error=None,
            )
    return {"failures": failures, "empty": empty, **state.summary()}


def sw_repair_candidates(
    start: str = "2016-01-01",
    end: str = "2026-10-15",
    switch_date: str = "2021-12-13",
) -> list[str]:
    """按原始分片找出 2016 年以来行业覆盖不全的股票代码。"""
    import numpy as np
    from quant.codes import normalize_code

    catalogs = {}
    code_to_classification: dict[str, str] = {}
    for classification in ("SW2014", "SW2021"):
        path = _industry_dir(classification) / "classify.parquet"
        if not path.exists():
            raise SourceError(f"缺少 {classification} 行业目录，请先运行 industry")
        catalog = pd.read_parquet(path)
        catalogs[classification] = catalog
        for code in catalog.loc[catalog["level"].astype(str) == "L3", "index_code"].astype(str):
            code_to_classification.setdefault(code, classification)

    frames = []
    for classification in ("SW2014", "SW2021"):
        history_dir = _industry_dir(classification) / "history_members"
        for path in sorted(history_dir.glob("l3_code=*.parquet")):
            frame = pd.read_parquet(path)
            frame["classification"] = classification
            frames.append(frame)
    current_dir = _industry_dir("SW2021") / "members"
    for path in sorted(current_dir.glob("l3_code=*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame.rename(columns={"l3_code": "index_code", "ts_code": "con_code"})
        frame["classification"] = "SW2021"
        frames.append(frame)
    if INDUSTRY_REPAIR_DIR.exists():
        for path in sorted(INDUSTRY_REPAIR_DIR.glob("ts_code=*.parquet")):
            frame = pd.read_parquet(path)
            frame["classification"] = [
                code_to_classification.get(str(code)) for code in frame["index_code"]
            ]
            frames.append(frame)
    if not frames:
        raise SourceError("没有申万历史成员分片，请先运行 industry")
    members = pd.concat(frames, ignore_index=True)
    members = members.loc[
        members["classification"].notna()
        & members["index_code"].astype(str).isin(code_to_classification)
        & members["con_code"].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$")
    ].copy()
    members["stock_code"] = [normalize_code(code) for code in members["con_code"]]
    members["in_date"] = pd.to_datetime(members["in_date"], format="%Y%m%d", errors="coerce")
    members["out_date"] = pd.to_datetime(members["out_date"], format="%Y%m%d", errors="coerce")
    members = members.dropna(subset=["stock_code", "in_date"])

    calendar_path = TUSHARE_DIR / "trade_cal.parquet"
    if not calendar_path.exists():
        raise SourceError("缺少 trade_cal.parquet，请先运行 reference")
    calendar = pd.read_parquet(calendar_path)
    opened = calendar[calendar["is_open"].astype(int) == 1]
    stamps = pd.to_datetime(opened["cal_date"].astype(str), format="%Y%m%d")
    stamps = pd.DatetimeIndex(
        sorted(stamps[(stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))])
    )
    if stamps.empty:
        return []
    switch = pd.Timestamp(switch_date)

    basic = pd.read_parquet(TUSHARE_DIR / "stock_basic_all.parquet")
    basic = basic[["ts_code", "list_date", "delist_date"]].copy()
    basic = basic.loc[basic["ts_code"].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$")].copy()
    basic["stock_code"] = [normalize_code(code) for code in basic["ts_code"]]
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
    basic["delist_date"] = pd.to_datetime(basic["delist_date"], format="%Y%m%d", errors="coerce")

    grouped: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in members.itertuples(index=False):
        grouped.setdefault((row.stock_code, row.classification), []).append(
            (row.in_date, row.out_date)
        )

    candidates: list[str] = []
    for row in basic.drop_duplicates("stock_code").itertuples(index=False):
        first = stamps[0] if pd.isna(row.list_date) else max(row.list_date, stamps[0])
        last = min(row.delist_date, stamps[-1]) if pd.notna(row.delist_date) else stamps[-1]
        if pd.isna(first) or first > last:
            continue
        active_days = stamps[(stamps >= first) & (stamps <= last)]
        if active_days.empty:
            continue
        active_values = active_days.to_numpy(dtype="datetime64[ns]")
        covered_2014 = np.zeros(len(active_days), dtype=bool)
        covered_2021 = np.zeros(len(active_days), dtype=bool)
        for classification, target in (
            ("SW2014", covered_2014), ("SW2021", covered_2021)
        ):
            for lower, upper in grouped.get((row.stock_code, classification), []):
                start_at = int(np.searchsorted(active_values, np.datetime64(lower), side="left"))
                end_at = len(active_days) if pd.isna(upper) else int(
                    np.searchsorted(active_values, np.datetime64(upper), side="right")
                )
                if start_at < end_at:
                    target[start_at:end_at] = True
        before = active_days < switch
        after = ~before
        missing_before = bool(before.any() and not covered_2014[before].all())
        missing_after = bool(
            after.any() and not (covered_2021[after] | covered_2014[after]).all()
        )
        if missing_before or missing_after:
            candidates.append(str(row.ts_code))
    return sorted(set(candidates))


def _dividend_worker(code: str) -> tuple[str, pd.DataFrame | None, str]:
    """下载一只股票的完整分红记录。"""
    try:
        frame = tushare_src.call("dividend", ts_code=code)
        return code, pd.DataFrame() if frame is None else frame, ""
    except Exception as exc:
        return code, None, str(exc)


def dividend_bucket(code: str) -> str:
    """把股票稳定映射到 256 个分红分片，新增股票不会移动旧分片。"""
    return hashlib.sha1(code.encode("ascii")).hexdigest()[:2]


def _stock_codes() -> list[str]:
    """从已落盘股票基础表读取全部 Tushare 股票代码。"""
    path = TUSHARE_DIR / "stock_basic_all.parquet"
    if not path.exists():
        raise SourceError("缺少 stock_basic_all.parquet，请先运行 reference")
    frame = pd.read_parquet(path, columns=["ts_code"])
    codes = frame["ts_code"].dropna().astype(str).drop_duplicates()
    valid = codes[codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")]
    skipped = len(codes) - len(valid)
    if skipped:
        logger.warning("分红下载跳过 %d 个非标准历史代码", skipped)
    return sorted(valid.tolist())


def fetch_dividends(manifest: ResourceManifest | None = None) -> dict:
    """按稳定哈希桶下载分红送转记录。"""
    state = manifest or ResourceManifest()
    codes = _stock_codes()
    failures: list[str] = []
    groups: dict[str, list[str]] = {}
    for code in codes:
        groups.setdefault(dividend_bucket(code), []).append(code)
    for bucket, batch in sorted(groups.items()):
        path = EVENT_DIR / "dividend" / f"bucket={bucket}.parquet"
        key = f"dividend:{bucket}"
        if state.is_complete(key, path):
            continue
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        with cf.ThreadPoolExecutor(max_workers=tushare_src.CONCURRENCY) as pool:
            for code, frame, error in pool.map(_dividend_worker, batch):
                if error:
                    errors.append(f"{code}: {error}")
                elif frame is not None and not frame.empty:
                    frames.append(frame)
        if errors:
            message = "；".join(errors)
            state.update(
                key, status="failed", api="dividend", partition=bucket,
                path=_manifest_path(path), rows=0, error=message[:500],
            )
            failures.append(key)
            logger.error("dividend 分片 %s 失败：%s", bucket, message)
            continue
        if frames:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "The behavior of DataFrame concatenation with empty or all-NA entries "
                        "is deprecated"
                    ),
                    category=FutureWarning,
                )
                merged = pd.concat(frames, ignore_index=True)
            merged = merged.drop_duplicates().reset_index(drop=True)
        else:
            merged = pd.DataFrame()
        if not merged.empty:
            _write_parquet_atomic(path, merged)
        low, high = _date_range(merged)
        state.update(
            key, status="complete", api="dividend", partition=bucket,
            path=_manifest_path(path) if not merged.empty else None,
            rows=len(merged), min_date=low, max_date=high, codes=len(batch), error=None,
        )
        logger.info("dividend 分片 %s 完成：%d 只股票，%d 行", bucket, len(batch), len(merged))
    return {"failures": failures, **state.summary()}


def fetch_all(end: str, start_year: int = DEFAULT_START_YEAR) -> dict:
    """下载核心财务、事件与历史行业资源。"""
    manifest = ResourceManifest()
    periods = quarter_periods(start_year, end)
    sw_history = fetch_sw_all(manifest)
    if sw_history["failures"]:
        sw_repair = {
            "failures": [], "empty": [], "skipped": "行业历史分片仍有失败，先重跑补片"
        }
    else:
        repair_codes = sw_repair_candidates(end=f"{end[:4]}-{end[4:6]}-{end[6:]}")
        sw_repair = fetch_sw_repair(repair_codes, manifest)
    results = {
        "period_resources": fetch_period_resources(periods, manifest),
        "sw_history": sw_history,
        "sw_repair": sw_repair,
        "dividend": fetch_dividends(manifest),
    }
    results["manifest"] = manifest.summary()
    return results


def status() -> dict:
    """返回核心资源下载状态。"""
    return ResourceManifest().summary()
