"""Tushare 取数。

账号通过中转服务访问，必须把客户端的请求地址指到 ``http://jiaoch.site``，
否则官方接口会返回 40101「token 不对」。

中转服务实测约束：
    * 并发上限 2，超限会触发 45 秒冷却；
    * 单次调用按交易日批量取全市场，``daily`` 约 1~3 秒、``daily_basic`` 约 5 秒。

因此这里统一用「单进程 + 2 个工作线程 + 信号量限流 + 退避重试」，
并按「接口 / 年 / 月」分片落盘，进程被杀后可断点续跑。
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

from scripts.config import EXTRA_DIR, get_tushare_token, load_env_file
from scripts.errors import SourceError, SourcePermissionError, SourceRateLimitError
from scripts.logging_conf import get_logger
from scripts.sources.cache import write_frame

logger = get_logger("sources.tushare")

CONCURRENCY = 2

STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,market,exchange,curr_type,"
    "list_date,delist_date,list_status,is_hs"
)
MAX_ATTEMPTS = 6
COOLDOWN_SECONDS = 46.0
RAW_DIR = EXTRA_DIR / "tushare" / "raw"

_thread_local = threading.local()
_semaphore = threading.BoundedSemaphore(CONCURRENCY)


def relay_url() -> str:
    """返回中转服务地址，未配置时回落到官方地址。"""
    load_env_file()
    return os.environ.get("TUSHARE_HTTP_URL", "").strip()


def get_client():
    """返回当前线程独占的 Tushare 客户端。

    每个线程一个客户端，避免 requests.Session 被多线程共用。
    """
    client = getattr(_thread_local, "client", None)
    if client is None:
        token = get_tushare_token()
        client = ts.pro_api(token)
        client._DataApi__token = token
        url = relay_url()
        if url:
            client._DataApi__http_url = url
        _thread_local.client = client
    return client


def _classify(message: str) -> str:
    """把服务端报错归类。

    参数:
        message: 服务端返回的错误文本。
    """
    if "并发请求过多" in message:
        return "concurrency"
    if "频率超限" in message or "每分钟" in message:
        return "rate"
    if "没有接口" in message and "访问权限" in message:
        return "permission"
    if "token" in message and "不对" in message:
        return "token"
    if "接口用法错误" in message:
        # 中转服务偶发把正常请求误报成用法错误，短暂重试通常即可恢复。
        return "usage"
    return "other"


def call(api_name: str, **params) -> pd.DataFrame:
    """带限流与退避重试地调用一个 Tushare 接口。

    参数:
        api_name: 接口名，例如 daily。
        **params: 接口参数。

    返回:
        接口返回的 DataFrame。
    """
    last_error: Exception | None = None
    attempts = 12 if api_name == "index_member" else MAX_ATTEMPTS
    for attempt in range(attempts):
        try:
            with _semaphore:
                client = get_client()
                return client.query(api_name, **params)
        except Exception as exc:  # 限流与网络抖动都靠退避消化，权限类错误直接抛出
            message = str(exc)
            kind = _classify(message)
            last_error = exc
            if kind == "permission":
                raise SourcePermissionError(f"{api_name} 无权限: {message}") from exc
            if kind == "token":
                raise SourceError(f"{api_name} token 校验失败: {message}") from exc
            if kind == "concurrency":
                wait = COOLDOWN_SECONDS
            elif kind == "usage":
                wait = 1.0
            else:
                wait = min(4 * (attempt + 1), 30)
            logger.warning(
                "%s 第%d次失败(%s)，%.0fs 后重试: %s",
                api_name,
                attempt + 1,
                kind,
                wait,
                message[:90],
            )
            time.sleep(wait)
    raise SourceRateLimitError(f"{api_name} 重试 {attempts} 次仍失败: {last_error}")


def fetch_reference() -> dict[str, Path]:
    """拉取一次性使用的参照数据：交易日历、股票基础表、曾用名。

    返回:
        表名到落盘路径的映射。
    """
    paths = {}
    calendar = call("trade_cal", exchange="SSE", start_date="19900101", end_date="20261231")
    paths["trade_cal"] = write_frame("tushare", "trade_cal", calendar)
    logger.info("trade_cal %d 行", len(calendar))

    frames = []
    for status in ("L", "P", "D"):
        # 默认返回里没有 delist_date，必须显式指定字段
        part = call("stock_basic", list_status=status, fields=STOCK_BASIC_FIELDS)
        frames.append(part.assign(list_status=status))
        logger.info("stock_basic(%s) %d 行", status, len(part))
    paths["stock_basic_all"] = write_frame("tushare", "stock_basic_all", pd.concat(frames, ignore_index=True))

    change = call("namechange", start_date="19900101", end_date="20261231")
    paths["namechange_all"] = write_frame("tushare", "namechange_all", change)
    logger.info("namechange %d 行", len(change))
    return paths


def open_trade_days(start: str = "20000101", end: str = "20261231") -> list[str]:
    """从已有交易日历缓存里取交易日列表。

    参数:
        start: 起始日期。
        end: 截止日期。
    """
    path = EXTRA_DIR / "tushare" / "trade_cal.parquet"
    if not path.exists():
        calendar = call("trade_cal", exchange="SSE", start_date="19900101", end_date="20261231")
        write_frame("tushare", "trade_cal", calendar)
    calendar = pd.read_parquet(path)
    opened = calendar[calendar["is_open"].astype(int) == 1]
    days = [d for d in opened["cal_date"].astype(str) if start <= d <= end]
    return sorted(days)


def shard_path(api_name: str, day: str) -> Path:
    """返回某个交易日的分片路径。

    参数:
        api_name: 接口名。
        day: 交易日，YYYYMMDD。
    """
    return RAW_DIR / api_name / f"year={day[:4]}" / f"month={day[4:6]}.parquet"


def load_shard(api_name: str, day: str) -> pd.DataFrame | None:
    """读取交易日所在分片。

    参数:
        api_name: 接口名。
        day: 交易日。
    """
    path = shard_path(api_name, day)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def shard_has_day(api_name: str, day: str) -> bool:
    """判断某交易日是否已经在分片里。

    参数:
        api_name: 接口名。
        day: 交易日。
    """
    frame = load_shard(api_name, day)
    if frame is None or frame.empty:
        return False
    return bool((frame["trade_date"].astype(str) == day).any())


def completed_days(api_name: str, days: list[str]) -> set[str]:
    """一次读出所有已落盘的交易日。

    按月份分组读取分片，避免每个交易日都去打开同一个文件。

    参数:
        api_name: 接口名。
        days: 待检查的交易日列表。
    """
    months: dict[tuple[str, str], list[str]] = {}
    for day in days:
        months.setdefault((day[:4], day[4:6]), []).append(day)
    done: set[str] = set()
    for (year, month), group in months.items():
        path = RAW_DIR / api_name / f"year={year}" / f"month={month}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["trade_date"])
        present = set(frame["trade_date"].astype(str))
        done.update(day for day in group if day in present)
    return done


def append_day(api_name: str, day: str, frame: pd.DataFrame) -> None:
    """把单个交易日的返回追加进所在月份的分片。

    参数:
        api_name: 接口名。
        day: 交易日。
        frame: 该交易日的返回数据。
    """
    path = shard_path(api_name, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["trade_date"].astype(str) != day]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_parquet(path, index=False)


def _worker(api_name: str, day: str, params: dict) -> tuple[str, pd.DataFrame | None, str]:
    """在子线程里拉取一个交易日。

    参数:
        api_name: 接口名。
        day: 交易日。
        params: 附加参数。

    返回:
        (交易日, 数据或 None, 错误信息)。
    """
    try:
        frame = call(api_name, trade_date=day, **params)
        if frame is None:
            frame = pd.DataFrame(columns=["trade_date", "ts_code"])
        return day, frame, ""
    except Exception as exc:
        return day, None, str(exc)


def fetch_by_day(api_name: str, days: list[str], extra: dict | None = None) -> dict:
    """按交易日批量拉取某个接口，跳过已落盘的日子。

    用固定的 2 个工作线程并发，主线程负责落盘，避免并发写文件。

    参数:
        api_name: 接口名。
        days: 交易日列表。
        extra: 附加的固定参数。

    返回:
        统计信息，含总数、跳过数、成功数、失败数。
    """
    params = dict(extra or {})
    done_days = completed_days(api_name, days)
    pending = [day for day in days if day not in done_days]
    logger.info("%s 待拉取 %d 天 / 共 %d 天", api_name, len(pending), len(days))
    done = 0
    failed: list[str] = []
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        tasks = [(api_name, day, params) for day in pending]
        for day, frame, error in pool.map(lambda item: _worker(*item), tasks):
            if error:
                logger.error("%s %s 拉取失败: %s", api_name, day, error[:100])
                failed.append(day)
            else:
                append_day(api_name, day, frame)
                done += 1
            if done and done % 100 == 0:
                rate = done / max(time.time() - started, 1e-6)
                logger.info(
                    "%s 进度 %d/%d，%.2f 天/秒，剩余约 %.0f 分钟",
                    api_name,
                    done,
                    len(pending),
                    rate,
                    (len(pending) - done) / max(rate, 1e-6) / 60,
                )
    return {
        "api": api_name,
        "total": len(days),
        "pending": len(pending),
        "done": done,
        "failed": len(failed),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def fetch_daily_batch(apis: list[str], days: list[str]) -> list[dict]:
    """顺序拉取多个按交易日批量取的接口。

    参数:
        apis: 接口名列表。
        days: 交易日列表。
    """
    stats = []
    for api_name in apis:
        stats.append(fetch_by_day(api_name, days))
        logger.info("%s 完成: %s", api_name, stats[-1])
    return stats


def fetch_suspend(end: str = "20261015") -> Path:
    """按年拉取官方停牌记录。

    参数:
        end: 截止日期，YYYYMMDD。
    """
    frames = []
    end_year = int(end[:4])
    for year in range(2000, end_year + 1):
        frame = call(
            "suspend_d", suspend_type="S",
            start_date=f"{year}0101", end_date=min(f"{year}1231", end),
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
        logger.info("suspend_d %d 年 %d 行", year, 0 if frame is None else len(frame))
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return write_frame("tushare", "suspend_d", merged)


def fetch_index_weights(indices: dict[str, str], start: str = "20050101") -> Path:
    """按指数拉取历史成分权重。

    参数:
        indices: 指数代码到名称的映射。
        start: 起始日期。
    """
    frames = []
    for code, name in indices.items():
        for year in range(int(start[:4]), 2027):
            frame = call(
                "index_weight",
                index_code=code,
                start_date=f"{year}0101",
                end_date=f"{year}1231",
            )
            if frame is not None and not frame.empty:
                frames.append(frame.assign(index_name=name))
        logger.info("index_weight %s 完成", code)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return write_frame("tushare", "index_weight", merged)


def merge_index_daily(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """把新拉取的指数日线并入已有缓存。

    参数:
        existing: 已落盘的指数日线；None 表示没有旧文件。
        incoming: 本次拉取的指数日线，允许只包含部分指数。
    """
    frames = [frame for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return (
        merged.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )


def fetch_index_daily(indices: dict[str, str], start: str = "20000101",
                      end: str = "20261015", *, merge_existing: bool = True) -> Path:
    """按指数拉取日线行情。

    指数日线单次返回上限足够覆盖全历史（实测沪深300 自 2005 年起 5280 行），
    因此每个指数只请求一次，不做分片与断点续跑。

    参数:
        indices: 指数代码到名称的映射。
        start: 起始日期。
        end: 截止日期。
        merge_existing: True 时与已落盘文件合并，允许只刷新部分指数。
    """
    frames = []
    for code, name in indices.items():
        frame = call("index_daily", ts_code=code, start_date=start, end_date=end)
        if frame is None or frame.empty:
            raise SourceError(f"指数日线 {code} 返回空数据，请检查代码与权限")
        frames.append(frame.assign(index_name=name))
        logger.info("index_daily %s(%s) %d 行", code, name, len(frame))
    incoming = pd.concat(frames, ignore_index=True)
    existing = None
    path = EXTRA_DIR / "tushare" / "index_daily.parquet"
    if merge_existing and path.exists():
        existing = pd.read_parquet(path)
    merged = merge_index_daily(existing, incoming)
    return write_frame("tushare", "index_daily", merged)


# 指数目录来源；CSI 不加类别过滤会被服务端截断在 8000 行，必须按类别逐个拉
INDEX_BASIC_MARKETS = ("SSE", "SZSE", "SW")
INDEX_BASIC_CSI_CATEGORIES = (
    "规模指数", "风格指数", "策略指数", "综合指数", "行业指数", "主题指数",
    "债券指数", "基金指数", "多资产指数", "期货指数", "其他指数",
)

# index_weight 单次调用实测只返回区间内最近的 7000 行，超限会静默丢掉早期数据
INDEX_WEIGHT_HARD_CAP = 7000
INDEX_WEIGHT_CALL_BUDGET = 6000
# 分片覆盖自身月份跨度低于这个比例就不纳入合并产物
INDEX_WEIGHT_MIN_COVERAGE = 0.98


def fetch_index_basic() -> Path:
    """拉取指数目录。

    CSI 来源的指数数量远超单次返回上限，因此按类别分多次拉取后按代码去重。
    """
    frames = []
    for market in INDEX_BASIC_MARKETS:
        frame = call("index_basic", market=market)
        if frame is not None and not frame.empty:
            frames.append(frame.assign(market=market))
            logger.info("index_basic %s %d 条", market, len(frame))
    for category in INDEX_BASIC_CSI_CATEGORIES:
        frame = call("index_basic", market="CSI", category=category)
        if frame is not None and not frame.empty:
            frames.append(frame.assign(market="CSI"))
        logger.info("index_basic CSI/%s %d 条", category, 0 if frame is None else len(frame))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"])
    merged = merged.reset_index(drop=True)
    logger.info("index_basic 合计 %d 条", len(merged))
    return write_frame("tushare", "index_basic", merged)


def fetch_sw_index_daily(indices: dict[str, str], start: str = "20000101",
                         end: str = "20261015") -> Path:
    """按指数拉取申万行业指数日线。

    申万指数不在 ``index_daily`` 的覆盖范围内，必须走 ``sw_daily``；
    单指数一次调用即可取回全历史（实测约 0.3 秒）。

    参数:
        indices: 指数代码到名称的映射。
        start: 起始日期。
        end: 截止日期。
    """
    frames = []
    for code, name in indices.items():
        frame = call("sw_daily", ts_code=code, start_date=start, end_date=end)
        if frame is None or frame.empty:
            raise SourceError(f"申万行业日线 {code}({name}) 返回空数据")
        frames.append(frame.drop(columns=["name"], errors="ignore").assign(index_name=name))
        logger.info("sw_daily %s(%s) %d 行", code, name, len(frame))
    merged = pd.concat(frames, ignore_index=True)
    return write_frame("tushare", "sw_index_daily", merged)


def _to_date(value: str) -> date:
    """把 YYYYMMDD 形式的日期转成 date。

    参数:
        value: 日期字符串。
    """
    text = str(value)[:8]
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _add_months(day: date, months: int) -> date:
    """按月平移日期，返回目标月的 1 号。

    参数:
        day: 基准日期。
        months: 平移月数，可为负。
    """
    total = day.year * 12 + (day.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def _weight_step(members: int) -> int:
    """按成分数选一个能整除 12 的窗口步长，保证分片始终对齐日历。

    参数:
        members: 该指数最近一期的成分数。
    """
    for step in (12, 6, 4, 3, 2, 1):
        if step * max(members, 1) <= INDEX_WEIGHT_CALL_BUDGET:
            return step
    return 1


def _coverage_ratio(frame: pd.DataFrame) -> float:
    """计算分片覆盖自身月份跨度的比例，用来判断有没有区间被漏拉。

    参数:
        frame: 单只指数的成分权重。
    """
    if frame.empty:
        return 0.0
    periods = frame["trade_date"].nunique()
    span = _month_span(str(frame["trade_date"].min()), str(frame["trade_date"].max()))
    return periods / max(span, 1)


def _month_span(start: str, end: str) -> int:
    """两个 YYYYMMDD 之间相隔的自然月数（含首尾）。

    参数:
        start: 起始日期。
        end: 结束日期。
    """
    first, last = _to_date(start), _to_date(end)
    return (last.year - first.year) * 12 + (last.month - first.month) + 1


def _probe_member_count(code: str, anchor: date) -> int:
    """取最近一期的成分数，用来估算分片步长。

    参数:
        code: 指数代码。
        anchor: 最近的月份。
    """
    for back in range(13):
        start = _add_months(anchor, -back)
        end = _add_months(anchor, 1 - back) - timedelta(days=1)
        frame = call("index_weight", index_code=code,
                     start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if frame is not None and not frame.empty:
            return int(frame.groupby("trade_date").size().max())
    return 300


def _weight_call(code: str, start: date, end: date, retry_empty: bool) -> pd.DataFrame:
    """调用一次成分权重接口，空结果重试。

    服务端会间歇性返回空表，不重试就会把「整段没拉到」当成「该区间没有数据」，
    直接丢掉一整年。

    参数:
        code: 指数代码。
        start: 窗口起始日。
        end: 窗口结束日。
        retry_empty: 是否对空结果重试；指数尚未发布之前的窗口不必重试。
    """
    attempts = 3 if retry_empty else 1
    for attempt in range(attempts):
        frame = call("index_weight", index_code=code,
                     start_date=start.strftime("%Y%m%d"),
                     end_date=end.strftime("%Y%m%d"))
        if frame is not None and not frame.empty:
            return frame
        if attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def _pull_weight_window(code: str, start: date, end: date,
                        retry_empty: bool = True) -> pd.DataFrame:
    """拉取一个日期窗口的成分权重，返回行数触顶时对半拆分重试。

    参数:
        code: 指数代码。
        start: 窗口起始日。
        end: 窗口结束日。
        retry_empty: 空结果是否重试。
    """
    frame = _weight_call(code, start, end, retry_empty)
    if frame.empty:
        return frame
    if len(frame) < INDEX_WEIGHT_HARD_CAP or start >= end:
        return frame
    logger.warning("%s %s~%s 返回触顶 %d 行，对半拆分重拉", code, start, end, len(frame))
    middle = start + (end - start) / 2
    left = _pull_weight_window(code, start, middle)
    right = _pull_weight_window(code, middle + timedelta(days=1), end)
    return pd.concat([left, right], ignore_index=True)


def _periods_in(frame: pd.DataFrame, start: date, end: date) -> int:
    """统计已收集的数据落在某个窗口内的快照期数。

    参数:
        frame: 已收集的成分权重。
        start: 窗口起始日。
        end: 窗口结束日。
    """
    if frame.empty:
        return 0
    low, high = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    inside = frame[(frame["trade_date"] >= low) & (frame["trade_date"] <= high)]
    return int(inside["trade_date"].nunique())


def _pull_index_weight(code: str, start: str, end: str,
                       existing: pd.DataFrame | None = None,
                       max_passes: int = 4) -> pd.DataFrame:
    """按自适应步长拉取单只指数的历史成分权重。

    中转服务对 ``index_weight`` 的返回不稳定：同一个区间多次请求会给出不同数量的快照
    （整段为空、12 期只给 1 期，甚至比上一次还少）。因此这里做两件事：
    与已有分片取并集，只增不减；再对仍然缺期数的窗口反复补拉。

    参数:
        code: 指数代码。
        start: 起始日期，YYYYMMDD。
        end: 截止日期，YYYYMMDD。
        existing: 已落盘的分片数据，作为起点合并。
        max_passes: 最多补拉轮数。
    """
    first, last = _to_date(start), _to_date(end)
    members = _probe_member_count(code, last)
    step = _weight_step(members)
    windows: list[tuple[date, date]] = []
    cursor = first
    while cursor <= last:
        windows.append((cursor, min(_add_months(cursor, step) - timedelta(days=1), last)))
        cursor = _add_months(cursor, step)

    keep = ["index_code", "con_code", "trade_date", "weight"]
    if existing is not None and not existing.empty:
        collected = existing[keep].copy()
        collected["trade_date"] = collected["trade_date"].astype(str)
    else:
        collected = pd.DataFrame(columns=keep)

    for attempt in range(max_passes):
        improved = False
        # 指数发布之前的窗口只在前两轮尝试，避免为没上市的年份反复空跑
        anchor = None if collected.empty else str(collected["trade_date"].min())
        for window_start, window_end in windows:
            if attempt >= 2 and anchor and window_end.strftime("%Y%m%d") < anchor:
                continue
            if _periods_in(collected, window_start, window_end) >= step:
                continue
            frame = _pull_weight_window(code, window_start, window_end,
                                        retry_empty=attempt > 0)
            if frame.empty:
                continue
            before = len(collected)
            collected = pd.concat([collected, frame], ignore_index=True)
            collected = collected.drop_duplicates(subset=["index_code", "con_code", "trade_date"])
            if len(collected) > before:
                improved = True
        if not improved and attempt > 0:
            break

    if collected.empty:
        return pd.DataFrame(columns=keep)
    merged = collected.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
    logger.info("%s 成分权重 %d 期 %d 行（成分 %d，步长 %d 个月，覆盖 %.1f%%）",
                code, merged["trade_date"].nunique(), len(merged),
                members, step, _coverage_ratio(merged) * 100)
    return merged


def fetch_index_weight_parts(indices: dict[str, str], start: str = "20050101",
                             end: str = "20261015") -> Path:
    """按指数拉取历史成分权重，分片落盘后再合并。

    单次调用只返回区间内最近的 7000 行，所以先按成分数决定步长，触顶时再对半拆分。
    每个指数落一个分片文件，重跑时已经拉到截止日附近的指数直接跳过。

    起点不按指数发布日裁剪：实测中证500 在公布日之前就有历史权重数据，
    裁剪会丢掉这部分。

    参数:
        indices: 指数代码到名称的映射。
        start: 起始日期。
        end: 截止日期。
    """
    parts_dir = EXTRA_DIR / "tushare" / "index_weight_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for code, name in indices.items():
        path = parts_dir / f"{code}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        frame = _pull_index_weight(code, start, end, existing=existing)
        if frame.empty:
            logger.warning("%s 成分权重没取到数据", code)
            continue
        frame = frame.assign(index_name=name)
        frame.to_parquet(path, index=False)
        # 覆盖不足的分片留在缓存里等以后补，但不进合并产物，避免半截数据流向研究层
        coverage = _coverage_ratio(frame)
        if coverage < INDEX_WEIGHT_MIN_COVERAGE:
            logger.warning("%s 成分权重只覆盖 %.1f%%，不纳入合并产物", code, coverage * 100)
            continue
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["trade_date"] = merged["trade_date"].astype(str)
    return write_frame("tushare", "index_weight", merged)
