"""外部数据取数编排。

用法：
    python -m scripts.sources.fetch_all reference              # 交易日历/股票基础表/曾用名
    python -m scripts.sources.fetch_all daily --start 20000101 --end 20261015
    python -m scripts.sources.fetch_all extra                  # 停牌记录 + 指数成分
    python -m scripts.sources.fetch_all index                  # 目录 + 指数日线 + 申万行业 + 成分权重
    python -m scripts.sources.fetch_all index-daily --codes H00300.CSI,H00905.CSI \
        --start 20261015 --end 20261015                       # 只刷新指定指数日线
    python -m scripts.sources.fetch_all core --end 20261015    # 财务/事件/历史行业原始资源
    python -m scripts.sources.fetch_all industry --end 20261015  # SW2014/SW2021 行业成员与缺口修复
    python -m scripts.sources.fetch_all gbbq                   # 公司行动辅助数据（固定版本）
    python -m scripts.sources.fetch_all refresh --start 20260929 --end 20261015
    python -m scripts.sources.fetch_all status                 # 查看落盘进度
    python -m scripts.sources.fetch_all all
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from scripts.config import EXTRA_DIR, ensure_dirs
from scripts.errors import SourceError
from scripts.logging_conf import get_logger
from scripts.sources import tushare_src
from scripts.sources import core_resources
from scripts.sources import gbbq
from scripts.sources.cache import write_frame

logger = get_logger("sources.fetch_all")

DAILY_APIS = ("daily", "adj_factor", "stk_limit", "daily_basic")

# 指数日线清单：宽基与规模、风格、常用策略、中证全指行业
INDEX_CODES = {
    # 宽基与规模
    "000001.SH": "上证指数",
    "000009.SH": "上证380",
    "000010.SH": "上证180",
    "000016.SH": "上证50",
    "000132.SH": "上证100",
    "000300.SH": "沪深300",
    "H00300.CSI": "沪深300全收益",
    "000510.SH": "中证A500",
    "000680.SH": "科创综指",
    "000688.SH": "科创50",
    "000698.SH": "科创100",
    "000852.SH": "中证1000",
    "000903.SH": "中证A100",
    "000905.SH": "中证500",
    "H00905.CSI": "中证500全收益",
    "000906.SH": "中证800",
    "000985.CSI": "中证全指",
    "399001.SZ": "深证成指",
    "399005.SZ": "中小100",
    "399006.SZ": "创业板指",
    "399019.SZ": "创业200",
    "399088.SZ": "深创100",
    "399102.SZ": "创业板综",
    "399106.SZ": "深证综指",
    "399303.SZ": "国证2000",
    "399310.SZ": "国证A50",
    "399311.SZ": "国证1000",
    "399330.SZ": "深证100",
    "399673.SZ": "创业板50",
    "399850.SZ": "深证50",
    "899050.BJ": "北证50",
    "932000.CSI": "中证2000",
    # 风格：成长 / 价值
    "000028.SH": "180成长",
    "000029.SH": "180价值",
    "399370.SZ": "国证成长",
    "399371.SZ": "国证价值",
    "399372.SZ": "大盘成长",
    "399373.SZ": "大盘价值",
    "399374.SZ": "中盘成长",
    "399375.SZ": "中盘价值",
    "399376.SZ": "小盘成长",
    "399377.SZ": "小盘价值",
    # 常用策略
    "000015.SH": "红利指数",
    "000922.CSI": "中证红利",
    "399324.SZ": "深证红利",
    "399295.SZ": "创价值",
    "399296.SZ": "创成长",
    "399645.SZ": "100低波",
    # 下面两个已经停编，只保留历史行情，新研究别再拿它们当基准
    "399925.SZ": "基本面50",     # 2009-02-26 ~ 2019-01-25
    "399984.SZ": "300等权",      # 2011-08-02 ~ 2015-04-29
    # 中证全指一级行业
    "000986.SH": "全指能源",
    "000987.SH": "全指材料",
    "000988.CSI": "全指工业",
    "000989.SH": "全指可选",
    "000990.CSI": "全指消费",
    "000991.SH": "全指医药",
    "000992.SH": "金融地产",
    "000993.SH": "全指信息",
    "000994.CSI": "全指通信",
    "000995.CSI": "全指公用",
}

# 需要历史成分权重的宽基。中证全指不取：成分是全市场，按月拉要两百多次请求且用途小。
INDEX_WEIGHT_CODES = {
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000510.SH": "中证A500",
    "000688.SH": "科创50",
    "000852.SH": "中证1000",
    "000903.SH": "中证A100",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "399303.SZ": "国证2000",
    "899050.BJ": "北证50",
    "932000.CSI": "中证2000",
}


def run_reference() -> None:
    """拉取一次性参照数据。"""
    tushare_src.fetch_reference()


def run_daily(start: str, end: str) -> None:
    """按交易日拉取四个行情类接口。

    参数:
        start: 起始日期，YYYYMMDD。
        end: 截止日期，YYYYMMDD。
    """
    days = tushare_src.open_trade_days(start, end)
    logger.info("交易日 %d 天：%s ~ %s", len(days), days[0], days[-1])
    for api_name in DAILY_APIS:
        stats = tushare_src.fetch_by_day(api_name, days)
        logger.info("%s 完成 %s", api_name, stats)
        print(stats, flush=True)


def sw_level1() -> dict[str, str]:
    """取申万一级行业代码与名称，清单以接口为准而不是写死。"""
    frame = tushare_src.call("index_classify", level="L1", src="SW2021")
    if frame is None or frame.empty:
        raise SourceError("index_classify 未返回申万一级行业")
    return {row["index_code"]: row["industry_name"] for row in frame.to_dict("records")}


def run_extra(end: str = "20261015") -> None:
    """拉取停牌记录与指数历史成分权重。"""
    path = tushare_src.fetch_suspend(end=end)
    logger.info("停牌记录写入 %s", path)
    path = tushare_src.fetch_index_weight_parts(INDEX_WEIGHT_CODES, end=end)
    logger.info("指数权重写入 %s", path)


def run_index(end: str = "20261015") -> None:
    """补齐指数数据：目录 -> 精选日线 -> 申万行业日线 -> 历史成分权重。"""
    tushare_src.fetch_index_basic()
    path = tushare_src.fetch_index_daily(INDEX_CODES, end=end)
    logger.info("指数日线写入 %s", path)
    path = tushare_src.fetch_sw_index_daily(sw_level1(), end=end)
    logger.info("申万行业日线写入 %s", path)
    path = tushare_src.fetch_index_weight_parts(INDEX_WEIGHT_CODES, end=end)
    logger.info("指数权重写入 %s", path)


def run_index_daily(start: str, end: str, codes: list[str]) -> None:
    """只刷新指定指数的日线，并与已有缓存合并。

    参数:
        start: 起始日期，YYYYMMDD。
        end: 截止日期，YYYYMMDD。
        codes: 指数代码列表，必须都在 ``INDEX_CODES`` 清单内。
    """
    unknown = sorted(set(codes) - set(INDEX_CODES))
    if unknown:
        raise SourceError(f"未知指数代码：{unknown}")
    selected = {code: INDEX_CODES[code] for code in codes}
    path = tushare_src.fetch_index_daily(selected, start=start, end=end, merge_existing=True)
    logger.info("指数日线定向刷新写入 %s", path)
    print({"index_daily": path, "codes": codes, "start": start, "end": end}, flush=True)


def run_core(end: str) -> None:
    """拉取财务、事件和申万历史成员原始资源。"""
    # 分红按股票清单分片，先刷新参照表以包含最近上市/退市股票。
    run_reference()
    result = core_resources.fetch_all(end=end)
    print(result, flush=True)
    if result["manifest"]["failed"]:
        raise SourceError(f"核心资源仍有 {result['manifest']['failed']} 个失败分片，请直接重跑 core")


def run_industry(end: str) -> None:
    """只拉取申万历史成员和逐股缺口修复。"""
    result = core_resources.fetch_sw_all()
    if result["failures"]:
        print({"industry": result, "repair": "skipped"}, flush=True)
        raise SourceError(
            f"行业资源仍有 {len(result['failures'])} 个失败分片，请直接重跑 industry"
        )
    codes = core_resources.sw_repair_candidates(
        end=f"{end[:4]}-{end[4:6]}-{end[6:]}"
    )
    repair = core_resources.fetch_sw_repair(codes)
    print({"industry": result, "repair_candidates": len(codes), "repair": repair}, flush=True)
    failures = list(repair["failures"])
    if failures:
        raise SourceError(f"行业资源仍有 {len(failures)} 个失败分片，请直接重跑 industry")


def run_refresh(start: str, end: str) -> None:
    """刷新参照、日频行情、停牌与指数资源。"""
    run_reference()
    run_daily(start, end)
    path = tushare_src.fetch_suspend(end=end)
    logger.info("停牌记录写入 %s", path)
    run_index(end)


def run_status() -> None:
    """统计各接口已落盘的交易日数与行数。"""
    for api_name in DAILY_APIS:
        directory = tushare_src.RAW_DIR / api_name
        if not directory.exists():
            print(f"{api_name}: 无数据", flush=True)
            continue
        parts = sorted(directory.glob("year=*/month=*.parquet"))
        days = 0
        rows = 0
        for part in parts:
            frame = pd.read_parquet(part, columns=["trade_date"])
            days += frame["trade_date"].nunique()
            rows += len(frame)
        print(f"{api_name}: 月份分片 {len(parts)} 个，交易日 {days} 天，行数 {rows}", flush=True)
    print(f"core_resources: {core_resources.status()}", flush=True)
    try:
        digest = gbbq.verify_file()
    except SourceError as error:
        print(f"gbbq: 不可用（{error}）", flush=True)
    else:
        print(f"gbbq: 校验通过 {digest[:12]}…", flush=True)


def main(argv: list[str]) -> int:
    """命令行入口。

    参数:
        argv: 命令行参数。
    """
    ensure_dirs()
    action = argv[1] if len(argv) > 1 else "status"
    start = argv[argv.index("--start") + 1] if "--start" in argv else "20000101"
    end = argv[argv.index("--end") + 1] if "--end" in argv else "20261015"

    if action == "reference":
        run_reference()
    elif action == "daily":
        run_daily(start, end)
    elif action == "extra":
        run_extra(end)
    elif action == "index":
        run_index(end)
    elif action == "index-daily":
        codes_arg = argv[argv.index("--codes") + 1] if "--codes" in argv else ""
        codes = [item.strip() for item in codes_arg.split(",") if item.strip()]
        if not codes:
            print("index-daily 需要 --codes，例如 H00300.CSI,H00905.CSI", file=sys.stderr)
            return 2
        run_index_daily(start, end, codes)
    elif action == "core":
        run_core(end)
    elif action == "gbbq":
        print(gbbq.download(), flush=True)
    elif action == "industry":
        run_industry(end)
    elif action == "refresh":
        run_refresh(start, end)
    elif action == "status":
        run_status()
    elif action == "all":
        run_reference()
        run_daily(start, end)
        run_extra(end)
    else:
        print(f"未知动作: {action}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
