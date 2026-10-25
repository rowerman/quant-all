"""quant 数据读写库用法速查：A 股日线、指数行情与未来收益标签。

用途：分块浏览，或逐块复制进 ipynb 探索。每块以 ``# %%`` 开头（VSCode 里可直接逐块运行），
紧跟一行 ``# ==== 编号 · 标题 ====`` 说明这块讲什么；块尾是一句表达式，
粘进 notebook 单元格后会自动渲染成表格。变量名逐块区分，多块粘进同一个 notebook
不会互相覆盖。

约定：
    1. 必须在项目根目录运行，``import quant`` 才找得到包；
    2. 本文件只读本地库 ``data/databases/quant.duckdb``，不触发 Tushare 取数，也不写库；
    3. 每块都依赖 B1 的环境，示例区间统一用 2024 年，全年约 130 万行，秒级返回；
    4. 写库不走读取层：整库重来用 ``.venv/bin/python -m scripts.build_db.build``。

块索引：
    B1  环境与常量                  B14 指数目录 load_index_basic
    B2  库总览：表清单与行数         B15 指数日线 load_index_daily
    B3  字段单位口径                 B16 申万行业日线 load_sw_index_daily
    B4  清洗异常明细                 B17 指数成分权重 load_index_weight
    B5  直连 SQL                    B18 标签长表 load_label
    B6  代码归一                     B19 标签宽表 load_label_wide
    B7  可用字段清单                 B20 未入库接口的报错
    B8  交易日历                     B21 组合：指数成分 → 股票池 → 行情 → 标签
    B9  股票基础表                   B22 组合：算因子 + 单日截面 IC
    B10 停牌区间                     B23 组合：IC 序列与 ICIR
    B11 日线长表 load_panel          B24 组合：可交易性过滤
    B12 三种复权口径对比             B25 组合：个股 vs 指数累计净值
    B13 日线宽表 load_panel_wide
"""

# %%
# ==== B1 · 环境与常量 ====
# 所有块都依赖这里：读连接、SQL 入口、字段常量，以及统一的示例参数。
from pathlib import Path

import pandas as pd

import quant
from quant import schema
from quant.db import get_conn, query

DB_PATH = Path("data/databases/quant.duckdb")  # 相对项目根目录；换库只改这一行
START, END = "2024-01-01", "2024-12-31"  # 统一示例区间
STOCK = "SH600000"  # 个股示例：浦发银行
INDEX = "000300.SH"  # 指数示例：沪深300
SW_INDEX = "801010.SI"  # 申万一级行业示例：农林牧渔

pd.set_option("display.max_columns", 60)  # 日线有 43 列，别被 pandas 折叠掉
pd.set_option("display.width", 200)

quant.__all__

# %%
# ==== B2 · 库总览：表清单与行数 ====
# md 是行情与标签，ref 是参照表，evt 是事件明细，meta 是版本/口径/异常。
# 表名要写全 ``schema.table``，SQL 里不带 schema 会找不到表。
tables = query(
    "SELECT schema_name AS schema, table_name AS tbl FROM duckdb_tables() ORDER BY 1, 2"
)
tables["rows"] = [
    query(f"SELECT count(*) AS n FROM {s}.{t}").n[0]
    for s, t in zip(tables.schema, tables.tbl)
]
tables

# %%
# ==== B3 · 字段单位口径：meta.data_dictionary ====
# 不知道某列的单位就看这里；库里的字段口径以这张表为准。
dictionary = query(
    """
    SELECT table_name, column_name, unit, description
    FROM meta.data_dictionary
    WHERE table_name IN ('daily_bar', 'daily_hfq', 'daily_qfq', 'fwd_return')
    ORDER BY table_name, column_name
    """
)
dictionary

# %%
# ==== B4 · 清洗异常明细：meta.anomaly ====
# 四类异常：收益口径差异、有因子跳变无除权、涨跌停价越界、有除权无因子跳变。
# 注意 trade_date 在这张表里是字符串，跨表 join 要先 CAST 成 DATE。
anomaly = query(
    """
    SELECT kind, count(*) AS n_rows,
           min(trade_date) AS first_day, max(trade_date) AS last_day
    FROM meta.anomaly GROUP BY 1 ORDER BY 2 DESC
    """
)
anomaly

# %%
# ==== B5 · 直连 SQL：get_conn / query ====
# 读取层没封装的东西（聚合、窗口、自定义 join）直接用 SQL；连接默认只读，
# 可以和其它进程并存。写库要显式传 read_only=False，正式库请统一用建库脚本重建。
with get_conn() as conn:  # 等价于 get_conn(DB_PATH, read_only=True)
    manifest = conn.execute("SELECT data_version, generated_at FROM meta.manifest").df()
    n_calendar = conn.execute("SELECT count(*) AS n FROM md.calendar").fetchone()[0]

# query() 每次新开一个只读连接；参数用 ? 占位，别做字符串拼接
recent = query(
    """
    SELECT stock_code, trade_date, close
    FROM md.daily_hfq
    WHERE stock_code = ? AND trade_date >= ?
    ORDER BY trade_date DESC LIMIT 5
    """,
    [STOCK, "2026-09-01"],
)
recent

# %%
# ==== B6 · 代码归一：normalize_code / infer_exchange / board_of ====
# 库里的股票代码统一是 SH600000 / SZ000001 / BJ920000，其它写法的原始数据
# 都要先过 normalize_code；指数代码例外，保留 Tushare 写法 000300.SH。
raw_codes = ["600000.SH", "sh.600000", "000001", "T600018.SH", "920000", "688001.SH"]
code_map = pd.DataFrame({"原始写法": raw_codes})
code_map["归一结果"] = [quant.normalize_code(code) for code in raw_codes]
code_map["交易所"] = [quant.infer_exchange(code[-6:]) for code in code_map["归一结果"]]
code_map["板块"] = [quant.board_of(code[-6:]) for code in code_map["归一结果"]]
code_map

# %%
# ==== B7 · 可用字段清单：quant.schema ====
# load_panel 的 fields 只能从这些列里选，选错会报「未知字段」。
# 口径组：ADJUST_MODES = ('raw', 'hfq', 'qfq')，对应三张同构日线表 DAILY_TABLES；
# 标签组：LABEL_HORIZONS = (1, 3, 5, 10, 15, 20)，LABEL_PRICES = ('close', 'open')。
field_groups = {
    "价格": schema.PRICE_COLUMNS,
    "量额": schema.MARKET_COLUMNS,
    "复权与收益": schema.ADJUST_COLUMNS,
    "生命周期与停牌": schema.LIFECYCLE_COLUMNS,
    "估值": schema.VALUATION_COLUMNS,
    "涨跌停": schema.LIMIT_COLUMNS,
    "可交易性": schema.TRADEABLE_COLUMNS,
    "审计来源": schema.AUDIT_COLUMNS,
}
fields = pd.DataFrame(
    [{"分组": group, "字段": name} for group, cols in field_groups.items() for name in cols]
)
fields

# %%
# ==== B8 · 交易日历：load_calendar(start, end) ====
# 所有对齐都要靠它：行情只覆盖每个交易日，日历用来生成完整交易日轴。
cal = quant.load_calendar(start=START, end=END)
cal

# %%
# ==== B9 · 股票基础表：load_stock_basic(status) ====
# status 取 "listed" / "delisted"，也接受 Tushare 的 "L" / "D" / "P"，不传就是全部；
# 非法值会直接抛 ValueError，不会悄悄返回全表。单行明细看 basic.head()。
basic = quant.load_stock_basic()
listed = quant.load_stock_basic(status="listed")
delisted = quant.load_stock_basic(status="delisted")
basic.groupby(["list_status", "board"]).size().rename("只数").reset_index()

# %%
# ==== B10 · 停牌区间：load_suspend_segments() ====
# 全表 71,319 段；is_long 是清洗阶段的长停牌标记。日线表的 status == 'suspended'
# 只能看单日，要整段剔除就用这张表。
seg = quant.load_suspend_segments()
seg.sort_values("n_days", ascending=False).head(8)

# %%
# ==== B11 · 日线长表：load_panel(...) ====
# 主力接口。fields 不传返回全部 43 列；adjust 取 raw / hfq / qfq；
# drop_suspended=True 剔掉停牌行。不做全市场研究时不要省略 start / end，
# 否则会把 1795 万行整表读进内存。
panel = quant.load_panel(
    start=START,
    end="2024-01-31",
    stocks=[STOCK],
    fields=["open", "high", "low", "close", "ret", "limit_up", "is_limit_up",
            "status", "can_buy_close"],
    adjust="hfq",
    drop_suspended=True,
)
panel.head()

# %%
# ==== B12 · 三种复权口径对比：raw / hfq / qfq ====
# 三张表结构完全一致，只有四价与 vwap 不同：
#   hfq = raw × factor；qfq = raw × factor / factor_T，factor_T 是该股最后一个交易日因子。
# 所以同一只股票在不同时间取 qfq，数值会变；做回测统一用 hfq。
raw = quant.load_panel(start=START, end="2024-01-10", stocks=[STOCK],
                       fields=["close"], adjust="raw")
hfq = quant.load_panel(start=START, end="2024-01-10", stocks=[STOCK],
                       fields=["close", "factor"], adjust="hfq")
qfq = quant.load_panel(start=START, end="2024-01-10", stocks=[STOCK],
                       fields=["close"], adjust="qfq")
compare = (
    raw.rename(columns={"close": "close_raw"})
    .merge(hfq.rename(columns={"close": "close_hfq"}), on=["stock_code", "trade_date"])
    .merge(qfq.rename(columns={"close": "close_qfq"}), on=["stock_code", "trade_date"])
)
compare["hfq/raw"] = (compare.close_hfq / compare.close_raw).round(6)
compare.head()

# %%
# ==== B13 · 日线宽表：load_panel_wide(field, ...) ====
# 透视成「交易日 × 股票」，横截面运算（排序、标准化、分组）在宽表上做最省事。
close_wide = quant.load_panel_wide("close", start=START, end=END, adjust="hfq")
close_wide.head(3)

# %%
# ==== B14 · 指数目录：load_index_basic(...) ====
# market 取 SSE / SZSE / CSI / SW，category 如「规模指数」「一级行业」。
# 目录有 10,609 行，含大量分级/货币等变体代码；只有 md.index_daily 里那 58 个
# 才真有日线，查代码时用 has_daily 过滤一下最稳。
live_codes = query("SELECT DISTINCT index_code FROM md.index_daily").index_code
index_catalog = quant.load_index_basic()
index_catalog["has_daily"] = index_catalog.index_code.isin(live_codes)
index_catalog[index_catalog.has_daily]

# %%
# ==== B15 · 指数日线：load_index_daily(index_code, ...) ====
# md.index_daily 共 58 个指数：30 宽基 + 10 风格 + 8 策略 + 10 中证全指行业。
# 399925.SZ（基本面50，到 2019-01-25）与 399984.SZ（300等权，到 2015-04-29）
# 已经停编，只有历史行情，别拿它们当基准。
idx_daily = quant.load_index_daily(INDEX, start=START, end=END)
idx_daily.head()

# %%
# ==== B16 · 申万行业日线：load_sw_index_daily(index_code, ...) ====
# 31 个申万一级行业，比 index_daily 多出行业整体 pe / pb 与市值列；
# 覆盖从 2010-04-13 起，比中证系指数短。
sw_daily = quant.load_sw_index_daily(SW_INDEX, start=START, end=END)
sw_daily.head()

# %%
# ==== B17 · 指数成分权重：load_index_weight(index_code, start, end) ====
# 12 个宽基的月末快照。con_code 是 Tushare 原写法（600050.SH），
# stock_code 是归一后的项目口径（SH600050），join 日线表必须用 stock_code。
# 旧名别名 load_index_constituents 只接 index_code，没有 start / end 参数，
# 其余行为与 load_index_weight 一致（下面 assert 了一下）。
snapshot = quant.load_index_weight(INDEX, start="2024-01-31", end="2024-01-31")
alias_full = quant.load_index_constituents(INDEX)
assert alias_full.equals(quant.load_index_weight(INDEX))
snapshot.head()

# %%
# ==== B18 · 标签长表：load_label(start, end, horizons, price, stocks) ====
# 列名 fwd_ret_close_N / fwd_ret_open_N；price="close" 指 t 日收盘买入、t+N 收盘卖出，
# price="open" 指 t+1 开盘买入、t+1+N 开盘卖出，两者持有交易日数一致。
# 停牌段用上一有效交易日价格填平，停牌行与尾部不足 N 的行留空。
label = quant.load_label(start=START, end="2024-01-31", horizons=[1, 5, 20], price="close")
label.head()

# %%
# ==== B19 · 标签宽表：load_label_wide(horizon, ...) ====
# 单档标签的「交易日 × 股票」形态，和 load_panel_wide 对齐后可直接算 IC。
fwd5_wide = quant.load_label_wide(5, start=START, end=END, price="close")
fwd5_wide.head(3)

# %%
# ==== B20 · 财务数据与行业成分：load_financial / load_financial_panel ====
# 财务原始记录按报告期读取；日面板按公告日次一交易日对齐，避免前视。
financial = quant.load_financial(
    STOCK, table="fina_indicator", start="2023-01-01", end="2024-12-31",
    fields=["report_date", "ann_date", "available_date", "roe_yearly", "netprofit_yoy"],
)
financial_panel = quant.load_financial_panel(
    "fina_indicator", start="2024-01-01", end="2024-12-31",
    fields=["roe_yearly", "netprofit_yoy"], stocks=[STOCK],
)
industry = quant.load_members("industry_member", name="银行")
quant.load_financial_manifest().head()

# %%
# ==== B21 · 组合：指数成分 → 股票池 → 行情 → 标签 ====
# 用某个月末的成分快照当股票池，是规避幸存者偏差的做法：池子里只放当时真实
# 纳入指数的股票。快照是月末的，实际研究里按月滚动取。
pool_snap = quant.load_index_weight(INDEX, start="2024-01-31", end="2024-01-31")
pool = pool_snap.stock_code.tolist()
pool_px = quant.load_panel(
    start="2024-02-01", end="2024-02-29", stocks=pool,
    fields=["close"], adjust="hfq", drop_suspended=True,
)
pool_label = quant.load_label(
    start="2024-02-01", end="2024-02-29", horizons=[5], price="close", stocks=pool,
)
pool_paired = pool_px.merge(pool_label, on=["stock_code", "trade_date"], how="left")
pool_paired.head()

# %%
# ==== B22 · 组合：算因子 + 单日截面 IC ====
# 20 日动量 → 截面 z-score → 与未来 5 日收益对齐算 Spearman 相关。
# 因子只能用 t 日及以前的信息，标签在 t 日之后，方向别弄反。
history = quant.load_panel_wide("close", start="2023-12-01", end="2024-01-31", adjust="hfq")
mom20 = history / history.shift(20) - 1
mom20_z = mom20.sub(mom20.mean(axis=1), axis=0).div(mom20.std(axis=1), axis=0)
day = mom20_z.index[-1]
factor = mom20_z.loc[day].rename("mom20_z")
target = (
    quant.load_label(start=str(day.date()), end=str(day.date()), horizons=[5], price="close")
    .set_index("stock_code")["fwd_ret_close_5"]
)
single_day = pd.concat([factor, target], axis=1).dropna()
pd.DataFrame({
    "交易日": [day.date()],
    "样本数": [len(single_day)],
    "RankIC": [round(single_day.mom20_z.corr(single_day.fwd_ret_close_5, method="spearman"), 4)],
})

# %%
# ==== B23 · 组合：IC 序列与 ICIR ====
# 逐日算截面 Rank IC，再看均值、波动、ICIR（均值 / 标准差）与胜率。
# 换别的因子时，只要把 ic_mom20 那一行替换掉即可。
ic_close = quant.load_panel_wide("close", start=START, end=END, adjust="hfq")
ic_mom20 = ic_close / ic_close.shift(20) - 1
ic_fwd5 = quant.load_label_wide(5, start=START, end=END, price="close")
ic_joined = pd.concat(
    [ic_mom20.stack().rename("factor"), ic_fwd5.stack().rename("fwd")], axis=1
).dropna()
ic_series = ic_joined.groupby(level=0).apply(
    lambda g: g["factor"].corr(g["fwd"], method="spearman")
)
pd.DataFrame({
    "交易日数": [len(ic_series)],
    "IC 均值": [round(ic_series.mean(), 4)],
    "IC 标准差": [round(ic_series.std(), 4)],
    "ICIR": [round(ic_series.mean() / ic_series.std(), 4)],
    "IC > 0 占比": [round((ic_series > 0).mean(), 4)],
})

# %%
# ==== B24 · 组合：可交易性过滤 ====
# 回测里「有行情」不等于「成交得了」：涨跌停、一字板都别当成有效成交。
# can_buy_close / can_sell_close 是清洗阶段给的保守口径，直接拿来过滤最省事。
trade = quant.load_panel(
    start="2024-01-02", end="2024-01-31",
    fields=["close", "is_st", "is_limit_up", "is_limit_down", "is_one_word_board",
            "can_buy_close", "can_sell_close", "status"],
    adjust="hfq", drop_suspended=True,
)
pd.DataFrame({
    "交易日数": [trade.trade_date.nunique()],
    "样本行数": [len(trade)],
    "ST 行": [int(trade.is_st.sum())],
    "涨停行": [int(trade.is_limit_up.sum())],
    "跌停行": [int(trade.is_limit_down.sum())],
    "一字板行": [int(trade.is_one_word_board.sum())],
    "次日可买行": [int(trade.can_buy_close.sum())],
    "次日可卖行": [int(trade.can_sell_close.sum())],
})

# %%
# ==== B25 · 组合：个股 vs 指数累计净值 ====
# 个股用后复权 ret 复利成净值，指数用收盘点位算收益，两条曲线对齐后再看超额。
nav_stock = quant.load_panel(start=START, end=END, stocks=[STOCK], fields=["ret"], adjust="hfq")
nav_index = quant.load_index_daily(INDEX, start=START, end=END)
nav_ret = pd.concat(
    [
        nav_stock.set_index("trade_date")["ret"].rename(STOCK),
        nav_index.set_index("trade_date")["close"].pct_change().rename(INDEX),
    ],
    axis=1,
).dropna()
nav = (1 + nav_ret).cumprod()
nav["超额"] = nav[STOCK] / nav[INDEX] - 1
nav.iloc[[0, -1]]
