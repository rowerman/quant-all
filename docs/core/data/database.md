# 数据库结构与访问方式

## 数据库

单文件 DuckDB：`data/databases/quant.duckdb`，由 `python -m scripts.build_db.build` 生成。
当前库版本 `tushare_fullA_20000104_20261015`，2026-10-18 增量加入公司行动表（26 张表，约 7.3 GB）：
日线覆盖 2000-01-04 ~ 2026-10-15 共 6488 个交易日、5913 只股票，三张日线表各 17,999,882 行。


## schema 划分

| schema | 表 | 说明 |
|---|---|---|
| `md` | `daily_bar` / `daily_hfq` / `daily_qfq` | 三张同构日线表，键为 `stock_code` + `trade_date`，只有四价与 `vwap` 口径不同 |
| `md` | `fwd_return` | 未来收益标签，行集与 `daily_hfq` 一一对应，口径见 `docs/core/data/label.md` |
| `md` | `index_daily` | 60 个指数的日线行情（含沪深300/中证500全收益） |
| `md` | `sw_index_daily` | 31 个申万一级行业指数日线，含行业估值与市值 |
| `ref` | `index_basic` | 指数目录，含来源、类别、发布日 |
| `md` | `calendar` | 交易日历 |
| `md` | `suspend_segment` | 停牌区间，含持续天数与是否长停牌 |
| `ref` | `stock_basic` | 股票基础表，含退市股的上市日与退市日 |
| `ref` | `namechange` | 证券简称变更历史，可推 ST 与退市整理期区间 |
| `ref` | `index_weight` | 指数历史成分权重 |
| `ref` | `sw_industry_classify` | SW2014 / SW2021 一级、二级、三级行业目录 |
| `ref` | `sw_industry_member` | 股票申万行业历史生效区间，含 L1/L2/L3 |
| `ref` | `corporate_action` | 分红、送转、配股和可交叉验证拆并的除权日账本；键为股票 + 除权日 |
| `fin` | `income_statement` / `balance_sheet` / `cash_flow` / `fina_indicator` | 财务三表与财务指标核心字段，键为 `stock_code` + `report_date` + `ann_date` |
| `evt` | `suspend_daily` | 官方停牌记录明细 |
| `meta` | `manifest` / `sw_industry_manifest` / `financial_manifest` / `corporate_action_manifest` / `anomaly` / `data_dictionary` | 数据版本、行业摘要、财务摘要、公司行动摘要、异常明细、字段口径 |

## 代码口径

统一为 `SH600000` / `SZ000001` / `BJ920000`（交易所前缀 + 6 位数字）。
原始数据里存在 `600000.SH`、`sh.600000`、`T600018.SH`、`600000` 等写法，全部在入库时归一。

两个例外要注意：

1. **指数代码不归一**，保持 Tushare 写法 `000300.SH`，与 `md.index_daily.index_code`、
   `ref.index_weight.index_code` 一致；
2. `ref.index_weight` 同时保留了原始列 `con_code`（`600050.SH`）和归一后的 `stock_code`
   （`SH600050`），**join 日线表要用 `stock_code`**。归一前该表与日线表 join 命中 0 行，
   归一后 984,140 行里 984,039 行能对上（99.99%），剩下 101 行是几只已不在
   `stock_basic` 里的历史代码（如 `SZ000022` 深赤湾A、`SH601313` 江南嘉捷）。

## 访问方式

完整的分块示例见项目根目录 `demo.py`（25 块，覆盖全部读取接口与常用组合用法，
每块以 `# %%` 开头，可直接粘进 ipynb）。

```python
import quant

# 长表：raw / hfq / qfq 三种口径
df = quant.load_panel(
    start="2020-01-01", end="2020-12-31",
    fields=["open", "high", "low", "close", "limit_up", "is_limit_up"],
    adjust="hfq", drop_suspended=True,
)

# 宽表：交易日 × 股票，便于横截面运算
close = quant.load_panel_wide("close", start="2020-01-01", end="2020-12-31", adjust="hfq")

quant.load_calendar()
quant.load_stock_basic(status="delisted")
quant.load_suspend_segments()
quant.load_index_weight("000300.SH")                       # 指数历史成分权重（月末快照）
quant.load_index_weight("000300.SH", start="2026-01-01")   # 也可按日期区间取
quant.load_index_daily("000300.SH", start="2026-01-01")    # 指数日线行情
quant.load_sw_index_daily("801010.SI")                     # 申万一级行业日线
quant.load_sw_industry_member(start="2020-01-01", end="2020-12-31")  # 时点行业区间
quant.load_corporate_actions(start="2024-01-01", end="2024-12-31")    # 除权日公司行动
quant.load_index_basic(market="CSI", category="规模指数")   # 指数目录
quant.load_financial("SH600000", table="fina_indicator")     # 单只股票财务记录
quant.load_financial_panel("fina_indicator", start="2024-01-01", end="2024-12-31",
                           fields=["roe_yearly", "or_yoy"]) # 公告日时点日面板
quant.load_financial_manifest()                              # 财务入库摘要

# 未来收益标签：N 取 1/3/5/10/15/20，price 取 close（t 收盘买）或 open（t+1 开盘买）
lab = quant.load_label(start="2024-01-01", end="2024-12-31", horizons=[1, 5], price="close")
fwd = quant.load_label_wide(5, start="2024-01-01", end="2024-12-31", price="close")
```

也可以直接连库：

```python
from quant.db import get_conn
with get_conn() as conn:
    conn.execute("SELECT count(*) FROM md.daily_bar").fetchone()
```

## 单位约定（详见 `meta.data_dictionary`）

| 字段 | 单位 |
|---|---|
| 四价 / 涨跌停价 / vwap / pre_close / change | 元 |
| `volume_share` | 股 |
| `amount_cny` | 元 |
| `total_mv` / `circ_mv` | 万元 |
| `turnover_rate` / `dv_ttm` | % |
| `factor` | 无量纲 |
| 财务金额（收入、资产、现金流等） | 元 |
| 财务指标（ROE、增速、毛利率等） | %，因子层再转小数 |

## 修改时注意

1. `core/data/quant/schema.py` 的 `DAILY_COLUMNS` 是唯一列定义来源，建库与读取层都引用它；
   标签周期与成交通道同理，唯一来源是 `LABEL_HORIZONS` / `LABEL_PRICES`；
2. 三张 `daily_*` 表必须保持同构，否则 `adjust` 切换会出现列不匹配；
3. 读连接默认 `read_only=True`，写入需要显式传 `read_only=False`；
4. 所有日期列都必须是 `DATE`：Tushare 原始写法的 `20240131` 存成字符串后，
   `trade_date >= '2024-01-02'` 这类过滤会因为字符串比较而静默返回空集
   （`ref.index_weight` 与 `evt.suspend_daily` 已修正）。
5. `index_member` 的来源区间存在重叠，且部分新股只出现在 `index_member_all` 最新快照。
   `ref.sw_industry_member` 合并 SW2014、SW2021、当前快照和逐股修复记录后，按分类分别按
   「最新调入日优先；同日起优先有明确调出日的历史记录，再取较早结束者」解析成唯一闭区间，
   并保留 `source_in_date/source_out_date` 供追溯。`2021-12-12` 及以前使用 SW2014，
   `2021-12-13` 起使用 SW2021；SW2021 完全没有记录时沿用最近已知行业，
   `carry_forward=true`；完全没有记录时，若最早一条记录晚于上市日，则按该记录向前补到上市日，
   `source_api=backfill_earliest_*`。`classification`、`source_l3_code` 和 `source_api` 用于追溯
   分类版本与代码别名。前后三级行业完全相同且期间没有其它记录时，填平中间空档；其余情况按最近
   已知行业延续。不要直接读取原始分片做时点连接。
6. 财务时点口径固定为 `trade_date > available_date`：`available_date` 优先取
   `coalesce(f_ann_date, ann_date)`，公告日次一交易日起才可用于信号，避免收盘后公告前视。
   财务表只保留因子和追溯所需核心字段，完整原始字段仍在 `data/source/tushare/raw_financial/`。

## 读取接口（2026-09-28 全量建库后核对）

| 接口 | 说明 |
|---|---|
| `load_panel` / `load_panel_wide` | 日线长表 / 宽表，支持 `raw` / `hfq` / `qfq` |
| `load_calendar` | 交易日历 |
| `load_stock_basic(status=...)` | 支持 `"listed"` / `"delisted"`，也接受 Tushare 的 `"L"` / `"D"` / `"P"`；非法值直接抛错 |
| `load_suspend_segments` | 停牌区间 |
| `load_index_weight(index_code, start, end)` | 指数历史成分权重（`ref.index_weight`，月末快照，带归一后的 `stock_code`）；`load_index_constituents` 为兼容旧名的别名 |
| `load_index_daily(index_code, start, end)` | 指数日线行情（`md.index_daily`，60 个指数） |
| `load_sw_index_daily(index_code, start, end)` | 申万一级行业日线（`md.sw_index_daily`，31 个行业） |
| `load_sw_industry_member(start, end, stocks, db_path=None, classification=None)` | 与日期区间相交的时点申万历史行业成员，可按分类版本过滤 |
| `load_corporate_actions(start, end, stocks)` | 除权日公司行动账本，含税前现金分红、送转比例、配股比例/价格、登记日与到账日 |
| `load_index_basic(index_code, market, category)` | 指数目录（`ref.index_basic`）；`market` 取 `SSE`/`SZSE`/`CSI`/`SW` |
| `load_label(start, end, horizons, price, stocks)` | 未来收益标签长表（`md.fwd_return`），口径见 `docs/core/data/label.md` |
| `load_label_wide(horizon, start, end, price)` | 把某一档标签透视成「交易日 × 股票」宽表 |
| `load_financial(stock_code, table, start, end, fields)` | 读取单只股票的财务原始记录；支持 `income_statement` / `balance_sheet` / `cash_flow` / `fina_indicator` |
| `load_financial_panel(table, start, end, fields, stocks)` | 按公告日时点对齐的财务日面板，默认剔除停牌行 |
| `load_financial_manifest()` | 财务入库行数、股票数、日期范围、去重与异常统计 |
| `load_members(table, name)` | `industry_member` 读取申万历史行业；`concept_member` / `block_member` 尚未入库 |
