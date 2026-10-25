# 待补数据清单

当前因子库只用了库里已有的日线、估值、指数与生命周期数据，所以做不了
基本面、资金面、事件类因子。下表列出要补的数据、能解锁的因子，以及取数时要注意什么。

取数脚本在 `core/data/scripts/sources/`：新增接口时在 `tushare_src.py` 里加拉取函数、
在 `fetch_all.py` 里加子命令，落到 `data/source/tushare/`，再走清洗与建库。
**硬约束不变：全局并发上限 2 个请求，同一时刻只跑一个取数进程。**

## 一、财务数据（已接入）

V46 已把 `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip` 的
核心字段导入 `fin.*`，并生成公告日时点读取；财务因子族 `fundamental` 已补 18 个因子。
时点口径为 `trade_date > available_date`，公告日次一交易日才可用于信号。
`forecast` / `express` / `disclosure_date` / `fina_audit` 仍只保留原始数据，尚未入库。

| 接口 | 内容 | 能解锁的因子 | 注意事项 |
|---|---|---|---|
| `fina_indicator` | ROE、毛利率、净利率、资产负债率、同比增速 | 盈利能力、质量因子 | 必须按 `ann_date`（公告日）对齐，不能用 `end_date`，否则前视 |
| `income` | 营业收入、净利润、扣非净利润 | 营收/利润增速、SUE | 单季与累计口径要区分，`report_type` 要筛 |
| `balancesheet` | 总资产、净资产、存货、应收账款 | 应计、资产周转、杠杆 | 银行与非银口径差异大 |
| `cashflow` | 经营现金流、资本开支 | 现金流质量、应计异象 | 与净利润配合算应计 |
| `fina_mainbz` | 主营业务构成 | 业务集中度、行业切换 | 披露不完整，覆盖年代较晚 |

正式表按 `(stock_code, report_date, available_date, report_type)` 去重，读取层提供
`quant.load_financial`（单股票原始记录）和 `quant.load_financial_panel`（日频时点面板）。
因子构建通过 `ASOF LEFT JOIN` 直接对齐，避免每个因子各写一遍时点逻辑。

## 二、行业与板块成分

V22 下载的 `index_member_all` 分片实际是最新分类快照，不含历史调出记录，不能用于时点研究。
V44 已补齐 SW2014 历史成员、SW2021 分片和逐股修复记录，生成按 `2021-12-13` 切换的
`ref.sw_industry_classify` / `ref.sw_industry_member`；SW2021 缺失时沿用最近 SW2014 行业并标记
`carry_forward`。行业中性化仍需在因子评估链单独接入。

| 接口 | 内容 | 能解锁的因子 | 注意事项 |
|---|---|---|---|
| `index_member` | SW2014/SW2021 行业成分（含历史变更） | 时点行业中性化、行业动量、行业相对强度 | 使用正式表的规范区间按 `in_date <= trade_date <= out_date` 对齐，并用 `classification` 区分版本；原始重叠区间不能直接连接 |
| `concept_detail` / `ths_member` | 概念板块成分 | 概念热度、板块联动 | 尚未下载入库；概念成分变动频繁，注意生效日期 |
| `stock_basic` 的 `industry` | 已入库的静态行业 | 粗粒度行业因子 | 退市股基本缺失，仅作兜底 |

## 三、资金与交易行为

| 接口 | 内容 | 能解锁的因子 | 注意事项 |
|---|---|---|---|
| `hsgt_top10` / `hk_hold` | 陆股通持股与十大成交 | 北向持股比例、北向净流入 | 覆盖年代较晚，早期为空 |
| `moneyflow` / `moneyflow_dc` | 大中小单资金流向 | 资金流强度、主力净流入 | 口径与交易所披露规则变化有关 |
| `margin_detail` / `margin` | 融资融券余额与买入额 | 杠杆情绪、融券做空压力 | 仅两融标的，非标的为空 |
| `top_list` / `top_inst` | 龙虎榜买卖明细 | 游资参与度、机构席位净买 | 只在异动日有数据，稀疏 |
| `stk_holdernumber` | 股东户数 | 筹码集中度 | 季报频率 |
| `stk_holdertrade` | 股东增减持 | 减持压力、增持信号 | 按 `ann_date` 对齐 |
| `repurchase` | 回购 | 回购力度 | |
| `stk_limit` 已有 | 涨跌停价 | 已有的涨跌停因子 | |

## 四、预期与情绪

| 接口 | 内容 | 能解锁的因子 | 注意事项 |
|---|---|---|---|
| `report_rc` / `forecast` / `express` | 分析师预测、业绩预告/快报 | 预期修正、超预期（SUE） | 预告数据按 `ann_date` 对齐，注意修正与首次披露 |
| `news` / `anns` | 新闻与公告 | 事件驱动因子 | 需要文本处理，工程量另算 |
| `stk_mins` | 分钟行情 | 日内动量、尾盘效应、真实 VWAP 偏离、隔夜/日内收益分解 | 数据量极大（全 A 分钟级数百 GB），建议先按指数成分股与最近 3~5 年试水 |

## 五、其它

| 接口 | 内容 | 用途 |
|---|---|---|
| `index_dailybasic` | 指数估值（PE/PB/股息率） | 指数层面的估值分位与择时 |
| `broker_recommend` | 券商月度金股 | 情绪类因子 |
| `fund_share` / `fund_daily` | ETF 份额与行情 | 份额变动、资金流向 |
| `index_classify` / `index_member` | 指数分类与成分 | 补全 `ref.index_weight` 没覆盖的指数 |

## 补数之后要做的三件事

1. 在 `core/data/quant/schema.py` 里加对应的列定义或 schema 常量，保持「唯一列定义来源」；
2. 在建库阶段生成 point-in-time 对齐视图，并在 `docs/core/data/database.md` 里写清口径；
3. 在 `core/factors/` 里新增族（例如 `fundamental`）或给现有族加因子，
   用 `extra_sources` 引用新的数据源（`core/factors/sources.py` 里加一条即可）。
