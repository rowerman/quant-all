# 取数模块（core/data/scripts/sources）

## 负责功能

经中转服务从 Tushare 拉取全 A 股日线与清洗所需参照数据；公司行动另使用固定版本的
gbbq 辅助文件补齐配股与送转条款。所有原始产物分片落盘并支持断点续跑。

## 关键约束（实测）

| 项 | 实测结果 |
|---|---|
| 接口地址 | 必须指向 `http://jiaoch.site`，直连 `api.tushare.pro` 会返回 `40101 token 不对` |
| 并发上限 | 2，超限触发 45 秒冷却，因此固定「单进程 + 2 线程 + 信号量」 |
| 单请求耗时 | `daily` 约 1.5 秒、`adj_factor` 约 3 秒、`daily_basic` 约 3 秒、`stk_limit` 约 0.8 秒 |
| 交易日跨度 | 2000-01-04 ~ 2026-10-15，共 6488 个交易日 |
| 接口覆盖差异 | `stk_limit` 服务端只有 2007-01-04 之后的数据（4805/6488 天）；其余三个接口 6488 天全覆盖 |

全量拉取与清洗入库的具体命令、失败门槛、诊断脚本见 `docs/core/ops/v5_runbook.md`；
指数相关数据的自查用 `tools/data_audit/audit_indexes.py`（检查日线空洞、停编指数、成分权重覆盖与归一命中率）。

## 模块划分

| 文件 | 职责 |
|---|---|
| `tushare_src.py` | 客户端构建（含中转地址）、限流退避、按交易日批量拉取、按「接口/年/月」分片落盘与续跑检测 |
| `core_resources.py` | 财务、事件、分红与申万历史成员的分页、分片、断点和资源清单 |
| `gbbq.py` | 固定版本公司行动辅助数据下载、SHA-256 校验与来源清单 |
| `fetch_all.py` | 编排：`reference` / `daily` / `extra` / `index` / `index-daily` / `core` / `gbbq` / `refresh` / `status` |
| `cache.py` | 一次性参照表的读写 |

## 分片布局

```
data/source/tushare/
  trade_cal.parquet          交易日历
  stock_basic_all.parquet    全市场股票（L/P/D 三态，含退市日）
  namechange_all.parquet     证券简称变更历史
  suspend_d.parquet          官方停牌记录
  index_daily.parquet        精选指数日线（含沪深300/中证500全收益）
  sw_index_daily.parquet     31 个申万一级行业指数日线
  index_basic.parquet        指数目录（四个来源按类别拉全）
  index_weight.parquet       12 个宽基的历史成分权重（合并产物）
  index_weight_parts/<指数>.parquet   成分权重分片，续跑与增量补拉的单元
  raw/<接口>/year=YYYY/month=MM.parquet
  raw_financial/<接口>/period=YYYYMMDD.parquet
  raw_events/<接口>/period=YYYYMMDD.parquet
  raw_events/dividend/batch=NNNN.parquet
  raw_industry/sw2014/classify.parquet
  raw_industry/sw2014/history_members/l3_code=*.parquet
  raw_industry/sw2021/classify.parquet
  raw_industry/sw2021/history_members/l3_code=*.parquet
  raw_industry/sw2021/members/l3_code=*.parquet   index_member_all 最新开放区间补充
  raw_industry/repair/ts_code=*.parquet           index_member 逐股跨版本修复
  manifests/core_resources.json
data/source/gbbq/
  gbbq.parquet               公司行动辅助数据（固定提交与 SHA-256）
  manifest.json              来源、提交、大小和摘要
```

只补指定指数的日线时使用：

```bash
.venv/bin/python -m scripts.sources.fetch_all index-daily \
  --codes H00300.CSI,H00905.CSI --start 20261015 --end 20261015
```

`fetch_index_daily` 会与已有 `index_daily.parquet` 按 `(ts_code, trade_date)` 合并，
新值覆盖旧值，未指定指数保持不变；因此可以安全地只刷新单只基准的缺失交易日。

## 核心研究资源

限时 token 下先把不可替代的原始历史数据落盘。行业与财务正式入库已经接入建库流程：

```bash
.venv/bin/python -m scripts.sources.fetch_all core --end 20261015
.venv/bin/python -m scripts.sources.fetch_all status
```

`core` 拉取四张财务主表、财务指标、预告、快报、披露日、审计意见、分红送转和
申万历史行业成员。季度类接口用 `limit/offset` 拉全市场，分红按股票代码稳定哈希分片。
行业成员分别遍历 SW2014 的 227 个三级行业和 SW2021 的 `is_pub=1` 有效三级行业，调用
`index_member(index_code=...)`，按分类和三级行业分片到 `history_members/`。成功分片写入
`manifests/core_resources.json`，重跑只补失败或缺失分片；空响应记为 `empty`，不伪装成有文件。
`index_member_all` 结果只是最新分类快照，不能单独当历史使用；建库仅把 `members/` 作为开放区间
补充，与 `index_member` 的有限历史区间一起规范化。较新调入日和有限历史区间优先，不能把快照
直接回填覆盖更具体的历史记录。快照中的停用三级行业仍可映射到有效一级行业，因此一级行业约束
保留这些记录。2016~2026 仍有缺口的股票会再用 `index_member(ts_code=...)` 逐股补拉，落在
`repair/`；SW2021 完全没有记录时，建库沿用最近已知一级行业并标记 `carry_forward`。
最早一条记录晚于上市日时，按该记录向前补到上市日并标记 `backfill_earliest_*`。
2026-07 以后新上市的少量股票目前尚无任何 Tushare 申万记录，审计列为 `unclassified`；
不使用 `stock_basic.industry` 静态分类兜底，进入行业上限目标时仍严格失败。
财务修订记录原样保留。建库时把利润表、资产负债表、现金流量表和财务指标导入 `fin.*`，
核心字段按 `(stock_code, report_date, available_date)` 去重；时点读取使用
`trade_date > available_date`，公告日次一交易日才可见，不能按报告期直接前填。

分红实施记录在建库阶段归一到 `ref.corporate_action`：同一股票同一除权日去重，
现金分红保留税前 `cash_div_tax`，送转保留 `stk_div`。经典配股的每股比例和配股价
Tushare 没有完整结构化字段，因此公司行动以 gbbq 类别 1 为主数据源：

```bash
.venv/bin/python -m scripts.sources.fetch_all gbbq
.venv/bin/python -m scripts.build_db.corporate_actions
```

gbbq 固定到提交 `d16be984…`，文件 SHA-256 必须等于 `0ca271a5…`。Tushare 分红表
用于补缺和交叉校验；两边都缺失时，只有总股本与复权因子同日按相同比例变化才用
`source=share_count` 兜底。冲突和未解释跳变见 `tools/data_audit/audit_corporate_actions.py`。

只补行业数据的命令：

```bash
.venv/bin/python -m scripts.sources.fetch_all industry --end 20261015
```

## 指数数据

`python -m scripts.sources.fetch_all index` 一个命令把指数相关数据补齐，顺序是：
目录 → 指数日线 → 申万行业日线 → 历史成分权重。

**指数日线**：与按交易日批量拉的四个行情接口不同，单个指数一次调用就能取回全历史
（实测沪深300 自 2002-01-04 起 6001 行，与交易日历逐日对齐、零缺失），
因此精选指数各请求一次，不做分片、不做断点续跑，直接覆盖写
`index_daily.parquet`。清单写在 `fetch_all.INDEX_CODES`，分四组：

| 组别 | 个数 | 说明 |
|---|---|---|
| 宽基与规模 | 30 | 上证指数、深证成指、上证50/180/380/100、沪深300、中证A500/A100/500/800/1000/2000、中证全指、科创综指/50/100、创业板指/综/50/200、深证100/50/深创100、深证综指、中小100、国证A50/1000/2000、北证50 |
| 风格 | 10 | 180成长/价值、国证成长/价值、大盘/中盘/小盘 × 成长/价值 |
| 常用策略 | 8 | 红利指数、中证红利、深证红利、创价值、创成长、100低波，以及**已停编**的基本面50（2009-02-26~2019-01-25）与 300等权（2011-08-02~2015-04-29） |
| 中证全指行业 | 10 | 能源、材料、工业、可选、消费、医药、金融地产、信息、通信、公用 |

另含 `H00300.CSI`（沪深300全收益）和 `H00905.CSI`（中证500全收益），只拉行情，
成分池仍使用 `000300.SH` / `000905.SH` 的历史权重。

**申万行业日线**：申万指数不在 `index_daily` 覆盖范围内，必须走 `sw_daily`。
只拉一级行业 31 个（清单由 `index_classify(level='L1', src='SW2021')` 生成，不写死），
单指数一次调用取全历史，落 `sw_index_daily.parquet`。历史从 **2010-04-13** 起，
比中证系指数短，这是服务端限制。

**指数目录**：`index_basic` 按来源拉，SSE/SZSE/SW 各一次全量就够了，但 CSI 不加
`category` 过滤会被截断在 8000 行，因此按 11 个类别逐个拉，最后按 `ts_code` 去重。

## 历史成分权重

这部分接口行为最坑，动手前务必读这段。

**两个硬约束（实测）**：

1. 单次调用只返回区间内**最近的 7000 行**。沪深300 全历史一次拉只会得到
   2025-01 之后的 24 期，早期数据全被丢掉却没有任何报错。
2. 返回**不稳定且不单调**：同一个区间多次请求会给出不同数量的快照——整段为空、
   12 期只给 1 期，甚至这次比上次还少。单个指数一次请求拿不全，是常态。

**应对**：

| 机制 | 做法 |
|---|---|
| 分片步长 | 按成分数选一个能整除 12 的步长，保证「期数 × 成分数」≤ 6000：≤500 只按年、800~1000 只按半年、2000 只按季、更大按月 |
| 触顶拆分 | 单次返回 ≥ 7000 行视为触顶，把窗口对半拆开重拉 |
| 空结果重试 | 空结果重试 3 次；指数发布之前的窗口不重试，避免空跑 |
| 并集增量 | 与已落盘分片取并集，**只增不减**，重跑只会更全 |
| 覆盖度门槛 | 分片覆盖自身月份跨度不足 98% 就不纳入合并产物，避免半截数据流向研究层 |

产物覆盖 12 个宽基：上证50、沪深300、中证A500、科创50、中证1000、中证A100、
中证500、中证800、深证成指、创业板指、国证2000、北证50。

几处实测结论：

- **中证1000 的旧数据是残缺的**：老实现按年拉，12 期 × 1000 只 = 12000 行触顶，
  只留下 87 期（应为 143 期）。新实现修到 143 期。
- **中证500 / 中证800 的权重早于其官方发布日**（分别从 2005-01-31 起，
  发布日是 2007-01-15），起点**不要按发布日裁剪**，否则会丢历史。
- **中证2000 未纳入产物**：拉取时缺 2024-04、2024-06 两期，覆盖率 94.6%，
  按门槛被过滤掉；分片留在缓存里，等 token 恢复后重跑同一条命令即可补齐。

## 被谁调用 / 调用了谁

- 被 `core/data/scripts/clean/sources.py` 通过读取分片的方式间接依赖；
- 只依赖 `core/data/scripts/config.py`（token 与环境变量）、`core/data/scripts/errors.py`、`core/data/scripts/logging_conf.py`。

## 修改时注意

1. `stock_basic` 默认不返回 `delist_date`，必须显式传 `fields`；
2. 同一时刻只允许一个进程在拉取，否则必然触发并发上限；
3. 退市清单里存在 `T600018.SH` 这类历史代码，归一后会与在市股撞车，必须按「在市优先」去重；
4. 续跑检测按月分片整块读取，不要改回「每个交易日打开一次文件」。
5. 某些接口在早年会返回空（如 `stk_limit` 2007 年之前），这类交易日不算失败、
   也不会被 `completed_days` 记为已完成；判断「拉没拉完」要看统计里的 `failed`，
不要用交易日覆盖数是否等于 6488 来卡。详见 `docs/core/ops/v5_runbook.md`。
