# 清洗模块（core/data/scripts/clean）

## 负责功能

把 Tushare 落地分片加工成可直接做因子与回测的日线长表，落盘到 `data/clean/daily/year=YYYY/part.parquet`。

## 模块划分

| 文件 | 职责 |
|---|---|
| `sources.py` | 分片读取、交易日历、股票基础表、曾用名区间、停牌记录 |
| `io.py` | 把分片组装成按年的二维面板（股票 × 交易日） |
| `adjust.py` | 复权乘数、前收、日收益、除权日识别 |
| `lifecycle.py` | 上市/退市/停牌状态、ST 与退市整理期区间 |
| `limits.py` | 涨跌停比例规则、四舍五入取整、官方价优先、可交易性 |
| `validate.py` | 基于 DuckDB 的 A1~A8 校验与异常表 |
| `manifest.py` | 产物清单、校验结果与已知限制 |
| `pipeline.py` | 编排：按年处理，跨年只传上一年的最后收盘价 |

## 为什么按年切分

全 A 自 2000 年起约 3600 万个「股票 × 交易日」单元格，整块载入需要十几 GB 内存；
按年后单年只有百万级单元格，峰值内存约 700 MB。

## 口径要点

1. **复权**：`后复权价 = 原始价 × factor`；`前复权价 = 原始价 × factor / factor_T`，
   `factor_T` 取该股最后一个有效交易日的因子。三张同构表在入库阶段生成，清洗层只存原始价与因子。
2. **收益**：主口径 `ret = change / (close - change)`，与交易所涨跌幅一致；因子口径用于交叉校验。
3. **停牌**：官方 `suspend_d` 与「上市期内无行情」两个条件取并集。
4. **涨跌停**：优先用官方 `stk_limit`；缺失时才按「板块 + 日期 + ST + 退市整理期」推算。
5. **行集**：只落 `[list_date, delist_date]` 区间内的行，上市前与退市后不铺空行，
  区间由 `ref.stock_basic` 表达。
6. **历史遗留代码**：Tushare 对「代码曾被另一家历史主体占用过」的情况加 `T` 前缀。
   例如上港集箱在 `stock_basic` 里是 `T600018.SH`（20000719~20061020），
   在 `adj_factor` 里写作 `T00018.SH`，而在市的 `600018.SH`（上港集团，20061026 上市）
   沿用同一代码。按「在市优先」合并到 `SH600018` 后，历史主体的行情全部落在上市日之前，
   清洗的上市期过滤会排除它们，产物里不会出现。

## 被谁调用 / 调用了谁

- 被 `core/data/scripts/build_db/build.py` 读取产物建库；
- 依赖 `core/data/quant/codes.py`（代码归一与板块判定）、`core/data/scripts/config.py`、`core/data/scripts/errors.py`、`core/data/scripts/logging_conf.py`；
- `pipeline.py` 是唯一编排入口，其余模块之间不互相调用。

## 校验口径（A1~A8，2026-09-28 全量实测）

- **A1**：`status` 无 NULL、取值只有 `trading` / `suspended`，
  且每只股票在 `[list_date, delist_date] ∩ 数据区间` 内的交易日数等于实落行数
  （`A1_calendar_gaps` 统计空洞）。全量实测：空洞 0，覆盖 5906 只股票。
- **A2**：除权日 = 官方涨跌额与「今收 − 上一**有效**交易日收盘价」对不上
  （容差 `1e-4 × 收盘价`）。昨收必须跨过停牌日，否则停牌期间除权的个股在复牌首日标不出来。
  因子比较同样要跨过停牌日：Tushare 的 `adj_factor` 在停牌期间照样逐日更新（`daily` 却没有行），
  按日历行比会把停牌日的因子变化误判成跳变。按「上一有效交易日」口径实测：
  `A2_ex_without_jump` 只剩 150 行 = 102 行个股首日行 + 48 行小额现金分红
  （因子只保留 3 位小数，相对价格小于约 0.05% 的分红在因子上看不到）。
- **A3**：官方涨跌额口径与因子口径收益差异超过 `1e-4` 的占比。
  实测 28,715 行（0.165%），集中在复牌 / 重组 / 换股等事件日——
  这类日子 Tushare 的 `change` 与 `adj_factor` 本身就是两种口径，不是实现缺陷，
  逐条可在 `data/clean/anomalies.parquet` 里按 `stock_code + trade_date` 定位
  （异常表与 A2/A3/A4 校验使用同一比较口径，四类条数与校验值逐条相等，
  可用 `tools/v5/audit_checks.py` 自查）。
- **A4 / A6 / A7** 见 `data/clean/manifest.json`；**A5** 对应 `year_stats` 里的规则/官方涨停价一致率
  （2000~2006 无官方数据不参与比较；2007 年 98.09% 最低，2008 年 99.55%，
  2009~2025 年 99.73%~99.97%，2026 年（至 9-28）99.04%）。

## 修改时注意

1. `build_status` 依赖 `reference` 的行序与 `codes` 严格一致，改 `build_reference` 时不能破坏顺序；
2. Tushare 的 `vol` 是手、`amount` 是千元，落盘前必须换算成股与元；
3. 除权日识别要用「昨收」而不是官方 `pre_close`，否则永远判定不出除权；
4. 跨年的 `prev_close` 由 `carry_close` 传递，不要改成每年独立计算。
5. 原始分片里会出现无法归一的 T 前缀历史遗留代码，`sources.normalize_codes` 记 WARNING 后跳过；
   不要改回逐行严格归一，否则整条链路会中断；
6. 昨收走 `adjust.forward_fill_close` + `adjust.shift_down`，即「上一有效交易日收盘价」，
   跨年种子与 `carry_close` 也都取最后一个有效交易日。不要把昨收改回单纯的下移一行：
   停牌处是 NaN，会让停牌期间除权的个股在复牌首日漏标。
