# V5 取数与清洗入库的运行手册

对应需求：`AiRecord/progress/V5.md`（V5-1 ~ V5-4）。
本文说明全链路怎么跑、断点怎么续、产物与日志落在哪里，以及本次运行遇到的问题。

## 一、数据与产物的位置

全部落在项目根目录 `/home/hanwen/quant` 下：

| 路径 | 内容 |
|---|---|
| `data/source/tushare/*.parquet` | 一次性参考表：交易日历、股票基础表、曾用名、停牌记录、指数权重 |
| `data/source/tushare/raw/<接口>/year=YYYY/month=MM.parquet` | 按交易日批量拉取的行情分片（接口：daily / adj_factor / stk_limit / daily_basic） |
| `data/clean/daily/year=YYYY/part.parquet` | 清洗后的日线长表 |
| `data/clean/{calendar,anomalies,suspend_segments}.parquet`、`data/clean/manifest.json`、`data/clean/build_stats.json` | 衍生表与产物清单 |
| `data/databases/quant.duckdb` | DuckDB 单文件数据库（每次建库删库重建） |
| `logs/core/data/<模块名>/起止时间.log` | 各模块日志，128MB 轮转 |
| `logs/core/data/ops_v5/` | 串联运行的新阶段日志（取数/清洗/建库的 stdout 与运行时序） |
| `AiRecord/evidence/ops_v5/` | 已完成验收对应的历史阶段证据 |

注意：`core/data/quant/` 目录是**代码包**（读取层 `quant.load_panel` 等），不是数据目录。

## 二、全链路怎么跑

依赖安装与版本锁定见 `docs/core/ops/dependencies.md`；运行前先执行
`.venv/bin/pip check`。

```bash
# 取数 + 清洗 + 建库 + 自检，一条命令跑完（后台挂机见下）
bash tools/v5/run_full_chain.sh

# 取数与 extra 已完成，只想重跑后段
bash tools/v5/run_clean_build.sh

# 只看进度
bash tools/v5/watch.sh          # 立刻打一次看板
bash tools/v5/poll.sh 290       # 等 290 秒再打一次
```

单步命令（等价于串联脚本里的阶段）：

```bash
.venv/bin/python -m scripts.sources.fetch_all daily --start 20000101 --end 20260928
.venv/bin/python -m scripts.sources.fetch_all extra
.venv/bin/python -m scripts.sources.fetch_all industry --end 20261015
.venv/bin/python -m scripts.sources.fetch_all gbbq
.venv/bin/python -m scripts.clean.pipeline
.venv/bin/python -m scripts.build_db.build
.venv/bin/python -m scripts.sources.fetch_all status
```

`gbbq` 只下载 3.4MB 的公司行动辅助文件，固定提交和 SHA-256；已有正确文件时不会联网。
只更新公司行动账本时可直接运行：

```bash
.venv/bin/python -m scripts.build_db.corporate_actions
.venv/bin/python tools/v5/audit_corporate_actions.py
```

建库会把财务三表与财务指标核心字段导入 `fin.*`，并生成 `meta.financial_manifest`。
建库前会检查财务 parquet 文件和磁盘空间；空间不足时不会删除旧库。
临时排查可用 `python -m scripts.build_db.build --skip-financial`，正式建库不要使用。

后台挂机（`setsid` 让进程脱离当前终端）：

```bash
setsid nohup bash tools/v5/run_full_chain.sh > logs/core/data/ops_v5/chain_boot.log 2>&1 < /dev/null &
```

停止：`pkill -f "fetch_all dail[y]"`（模式里用 `[y]`，否则会匹配到自身 shell）。

## 三、串联脚本的失败门槛

- 预检：先拉 `20260901~20260928` 约 20 个交易日 × 四接口，验证 token、中转服务与权限；
  预检出现失败交易日则整条链路中止，不浪费几小时。
- 取数：每个接口统计里的 `failed` 必须为 0；出现失败交易日就重跑一轮补拉（最多 3 轮），
  已完成的交易日会自动跳过。
- `extra` / 清洗 / 建库：退出码非 0 立即中止链路，不带着残缺数据往下走。
- 收尾：`fetch_all status` 与 `pytest -q` 结果都写进 `logs/core/data/ops_v5/`。

## 四、诊断脚本

| 脚本 | 用途 |
|---|---|
| `tools/v5/diag_coverage.py` | 各接口分片的交易日覆盖、起止年份（判断是否有接口缺历史） |
| `tools/v5/scan_codes.py` | 扫描原始分片与参考表里无法归一化的股票代码 |
| `tools/v5/probe_legacy_code.py` | 追查某个 T 前缀历史遗留代码（如 `T00018.SH`）的归属 |
| `tools/v5/apply_fix_20260928_2.py` | 20260928-2 修复的定点替换脚本，保留用于复现与追溯 |
| `tools/v5/audit_core_resources.py` | 审计财务、事件与 SW2014/SW2021 行业原始分片 |
| `tools/v5/audit_sw_industry.py` | 按年份审计行业覆盖、缺失和 carry-forward |

## 五、已知限制（实测）

- `stk_limit` 只有 `20070104` 之后的数据（4797/6480 个交易日），
  2007 年之前清洗会走「按板块 + 日期 + ST + 退市整理期」的规则兜底价；
  这不是漏拉，是服务端没有该区间的数据。
- Tushare 对「代码被历史主体占用过」的情况加 `T` 前缀：
  `adj_factor` 里上港集箱（`stock_basic` 记为 `T600018.SH`，20000719~20061020）
  被写成 `T00018.SH`（5 位数字），而 `daily` 里这段行情直接挂在 `600018.SH` 下。
  按「在市优先」的口径，上港集箱并入在市的 `SH600018`（上港集团，20061026 上市），
  其行情全部落在上市日之前，清洗的上市期过滤会排除这 1514 行；
  输入层对这些行记 WARNING 日志后跳过，不再中断整条链路。
