# 工程支撑

本章说明配置、命令行脚本、测试、日志、错误、Docker、运维工具和维护记录。它们不是业务
模型本身，但决定了项目能否复现、排障和持续演进。

## 1. 技术栈

| 技术 | 用途 |
|---|---|
| pytest | 单元、联合、真实数据和 API 测试 |
| JSON 配置 | 训练、挖掘目录、评估计划和阈值 |
| Docker Compose | 独立 RD-Agent Worker |
| shell 脚本 | 服务启动和批处理入口 |
| Python CLI (`argparse`) | 数据、因子、模型、回测和审计命令 |
| 统一错误类 | 稳定错误码和 HTTP 映射 |
| 时间戳日志 | 按模块、按 128 MB 轮转 |
| Git / Markdown 记录 | 计划、变更、差异、版本和进度 |

## 2. 提供功能

- 提供统一路径、运行目录和本地数据组织。
- 用 JSON 固定模型、挖掘目录、质量门槛和运行预算。
- 提供数据、因子、模型、挖掘、回测和监测站的命令行入口。
- 用分层测试覆盖单元、联合、真实数据、API 和浏览器流程。
- 用结构化日志和稳定错误码支持跨模块排障。
- 用 Docker 隔离 RD-Agent Worker 和模型密钥。
- 提供性能、审计、浏览器和数据完整性工具。
- 用计划、进度、变更和差异记录维护需求可追溯性。

## 3. 配置体系

### 3.1 根目录环境变量

本地 `.env` 至少包含：

```text
TUSHARE_TOKEN=<本地 Token>
TUSHARE_HTTP_URL=<Tushare 中转地址>
FACTOR_DEEPSEEK_API=<Worker 模型 Key>
```

`.env` 在 Git 忽略范围内。DeepSeek Key 只映射到 Worker 容器，不进入主服务。

### 3.2 挖掘配置

| 文件 | 内容 |
|---|---|
| `configs/mining/catalog.json` | 84 个允许字段、34 个算子、单位和时点 |
| `configs/mining/evaluation.json` | 质量门禁与相关性阈值 |
| `configs/mining/orchestrator.json` | 轮次、候选、LLM、深度和失败率预算 |
| `configs/mining/worker.json` | Worker 版本、模型、超时和批量上限 |

修改这些文件会改变冻结上下文或证据哈希，必须让旧运行和审批失效。

### 3.3 模型配置

| 文件 | 内容 |
|---|---|
| `train_ridge.json` | 线性后端、alpha 网格和预处理 |
| `train_lightgbm_ensemble.json` | LightGBM 五种子集成 |
| `train_sklearn_hist_gbdt.json` | 直方图 GBDT |
| `development_evaluation_v85.json` | 候选、参照、经济门和风险门 |
| `holdout_windows.json` | 已使用和待使用的最终留出窗口 |
| `final_holdout_2024_2026.json` | 冻结的最终留出候选 |

训练规格决定 `model_id`；评估计划决定 `evaluation_id`；任何内容变化都不能覆盖旧产物。

## 4. 如何使用

公共 `pyproject.toml` 把物理目录映射成原有包名：

```text
core/data/scripts -> scripts
core/factors      -> factors
research/evaluation/calculate -> calculate
research/modeling/alpha       -> alpha
trading/backtest              -> backtest
apps/monitor/backend          -> backend
research/mining               -> mining
research/mining_worker        -> mining_worker
```

因此命令行入口保持：

### 4.1 数据

```bash
.venv/bin/python -m scripts.sources.fetch_all status
.venv/bin/python -m scripts.clean.pipeline
.venv/bin/python -m scripts.build_db.build
.venv/bin/python -m scripts.build_db.labels
```

### 4.2 因子

```bash
.venv/bin/python -m scripts.build_factors.build
.venv/bin/python -m scripts.build_factors.validate
.venv/bin/python -m scripts.build_factors.catalog --write-docs
```

### 4.3 模型

```bash
.venv/bin/python -m scripts.build_alpha.build_dataset
.venv/bin/python -m scripts.build_alpha.build_folds
.venv/bin/python -m scripts.build_alpha.train
.venv/bin/python -m scripts.build_alpha.predict
.venv/bin/python -m scripts.build_alpha.evaluate
.venv/bin/python -m scripts.build_alpha.registry --rebuild
.venv/bin/python -m scripts.build_alpha.monitor
.venv/bin/python -m scripts.build_alpha.lifecycle
```

### 4.4 回测

```bash
.venv/bin/python -m backtest --strategy mom20 --start 2024-01-01 --end 2024-09-30
```

### 4.5 服务

```bash
bash tools/monitor/serve.sh
```

## 5. 测试体系

### 5.1 测试位置

| 位置 | 内容 |
|---|---|
| `tests/` | 根级数据、因子、模型、挖掘联合与真实数据测试 |
| `research/evaluation/calculate/tests/` | 评估面板和指标 |
| `trading/backtest/tests/` | 信号、订单、账本、成本和结果 |
| `apps/monitor/backend/tests/` | API、SSE、调度、挖掘和回测 Web |

### 5.2 测试层级

1. **单元测试**：参数、类型、公式、边界和状态机。
2. **模块联合测试**：数据库到评估、训练到预测、信号到账本、API 到服务。
3. **真实数据短区间测试**：只读真实行情，使用临时输出目录。
4. **浏览器验收**：桌面、390px 移动端、SSE 断线重连和错误恢复。
5. **性能和资源测试**：因子双任务、回测双 worker 和峰值 RSS。

### 5.3 常用命令

```bash
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_project_layout.py
.venv/bin/python -m pytest -q trading/backtest/tests
.venv/bin/python -m pytest -q apps/monitor/backend/tests
```

全量测试耗时很长。项目规则是：若全量测试大部分通过，仅少数失败并修改后，只复跑与修改
相关的测试，不重复运行全部套件。

### 5.4 数据与产物隔离

- 发布测试使用临时因子目录、临时因子库和临时 `mined` 链接。
- Worker 测试使用确定性 Mock，不调用真实 DeepSeek。
- 浏览器测试使用临时候选库。
- 真实数据测试只读生产行情，不写正式因子或模型目录。

## 6. 日志体系

### 6.1 目录

```text
logs/
  core/data/
  core/factors/
  research/evaluation/
  research/modeling/
  research/mining/
  trading/backtest/
  apps/monitor/
```

每个模块再有一个子目录，例如 `calculate_panel/`。

### 6.2 轮转

`TimestampedFileHandler`：

- 文件名是首条和最后一条记录时间。
- 写入时不断重命名结束时间。
- 达到 128 MB 前切换新文件。
- 不自动删除历史日志。

### 6.3 安全

以下内容不能进日志：

- API Key、Tushare Token、Bearer Token。
- 原始 Prompt 和完整模型响应。
- 未脱敏的完整研究假设。
- 浏览器或 Worker secret。

Worker 日志只记录 job ID、轮次、候选数量、状态和错误码。

## 7. 错误体系

| 域 | 前缀 | 示例 |
|---|---|---|
| 数据与因子 | `SRC-*`、`CLEAN-*`、`FACTOR-*` | 取数、复权、建库、因子构建 |
| 因子评估 | `EVAL-*` | 参数、面板、指标和风格 |
| 模型 | `ALPHA-*` | 数据、训练、产物、登记和生命周期 |
| 挖掘 | `MINING-*` | DSL、Worker、评估、审批和发布 |
| 回测 | `BT-*` | 配置、数据、执行、快照和 provider |
| Web | `MONITOR-*` | 参数、容量、数据缺失和内部失败 |

错误类通常携带：

```text
code
message
建议 HTTP 状态
```

捕获后必须向上反馈或转换为用户错误，不能 `pass` 掉。

## 8. Docker Worker

### 8.1 镜像

[research/mining_worker/Dockerfile](../../research/mining_worker/Dockerfile)：

- 固定 Python 3.11 Debian 镜像 digest。
- 使用带哈希的 Worker `requirements.lock`。
- 只复制 Worker 代码、公共日志和 Worker 配置。
- 只暴露 8790。

### 8.2 Compose

[compose.mining.yaml](../../compose.mining.yaml)：

- 根文件系统只读。
- `/state` 挂载 Worker 作业库和 Token。
- `/logs` 挂载日志。
- `/tmp` 使用 tmpfs。
- Key 从 `.env` 进入容器环境。
- Token 使用 Compose secret。
- 健康检查请求 `/health`。

### 8.3 运维命令

```bash
docker compose --env-file .env -f compose.mining.yaml up -d --build
docker compose --env-file .env -f compose.mining.yaml ps
docker compose --env-file .env -f compose.mining.yaml logs --tail=100
docker compose --env-file .env -f compose.mining.yaml stop
docker compose --env-file .env -f compose.mining.yaml down
```

`down` 不会删除本地状态和候选数据，除非显式删除挂载目录。

## 9. 运维工具

### 9.1 监测站

| 工具 | 作用 |
|---|---|
| `tools/monitor/serve.sh` | 启动本地服务 |
| `browser_check.py` | 因子页面浏览器验收 |
| `benchmark.py` | 因子分析耗时与 RSS |
| `stress.py` | 并发、断连、背压和取消 |
| `mining_check.py` | 挖掘、审批、发布 UI 验收 |
| `preflight_concurrency` 测试 | 服务启动容量 |

### 9.2 回测

| 工具 | 作用 |
|---|---|
| `backtest_audit.py` | 费用、申报单位、现金裁剪和分配审计 |
| `backtest_check.py` | 短区间页面验收 |
| `backtest_profile.py` | 阶段耗时、峰值 RSS、单/双任务 |

### 9.3 数据审计

`tools/v5/` 包含指数、行业、公司行动、核心资源和检查项审计。运行这些工具前应确认它们
只读还是重建，避免把诊断脚本误当成无副作用命令。

## 10. 数据与产物目录

| 路径 | 生命周期 |
|---|---|
| `data/source/` | 原始数据，可在取数后重放 |
| `data/clean/` | 清洗产物，可由原始数据重建 |
| `data/databases/quant.duckdb` | 主库，可完整重建 |
| `data/factors/` | 内置因子可重建，`mined` 由发布服务管理 |
| `data/models/` | 数据集、模型、预测和运行，原则上不覆盖 |
| `data/factor_mining/` | 候选文件和评估证据 |
| `data/databases/backtest.duckdb` | 策略版本与回测结果 |
| `logs/` | 运行证据，需要人工保留/清理 |

`data/` 和 `logs/` 不进 Git，复制仓库或在新机器启动时不能假定它们已经存在。

## 11. 计划、版本和变更记录

`AiRecord/` 保存本地维护记录：

| 目录 | 内容 |
|---|---|
| `plans/` | 每次修改前的计划 |
| `changes/` | 完成后的变更总结 |
| `diffs/` | 文件行数增删，用于回滚溯源 |
| `progress/` | 每个版本 ID 的详细需求与验收 |
| `version.md` | 需求提出和解决的版本索引 |
| `evidence/` | 截图、浏览器和真实数据证据 |
| `benchmarks/` | 资源与性能原始结果 |

命名规则：

```text
YYYYMMDD-X.md
```

`X` 是当天第几次修改。一次修改通常对应一个 `fix`、`feat` 或 `refactor`。

`AiRecord/` 被 Git 忽略，只在本地用于维护和审计。文档和代码仍应以 Git 跟踪文件为准。

## 12. 修改工作流

推荐顺序：

```text
确认需求与边界
  -> 读取相关代码、配置和现有文档
  -> 写 plans/YYYYMMDD-X.md
  -> 更新 version.md 与 progress/Vxxx.md
  -> 修改代码或文档
  -> 运行分层和定向测试
  -> 对照 progress 验收
  -> 写 changes 和 diffs
  -> 检查 git diff 与工作区边界
```

## 13. 工程原则

1. 数据契约先于功能实现；跨模块传递的数据结构必须有明确字段。
2. 模块之间只依赖必须的公开接口，不共享未定义内部状态。
3. 不捕获后静默忽略错误；无法处理时向上抛出并保留错误码。
4. 不把大量逻辑堆在一个文件里；业务域、存储、契约和执行层分离。
5. 新建项目或新域时设计日志和错误体系，不能事后零散补。
6. 历史模型、预测、策略、因子和发布采用新 ID，不覆盖既有证据。
7. 写入正式产物前使用临时路径、校验和原子替换。
8. 诊断、测试和重建命令要区分只读与有副作用操作。

