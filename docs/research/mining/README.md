# 自动因子挖掘模块

当前实现覆盖 S2～S9 和 M1～M5：V1 领域契约、运行与候选状态机、独立候选存储、
幂等、中断恢复、受控 DSL、候选评估、RD-Agent Worker、Web 人工审核和批量正式发布；
V2 证据驱动搜索、四段时序窗口、尝试台账、受控分析工具、组合价值和统计诊断；M5
隔离代码研究沙箱及当前运行内的失败、成功和修复知识索引。V135 已复核路线图 34 项，
当前 M1～M5 均已完成；P2-02 已支持文本型 PDF 解析、金融研报分类和逐文件失败隔离，
P2-03 已支持带页内原文引用和证据状态的因子字段语义抽取，P2-04 已支持六字段筛查、
模型辅助变量映射、受控 DSL 转换和跨报告语义去重，P2-05 已接入资料驱动运行、来源
展示和 V2 审批发布闭环。P3-02 已支持在无初始假设时，根据冻结目录生成带机制类别、
数据依赖、表达复杂度和随机种子的首轮研究方向；自由探索页面入口仍由后续任务实现。

V1 仍是默认生产链路。V2 通过 `MINING_SEARCH_V2_ENABLED=true` 启用，运行停在
`review_ready` 后进入人工审核；候选批准、拒绝、整批正式发布和失败重试均已接入页面。
M5 默认关闭，只产生人工审核材料。
后续任务见[根目录进度记录](../../../FACTOR_MINING_PROGRESS.md)。

## 启动与停止

以下命令都从项目根目录 `/home/hanwen/quant` 执行。挖掘功能由 RD-Agent Worker
和监测站主服务两个进程组成；只启动监测站时页面可以打开，但提交研究假设会返回
`MINING-119`。

### 1. 准备

确认自带的 Python 环境已安装项目包：

```bash
cd /home/hanwen/quant
.venv/bin/python -m pip install -e . --no-deps
```

Worker 使用本地 Token 鉴权。首次启动前生成 Token，已有文件会直接复用，文件权限
应为 `0600`：

```bash
.venv/bin/python -c \
  'from common.paths import DATA_DIR; from mining.worker_client import ensure_worker_token; ensure_worker_token(DATA_DIR / "mining_worker" / "token")'
```

`.env` 只由 Compose 读取，至少需要提供 `FACTOR_DEEPSEEK_API`。该 Key 只映射到 Worker
容器，不传给监测站主服务，也不要导出到当前 shell。

### 2. 启动 Worker

```bash
docker compose --env-file .env -f compose.mining.yaml up -d --build
docker compose --env-file .env -f compose.mining.yaml ps
curl --fail --silent http://127.0.0.1:8790/health
```

健康接口应返回 `status=ready`、`mode=rdagent`、`rdagent_version=1.0.0` 和当前模型。
Worker 作业库与 Token 位于 `data/mining_worker/`，运行日志位于
`logs/research/mining/mining_worker_service/`。Worker 同时只执行一个作业。

### 3. 启动监测站

在两个不同终端中分别运行 Worker 和监测站。启动监测站前，必须在同一终端显式设置
Worker 地址：

```bash
cd /home/hanwen/quant
export MINING_WORKER_URL=http://127.0.0.1:8790
bash tools/monitor/serve.sh
```

默认仍使用 V1 生产编排。需要运行证据驱动 V2 实验时额外设置
`MINING_SEARCH_V2_ENABLED=true`；V2 默认关闭，运行进入待审核状态后可完成人工审批、
整批正式发布和失败重试。

监测站是前台服务，打开 http://127.0.0.1:8800/#/mining 进入挖掘页面。首次启动会创建
独立候选库 `data/databases/factor_mining.duckdb` 和 `data/factor_mining/`；只有人工批准
并执行正式发布后，才会写入 `mined` 正式因子。

设置 `MONITOR_PORT` 可以修改监测站端口，例如：

```bash
MONITOR_PORT=8801 bash tools/monitor/serve.sh
```

### 4. 验证

Worker 健康后，再检查监测站是否能读到挖掘入口：

```bash
curl --fail --silent http://127.0.0.1:8800/api/mining/options
```

返回结果中的 `worker_configured` 应为 `true`。页面提交失败时先检查：

| 错误码 | 常见原因 | 检查方式 |
|---|---|---|
| `MINING-119` | 未设置 `MINING_WORKER_URL`、Token 不一致或 Worker Key 未配置 | 检查主服务环境、Token 文件权限和 Worker `/health` |
| `MINING-103` | Worker 不可达或响应失败 | 检查容器状态、8790 端口和 Worker 日志 |
| `MINING-102` | Worker 已有作业在运行 | 等待当前作业结束，不要启动第二个 Worker |

生产路径不会回退到 Mock。Worker 不可用时应修复启动配置，不要用测试夹具代替真实挖掘。

### 5. 停止

监测站运行在前台，在对应终端按 `Ctrl+C` 停止。Worker 不再使用时执行：

```bash
docker compose --env-file .env -f compose.mining.yaml stop
```

需要同时移除 Compose 创建的容器和网络时执行：

```bash
docker compose --env-file .env -f compose.mining.yaml down
```

`down` 不会删除挂载的 Worker Token、作业库、候选数据或正式因子。

## 模块边界

```text
监测站 Web（S7）
  -> backend.mining_service 预算、运行视图、人工审批和发布入口
  -> mining.orchestrator   生成、筛选、完整评估、冻结和测试段编排
  -> mining.worker_client    Worker 作业调用与候选 S3 二次校验
       -> mining_worker      Python 3.11 独立 Worker 的纯协议和本地 HTTP 服务
  -> mining.contracts       请求、冻结上下文、候选、评估、审批和发布对象
  -> mining.catalog         字段、算子、时点与目录 schema 核对
  -> mining.expression_*    解析、安全校验、规范化、DAG 规划、单候选和批量 SQL
  -> mining.candidate_batch    同轮 DAG 批量拆分、失败隔离和单候选回退
  -> mining.candidate_materialization 单候选原子物化与 manifest 校验
  -> mining.evaluate_candidate 门禁、快速筛选与可选完整评估
  -> mining.states          运行、候选和发布状态机
  -> mining.storage         MiningStore 公共存储入口
       -> storage_runs      运行、冻结上下文、中断与恢复
       -> storage_candidates 候选定义、去重与状态迁移
       -> storage_evidence  评估报告与技术门禁
       -> storage_review    审批、测试段和发布记录
       -> storage_publications 发布批次、逐候选关联与状态收口
       -> storage_events    不可变事件与临时目录清理
       -> storage_schema    DuckDB schema 与版本

监测站 Web（V2 实验）
  -> backend.mining_search_service 搜索视图、分析工具和审核证据
  -> mining.search_orchestrator    逐轮搜索、选择、最终评估和测试段
  -> mining.free_exploration       自由探索首轮机制、字段和复杂度选择
  -> free_exploration_contracts    方向、种子和生成依据的稳定审计契约
  -> mining.storage_v2             mining_v2 轨迹、尝试、证据和工具审计
  -> storage_v2_publication        V2 发布输入、批次和候选状态事务
  -> mining.search_publication_contracts
                                     只读候选、证据和上下文发布映射

沙箱研究（M5，默认关闭）
  -> mining_sandbox.research       有限代码提案和恢复循环
  -> mining_sandbox.runtime        断网、只读、资源限制和故障终止
  -> mining_sandbox.evaluator      代码限制、因子明细和索引确定性评估
  -> mining_sandbox.feedback        结构化失败分类、稳定错误码和脱敏
  -> mining_sandbox.knowledge       当前运行的失败轨迹、成功实现和修复摘要
  -> mining_sandbox.knowledge_evidence
                                     当前运行知识的 Worker 白名单证据
  -> mining_sandbox.logging_conf    安全反馈事件日志
  -> mining_worker /v4/jobs        RD-Agent 模型生成未信任代码提案
  -> DockerSandboxRunner           断网、非 root、只读根和资源受限执行

资料驱动（P2-02 解析、P2-03 字段抽取、P2-04 筛选转换、P2-05 运行闭环）
  -> mining.research_materials     资料清单、PDF 页正文、分类、来源和解析失败
  -> mining.research_material_extractor
                                     页窗口、字段草稿、原文引用和证据状态
  -> mining_worker /v5/jobs        RD-Agent 严格 JSON 研报字段抽取
  -> mining.research_factor_screening
                                     六字段筛查、变量映射和受控 DSL 二次校验
  -> mining_worker /v6/jobs        RD-Agent 研报草案到 DSL 的批量映射建议
  -> mining.research_run_pipeline  预算内稳定准入和来源哈希绑定
  -> mining.search_orchestrator_research
                                     V2 种子、开发评估、选择、最终研究和测试段
  -> mining_v2.research_*          运行摘要、解析失败和候选页码证据
```

`mining.orchestrator` 组合研究与测试阶段。S8 由独立
`MiningPublicationService` 重建并原子提交正式 `mined` 族，详见
[正式入库](publication.md)。S6 的详细流程和恢复规则见
[挖掘编排](orchestration.md)，Web 路由、事件流和审批规则见 [挖掘 Web](web.md)。
V2 搜索候选进入现有审批和发布链路的字段边界见
[V2 到 V1 发布桥接](v2-publication-bridge.md)。

字段和算子的冻结配置位于 `configs/mining/catalog.json`。完整字段、单位和可用时点见
[字段与算子目录](catalog.md)，DSL 语法、编译和安全边界见 [受控 DSL](dsl.md)。

## 数据位置

```text
候选元数据   data/databases/factor_mining.duckdb
候选运行目录 data/factor_mining/runs/<run_id>/
候选缓存     data/factor_mining/cache/
候选因子值   data/factor_mining/runs/<run_id>/<candidate_id>/factor/year=YYYY/part.parquet
评估报告     data/factor_mining/runs/<run_id>/<candidate_id>/evaluations/
沙箱工作区   data/mining_sandbox/work/<sandbox_run_id>/
沙箱快照     data/mining_sandbox/snapshots/<snapshot_id>/
挖掘模块日志 logs/research/mining/<module>/
沙箱模块日志 logs/research/mining_sandbox/<module>/

正式 mined（仅 S8 临时验收或用户确认发布时写）
data/factors/mined -> .mined-versions/<batch_id>
data/factors/.mined-versions/<batch_id>/
data/databases/factors.duckdb
data/databases/quant.duckdb
```

候选库包含 V1 `mining` schema 和 V2 `mining_v2` schema。V1 使用 `mining_v1`，
V2 使用 `mining_v2`；两者相互隔离。发现未知 schema 版本时直接拒绝启动，普通初始化
不会把 V1 静默迁移成 V2。

## 基本用法

```python
from mining import FrozenContext, MiningRunRequest, MiningStore

store = MiningStore()
request = MiningRunRequest(
    idempotency_key="client-run-20261024-001",
    hypothesis="短期价格反转可能带来超额收益",
)
context = FrozenContext(data_fingerprint="<64位数据快照哈希>")
run = store.create_run(request, context)

store.transition_run(
    run["run_id"],
    "generating",
    event_key="generation_started",
    stage="generating",
)
```

同一 `idempotency_key` 和相同请求再次提交会返回 `reused=True` 的原运行；键相同但请求
或冻结上下文不同会返回 `MINING-105`。页面通过 `/api/mining/*` 的 S7 服务访问候选，
不得直接访问候选 DuckDB 表。HTTP 契约见 [监测站 API](../../apps/monitor/api.md)。

受控 DSL 的基本用法：

```python
from mining import validate_expression

validated = validate_expression("rank(mean(ret, 20))", direction=1)
candidate = validated.to_candidate(
    mechanism="短期反转",
    description="二十日收益的截面排名",
    steps=("计算二十日平均收益", "按交易日做截面排名"),
    discovery_round=1,
    discovery_index=1,
)
```

校验失败直接抛出 `MINING-114`，不会执行 SQL 或写入候选库。目录和行情库 schema 不一致
时抛出 `MINING-115`。

候选评估用法：

```python
from mining import CandidateEvaluationInput, evaluate_candidate

result = evaluate_candidate(
    CandidateEvaluationInput(
        run_id=run["run_id"],
        candidate_id=candidate["candidate_id"],
        candidate=definition,
        context=context,
        candidate_root=candidate_root,
        run_full=False,
    )
)
assert result.technical.stage == "technical"
```

快速筛选只计算研究期轻量 RankIC；完整评估只在 `run_full=True`、技术门禁和快速筛选
都通过时执行。`evaluate_candidate` 只返回报告和门禁对象，不修改候选状态，也不写候选
元数据库；S6 使用 `evaluate_full_candidate()` 和 `evaluate_holdout_candidate()`
补充单阶段入口，并通过 S2 存储接口推进状态。详细契约见
[候选评估与质量门禁](evaluation.md)。

Worker 用法与部署命令见 [独立 RD-Agent Worker](worker.md)。主服务只接受 Worker 返回的
未信任表达式草稿，稳定名称、依赖、回看窗口、语义哈希和候选 ID 均由 S3 重新计算。

V2 搜索契约、分层窗口和角色可见性见
[V2 搜索轨迹基础](search-v2.md)。M1～M4 已覆盖轨迹、逐轮闭环、组合价值、统计诊断
和受控分析工具。V2 仍不接管 V1 默认编排；候选审批、正式发布和失败重试已经接入
监测站 Web。

M5 的 Qlib 快照、代码提案循环、产物清单和人工提升流程见
[隔离代码研究沙箱](sandbox.md)。该能力不等同于直接运行上游 `FactorCoSTEER`，
产物不能自动成为正式候选。

已有候选库升级到 `mining_v2` 时必须停止服务并显式执行
`python -m mining.migrate_v1_to_v2 --db data/databases/factor_mining.duckdb`；
普通存储初始化不会静默迁移，V1 快照只读且不能恢复到 V2 搜索。

编排用法：

```python
from mining import MiningOrchestrator

orchestrator = MiningOrchestrator()
run = orchestrator.create_run(request)
progress = orchestrator.progress(run["run_id"])
```

默认后台执行；Worker 不可用时明确失败，不切换到 Mock。候选失败率达到 75% 且至少检查
4 个候选后停止研究和完整评估，测试段不会打开。

## 写入规则

- 所有状态写入都经过模块文件锁、DuckDB 事务和条件更新，不允许直接执行 SQL 跳状态。
- 运行首次进入 `generating` 时占用唯一执行槽；其他运行保持 `queued`。
- 进程重启后调用 `interrupt_active_runs()`，只把活动运行和候选标为 `interrupted`；
  用户显式调用 `resume_run()` 后才回到各自中断前阶段。
- 同运行内相同候选内容重复写入是幂等操作；语义相同但展示内容不同会冲突。
- 模块测试段由 `claim_holdout()` 唯一领取，同批次同证据重复调用幂等。
- 编排器只在入围候选冻结后领取测试段；测试段结果不反馈给 Worker 或用于候选重排。
- 数据、目录或正式因子清单变化时，运行和全部未终态候选原子进入 `stale`。
- 审批必须绑定当前候选内容、冻结上下文、通过的技术门禁、通过的测试段报告和证据哈希。
- 正式发布必须整批覆盖运行内全部已批准候选，批准和发布是两个独立动作。
- 发布先重建完整 `mined` 族并全量校验，再通过版本目录和原子链接切换；失败恢复旧版本。
- 普通源码因子构建不会覆盖 `mined`；已发布定义由 `definitions.json` 和因子目录库共同保存。

## 日志与错误

日志只写运行 ID、候选 ID、阶段、状态、错误码和哈希，不写 Key、完整假设或原始公式。
共享日志处理器按 128MB 轮转，文件名记录该段首末消息时间。

| 错误码 | 含义 | HTTP |
|---|---|---:|
| `MINING-100` | 未分类挖掘错误兜底 | 400 |
| `MINING-101` | 契约或参数非法 | 400 |
| `MINING-102` | 已有运行占用执行槽 | 429 |
| `MINING-103` | Worker 不可用 | 503 |
| `MINING-104` | 候选非法 | 400 |
| `MINING-105` | 幂等键或候选重复冲突 | 409 |
| `MINING-106` | 评估失败 | 422 |
| `MINING-107` | 测试段状态冲突 | 409 |
| `MINING-108` | 审批依据变化或冲突 | 409 |
| `MINING-109` | 发布失败 | 500 |
| `MINING-110` | 记录不存在 | 404 |
| `MINING-111` | 状态或并发冲突 | 409 |
| `MINING-112` | 候选存储不可用 | 503 |
| `MINING-113` | 中断运行不能恢复 | 409 |
| `MINING-114` | 表达式语法、字段、算子或安全规则非法 | 400 |
| `MINING-115` | 字段目录损坏或与实际 schema 漂移 | 500 |
| `MINING-116` | Worker 作业超时 | 504 |
| `MINING-117` | Worker 作业取消或重启中断 | 409 |
| `MINING-118` | Worker 响应或本地请求协议非法 | 502 |
| `MINING-119` | Worker 鉴权、Key 或模型配置不可用 | 503 |
| `MINING-120` | 数据、目录或正式因子快照已过期 | 409 |
| `MINING-121` | 发布资格、批次或并发冲突 | 409 |
| `MINING-122` | 正式因子全量校验失败 | 422 |
| `MINING-123` | 发布提交、回滚或恢复失败 | 503 |
| `MINING-124` | 候选批量边界或共享层物化失败 | 422 |
| `MINING-125` | 搜索窗口或角色可见性非法 | 400 |
| `MINING-126` | V2 存储迁移失败或源快照不可迁移 | 503 |
| `MINING-127` | 分析工具不存在 | 400 |
| `MINING-128` | 分析工具参数非法 | 400 |
| `MINING-129` | 分析工具越权 | 403 |
| `MINING-130` | 分析工具超时或取消 | 504 |
| `MINING-131` | 分析工具结果过大 | 422 |
| `MINING-132` | 分析工具执行失败 | 500 |
| `MINING-133` | 沙箱政策非法 | 400 |
| `MINING-134` | 沙箱容器或研究循环失败 | 500 |
| `MINING-135` | Qlib 快照生成或校验失败 | 422 |
| `MINING-136` | 沙箱路径、挂载或隔离边界破坏 | 403 |
| `MINING-137` | 沙箱产物或人工提升记录非法 | 422 |
| `MINING-147` | 资料清单或解析记录不符合契约 | 400 |
| `MINING-148` | 资料路径或脱敏输入越过安全边界 | 403 |
| `MINING-149` | 资料文件缺失、不可读或无法计算指纹 | 422 |
| `MINING-150` | 资料格式、正文或加密状态不允许解析 | 422 |
| `MINING-151` | 资料解析器发生未预期失败 | 500 |
| `MINING-152` | 资料文件、页数或正文字符超过解析限制 | 422 |
| `MINING-153` | 研报字段抽取请求或结果不符合契约 | 400 |
| `MINING-154` | 研报引用无法验证或抽取内容越过安全边界 | 422 |
| `MINING-155` | 研报正文、窗口或抽取输出超过固定上限 | 422 |
| `MINING-156` | 研报筛选、转换请求或结果不符合契约 | 400 |
| `MINING-157` | 研报变量映射、DSL 或来源证据无法通过校验 | 422 |
| `MINING-158` | 研报转换批次超过固定资源上限 | 422 |
| `MINING-159` | 资料运行输入、来源或摘要不符合契约 | 400 |
| `MINING-160` | 资料运行没有产生可进入 V2 验证的候选 | 422 |
| `MINING-161` | 自由探索无法从冻结目录生成研究方向 | 422 |

候选评估执行错误使用 `MINING-106`。门禁不通过是正常业务结论，通过
`TechnicalGate.passed=false` 和逐项 `checks` 表达，不伪装成系统异常。

M5 沙箱研究循环另外使用内部反馈错误码：`MINING-138` 编译、`MINING-139` 运行、
`MINING-140` 输出格式、`MINING-141` 索引、`MINING-142` 覆盖率、`MINING-143` 数值、
`MINING-144` 代码限制、`MINING-145` 资源或超时。它们记录在脱敏研究反馈和
`logs/research/mining_sandbox/` 业务日志中，不直接改变现有 Web 错误映射。
`MINING-146` 表示当前运行的 `knowledge.json`、来源引用、任务哈希或知识版本非法。
资料驱动的输入、PDF 解析、分类、哈希、来源引用和失败规则见
[资料输入与解析契约](research-materials.md)。

## 修改注意

1. 修改任一契约字段前先评估哈希、数据库 schema 和已有审批失效规则。
2. 不得把正式因子路径加入候选存储，也不得从归档原型直接复制未验收实现。
3. 新增状态时必须同步状态表、测试、日志阶段和文档；新增字段或算子必须先改目录并重新核对。
4. 候选大产物默认保留到人工清理，只有中断运行的超期临时目录允许自动清理。
5. 详细字段与哈希口径见 [契约与状态机](contracts.md)，表达式规则见 [受控 DSL](dsl.md)。
6. 修改评估阈值必须升级 `mining_quality_v1` 并同步配置、测试和文档；阈值全文进入
   `input_hash`，变化后不得复用旧评估结论。
7. 编排预算和失败阈值修改后必须同步 `orchestrator.json`、契约测试和恢复说明。
