# 挖掘 Web 与人工审核

## 职责

S7 在监测站内提供挖掘运行创建、状态查看、事件流、取消、恢复和人工审核。它只调用
`MiningOrchestrator` 与 `MiningStore` 的公开方法，不直接写候选表，也不构建或发布正式因子。

设置 `MINING_SEARCH_V2_ENABLED=true` 时，新运行改走实验性
`MiningSearchWebService` 和 `MiningSearchOrchestrator`；未设置或为 false 时继续使用
V1。已有 `searchrun_` 运行仍可按 ID 查询、取消和恢复，V1 运行不会因开关变化改变
行为。

V2 页面展示假设树、轮次决策、尝试台账、选择诊断、稳定性、研究质量和分析工具审计，
并在 `review_ready` 后提供候选详情、批准、拒绝、整批正式发布和失败重试入口。
V2 发布继续复用现有发布服务，不建立第二套正式因子目录。

新建入口还支持资料驱动模式。接口为 `POST /api/mining/research-runs`，字段固定为
`idempotency_key`、`material_path`、`budget` 和 `llm_profile`。`material_path`
只能是 `data/source/reports` 下的相对文件或目录；绝对路径、`..` 和符号链接会被拒绝。
服务端在创建运行时冻结资料指纹，后台执行 Worker v5 字段抽取和 Worker v6 DSL 转换。

```text
浏览器 #/mining
  -> backend.mining_routes       REST、SSE、HTTP 错误
  -> backend.mining_service      运行/候选视图、预算合并、审批用例
       -> mining.orchestrator    后台执行、取消、恢复和进度
       -> mining.storage         运行、事件、证据、审批
```

## 运行生命周期

监测站启动时初始化 `data/databases/factor_mining.duckdb`。上次进程遗留的活动运行和候选
只标记为 `interrupted`，不会自动执行；用户必须从页面显式恢复。监测站关闭时等待编排器
线程安全退出。

生产环境只在显式配置 `MINING_WORKER_URL` 后调用 Worker。Worker 缺失、未就绪、鉴权失败
或协议异常时返回 `MINING-103`、`MINING-118` 或 `MINING-119`，不会回退到 Mock。

## 预算

新建页默认使用 `configs/mining/orchestrator.json`，高级区域允许覆盖可调整预算。服务端将
部分覆盖合并到默认值，再通过 `MiningConstraints` 校验并冻结进上下文哈希。固定股票池、
标签和价格通道不能由页面改变。

V2 实验页使用 `configs/mining/search.json` 中的搜索轮次、分支、候选和完整评估预算，
并把控制策略一起冻结。V2 候选页提供审批和整批发布按钮，但只有在五阶段证据、快照和
审批哈希均有效时才能提交。

## 事件流

运行事件保存在 `mining.event`，按运行内递增序号读取。页面先读取已有事件，再从最后序号
连接 SSE：

```text
GET /api/mining/runs/<run_id>/stream?after=<sequence>
```

每条业务事件使用 `event: mining`，SSE 的 `id` 和 JSON 内的 `sequence` 相同。连接提前
结束或网络中断时，页面从最后序号重连，不重复展示事件。运行进入 `review_ready`、完成、
中断、失败、取消或过期状态后发送 `done`。断开 SSE 不会取消运行，只有显式取消接口会在
安全阶段边界停止任务。

事件轨迹只展示已持久化的阶段和摘要，不包含模型隐藏思维链、原始提示、Key 或敏感原文。
V2 还展示假设父子关系、候选父子关系、逐轮决策、停止原因、分支数、尝试分类和评分卡
缺失项。防过拟合区域同时展示成功与失败候选的相关簇、增量价值、稳定性、独立研究质量
结论、BH q-value，以及包括失败尝试和稳定性变体的统计规模。质量通过不等于人工批准，
模块测试段明确标记为一次性隔离验收。
V2 工具调用另列“工具补充证据”，按 turn 区分成功补充证据和工具失败，并显示同一
候选累计分析次数，提醒审核人关注重复切片风险。运行页只展示参数摘要和哈希，不展示
完整表达式、行情值、路径、Prompt 或密钥；候选审核页可按冻结定义展示公式和审核证据。

## 人工审核

候选详情同时返回技术门禁、快速筛选、完整评估、模块测试段和审批资格。完整评估图表复用
`renderMetric()`，页面不重算指标。

页面看到的 `candidate_content_hash` 与 `evidence_hash` 会随审批提交返回。服务端重新计算
当前证据并比较：

- 候选内容、上下文、门禁、测试段报告或领取批次变化时返回 `MINING-108`。
- 行情数据、字段目录、算子目录或正式因子清单变化时返回 `MINING-120`。V1 和 V2
  都会同时把运行与未终态候选原子标记为 `stale`；候选详情继续展示快照失效原因，
  相同失效重复触发不会重复写事件。
- 审核人和备注必须非空；相同决定重复提交保持幂等。

审批成功后候选进入 `approved` 或 `rejected`。批准页面随后出现“构建并发布本批”：
系统要求该运行全部已批准候选一起进入 S8，先全量构建和校验，再原子更新
`mined` 正式因子族。发布成功后候选进入 `published`；失败进入 `build_failed`，
旧正式文件和因子库保持不变，可由页面重试。详细规则见 [正式入库](publication.md)。

V2 候选详情使用相同的审批请求字段和错误码，但展示
`technical/development/selection/full/holdout` 五阶段证据、评分卡、审批阻塞原因和
最近发布批次。V2 发布请求使用运行顶层 `context_hash`，由现有
`MiningPublicationService` 读取全部 `approved/build_failed` 候选后整批执行。
审批或发布预检遇到快照失效时，V2 存储接口
`mark_search_run_stale()` 在同一事务内收口运行、未终态候选、链式轨迹和事件。

## 页面

| 路由 | 内容 |
|---|---|
| `#/mining` | 运行列表和新建入口 |
| `#/mining/new` | 手工假设或资料相对路径、折叠预算和创建 |
| `#/mining/{run_id}` | 状态、预算消耗、资料筛选摘要、候选和事件轨迹 |
| `#/mining/{run_id}/candidate/{candidate_id}` | 定义、研报来源、V1 四阶段或 V2 五阶段证据、审批和整批发布 |

页面支持桌面与 390px 移动端。Worker 不可用、数据过期、门禁失败、测试段失败、审批冲突
和事件流中断均显示稳定错误码与请求编号。

## 验收

```bash
.venv/bin/python -m pytest -q apps/monitor/backend/tests/test_mining_api.py
.venv/bin/python -m pytest -q apps/monitor/backend/tests/test_mining_search_api.py
.venv/bin/python tools/monitor/mining_check.py
.venv/bin/python tools/monitor/mining_search_check.py
.venv/bin/python tools/monitor/mining_research_check.py
```

浏览器脚本启动临时候选库和确定性 Worker，不访问 DeepSeek，不打开生产模块测试段。
脚本会主动中断首条 SSE 以验证按序号重连，注入 Worker 缺失和审批冲突错误，并完成
两个候选批准与整批发布。截图写入
`AiRecord/evidence/mining_s7/`；S8 发布页截图写入
`AiRecord/evidence/mining_s8/`。V2 验收脚本完成批准、拒绝、发布失败和重试，截图写入
`AiRecord/evidence/mining_v2_web/`。资料驱动验收脚本检查资料模式、运行筛选摘要、
候选页码来源和 390px 布局，截图写入 `AiRecord/evidence/mining_research_web/`。
