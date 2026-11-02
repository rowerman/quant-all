# V3 统一研究闭环设计

本文件冻结 P4-01 的 V3 集成边界。V3 在 V2 的冻结窗口、预算、尝试台账、分阶段证据、
人工审批和原子发布基础上，统一接入手工假设、自由探索、资料驱动和 CoSTEER 代码研究。
P4-01 只提供公共契约和设计，不接入运行、存储、API、页面或真实迁移。

## 目标与边界

V3 的唯一正式入口仍是人工批准和原子发布。四类来源只能帮助形成研究假设或受控候选，
不能替代候选内容哈希、分阶段证据或人工审批。

V3 不新增第二套候选、审批和发布体系：

- 新运行、来源、候选、尝试和证据事实写入新的 `mining_v3` schema。
- 正式发布批次、逐候选发布记录和正式因子继续写共享 `mining.publication*`。
- V1 `mining` 和 V2 `mining_v2` 保留原表，迁移完成后作为只读审计来源。
- Worker Prompt、原始异常、密钥、主机路径和沙箱源码不进入 V3 公开契约。

## 四类来源

| 输入模式 | 产生可控候选的路径 | 允许的来源身份 |
|---|---|---|
| `hypothesis` | 手工假设进入搜索 Worker，再由主服务校验候选 | 请求哈希和幂等请求身份 |
| `free_exploration` | 冻结种子和方向池生成方向，Worker 只在方向内生成候选 | 请求哈希、有效种子和方向哈希 |
| `research_report` | PDF 清单、页正文、字段抽取和 Worker 转换后形成种子批次 | 资料、抽取、草稿、转换和来源引用哈希 |
| `costeer` | 沙箱完成代码研究并封存产物，人工转写为受控 DSL 后导入种子批次 | 沙箱运行、产物清单、人工提升记录和导入候选哈希 |

四类来源都归一为 `ResearchInputEnvelope`。来源类型必须等于输入模式，未知模式、模式与
来源不一致或携带未知字段时直接拒绝。

CoSTEER 来源必须同时提供 `artifact_manifest_hash` 和 `promotion_hash`；其他来源禁止
携带这两个字段。CoSTEER 不直接在沙箱执行主服务 DSL，也不自动创建正式候选。导入前
必须重新执行受控 DSL 校验、候选哈希、技术门禁和全部研究窗口评估。

## 公共契约

版本常量位于 `mining.v3_contracts`：

| 常量 | 值 | 用途 |
|---|---|---|
| `MINING_V3_SCHEMA_VERSION` | `mining_v3` | V3 存储 schema 版本 |
| `V3_RUN_ID_PREFIX` | `researchrun` | V3 运行 ID 前缀 |
| `V3_PUBLICATION_SOURCE_KIND` | `v3_research` | 共享正式发布批次来源 |
| `V3_RESEARCH_INPUT_SCHEMA_VERSION` | `mining_research_input_v3` | 统一输入包络 |
| `V3_RESEARCH_SOURCE_SCHEMA_VERSION` | `mining_research_source_v3` | 来源引用 |
| `V3_RESEARCH_CANDIDATE_SCHEMA_VERSION` | `mining_research_candidate_v3` | 候选包络 |
| `V3_RESEARCH_EVIDENCE_SCHEMA_VERSION` | `mining_research_evidence_v3` | 分阶段证据引用 |

### 输入包络

`ResearchInputEnvelope` 保存 `run_id`、`idempotency_key`、`input_mode`、
`context_hash`、`request_hash` 和 `ResearchSourceReference`。其 `input_hash` 由规范化
JSON 计算，不包含数据库写入时间。

### 来源引用

`ResearchSourceReference` 固定保存：

- `source_kind`
- `source_id`
- `source_hash`
- 可空 `artifact_manifest_hash`
- 可空 `promotion_hash`

来源引用采用字段白名单，不允许附加任意 metadata。来源正文和大文件只保存在原有受控
模块目录，V3 数据库只保存可重算的身份和哈希。

### 候选包络

`CandidateEnvelope` 绑定冻结的 `CandidateDefinition`、来源引用和候选级证据引用。
`candidate.semantic_hash`、`candidate.content_hash` 和 `candidate.candidate_id` 的计算
保持不变，不包含来源。

同一候选从不同来源导入时应保持相同候选 ID。来源差异通过 `source.reference_hash`、
`CandidateEnvelope.envelope_hash` 和审批证据哈希区分。P4-02 不得再把资料或沙箱来源
哈希塞入 `CandidateDefinition.parameters` 来改变候选身份。

### 分阶段证据

`EvidenceReference` 只保存候选 ID、阶段、评估 ID、输入哈希、证据哈希和状态。阶段固定
为：

```text
technical -> development -> selection -> full -> holdout
```

`CandidateEnvelope.approval_material()` 要求每个阶段至少有一条 `passed` 证据，并把
候选内容、语义哈希、上下文、来源引用和全部证据引用纳入审批摘要。
`approval_evidence_hash()` 是后续 P4-03 审批记录必须复用的稳定哈希入口。

## 状态机

V3 不复制状态表，`V3ResearchStateMachine` 只委托既有规范状态机：

```text
queued -> searching -> selecting -> evaluating -> holding_out -> review_ready
```

- 手工假设、自由探索和资料解析都在 `searching` 开始。
- CoSTEER 在进入 V3 前已完成沙箱执行和人工转写，导入后同样从 `searching` 开始。
- `interrupted` 只能恢复到中断前阶段。
- `cancelled`、`failed`、`stale` 和 `review_ready` 不能回到活动阶段。
- 候选、审批和共享发布继续使用 V1/V2 的规范状态机。
- 选择期、最终研究期和模块测试段证据只进入确定性控制器，不反馈给任何 Worker。

运行和候选只有同时处于 `review_ready` 时才允许形成最终人工决定。失败、中断、
测试段未完成、已决定或状态缺失都会出现在审核阻断原因中。审批事务会重新读取运行和
候选状态，因此直接调用 API 也不能绕过页面门禁。

## 存储范围

目标 schema 为 `mining_v3`。全新数据库由 P4-02 直接初始化，已有 V1/V2 数据库由
P4-04 的显式迁移创建。V3 沿用 V2 的以下事实类别：

- 冻结上下文、运行、运行控制、turn 和链式轨迹。
- 假设、实验批次、候选、候选谱系和尝试台账。
- 分阶段评估证据、分析工具审计和审批记录。
- 事件、尝试计数和 legacy 只读快照。

V3 新增的事实类别为：

- `research_input`：统一输入包络和输入哈希。
- `source_reference`：来源白名单引用。
- `candidate_source`：候选包络、来源和候选级证据绑定。
- `run_context`：与输入哈希绑定的完整冻结搜索上下文。
- `candidate_approval`：单一来源最终人工决定和审批证据哈希。
- `candidate_state`：审批及共享发布阶段的候选状态。
- `event`：审批、候选状态和发布审计事件。

正式发布事实不进入 `mining_v3`。共享 `mining.publication*` 使用
`source_kind=v3_research`，并保留 `researchrun_*` 来源运行 ID、候选 ID、上下文哈希、
审批证据哈希和 `CandidateEnvelope.envelope_hash`。

### P4-02 已实现范围

P4-02 已在全新数据库初始化以下表：

| 表 | 内容 |
|---|---|
| `mining_v3.schema_meta` | `mining_v3` schema 版本 |
| `mining_v3.research_input` | 统一运行输入和主来源 ID |
| `mining_v3.source_reference` | 四类来源的白名单身份和哈希 |
| `mining_v3.candidate_source` | 来源无关候选与一个来源的绑定 |
| `mining_v3.evaluation_evidence` | 候选来源绑定的五阶段证据引用 |
| `mining_v3.migration_source` | V1/V2 来源版本、稳定快照哈希和迁移计数 |
| `mining_v3.legacy_context` | 迁移时固化的旧上下文只读摘要 |
| `mining_v3.legacy_run` | 迁移时固化的旧运行和输入模式 |
| `mining_v3.legacy_candidate` | 迁移时固化的旧候选定义和状态 |

`MiningV3Store` 原子登记输入、候选来源和证据，追加证据时保持候选身份不变。同一候选
可在同一运行中绑定多个来源；同一候选和来源重复写入必须产生相同内容。检测到
`mining` 或 `mining_v2` 时，V3 初始化直接失败并提示执行后续显式迁移，不会自动改变
旧库。

`mining.v3_sources` 为手工假设、自由探索、资料报告和 CoSTEER 构造稳定来源身份。
资料候选在进入 V3 前移除 V2 专用的 `source_kind` 和 `research_provenance_hash`
参数，因此资料差异只改变来源包络。CoSTEER 导入只接受已封存产物对应的人工提升记录
和受控 DSL 候选，不执行沙箱代码。V3 证据适配器只接收技术、开发、选择、完整和测试段
五个阶段；诊断和稳定性辅助证据仍留在原研究链。

### P4-03 已实现范围

`MiningV3Store` 现可登记完整 `FrozenSearchContext`，并在发布时复用
`publication_context()` 转换为现有构建器使用的 `FrozenContext`。上下文哈希必须与
`ResearchInputEnvelope.context_hash` 一致，缺失上下文不能审批或发布。

`V3CandidateReview` 返回候选、来源、五阶段证据、阻断原因、候选包络哈希和审批证据
哈希。五个阶段必须全部存在且为 `passed`。同一候选可以绑定多个来源，但只能选择一个
来源形成最终批准或拒绝；最终审批按 `(run_id, candidate_id)` 唯一，其他来源只保留
审计价值。

`mining_v3.candidate_state` 是 V3 候选批准后生命周期状态的唯一权威来源。V3 运行详情、
候选详情和决策响应都以该状态覆盖 runtime 轨迹中的候选状态，因此批准后统一返回
`approved`，发布期间返回 `building`，失败返回 `build_failed`，成功后返回
`published`。Runtime 轨迹继续提供定义、指标和搜索谱系，但不再单独决定页面状态。

`MiningV3Store` 实现现有 `MiningPublicationService` 所需仓储接口。`v3_research`
批次通过原有构建、全量校验、原子提交和回滚流程写共享 `mining.publication*`；
候选审批或证据在构建期间变化时，整批中止并收口为 `build_failed`。正式定义清单保留
`source_kind`、来源上下文、`candidate_envelope_hash`、来源引用哈希和审批证据哈希。
V3 没有另建因子读取接口，发布结果由现有因子清单、评估、模型特征和回测加载器直接
读取。

V3 在发布输入加载、批次启动和最终提交前都会重新确认运行仍为 `review_ready`。
运行已过期、中断、失败或状态记录缺失时不会启动或完成发布，已有构建状态保持原样等待
上层失败收口。

## 迁移与兼容

- 普通启动不得静默迁移 V1 或 V2。
- 新空库可由 V3 存储首次初始化；已有 V1/V2 数据库必须执行显式迁移命令。
- 迁移前停止监测站和 Worker，并确认没有活动 V1/V2 运行。
- V1/V2 源数据分别计算稳定快照哈希，复制为 `mining_v3` 内只读 legacy 记录。
- 迁移在一个事务内完成；中断整体回滚，相同源快照重复执行返回复用结果。
- 首次 V3 写入后禁止回滚迁移；旧 V1/V2 表继续保留且不被覆盖。
- 旧运行只读展示，不能恢复为 V3 运行；新运行全部使用 `researchrun_*`。
- V1/V2 的现有读取、评估、模型特征和回测接口在 V3 发布后继续可读。

P4-04 已实现以下显式命令：

```bash
.venv/bin/python -m mining.migrate_to_v3 \
  --db data/databases/factor_mining.duckdb status
.venv/bin/python -m mining.migrate_to_v3 \
  --db data/databases/factor_mining.duckdb migrate
.venv/bin/python -m mining.migrate_to_v3 \
  --db data/databases/factor_mining.duckdb rollback
```

`migrate` 分别计算 `mining_v1` 和 `mining_v2` 的上下文、运行和候选摘要哈希，
拒绝活动运行，然后在一个事务内创建 V3 表、复制 legacy 快照并写入迁移完成记录。
失败或进程中断会整体回滚，相同源快照重复执行返回 `reused=true`。旧 schema 后续增加
运行或改变摘要时，命令拒绝自动合并。

`rollback` 只允许删除由显式迁移创建、且尚无 V3 输入、候选、审批、状态、事件或正式
发布批次的 `mining_v3` schema。该操作不修改 V1/V2 原表；任何 V3 业务写入后回滚会被
拒绝。`status` 以只读方式返回来源哈希、计数、源漂移和回滚资格。

完整旧证据、轨迹和评估产物仍从原 V1/V2 表及文件目录读取。V3 legacy 表只保存旧运行
页面所需的上下文、运行和候选摘要，固定返回 `read_only=true`、`resumable=false`。

## P4-05 统一运行、页面与监控

设置 `MINING_SEARCH_V3_ENABLED=true` 后，新建运行由 V3 接管，V1 仍是默认生产链路。
同时开启 V2 和 V3 时，V3 优先；已有 `searchrun_`、V2 和 V1 运行仍按原运行 ID 读取。

V3 运行复用 V2 已验证的 Worker 搜索、受控评估、选择期、完整研究和一次性测试段逻辑。
运行表使用 `mining_v3.runtime_*`，来源、候选包络、五阶段证据、审批和候选状态继续使用
同一 schema 的 V3 契约表。V3 运行 ID 固定为 `researchrun_`。

V3 创建运行前会持久化输入包络和冻结上下文；CoSTEER 的人工转写候选另存于
`mining_v3.runtime_seed_candidate`。服务启动时，遗留 `queued` 和活动阶段统一标记为
`interrupted`。显式恢复从运行表、输入包络、冻结上下文、资料运行和预置候选重建执行
状态，重验候选包络及已有审批后再回到原阶段。校验失败返回 `MINING-165` 且不改变运行
状态；不完整的旧 CoSTEER 运行不会静默降级为普通失败。

| 输入模式 | 创建入口 | 候选产生方式 |
|---|---|---|
| `hypothesis` | `POST /api/mining/runs` | Worker 在手工假设下生成受控候选 |
| `free_exploration` | `POST /api/mining/runs` | 冻结种子和机制方向后由 Worker 生成候选 |
| `research_report` | `POST /api/mining/research-runs` | 研报解析、抽取和 Worker 转换 |
| `costeer` | `POST /api/mining/costeer-runs` | 选择服务端人工提升记录并重新校验 DSL |

CoSTEER 页面只列出审核根目录中产物树、清单哈希和提升记录均通过校验的
`target_type=dsl` 记录。创建请求携带 `promotion_id`、幂等键、方向、机制、说明和步骤，
表达式取自已校验提升记录的 `target_reference`。导入不执行沙箱代码，也不跳过五阶段
评估、人工审批和共享原子发布。

运行、候选和环境监控入口为 `GET /api/mining/v3/status`，返回来源运行数、活动/中断
运行、待审核候选、未完成发布批次和 legacy 数量。业务日志使用 `mining.v3.*` 模块，
只记录运行、来源、阶段、哈希和稳定错误码。

页面继续复用 `#/mining`、`#/mining/new`、运行详情和候选详情。新建页提供四类输入；
运行页显示 V3 来源和冻结预算；候选页显示来源引用、候选包络哈希、审批证据哈希、
五阶段证据和共享发布状态。迁移旧运行固定只读，不能恢复或重新发布。

## 后续实现顺序

| 任务 | 本设计对应的实现 |
|---|---|
| P4-02 | 把四类来源接入统一输入、候选、证据和来源契约，建立 `mining_v3` 表和来源绑定 |
| P4-03 | 已接入统一审核、单一来源审批、共享发布和现有下游读取联合验收 |
| P4-04 | 已实现 V1/V2 显式迁移、受限回滚、中断恢复和旧运行只读兼容 |
| P4-05 | 已完成：四类来源端到端、页面、监控、恢复、文档和最终交付验收 |

P5-01～P5-06 已关闭 P4-05 复核缺口。V3 运行详情从持久化轨迹读取假设、turn、
候选谱系、工具调用、资料摘要和自由探索方向；候选详情的 `technical_gate` 固定为
`passed/rule_version/checks`，资料候选返回经 provenance 校验的页码和短引用。
四类来源通过真实创建 API、V3 运行器和审批发布接口完成联合验收。

正式发布使用稳定行号保存因子值，再从行情库生成受信键集组装最终分区；键集以行数和
无序哈希签名校验，避免全历史复杂查询中的键损坏和重型反连接。失败批次保持旧正式
因子不变并可重试。

## 验收原则

- 四类来源产生同一受控候选契约。
- 来源变化不改变候选内容哈希，但改变来源包络和审批证据哈希。
- 人工批准与原子发布仍是唯一正式入口。
- 测试段数据不反馈给 Agent，也不用于继续调参。
- 迁移不覆盖旧运行，正式数据写入没有测试副作用。
