# 挖掘契约与状态机

## V2 搜索契约

`mining_v2` 在现有 V1 运行契约之外增加证据驱动搜索所需的严格对象，V1 的上下文、
候选和状态哈希保持不变。V2 对象均拒绝缺失或未知字段，时间统一规范为 UTC，字段
上限在对象构造和 JSON 解析时同时生效。

| 对象 | 核心内容 | 稳定哈希排除项 |
|---|---|---|
| `HypothesisCard` | 父假设、turn、分支、机制、预期方向、可证伪条件 | `created_at` |
| `ExperimentBatch` | 假设、候选及其父候选、开发期、参数和输入哈希 | `created_at` |
| `EvidencePacket` | 开发期指标、失败、相关簇、尝试计数和预算状态 | `created_at` |
| `SearchDecision` | `refine / branch / stop / promote`、理由和目标假设 | `created_at` |
| `TrialRecord` | 提出、拒绝、重复、非法、门禁、评估和测试段尝试 | `occurred_at` |
| `ResearchDirection` | 机制类别、字段依赖、复杂度、种子和生成依据 | 无时间字段 |

`SearchBudget` 冻结默认上限：6 个 turn、3 个假设分支、每分支 3 次修正、每 turn
4 个候选、每运行 24 个候选、4 个完整评估、连续 2 turn 无改善停止、每 turn
4 次工具调用。调用方可以收紧上限，但不能超过这些值。

`SearchRunRequest` 支持 `hypothesis` 和 `free_exploration` 两种输入模式。未提供
`input_mode` 的旧请求和旧持久化字典按 `hypothesis` 解析；该模式必须提供 8～4000
字符假设。`free_exploration` 必须令 `hypothesis=None`，预算和模型配置仍按原契约校验。
手工假设请求沿用原哈希输入；自由探索显式把模式和空假设纳入请求哈希。自由探索可选
提供 `0～2^64-1` 的 `exploration_seed`；未提供时从请求哈希派生有效种子。手工假设
不得携带该字段。旧自由探索持久化字典缺少种子字段时保持原请求哈希。

`HypothesisCard` 的假设正文为 8～4000 字符，机制最多 1000 字符，可证伪条件
最多 12 条且每条最多 500 字符。`ExperimentBatch` 每批最多 4 个候选，参数对象最多
16 KiB；`EvidencePacket` 的总规范化 JSON 不得超过 64 KiB；`TrialRecord` 的扩展
载荷不得超过 16 KiB。所有对象由 `mining.search_contracts` 统一导出。

V2 上下文把时间拆为开发期、选择期、最终研究期和模块测试段，具体日期和角色可见性见
[V2 搜索轨迹基础](search-v2.md)。实验批次和证据包只允许使用开发期，Agent 脱敏上下文
不会包含另外三段窗口。

## 运行请求

`MiningRunRequest` 包含 `idempotency_key`、研究假设、预算约束和模型配置。首版固定：

- 股票池 `all_a`、标签 `fwd_ret_open_5`、后复权价格通道；
- 默认 3 轮、12 个候选、4 个完整评估、30 次 LLM 调用；
- 时间窗口、数据版本和测试段以 [需求口径](requirements.md) 为准。

运行 ID 由 schema 版本和幂等键确定。请求哈希包含规范化后的假设、约束和模型配置。

## 冻结上下文

`FrozenContext` 在运行期间不可修改，内容包含：

| 类别 | 字段 |
|---|---|
| 数据 | 数据版本、64 位数据指纹 |
| 口径 | 标签、股票池、价格通道 |
| 时间 | 研究、快速筛选、完整评估、模块测试段起止 |
| 规则 | 字段目录、算子目录、已有因子清单及其哈希 |
| 预算 | 轮次、候选数、完整评估数、LLM 调用、表达式深度、覆盖率、随机种子 |
| Worker | 模型配置名和固定模型名称 |

字段目录和算子目录先按规范化 JSON 去重、排序；因此仅改变输入顺序不会改变上下文哈希。
S3 的实际目录版本为 `mining_catalog_v1`，字段、单位、时点和算子见
[字段与算子目录](catalog.md)，表达式编译见 [受控 DSL](dsl.md)。

S5 从该上下文构造候选 `FactorView`，在独立运行目录物化因子值，并复用现有评估面板和
全部指标。候选相关性只对正式因子计算；技术门禁、快速筛选、完整评估和输入哈希见
[候选评估与质量门禁](evaluation.md)。

`MiningCatalog.frozen_field_mappings()` 只返回 84 个获准字段；
`frozen_operator_mappings()` 返回 34 个冻结算子。两者可直接赋给 `FrozenContext`，
目录哈希会随任一字段或算子定义变化。
上下文 ID 由上下文哈希生成，同一上下文可被多个不同运行共享，但存储内容只写一次。

### 字段声明

下表中的“来源”表示字段由谁产生；“不可变”表示对象创建后是否禁止修改。
没有物理量纲的标识、枚举、文本和哈希统一为无单位，时间统一为 UTC。

| 对象 | 字段 | 类型与单位 | 必填 | 来源 | 不可变 |
|---|---|---|---|---|---|
| 请求 | `idempotency_key` | string | 是 | 调用方 | 是 |
| 请求 | `hypothesis` | string，8～4000 字符 | 是 | 用户 | 是 |
| 请求 | `constraints` | `MiningConstraints` | 是 | 调用方/默认配置 | 是 |
| 请求 | `llm_profile` | string | 是 | 调用方/默认配置 | 是 |
| 约束 | `universe` | 枚举 `all_a` | 是 | 冻结口径 | 是 |
| 约束 | `label` | 枚举 `fwd_ret_open_5` | 是 | 冻结口径 | 是 |
| 约束 | `price_channel` | 枚举 `hfq` | 是 | 冻结口径 | 是 |
| 约束 | `direction` | `either/positive/negative` | 是 | 调用方 | 是 |
| 约束 | `max_rounds` | 整数，轮 | 是 | 默认配置/调用方 | 是 |
| 约束 | `max_candidates` | 整数，个 | 是 | 默认配置/调用方 | 是 |
| 约束 | `full_eval_top_k` | 整数，个 | 是 | 默认配置/调用方 | 是 |
| 约束 | `max_llm_calls` | 整数，次 | 是 | 默认配置/调用方 | 是 |
| 约束 | `max_expression_depth` | 整数，层 | 是 | 默认配置/调用方 | 是 |
| 约束 | `min_coverage` | float，比例 0～1 | 是 | 默认配置/调用方 | 是 |
| 约束 | `max_failure_rate` | float，默认 0.75 | 是 | 默认配置/调用方 | 是 |
| 约束 | `min_quality_checks_for_failure_stop` | 整数，默认 4 | 是 | 默认配置/调用方 | 是 |
| 约束 | `seed` | 整数，无单位 | 是 | 默认配置/调用方 | 是 |
| 约束 | `excluded_fields` | string 集合 | 否 | 调用方 | 是 |
| 上下文 | `data_version` | string | 是 | 数据构建 | 是 |
| 上下文 | `data_fingerprint` | SHA-256 | 是 | S6 数据快照 | 是 |
| 上下文 | `research/full/holdout_*` | ISO date | 是 | 冻结口径 | 是 |
| 上下文 | `field_catalog` | 对象集合，内容见 S3 | 否 | 字段目录 | 是 |
| 上下文 | `operator_catalog` | 对象集合，内容见 S3 | 否 | 算子目录 | 是 |
| 上下文 | `existing_factors` | string 排序集合 | 否 | 正式因子目录快照 | 是 |
| 上下文 | `budget` | `MiningConstraints` | 是 | 请求 | 是 |
| 上下文 | `created_at` | UTC 时间 | 是 | 系统 | 是 |

## 候选哈希

候选有两种哈希，二者职责不同：

| 哈希 | 规范化输入 | 用途 |
|---|---|---|
| `semantic_hash` | 规范表达式、方向、回看窗口、排序后的依赖字段、输出类型、参数 | 识别计算语义重复 |
| `content_hash` | schema 版本、`semantic_hash`、名称、机制、说明、顺序固定的白话步骤 | 审核版本绑定和稳定候选 ID |

修改说明或白话步骤只改变 `content_hash`，会要求重新审核；修改表达式等计算内容会同时
改变两种哈希。同一运行内 `content_hash` 重复按幂等返回，`semantic_hash` 相同但内容
不同按重复冲突处理；不同运行可以保留相同候选用于审计。

| 对象 | 字段 | 类型与单位 | 必填 | 来源 | 不可变 |
|---|---|---|---|---|---|
| 候选 | `name` | 大写因子标识 | 是 | Worker | 是 |
| 候选 | `mechanism` / `description` | string | 是 | Worker | 是 |
| 候选 | `canonical_expression` | 规范化表达式文本 | 是 | S3 | 是 |
| 候选 | `direction` | `-1/1` | 是 | Worker | 是 |
| 候选 | `lookback` | 整数，交易日 | 是 | S3 | 是 |
| 候选 | `dependencies` | 字段名排序集合 | 是 | S3 | 是 |
| 候选 | `output_type` | 输出类型名 | 是 | S3 | 是 |
| 候选 | `steps` | string 有序集合 | 是 | Worker | 是 |
| 候选 | `discovery_round/index` | 正整数，轮/序号 | 是 | S6 | 是 |
| 候选 | `parameters` | JSON 对象 | 否 | S3 | 是 |
| 候选 | `semantic_hash/content_hash` | SHA-256 | 是 | 系统计算 | 是 |

## 运行状态

```text
queued -> generating -> screening -> evaluating -> holding_out
       -> review_ready -> completed
```

- 活动阶段失败或取消后进入 `failed` / `cancelled`。
- `review_ready` 在数据、上下文或规则变化后进入 `stale`。
- 活动阶段发现数据、目录或正式因子清单变化时直接进入 `stale`，不领取测试段。
- 重启恢复把活动阶段改为 `interrupted`，记录原阶段和中断次数。
- `interrupted` 只能显式恢复到中断前阶段，不能借恢复跨阶段。
- 同一时刻只允许一个运行处于活动阶段。

## 候选状态

```text
proposed -> validating -> technical_failed | screening
screening -> screened_out | evaluating
evaluating -> evaluated -> holdout_running
holdout_running -> holdout_failed | review_ready
review_ready -> approved | rejected
approved -> building -> published
building -> build_failed -> building
```

`technical_failed`、`screened_out`、`holdout_failed`、`published`、`rejected`、
`failed`、`cancelled` 和 `stale` 为终态。失败、过期和测试段未通过的候选不能进入审批。

## 证据与审批

- `EvaluationReport` 保存阶段、候选哈希、上下文哈希、输入哈希、指标、警告、产物路径和
  产物哈希。阶段为 `technical/quick/full/holdout`，每个尝试生成稳定报告 ID。
- `TechnicalGate` 保存规则版本、检查结果、通过状态和检查哈希。
- `ApprovalDecision` 必须包含审批人、备注、候选内容哈希、语义哈希、上下文哈希和证据哈希。
- 证据哈希由最近一次通过的技术门禁哈希、全部通过的测试段报告 ID、测试段批次哈希和
  领取证据哈希规范化得到。
- 首次批准或拒绝是最终决定；完全相同的重复请求幂等，相反决定或备注变化返回冲突。
- 发布尝试从 `staging -> validating -> published`，失败进入 `failed` 并可记录
  `rolled_back`。S8 使用批次记录原子推进同一冻结上下文内的全部已批准候选。

| 对象 | 字段 | 类型与单位 | 必填 | 来源 | 不可变 |
|---|---|---|---|---|---|
| 评估 | `stage` | `technical/quick/full/holdout` | 是 | S5/S6 | 是 |
| 评估 | `status` | `running/passed/failed/error` | 是 | S5/S6 | 是 |
| 评估 | `input_hash` | SHA-256 | 是 | S5/S6 | 是 |
| 评估 | `metrics/warnings` | JSON/string 集合 | 是/否 | S5 | 是 |
| 评估 | `artifact_path/sha256` | 相对路径/SHA-256 | 否 | S5 | 是 |
| 评估 | `started_at/finished_at` | UTC 时间 | 结束时间可空 | S5 | 是 |
| 门禁 | `rule_version` | string | 是 | S3/S5 | 是 |
| 门禁 | `passed` | boolean | 是 | S3/S5 | 是 |
| 门禁 | `checks/check_hash` | JSON/SHA-256 | 是 | S3/S5 | 是 |
| 审批 | `decision` | `approved/rejected` | 是 | 用户 | 是 |
| 审批 | `reviewer/note` | string | 是 | 用户 | 是 |
| 审批 | 候选、上下文和证据哈希 | SHA-256 | 是 | 系统 | 是 |
| 发布 | `status` | 发布状态枚举 | 是 | S8 | 否，按状态机迁移 |
| 发布 | `staging_path` | 隔离暂存相对路径 | 是 | S8 | 是 |
| 发布 | `manifest/artifact` 哈希 | SHA-256 | 完成时必填 | S8 | 写入后不变 |

## 数据库表

| 表 | 责任 |
|---|---|
| `mining.run` | 请求、运行状态、幂等键和执行槽信息 |
| `mining.run_context` | 可复用的不可变冻结上下文 |
| `mining.candidate` | 候选内容、状态和发现顺序 |
| `mining.evaluation` | 各阶段评估报告 |
| `mining.technical_gate` | 技术门禁及检查哈希 |
| `mining.approval` | 最终人工审批 |
| `mining.publication` | 发布尝试及状态 |
| `mining.publication_batch` | 同一上下文内整批发布、清单、产物哈希和 `source_kind` |
| `mining.event` | 按运行连续编号的不可变事件 |
| `mining.holdout_usage` | 模块测试段唯一领取记录 |

数据库使用 `VARCHAR` 保存 UTC ISO 时间，以保留微秒并便于事件审计。所有状态更新使用
`WHERE status=<预期旧状态>` 的条件更新。事实来源是数据库；文件系统只保存可重建产物。

索引覆盖运行状态、运行内候选状态、候选阶段评估查询；其余高频键由主键和唯一约束覆盖。
运行、候选、事件、审批和发布元数据长期保留。候选大产物保留到人工清理；只有中断运行
所属且超过 24 小时的 `.tmp` 和 `staging` 允许自动删除。
