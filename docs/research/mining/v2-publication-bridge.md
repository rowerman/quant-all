# V2 到 V1 发布桥接契约

P0-01 冻结 V2 搜索运行进入现有审批、整批发布和正式因子写入链路前的只读输入映射。
桥接层只把 V2 已冻结的候选定义、证据身份和上下文口径转换为现有构建器能重新校验的
视图；它不复制运行、审批或发布事实，也不修改任何状态。

桥接版本为 `mining_search_publication_bridge_v1`。数据结构与上下文转换位于
`mining.search_publication_contracts`，只读加载位于
`mining.search_publication_loader`。

## 唯一事实来源

| 事实 | 唯一来源 | 桥接规则 |
|---|---|---|
| V2 候选定义、假设、批次、尝试和分阶段证据 | `mining_v2` schema | 只读 |
| V1 候选、审批、正式发布批次和正式因子关联 | `mining` schema | P0-02/P0-03 才可写入 |
| 正式 `mined` 因子清单和目录 | 现有 `factors.published` 与因子目录 | 继续由 `MiningPublicationService` 独占 |
| 发布用临时 V1 上下文 | V2 冻结上下文的内存派生 | 不持久化，不作为审批或发布事实 |

## 字段映射

| V2 来源 | V1 发布目标 | 规则 | 说明 |
|---|---|---|---|
| `run_id` | 桥接 `search_run_id` | 复用 | 保留 `searchrun_*` 身份，不创建同 ID 的 V1 运行 |
| `context_hash` | `source_context_hash` | 复用 | 仅用于追溯 V2 冻结输入 |
| `trace_hash` | `source_trace_hash` | 复用 | 映射前必须通过完整轨迹校验 |
| `candidate_id` | `candidate_id` | 复用 | 必须等于冻结定义推导出的稳定候选 ID |
| `candidate_hash` | `content_hash` | 复用 | 必须重新计算一致 |
| `semantic_hash` | `semantic_hash` | 复用 | 必须重新计算一致 |
| `candidate_trace.name` | `name` | 复用 | 必须与冻结定义一致 |
| `candidate_trace.payload` | `definition` | 复用 | 严格恢复为 `CandidateDefinition` |
| 假设 ID、实验批次 ID、父候选 | 候选谱系 | 复用 | 保留审核来源，不进入正式因子清单 |
| 阶段评估 ID、输入哈希、证据哈希、状态 | 证据引用序列 | 复用 | 不携带指标正文、错误正文或文件路径 |
| 评分卡哈希 | `scorecard_hash` | 复用 | 由登记证据确定性生成 |
| `development_start` | `research_start` | 转换 | 临时发布上下文的研究起点 |
| `final_end` | `research_end` | 转换 | 临时发布上下文的研究终点 |
| `final_start/final_end` | `full_eval_start/full_eval_end` | 转换 | 用于发布全历史构建范围 |
| `holdout_start/holdout_end` | V1 持有窗口 | 复用 | 日期口径不变 |
| V2 `SearchBudget` | V1 `MiningConstraints` | 转换 | 只映射构建器需要的上限 |
| V2 `control_policy` | V1 覆盖率、失败率等约束 | 转换 | 不改变 V2 已冻结的搜索判定 |
| `review_ready` | 发布资格输入 | 转换 | 仅表示可构造桥接视图，不写 V1 候选行 |
| V1 `approved`、`building`、`published` | 不适用 | 禁止 | P0-02/P0-03 才能写入 |

临时 V1 上下文的哈希只用于构建器内部稳定性，不能替换
`source_context_hash`，也不能作为 V2 审批证据。

## 禁止进入发布输入

- 不得复制 V2 候选、运行、审批或发布记录来伪造 V1 事实。
- 不得把 V2 的 `review_ready` 直接改写成 V1 的 `approved`、`building` 或 `published`。
- 不得把 Worker Prompt、密钥、沙箱源码、原始异常文本、绝对路径或未脱敏诊断写入发布输入。
- 不得让 V2 评估结果绕过现有 DSL 重新校验、正式名称冲突、语义重复、全量构建和原子提交。
- 不得从选择期、最终研究期或测试段重新向 Agent 反馈数据。

## 加载规则

`load_search_publication_inputs()` 只接受 `review_ready` 运行及其
`review_ready` 候选，并按候选 ID 排序返回稳定结果。加载过程会：

1. 校验 V2 链式轨迹。
2. 重新构造候选定义，并核对内容哈希、语义哈希、候选 ID 和名称。
3. 核对评估证据表字段与 `SearchEvaluationEvidence` 正文哈希。
4. 生成全阶段证据引用、评分卡哈希和临时 V1 发布上下文。
5. 全程只读，不新增事件、轨迹、候选、审批或发布记录。

P0-02 在此基础上定义审批资格和证据哈希；P0-03 再适配现有
`MiningPublicationService`，两者都不能另建第二套发布器。
