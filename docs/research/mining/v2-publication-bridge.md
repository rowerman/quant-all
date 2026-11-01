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
| V2 候选最终审批 | `mining_v2.search_approval` | 只由 V2 审批存储写入 |
| V1 候选、正式发布批次和正式因子关联 | `mining` schema | P0-03 才可写入 |
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
| `review_ready` | 审批前置状态 | 转换 | 仅表示可构造桥接视图和审批资格 |
| V2 `approved/rejected` | V2 最终审批结果 | 复用 | 审批记录保存在 `mining_v2.search_approval` |
| V1 `approved`、`building`、`published` | 不适用 | 禁止 | 正式发布适配属于 P0-03 |

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

## 候选审批

P0-02 在桥接契约之上实现 V2 候选级审批。审批入口位于
`mining.storage_v2_review`，状态仍以 `mining_v2.candidate_trace` 为唯一事实，
最终决定保存在 `mining_v2.search_approval`。V2 运行保持 `review_ready`，只有候选从
`review_ready` 迁移到 `approved` 或 `rejected`。

审批资格要求链式轨迹完整，并且以下阶段的候选级证据均存在且全部通过：

- `technical`
- `development`
- `selection`
- `full`
- `holdout`

审批证据哈希覆盖候选内容哈希、语义哈希、冻结上下文哈希，以及按阶段、输入哈希和评估
ID 排序的全部候选级证据引用。全局运行轨迹哈希不进入该哈希，因此批准一个候选不会使
其他候选的审批证据失效；新增或修改任一候选证据仍会改变其证据哈希。

最终审批只能写入一次。完全相同的重复请求保持幂等，相反决定、审核人、备注或证据变化
返回 `MINING-108`。审批、候选状态、事件和链式轨迹在一个事务内提交，审计轨迹会校验
审批记录内容。

P0-03 还需让已批准 V2 候选适配现有 `MiningPublicationService`。V2 尚未接入正式发布，
也没有审批 API、页面入口或下游读取验收。
