# V2 搜索轨迹与闭环

`mining_v2` 为证据驱动搜索提供独立契约和存储，当前 M1 只建立基础，不切换既有 V1
生产编排。V2 窗口、预算和角色配置位于 `configs/mining/search.json`。

## 分层窗口

| 区间 | 日期 | 可访问角色 | 用途 |
|---|---|---|---|
| 开发期 | 2012-01-01～2019-12-31 | Agent、控制器 | 假设迭代、快速评估和证据反馈 |
| 选择期 | 2020-01-01～2022-12-31 | 仅控制器 | 最终候选排序和入围证据 |
| 最终研究期 | 2023-01-01～2025-10-15 | 仅最终评估器 | 入围候选完整评估 |
| 模块测试段 | 2025-10-16～2026-10-15 | 仅测试段执行器 | 一次性隔离验收 |

`SearchWindows` 要求四段顺序递增、首尾不交叠。`FrozenSearchContext` 对窗口、数据
指纹、目录、正式因子清单和 `SearchBudget` 生成稳定哈希；`created_at` 不参与哈希。
运行开始时必须一次性冻结配置，之后修改 `search.json` 只影响新上下文。

## 可见性

- `worker_view()` 只返回开发期、目录、模型和预算，不包含选择期、最终研究期或测试段。
- `ensure_agent_visible()` 递归拒绝 `selection`、`final`、`holdout`、`test_segment`
  和 `module_test` 字段；日期范围字段必须位于开发期。
- 控制器可读取开发期和选择期，最终评估器只能读取最终研究期，测试段执行器只能读取
  模块测试段。越权访问抛出 `MINING-125`。
- `ExperimentBatch` 和 `EvidencePacket` 在对象构造时再次校验开发期边界，不能依赖
  上层调用方自觉过滤。

## Worker v2

Worker v2 使用独立 `mining_worker_v2` 协议和 `/v2/jobs`，只接收开发期证据摘要。
请求包含当前假设、候选谱系、证据、预算和允许动作；响应包含决策建议、下一假设和
候选草稿。主服务仍会重新校验表达式并计算全部候选身份。

Worker 响应中的动作只是建议。逐轮编排接入后，预算、停止、分支和入围动作由确定性
控制器决定，不能把选择期证据交给 Agent 或用于继续调参。

## 开发期证据

`build_development_evidence()` 只接收当前假设、实验批次及其候选开发期报告。指标字段
固定为 RankIC、ICIR、年度同号率、覆盖率、有效日数、样本量、换手和 IC 衰减；未产生
的字段保留 `None`，不会被解释成零。失败原因只保存错误码和去除路径、密钥后的摘要。

证据包同时保存窗口、分类尝试计数和预算状态。相同输入产生相同证据哈希，创建时间不
参与哈希；事件日志只记录运行、轮次、批次、证据哈希和数量。

## 确定性控制

`SearchControlPolicy` 在运行开始前冻结质量门槛和失败停止阈值，并进入搜索上下文哈希。
默认质量门槛为 `|RankIC|>=0.02`、`|ICIR|>=0.20`、年度同号率 `>=0.60`、覆盖率
`>=0.30`；失败率达到 `0.75` 且至少检查 4 个候选时可以停止当前分支。

`decide_search_action()` 的硬优先级为取消/快照失效、无有效指标、轮次或候选预算耗尽、
连续无改善、失败率、每分支修正上限，最后才考虑模型建议。模型提出越权动作时，控制器
按修正、换分支、入围或停止的合法顺序回退，并记录实际规则原因。

## 逐轮闭环

`MiningSearchOrchestrator` 独立于 V1 编排执行以下状态链：

```text
queued -> searching -> selecting -> evaluating -> holding_out -> review_ready
```

搜索阶段每轮先由 Worker 返回假设和候选草稿，再由主服务规范化、批量物化并执行开发期
评估。技术门禁失败、快速评估失败和非法草稿分别进入尝试台账，单个候选失败不会中断
其他候选。选择期只由确定性选择器读取，按绝对 RankIC、ICIR、覆盖率和候选 ID 排序；
选择结果不进入下一轮 Worker。最终研究期只执行入围候选完整评估，模块测试段每个运行
最多领取一次。

V2 评估证据表只保存指标、状态、哈希和脱敏警告，不保存行情值或评估文件路径。运行、
中断、取消和评估结果均可在同一运行内按事件序号读取。

## 恢复与取消

`MiningSearchOrchestrator.progress()` 返回当前 turn、分支、动作、停止原因、已完成
轮次、尝试分类和冻结预算。主服务启动时把遗留活动运行标记为 `interrupted`；恢复只
允许回到中断前阶段，重复恢复或跨阶段跳转会被拒绝。

恢复从最后一个未完成 turn 继续。已提交的 Worker 作业通过稳定请求 ID 和作业库复用，
已登记的评估证据直接重建候选摘要，不重新物化或执行已完成评估。取消请求先持久化并
取消当前 Worker 作业，搜索在 turn 边界转为 `cancelled`，不会改变窗口、模型或预算。

## 候选评分卡

`build_candidate_scorecard()` 只读取该候选在 V2 评估证据表中的技术、开发、选择、完整
和测试段记录。评分卡包含 RankIC、ICIR、年度同号率、覆盖率、换手、IC 衰减、样本量、
有效日数、门禁结论、失败原因、分层窗口、运行尝试计数和候选尝试计数。

同一指标按完整、选择、开发、技术阶段依次取首个可用值，并保留 `metrics_by_stage`
供审核追踪。没有证据的字段保持 `null` 并列入 `missing_fields`，不得按零值处理；
相同候选和证据集合产生相同评分卡哈希。评分卡不自动批准、发布或修改候选状态。

## 存储与迁移

V2 使用同一个候选 DuckDB 中独立的 `mining_v2` schema，V1 的 `mining` schema 保持
原样。新库由 `MiningV2Store` 首次创建 V2 schema；已有 V1 库不会在普通存储初始化时
自动升级。迁移前必须停止监测站和 Worker，并确保没有活动 V1 运行，然后显式执行：

```bash
.venv/bin/python -m mining.migrate_v1_to_v2 \
  --db data/databases/factor_mining.duckdb
```

迁移在同一事务中创建 V2 表并复制运行、上下文和候选的只读快照，最后写入源快照哈希。
中断时整体回滚；同一源快照重复执行返回 `reused=true`；源 V1 库在迁移后发生变化时
拒绝静默合并。只读快照的 `read_only=true` 且 `resumable=false`，V1 运行不能恢复成
V2 搜索。

```python
from mining import MiningV2Store

store = MiningV2Store()
legacy_runs = store.list_legacy_runs()
legacy = store.get_legacy_run(legacy_runs[0]["run_id"])
```

## 幂等轨迹

`MiningV2Store.begin_turn()` 在同一事务中登记假设、实验批次和 turn。假设父节点、
候选父节点、批次和候选必须属于同一运行；同一 turn 的相同请求返回原记录，内容变化
明确冲突。

一次 turn 的写入顺序：

```text
begin_turn
  -> record_candidate
  -> record_evidence
  -> record_decision
  -> transition_turn(decided -> completed)
```

重复调用相同对象不会追加事件或改变轨迹哈希。每个新对象和 turn 状态变化向
`mining_v2.trace_entry` 追加链式哈希，`run.trace_hash` 保存当前末端。
`validate_search_trace()` 会复算 previous/trace 链，并从当前假设、批次、候选、证据
和决策记录重新计算内容哈希；任一对象被外部修改都会抛出 `MINING-112`。

```python
trace = store.get_search_trace(run["run_id"])
store.validate_search_trace(run["run_id"])
```

轨迹只读取当前运行，不查询其他运行的候选、证据或结论。日志只记录运行、turn、对象
ID、请求哈希和重用状态，不记录完整假设、表达式或证据正文。

## 尝试台账与预算

`record_trial()` 只追加记录，不允许删除或覆盖。相同 `trial_key` 和相同内容幂等，
不同内容返回冲突；写入尝试、更新分类计数和推进轨迹哈希在同一事务完成。计数包括
提出、拒绝、重复、非法、门禁失败、快速评估、完整评估和测试段领取；作废操作追加
`trial_invalidation`，原尝试和原计数继续保留。

```python
trial = TrialRecord(
    run_id=run["run_id"],
    trial_key="turn-1-proposed-1",
    kind="proposed",
    turn_index=1,
    hypothesis_id=hypothesis_id,
    batch_id=batch_id,
    candidate_id=candidate_id,
)
store.record_trial(trial)
summary = store.trial_summary(run["run_id"])
```

`trial_summary()` 同时返回尝试总数、失败数、评估数、完整评估数和测试段领取数，避免
审核人只看到成功候选。`transition_candidate(..., trial=...)` 把候选状态和尝试记录
放在同一事务中；任何一步失败都会整体回滚。

预算在运行上下文中冻结。turn、分支数、每分支修正数、每 turn 候选数、每运行候选数、
完整评估数和测试段领取都会在执行入口校验；超过上限抛出 `MINING-111`。后续修改
`search.json` 只能影响新运行，不能为当前运行扩容。
