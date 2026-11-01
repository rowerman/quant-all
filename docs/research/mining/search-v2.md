# V2 搜索轨迹与闭环

`mining_v2` 为证据驱动搜索提供独立契约和存储。M1～M4 已覆盖轨迹、逐轮闭环、
组合价值、稳定性、统计诊断和受控分析工具；M5 代码沙箱独立于 V2 搜索。
V1 生产编排仍保持默认，V2 通过 `MINING_SEARCH_V2_ENABLED=true` 启用。
V2 窗口、预算和角色配置位于 `configs/mining/search.json`。

当前 V2 运行在一次性测试段后进入 `review_ready`。它尚未接入人工审批和正式发布，
后续 V2 链路补齐及 V3 最终闭环见
[根目录进度记录](../../../FACTOR_MINING_PROGRESS.md)。

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

## 受控分析工具

M4 在 Worker v2 上增加工具感知的 `mining_worker_v3` 补充协议。每轮首次请求可以返回
最多 4 个固定工具调用；主服务校验并执行后，以同一 turn 的一次 finalize 请求返回脱敏
结果摘要，Worker 随后必须给出最终决策。没有工具需求时仍可直接返回最终响应，历史
Worker v2 响应继续可读。

首批工具为 `lookup_fields`、`lookup_operators`、`run_summary`、
`candidate_evidence`、`candidate_correlation`、`candidate_ablation` 和
`window_neighborhood`。工具没有 SQL、表名、路径、日期切片或任意代码参数，结果只来自
当前运行的冻结开发期上下文。

工具目录使用 `mining_analysis_tool_v1`；单次结果不超过 64 KiB，默认超时 5 秒。缓存
键包含运行、上下文哈希、turn 请求哈希、候选哈希、工具版本和规范化参数，缓存只存在
于同一运行；命中缓存也仍计一次工具调用。缓存和每次调用审计分别保存在
`mining_v2.analysis_tool_cache` 与 `mining_v2.analysis_tool_audit`。

工具失败使用稳定错误码：`MINING-127` 未知工具、`MINING-128` 参数非法、
`MINING-129` 越权、`MINING-130` 超时或取消、`MINING-131` 结果过大、`MINING-132`
执行失败。失败会登记审计并阻断当前 turn，不会转换成空结果，也不会扩大日期或回退
查询。审计保存候选、参数摘要、输入与结果哈希、缓存命中、耗时和脱敏错误；日志只记录
工具名、版本、哈希和耗时。

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

## 选择期诊断

选择期确定性控制器会为通过快速筛选的候选登记 `selection_diagnostics` 证据。同批
候选相关性使用冻结选择期内最多 100 个均匀采样交易日，所有候选对齐到同一日期和股票
网格后逐日计算 Spearman。缺失配对不补零；常数截面、有效日不足和单候选场景分别保留
明确状态。绝对相关度不低于 `0.90` 的候选通过连通分量归入同一相关簇。

正式因子相关性复用选择期技术门禁的最高绝对相关度。增量价值使用选择期评估面板上的
风格残差 RankIC，同时保存原始 RankIC、残差 RankIC、保留比例和有效日数。诊断只写入
当前运行的 V2 评估证据，不进入 Worker 请求，也不读取最终研究期或模块测试段。

`selection_diagnostics` 报告包含稳定内容哈希；评分卡 v2 在 `analysis_by_stage` 中保留
完整诊断。重复打开运行直接复用已登记诊断，不以失败、缺失或不可计算结果改变候选状态。

## 邻域稳定性

每个进入确定性选择的候选从受控 AST 深度优先选择第一个可调整整数窗口，只生成窗口
左右各一个合法相邻版本，总数最多两个。变体重新执行主服务的表达式解析、目录校验、
稳定命名和候选哈希，随后只在开发期执行技术门禁和快速评估。

稳定性报告记录基础 RankIC、所有变体的窗口、方向、RankIC、ICIR、门禁和错误码。
基础版本和变体少于两个方向一致，或变体 `|RankIC|` 中位数低于基础版本的 `50%` 时
标记不稳定；有效指标不足时标记不可计算，不以零值补齐。报告写入 `stability` 评估
证据，恢复时直接复用，不进入假设树或 Worker 下一轮候选。

## Pareto 入围与研究质量

选择期诊断和稳定性证据完成后，控制器按 `|RankIC|`、`|增量 RankIC|`、稳定性分数和
覆盖率建立四目标 Pareto 前沿。目标均越大越好；证据缺失不按零填充，相关簇通过同批
候选相关性连通分量确定，每个簇最多一个候选进入最终研究期评估。

Pareto 同层按质量、增量价值、稳定性、覆盖率降序，再以候选内容哈希和候选 ID 升序
稳定排序。选择结果、Pareto 层级、排除原因和内容哈希写入 `finalists_selected` 事件；
相关性阈值、稳定性变体上限和 RankIC 保留比例在运行上下文中冻结。

执行状态和研究质量分开展示。`execution_failed` 表示该阶段报告失败或报错；
`evidence_insufficient` 表示执行成功但所需指标缺失；`quality_failed` 表示执行成功且
指标完整但未达到冻结门槛；`quality_passed` 表示执行成功且达到门槛。质量通过只用于
审核判断，不等于人工审批或正式发布。

## 多重检验诊断

选择期诊断保存逐日 RankIC 的日期和数值。统计报告使用 Bartlett 核 Newey-West HAC
标准误检验平均 RankIC 是否为零，带宽为
`floor(4 * (n / 100) ** (2 / 9))`；少于 20 个有效交易日或标准差为零时标记不可计算。

测试得到的两侧 p 值使用 BH 方法生成 q-value。分母取主候选尝试、非法或重复尝试和
稳定性变体与可检验候选数中的最大值，避免把大量失败搜索从搜索规模中隐藏。报告保存
尝试数、评估数、可检验数和选择规则，但 q-value 不参与 Pareto 排序，也不自动修改
候选状态、审批或发布。

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
