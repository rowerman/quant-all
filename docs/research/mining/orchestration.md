# 挖掘编排

## 职责

S6 `MiningOrchestrator` 负责把已验收的 Worker、DSL、评估和候选存储组合成一条运行。
它不修改正式因子库，不审批候选，也不执行正式发布。

```text
创建运行并冻结数据/目录/预算
  -> 多轮 Worker 生成
  -> S3 二次校验和去重
  -> 技术门禁
  -> 研究期快速筛选
  -> 前 K 个候选完整评估
  -> 冻结候选批次并唯一领取测试段
  -> review_ready
```

## 预算与停止

冻结预算包括轮次、候选数、完整评估数、LLM 调用数、表达式深度、覆盖率、失败率和失败
检查门限。默认配置见 `configs/mining/orchestrator.json`。

- 每轮 Worker 作业计一次 LLM 调用，最多请求 `max_candidates_per_round` 个候选。
- 每个入库候选执行一次技术和快速筛选，质量检查总数受 `max_candidates` 约束。
- 至少检查 4 个候选后，失败率达到 75% 即终止运行。
- 一轮没有新增合法候选时停止生成。
- Worker、协议、存储或评估引擎错误直接失败，不做 Mock 或旧候选回退。

## 候选冻结与测试段

快速筛选按 RankIC 绝对值、ICIR 绝对值、覆盖率和候选哈希确定性排序。测试段领取前，
技术、quick 和 full 报告必须齐全，并以候选哈希和报告 ID 生成批次证据哈希。

测试段在同一候选数据库中只能领取一次。恢复只允许复用原领取批次完成评估，不能换候选、
改顺序或再次领取。测试段结果不会进入后续 Worker 请求。

## 快照与恢复

创建运行和每个阶段边界都会重新核对行情数据指纹、字段目录、算子目录和正式因子清单。
任一变化会把运行和全部未终态候选原子标记为 `stale`，错误码为 `MINING-120`。

进程重启时只把活动运行标为 `interrupted`，不自动重放。显式 `resume()` 后：

- 已完成的 Worker 轮次从事件恢复，不再次调用；
- 已持久化的报告、门禁和测试段领取记录直接复用；
- 未完成阶段从中断前状态继续；
- 已完成预算从事件和候选记录重新统计。

## 接口

```python
orchestrator = MiningOrchestrator()
run = orchestrator.create_run(request)
progress = orchestrator.progress(run["run_id"])
orchestrator.cancel(run["run_id"])
orchestrator.resume(run["run_id"])
orchestrator.close()
```

`create_run()` 默认后台执行。测试或批处理可传入 `background=False`，再显式调用
`execute()`。S7 只能通过这些公开方法操作运行，不直接写候选数据库。
