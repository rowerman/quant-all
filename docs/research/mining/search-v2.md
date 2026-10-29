# V2 搜索轨迹基础

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

## 当前边界

M1 不创建 Worker V2 请求，不改变 V1 快速筛选和完整评估窗口。P2 接入逐轮闭环时，
主服务必须使用本文件的角色访问接口构造请求和证据，不得把完整 V2 上下文直接交给
Agent。

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
