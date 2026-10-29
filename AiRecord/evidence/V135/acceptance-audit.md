# V135 智能因子挖掘路线图验收审计

- 日期：2026-10-29
- 范围：`INTELLIGENT_FACTOR_MINING_TODO.md` 中 P1～P5 共 34 项
- 结论：34 项均保持“已完成”；未发现需要修改运行代码、测试、配置或数据的缺口

## M1 轨迹与防泄漏基础

| 条目 | 审计证据 | 结论 |
|---|---|---|
| P1.1 | `tests/test_mining_search_contracts.py` 严格字段、上限、动作和稳定哈希 | 通过 |
| P1.2 | `tests/test_mining_v2_storage.py` 事务迁移、回滚、只读兼容和未知 schema | 通过 |
| P1.3 | `tests/test_mining_v2_trace.py` 幂等 turn、父节点隔离和链式哈希 | 通过 |
| P4.1 | `tests/test_mining_search_context.py` 冻结窗口、角色隔离和越界字段拒绝 | 通过 |
| P4.2 | `tests/test_mining_v2_trials.py` 不可删除台账、分类计数和预算事务 | 通过 |

## M2 证据驱动闭环

| 条目 | 审计证据 | 结论 |
|---|---|---|
| P1.4 | `tests/test_mining_search_orchestrator.py` 恢复、取消和 turn 边界 | 通过 |
| P1.5 | `apps/monitor/backend/tests/test_mining_search_api.py` 页面所需轨迹与进度数据 | 通过 |
| P2.1 | `tests/test_mining_worker_protocol_v2.py`、Worker 服务和主服务二次校验 | 通过 |
| P2.2 | `tests/test_mining_search_evidence_builder.py` 脱敏开发期证据和稳定哈希 | 通过 |
| P2.3 | `tests/test_mining_search_control.py` 预算、无改善、失败和确定性停止 | 通过 |
| P2.4 | `tests/test_mining_search_orchestrator.py` 逐轮提出、评估、反馈和决策 | 通过 |
| P2.5 | `tests/test_mining_v2_search_control_storage.py` 运行/候选状态迁移 | 通过 |
| P2.6 | 编排恢复测试覆盖超时、重启、快照失效、取消和重复恢复 | 通过 |
| P2.7 | `AiRecord/evidence/V129/search-v2-short-e2e.json` 为 `review_ready`，6 条评估证据 | 通过 |

## M3 组合价值与统计质量

| 条目 | 审计证据 | 结论 |
|---|---|---|
| P4.3 | `tests/test_mining_search_scorecard.py` 证据引用、缺失字段和稳定评分卡 | 通过 |
| P4.4 | `tests/test_mining_selection_diagnostics.py` 相关簇、正式因子相关和增量价值 | 通过 |
| P4.5 | `tests/test_mining_search_stability.py` 邻域变体、方向一致和 50% 门槛 | 通过 |
| P4.6 | `tests/test_mining_search_portfolio.py` Pareto、相关簇去重和稳定排序 | 通过 |
| P4.7 | 质量测试与 API 测试区分执行失败、质量不足、通过和证据不足 | 通过 |
| P4.8 | `tests/test_mining_statistical_diagnostics.py` Newey-West、BH 和保守分母 | 通过 |
| P4.9 | `apps/monitor/backend/tests/test_mining_search_api.py` 尝试、质量、稳定性和统计字段 | 通过 |

## M4 受控分析工具

| 条目 | 审计证据 | 结论 |
|---|---|---|
| P3.1 | `tests/test_mining_analysis_tools.py` 固定工具名称、版本和严格参数/响应 | 通过 |
| P3.2 | 工具网关测试拒绝 SQL、路径、表名、连接和任意代码 | 通过 |
| P3.3 | 工具测试覆盖每轮 4 次预算、运行内缓存、上下文漂移和超时 | 通过 |
| P3.4 | `MINING-127`～`MINING-132` 稳定错误码和失败阻断 | 通过 |
| P3.5 | 审核 API 区分主证据、工具补充证据和失败调用 | 通过 |
| P3.6 | 工具、Worker v3 恢复、并发和越界日期负向测试 | 通过 |

## M5 隔离代码沙箱

| 条目 | 审计证据 | 结论 |
|---|---|---|
| P5.1 | `tests/test_mining_sandbox.py` 政策、威胁模型和稳定错误码 | 通过 |
| P5.2 | 运行器测试和源码检查确认断网、只读根、非 root、资源限制 | 通过 |
| P5.3 | V133 证据和实际清单复算哈希 `a52227aafc54df39…` | 通过 |
| P5.4 | `tests/test_mining_sandbox_research.py` 有限代码提案、执行和恢复 | 通过 |
| P5.5 | V134 产物文件树校验，清单哈希 `c4d266d3dae3160b…` | 通过 |
| P5.6 | 人工提升测试确认只写 review，固定 `formal_candidate_created=false` | 通过 |
| P5.7 | 安全恢复测试、固定镜像检查和无残留 `mining-sandbox-*` 容器 | 通过 |

## 命令与结果

| 批次 | 定向测试 |
|---|---:|
| M1 | 102 passed |
| M2 | 124 passed |
| M3 | 85 passed |
| M4 | 83 passed |
| M5 | 36 passed |

测试中出现 2 个既有 Starlette/httpx 弃用警告。没有调用付费模型，没有运行新的真实搜索或
沙箱容器，没有领取测试段。

## 追踪检查

- TODO 中任务数和“已完成”状态数均为 34。
- 未发现未开始、进行中、阻塞、已取消或未勾选条目。
- TODO 引用的代码、文档和 AiRecord 路径均存在。
- TODO 引用的 AiRecord Markdown/JSON 证据均受 Git 跟踪。
- V129、V133、V134 三份 JSON 证据可解析。
- V128 的 18 份计划、变更和差异记录已从本地忽略状态补入 Git。

## 限制

- 本次没有重跑全量测试；审计未修改运行代码，最近提交已有 `1483 passed` 记录。
- 本次复用 V129、V133、V134 的真实运行证据，没有重复产生付费或大数据副作用。
