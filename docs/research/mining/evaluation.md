# 候选评估与质量门禁

## 职责边界

S5 接收一个已冻结的 `CandidateDefinition` 和 `FrozenContext`，重新执行 S3 校验，
把候选值物化到独立临时目录，再依次执行技术门禁、研究期快速筛选和可选完整评估。
它不改正式因子注册表、不写正式因子目录，也不推进候选状态或领取模块测试段。

## 公共接口

`CandidateEvaluationInput` 的必填内容是运行 ID、候选 ID、候选定义、冻结上下文和候选
根目录。调用方可以用 `run_full` 明确请求完整评估；默认只做技术门禁和快速筛选。

`evaluate_candidate()` 返回 `CandidateEvaluationResult`：

| 字段 | 含义 |
|---|---|
| `gate` | 可持久化的 `TechnicalGate`，逐项保存硬门禁证据 |
| `technical` | technical 阶段 `EvaluationReport` |
| `quick` | 研究期快速筛选报告；门禁失败时为空 |
| `full` | 完整评估报告；未请求、门禁失败或快速筛选失败时为空 |
| `warnings` | 技术、预测强度和完整评估警告 |
| `factor_glob` / `factor_sha256` | 候选 Parquet 年分区来源和树哈希 |

公开函数只返回结果。S6 应根据候选状态机选择调用时机，并通过 `record_evaluation()`、
`record_technical_gate()` 持久化。

S6 另有两个公开单阶段入口：

- `evaluate_full_candidate()` 复用研究期候选产物，只计算完整评估；
- `evaluate_holdout_candidate()` 在候选目录的 `holdout/factor/` 中独立物化测试段产物，
  不覆盖研究期因子值。

模块测试段的通过标准只检查评估链路完整、无执行错误且至少 20 个有效交易日。
RankIC 等预测指标原样保存，但不用于通过判定、候选重排或再次生成。

## 数据流

```text
CandidateDefinition + FrozenContext
  -> S3 重新校验表达式和语义哈希
  -> 生成 staging/year=YYYY/part.parquet
  -> 原子替换候选 factor/
  -> 技术门禁
  -> 研究期轻量 RankIC
  -> [调用方要求] 完整评估全部 stage
```

候选因子路径固定为：

```text
data/factor_mining/runs/<run_id>/<candidate_id>/factor/year=YYYY/part.parquet
```

评估报告写入同级 `evaluations/<stage>.json`。写入前先写 `.tmp` 再原子替换；任何失败
都不留下 staging 或半完成年分区。正式因子目录只能作为只读对照，候选根目录落在
`data/factors/` 内会直接失败。

## 共享评估口径

`calculate.FactorView` 描述一个因子的展示元数据、目标 Parquet glob 和正式因子对照
目录。正式注册因子由 `resolve_factor()` 构造；候选由 mining 构造。`build_panel()`、
`load_daily_ic()` 和 `evaluate_context()` 接受该视图，因此两条路径共用：

- 全 A 池、停牌过滤、`fwd_ret_open_5` 标签和后复权因子值；
- float32 因子落盘、非有限值置空；
- Pearson IC、RankIC、分位、衰减、风格暴露和增量 IC 的全部既有计算。

候选相关性只计算候选对正式因子，避免在 S5 中重复构建正式因子两两全矩阵。监测站
正式因子的完整相关性行为不变。

## 质量门禁

配置位于 `configs/mining/evaluation.json`，当前规则版本为 `mining_quality_v1`：

| 检查 | 硬门禁 |
|---|---|
| 行集 | 与研究期 `md.daily_hfq` 双向差集为 0 |
| 取值 | 非有限值为 0；覆盖率不低于冻结上下文 `min_coverage`，默认 30% |
| 截面退化 | 至少 5 个非空值的交易日中，常数日不超过 1% |
| 前视 | 首、中、末三个交易日截断重算，相对误差不超过 `1e-5`，空值差异为 0 |
| 正式因子相关性 | 最多 100 个确定性交易日，单日至少 5 只股票，成对至少 20 日；最大绝对平均秩相关小于 0.90 |

无法建立足够相关性样本时 fail-closed。RankIC 均值、ICIR 和年度同号率只作为筛选指标
与警告；有效交易日不足 20 天时快速筛选判失败。

## 输入哈希

每个阶段报告都包含 `input_hash`，输入覆盖：

- 运行 ID、候选 ID、候选内容哈希和冻结上下文哈希；
- 候选因子树哈希、数据版本和数据指纹；
- 标签、股票池、价格通道、时间和评估参数；
- 完整质量规则版本与阈值。

修改阈值、正式因子清单、字段目录、算子目录、数据快照或候选值都会使旧评估证据
失效，不能继续用于审批。
