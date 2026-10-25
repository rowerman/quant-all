# 通用完整账本验收

日期：2026-10-23

## 职责

`scripts.build_alpha.accept_plan` 把一次已通过的通用评估扩展到完整账本。候选集合、
评分规则、平滑周数和持仓规则全部读取评估 manifest，不维护模型类型或 P4 固定候选
分支。

```text
评估计划
  -> 不可变候选分数与风险时序
  -> 通用完整账本
  -> 监控 / 生命周期 / 运行索引
```

## 命令

```bash
.venv/bin/python -m scripts.build_alpha.accept_plan \
  --evaluation-id <evaluation_id> \
  --candidate selected \
  --resume
```

`--candidate` 支持 `selected`、`reference`、`control`、`all` 或评估计划中的精确
候选名。`all` 会按计划顺序串行执行，建议先用小范围验证再运行全量。

## 来源与失败语义

- 评估状态必须是 `pass_evaluation`，manifest 的 ID 必须与目录一致。
- 每个候选读取 `scores/<candidate>.parquet`，并校验 manifest 中的 SHA256。
- 分析 ID 绑定评估 manifest、候选顺序、范围、资金参数和代码哈希。
- 同一 ID 下只有完成标记及全部文件哈希通过时才允许 `--resume` 跳过。
- 资金分配、补回、成本、日期、快照和信号日检查沿用完整账本统一验收；任一失败
  时总状态不能是 `pass_acceptance`。

## 产物

```text
data/models/runs/acceptance/<acceptance_id>/
  manifest.json
  sources.json
  comparison.csv
  yearly.csv
  report.md
  <candidate>/
    metrics.json
    strategy.json
    yearly.csv
    daily.parquet
    orders.parquet
    holdings.parquet
    run_manifest.json
    complete.json
```

`strategy.json` 使用 `plan_rank` v1 provider，只带评估 ID、候选名和显式调仓日；
模型、基线或融合的分数差异已经固化在评估候选中。

## 新模型正式结果

- 评估：`evaluation_d57dd9e6e750`
- 候选：`SKLEARN_HIST_RAW`
- 账本：`acceptance_2c80425c2cfe`
- 快照：`3435962234b9542281027c01ccb437bde6836aa776a482d0fd0a7621e433767a`
- 300 个信号日、299 个可执行段
- 年化净收益 21.85%、净超额 23.18%、Sharpe 0.8237、最大回撤 -34.39%、
  双边年换手 76.58
- 资金分配、补回、成本、日期、快照和信号日检查全部通过

## 修改注意

1. 不得让完整账本重新训练模型或重新选择候选。
2. 不得把周频研究模拟器结果写成完整账本结果。
3. 不得覆盖相同评估下的已有账本；代码或输入变化应生成新 ID。
4. 监控和生命周期必须引用同一个候选、评分规则和账本来源。
