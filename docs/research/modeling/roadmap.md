# ML 模型层施工计划

日期：2026-10-19  
最后整理：2026-10-23（V86）

## 当前定位

ML 模型层负责把可见因子转换成截面预测分数，不负责组合约束、交易执行和成本。
系统的目标是提供可替换、可复现、可独立验收的模型模块，不维护某个固定主模型。

当前模型：

- Ridge：活跃，用于线性参照和生命周期回退。
- 五种子 LightGBM：活跃，作为已完成的实验主模型保留。
- 直方图 GBDT：训练候选，已完成开发期评估、完整账本、监控和生命周期。
- 单种子 LightGBM：已归档，不再参与当前选择。

2024-01-01～2026-10-15 已经作为最终留出样本消耗，不能用于新模型调参。

## 已完成

| 阶段 | 结果 |
|---|---|
| P0～P1 | 冻结 5 日标签、周频调仓和全 A 股票池；完成 47 因子审计、数据集与 purged walk-forward |
| P2～P3 | 完成 Ridge、LightGBM 单模型、五种子集成、重要性消融和 P3 候选决策 |
| P4 | 标准预测接入 `ml_rank`，完成完整账本、容量和生命周期验收 |
| P5 | 完成模型登记、离线监控、生命周期规则和模型卡 |
| V83 | 2024～2026 最终留出验证失败；收益高度集中于 2025 年，2024 年净超额 -31.46% |
| V84 | 完成模型无关的通用训练、预测、评估计划和分层产物目录 |
| V85-ML-1 | 新增直方图 GBDT 后端并实际训练登记 `sklearn_hist_gbdt_label_rank_83e1200e554f` |
| V85-ML-2 | 历史 P3 八候选迁移到通用计划执行器，选中结果和指标与旧结果一致 |
| V85-ML-3 | 登记 2026-10-16～2027-10-15 新窗口，2024～2026 只读封存 |
| V85-ML-4 | 增加 13 周滚动组合风险门槛，覆盖市值、流动性和行业主动暴露 |
| V85-ML-5 | 新模型完成完整账本、监控和生命周期，产物与快照来源一致 |
| V85-ML-6 | 新增计划驱动完整账本入口，候选、分数和持仓规则全部来自评估产物 |
| V86 | 后端自动发现、训练自动登记、索引聚合下游运行、恢复复用失败关闭 |

V83 的正式验证 ID 为 `final_holdout_2ba777e855f5`，状态
`fail_final_holdout`。该结果证明当前模型相对 B4 没有稳定经济增量，但不影响
继续建设可替换模型体系。

## 当前架构

```text
数据集与标签
  -> 通用训练流水线
  -> ModelBackend
  -> 标准预测与 manifest
  -> 模型索引与下游运行关联
  -> ModelEvaluationPlan
  -> 通用评估执行器
  -> P3 / 完整账本 / 最终留出
```

核心约束：

- 新模型实现并注册 `ModelBackend`，不修改共享折循环。
- 模型只输出 `trade_date`、`stock_code`、`score` 标准预测。
- 历史 v1 manifest 读取时归一化；新模型写 v2 manifest。
- 评估、回测、监控和验收只通过标准产物和模型 ID 使用模型。
- 模型 ID 可以出现在文档示例和执行计划中，不能成为主链路分支条件。
- ML 侧不要求回测 provider 识别模型类型；回测只按 `model_id` 读取预测。
- 模型索引以 manifest 为训练事实，并聚合下游运行；多个评估可以引用同一模型。

## 目录

```text
research/modeling/alpha/modeling/
  contracts.py       # TrainingSpec、ModelArtifact、ModelBackend
  pipeline.py        # 通用折循环、重拟合、预测和产物
  backends/          # linear、lightgbm、sklearn_hist_gbdt
  provenance.py      # 模型 ID 和源码哈希
  manifests.py       # v1/v2 manifest 归一化
  forecast.py        # 按冻结折生成外部预测

data/models/
  models/            # 活跃模型
  archive/models/    # 归档模型
  runs/acceptance/   # 完整账本与生命周期验收
  runs/final_holdout/# 独立最终留出
  runs/index.json    # 运行状态和留出样本消耗记录
  index.json         # 模型 v3 索引与下游运行链接
```

## 当前接口

```python
fit_model(dataset_id: str, spec: TrainingSpec) -> ModelArtifact
register_backend(backend: ModelBackend) -> None
get_backend(name: str) -> ModelBackend
load_predictions(model_id: str, start=None, end=None) -> tuple
load_model_manifest(model_id: str) -> dict
rebuild_registry(root: Path | None = None) -> dict
```

训练命令：

```bash
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id <dataset_id> \
  --spec configs/alpha/<training_spec>.json
```

模型接入、manifest 和扩展方式见：

- `docs/research/modeling/alpha/model_layer_architecture.md`
- `docs/research/modeling/alpha/training_backends.md`
- `docs/research/modeling/alpha/model_evaluation_plan.md`
- `docs/research/modeling/alpha/model_inventory.md`

## 后续待办

V85-ML-4～6 已完成。新评估 ID 为 `evaluation_d57dd9e6e750`；新模型
`SKLEARN_HIST_RAW` 的完整账本、监控和生命周期 ID 分别为
`acceptance_2c80425c2cfe`、`p5_2_5494d7f2b910`、`p5_3_65282c575784`。
V86 已完成工程闭环。下一步需要等 2026-10-16～2027-10-15 数据齐备、5 日标签
完整度至少 95% 且有效周频信号至少 100 个后，执行新的独立留出验证。

## 验收标准

模块完整的标准不是当前模型收益为正，而是：

1. 新模型族只需实现后端即可完成训练、预测和产物登记。
2. 模型产物可从标准 manifest 和预测独立读取。
3. 评估计划可以更换候选和参照，不需要修改共享代码。
4. 回测、监控和验收不依赖具体模型类型。
5. 历史模型可审计，新模型有独立未来样本验证。

当前回归结果：仓库根级全集 `908 passed`，回测子仓全集 `351 passed`，
`pip check`、编译和索引重建通过。
V85-ML-1～6、V86 已完成。

详细历史和逐项结果见 `AiRecord/progress/`、`AiRecord/changes/` 与
`AiRecord/diffs/`。
