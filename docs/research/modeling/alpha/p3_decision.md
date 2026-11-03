# 通用评估与 P3 决策

日期：2026-10-23

## 结论

历史 P3 候选矩阵已迁移到通用评估计划。正式结果仍是四周平滑的五种子 LightGBM：
`lightgbm_ensemble_label_rank_6aea5c9299d5` + `smooth_rank(window=4)`，
Top-30 等权。加入直方图 GBDT 和组合风险门槛后的分析 ID 为
`evaluation_d57dd9e6e750`。

新执行器的 8 个候选与旧 P3 结果使用同一数据、指标和模拟器；核心指标最大差异
为 `5.4e-8`，选中候选一致。旧分析 `p3_5_5c02d3520f25` 保持只读。

## 计划驱动规则

- 参照、候选、控制、预期年度和决策门全部来自评估计划；
- RankIC、ICIR、年化净收益、最大回撤至少两项严格优于参照；
- 双边年换手不超过参照两倍；
- 每个预期年度的 RankIC 和净超额必须为正；
- 可选中候选按计划指定的指标顺序排序，再使用候选优先级和名称打破平局；
- 没有候选通过时保留参照；
- 基线和融合候选可以评估，但必须显式标记为不可自动选中。

开发期计划为 `configs/alpha/development_evaluation_v85.json`。其中 50/50 融合
仍保留在候选矩阵中，但不会伪装成独立模型产物。

## 候选结果

区间 2018-01-12～2023-12-29，周频研究模拟器，Top-30 等权。

| 候选 | RankIC | ICIR | 年化净收益 | 最大回撤 | 双边换手 | 决策门 |
|---|---:|---:|---:|---:|---:|---|
| Ridge 原始 | 0.0911 | 0.7808 | 14.07% | -32.47% | 88.63 | 参照 |
| LightGBM 原始 | 0.0973 | 0.8288 | 17.42% | -26.53% | 80.70 | pass |
| Ridge 平滑 4 周 | 0.0812 | 0.6576 | 17.98% | -31.57% | 48.19 | fail |
| LightGBM 平滑 4 周 | 0.0851 | 0.6815 | 30.92% | -28.13% | 43.00 | pass，选中 |
| Ridge buffer=45 | 0.0911 | 0.7808 | 15.84% | -32.32% | 83.36 | fail |
| LightGBM buffer=45 | 0.0973 | 0.8288 | 19.68% | -26.68% | 73.15 | pass |
| 直方图 GBDT 原始 | 0.0965 | 0.8611 | 25.57% | -29.89% | 85.55 | pass |
| 直方图 GBDT 平滑 4 周 | 0.0838 | 0.7096 | 28.38% | -29.55% | 44.28 | fail |
| LightGBM/Ridge 50/50 融合 | 0.0977 | 0.8175 | 20.33% | -33.07% | 85.01 | fail |
| B4 全因子 rank | 0.0562 | 0.5320 | 11.03% | -33.44% | 65.57 | 控制 |

平滑候选以年化净收益和最大回撤两项改善通过决策门，但 RankIC 和 ICIR 低于
Ridge。它是固定规则下的收益与换手取舍，不应表述为预测能力全面提升。

## 执行与产物

```bash
.venv/bin/python -m scripts.build_alpha.evaluate_plan \
  --plan configs/alpha/development_evaluation_v85.json
```

输出目录：`data/models/runs/evaluation/evaluation_d57dd9e6e750/`，包含
`candidates.csv`、`yearly.csv`、`weekly_results.parquet`、`orders.parquet`、
`plan.json`、`decision.json`、`report.md` 和 `manifest.json`。

旧 `decide_p3`、`p3_decision` 和 `p3_report` 入口已删除。新链路由
`alpha.evaluation_plan`、`alpha.evaluate` 和
`scripts.build_alpha.evaluation_executor` 组成。

## 边界

- 结果来自周频研究模拟器，不是完整回测账本；
- 2024～2026 已封存，不得用于本轮或后续调参；
- 新模型已进入候选评估并通过经济与风险门槛，但全量选择仍保留历史 LightGBM
  四周平滑；
- 新模型监控、生命周期和完整账本由 V85-ML-5 独立生成，不代表模型晋升。
