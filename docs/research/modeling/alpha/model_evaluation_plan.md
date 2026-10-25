# 通用模型评估计划

日期：2026-10-23

## 职责

评估计划描述比较哪些来源、如何变换分数、如何持仓、谁是参照，以及候选通过后
如何排序。执行器不识别具体模型类型或固定候选名。

## v3 计划

```json
{
  "schema_version": "alpha_model_evaluation_plan_v3",
  "reference": "RIDGE_RAW",
  "risk_gate": {
    "rolling_weeks": 13,
    "min_periods": 8,
    "max_size_abs_exposure": 2.0,
    "max_liquidity_abs_exposure": 2.0,
    "max_industry_active_weight": 0.35
  },
  "gate": {
    "expected_years": [2018, 2019, 2020, 2021, 2022, 2023],
    "required_improvements": 2,
    "turnover_multiple_limit": 2.0
  },
  "selection_metrics": [
    "annual_return",
    "sharpe",
    "max_drawdown",
    "turnover_two_way_annual"
  ],
  "candidates": [
    {
      "name": "CHALLENGER",
      "source": {"kind": "model", "id": "<model_id>"},
      "score_policy": {"kind": "smooth_rank", "smoothing_weeks": 4},
      "holding_policy": {"kind": "top_n", "top_n": 30},
      "priority": 2,
      "selectable": true
    }
  ],
  "controls": []
}
```

来源支持：

- `model`：按模型 ID 读取标准预测；
- `baseline`：按基线 ID 从数据集计算分数；
- `blend`：由多个模型 ID 和权重做逐日 rank 融合。

基线和融合候选必须设置 `selectable=false`。融合可以评估，但不会伪装成已有独立
模型产物，也不会被自动选为主模型。

## 兼容性

- v1 计划继续可读；缺失的新字段使用历史默认值。
- 新计划写 v2。
- 同一计划引用的模型必须具有相同数据集、目标、折和预测日期/股票集合。
- 计划内容、模型 manifest、预测、数据集和代码哈希共同生成确定性分析 ID。
- v1/v2 计划继续可读；未写 `risk_gate` 时保持旧行为。
- v3 风险门槛按候选目标持仓计算 13 周滚动绝对暴露，并要求至少 8 个有效观测；
  经济门槛和风险门槛都通过才允许选中。
- 评估产物保存每个候选的规范化分数、风格时序、行业时序和风险摘要，后续完整
  账本直接读取这些不可变文件，不再重算融合或平滑。

## 执行入口

```bash
.venv/bin/python -m scripts.build_alpha.evaluate_plan \
  --plan configs/alpha/development_evaluation_v85.json
```

输出在 `data/models/runs/evaluation/<analysis_id>/`，其中 `scores/<candidate>.parquet`
供完整账本使用，`risk/<candidate>/` 保存风险检查证据。计划非法、来源不一致、
风险观测不足或产物损坏时整体失败关闭，不产生部分决策。

相同输入命中已有评估目录时，会重新核对计划、来源、数据集、代码哈希以及 manifest
声明的全部输出文件。任一文件损坏时拒绝复用并要求显式 `--force` 重建。
