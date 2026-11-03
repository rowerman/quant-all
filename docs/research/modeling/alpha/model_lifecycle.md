# 模型生命周期

日期：2026-10-22

## 职责

`alpha.lifecycle` 把 P5-2 监控结果转换为可复现的生命周期状态和回测动作。它不训练
新模型、不重跑回测，也不修改历史模型、预测或监控产物。`ml_rank` v3 是生命周期
规则在回测中的执行边界。

生命周期状态：

| 状态 | 活动模型 | 仓位乘数 | 含义 |
|---|---|---:|---|
| `normal` | 五种子 LightGBM | 1.0 | 正常持有 Top-30 |
| `reduced` | 五种子 LightGBM | 0.5 | 降低模型风险暴露，其余保留现金 |
| `rollback` | Ridge | 1.0 | 回滚到已通过 P2 决策的模型 |
| `paused` | 无 | 0.0 | 下一可执行日清仓 |

## 固定规则

- 主模型：`lightgbm_ensemble_label_rank_6aea5c9299d5`
- 回滚模型：`ridge_label_rank_aaff2cb2fbcf`
- 监控滚动窗口：13 个信号周，至少 8 个有效观测
- 连续 4 期 warning 或 alert：降权
- 连续 2 期 alert：回滚 Ridge
- 连续 4 期 alert：暂停
- 可交易股票分数缺失率超过 5%：立即暂停
- 连续 4 期 `ok`：恢复主模型满仓
- `insufficient_data` 不触发恶化，也不计入恢复

阈值复用 `MonitorPolicy`，在正式回测前固定。结果出来后不为改善收益调整阈值。

## 因果口径

状态在信号日收盘后生成，次一交易日开盘执行：

1. 滚动 RankIC 使用上一信号日值，因为当日标签包含未来 5 日收益。
2. 超额、回撤、换手、缺失和暴露使用信号日收盘时已经可知的数据。
3. 不直接使用监控表中的 `status`，该状态包含当日未来标签 RankIC。
4. 未来监控数据变化不得改变此前状态。

## 重训规则

年度重训使用扩展窗口，与既有六折一致：

- 起始训练样本固定从 `2012-01-04` 开始。
- 每年首个信号日记录 `retrain_due`。
- 训练截止于前一年，并留 6 个交易日 purge。
- 使用前一完整自然年作为验证集，测试年开始前留 6 个交易日 embargo。
- 挑战者必须通过既有 P3-5 决策门，并具备完整样本外预测。

本轮只执行和审计重训日程，不自动训练或晋升挑战者。未经完整样本外验证的新模型
不能因为重训到期直接进入回测。

## 正式结果

- 生命周期 ID：`p5_3_dd7c9b4bf235`
- 完整账本验收 ID：`p5_3_4f91d5b68fba`
- 区间：2018-01-12～2023-12-29，300 个信号日、299 个可执行段
- 状态日数：normal 170、reduced 107、rollback 6、paused 17
- 动作：6 次降权、2 次回滚 Ridge、1 次暂停、5 次恢复、5 次重训到期
- 年化净收益 20.40%、净超额 21.73%、最大回撤 -27.87%、双边年换手 31.68
- 资金分配和补回均收敛，无未收敛段、无现金裁剪格

未启用生命周期的 P4-4A 候选年化净收益为 28.35%、最大回撤 -28.13%。
本轮生命周期降低了收益和换手，并将最大回撤小幅降至 -27.87%；正式验收只判断
规则可执行性，不据此调整阈值。

## 接口与命令

```python
from alpha.lifecycle import (
    LifecyclePolicy,
    evaluate_lifecycle,
    load_model_lifecycle,
    run_model_lifecycle,
)
```

```bash
.venv/bin/python -m scripts.build_alpha.lifecycle
.venv/bin/python -m scripts.build_alpha.lifecycle --force
```

通用账本模型可显式指定主模型、回退模型和候选平滑周数：

```bash
.venv/bin/python -m scripts.build_alpha.lifecycle \
  --model-id <model_id> \
  --fallback-model-id ridge_label_rank_aaff2cb2fbcf \
  --monitor-id <monitor_id>

.venv/bin/python -m scripts.build_alpha.lifecycle_accept \
  --lifecycle-id <lifecycle_id> \
  --model-id <model_id> \
  --fallback-model-id ridge_label_rank_aaff2cb2fbcf \
  --smoothing-weeks 1
```

正式生命周期包：

```text
data/models/<主模型ID>/lifecycle/<lifecycle_id>/
```

包含：

- `timeseries.csv`：逐信号日的因果观测、计数、状态、活动模型和仓位乘数；
- `events.csv`：降权、回滚、暂停、恢复和年度重训事件；
- `report.json`、`report.md`：机器和人工可读摘要；
- `manifest.json`：来源、策略、代码和文件哈希；
- `complete.json`：完成标记和全部结果文件 SHA256。

V85 新模型生命周期：

- 生命周期 ID：`p5_3_65282c575784`
- 主模型：`sklearn_hist_gbdt_label_rank_83e1200e554f`
- 回退模型：`ridge_label_rank_aaff2cb2fbcf`
- 监控：`p5_2_5494d7f2b910`
- 状态日数：normal 197、reduced 56、rollback 29、paused 18
- 动作：5 次降权、3 次回滚、2 次暂停、7 次恢复、5 次重训到期

## 错误语义

- `ALPHA-404`：生命周期策略、来源、状态、日期或哈希非法。
- 生命周期产物任一文件缺失或哈希不一致时拒绝读取。
- `ml_rank` v3 遇到生命周期缺失日期、模型预测缺失或活动模型不一致时返回
  `BT-201`，不静默跳过。

## 调用关系

- 上游：`alpha.monitor`、`alpha.registry`、`alpha.predict`
- 执行：`backtest.strategies.ml_rank_v3`
- 完整验收：`scripts.build_alpha.lifecycle_accept`
- 日志：`logs/research/modeling/alpha_lifecycle/`

## 修改注意

1. 修改状态阈值、去抖次数或仓位乘数时同步策略、文档和测试。
2. 不得使用信号日之后才能知道的标签或收益。
3. 不得覆盖已有生命周期 ID；输入或代码变化应生成新 ID。
4. 不得把重训到期解释成挑战者已自动晋升。
5. 不得读取 2024-2026 最终测试集。
