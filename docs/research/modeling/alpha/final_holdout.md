# 最终留出窗口与验证

日期：2026-10-22

## 职责

`alpha.final_holdout` 对已冻结的模型和策略执行一次独立留出期验证。窗口边界和
封存状态统一登记在 `configs/alpha/holdout_windows.json`。它不训练模型、不调参、
不按留出结果替换候选，也不覆盖开发期模型或预测。

## 窗口

| 窗口 | 区间 | 状态 |
|---|---|---|
| `holdout_2024_2026` | 2024-01-01～2026-10-15 | 已消耗，只读封存 |
| `holdout_2026_2027` | 2026-10-16～2027-10-15 | 待使用，至少 100 个完整标签信号日 |

新窗口当前数据尚未覆盖，只出现在运行索引中，状态为 `planned`。

正式验证 ID：`final_holdout_2ba777e855f5`

状态：`fail_final_holdout`

## 固定口径

- 数据区间：2024-01-01～2026-10-15。
- 模型评估：2024-01-05～2026-09-30，共 142 个 5 日标签完整的周频信号日。
- 完整账本：同一起点至 2026-10-15，142 个信号日和 142 个执行段。
- 候选、参照和冻结折由 `configs/alpha/final_holdout_2024_2026.json` 指定；
  本次历史运行使用最后的 `wf_2023` 折。
- 五种子 LightGBM 保留全部成员；Top-30、四周 rank 平滑和 P5 生命周期规则不变。
- 100 万元初始资金、10% 成交额参与率和默认成本。

## 接口

```python
from alpha.final_holdout import (
    FinalHoldoutSpec,
    load_final_holdout,
    run_final_holdout,
)

spec = FinalHoldoutSpec(
    window_id="holdout_2026_2027",
    plan_path="configs/alpha/<frozen_evaluation_plan>.json",
)
result = run_final_holdout(spec)
loaded = load_final_holdout(result["holdout_id"])
```

```bash
.venv/bin/python -m scripts.build_alpha.final_holdout \
  --load final_holdout_2ba777e855f5
.venv/bin/python -m scripts.build_alpha.final_holdout \
  --window-id holdout_2026_2027 \
  --plan configs/alpha/<frozen_evaluation_plan>.json
```

`--resume-from` 只用于运行中断后复用已通过完整哈希校验的数据集、冻结预测和候选账本；
它不接受不同的窗口、模型、数据、预测或验收门槛。已封存窗口不会接受新运行。

## 结果

### 模型层

| 候选 | RankIC | ICIR | 胜率 |
|---|---:|---:|---:|
| LightGBM 原始分数 | 0.0932 | 0.6480 | 76.06% |
| LightGBM 四周平滑 | 0.0794 | 0.4781 | 69.72% |
| Ridge 原始分数 | 0.0849 | 0.6045 | 76.06% |
| B4 全因子 rank | 0.0546 | 0.3604 | 65.49% |

原始 LightGBM 的 RankIC 仍高于 Ridge 和 B4，但开发期选中的四周平滑把 RankIC 降到
0.0794，低于 Ridge。

### 完整账本

| 候选 | 年化净收益 | 净超额 | Sharpe | 最大回撤 | 双边年换手 |
|---|---:|---:|---:|---:|---:|
| LightGBM 四周平滑 | 15.37% | 3.33% | 0.5263 | -45.54% | 50.08 |
| Ridge 原始分数 | 10.96% | -1.08% | 0.4242 | -40.02% | 87.08 |
| B4 全因子 rank | 19.43% | 7.39% | 0.7795 | -19.89% | 66.41 |
| P5 生命周期 | 17.66% | 5.62% | 0.6144 | -45.54% | 34.83 |

主候选相对 Ridge 只有年化净收益一项严格改善；RankIC、ICIR 和最大回撤均更差，
2024 年净超额为 -31.46%，因此预注册 P3 门失败。

### 分年

主候选 2024 年净收益 -9.60%、基准 +21.86%、净超额 -31.46%；2025 年净超额
+46.13%；2026 年至 10 月净超额 +5.02%。收益高度集中在 2025 年。

B4 三年净超额分别为 +2.13%、+14.72%、+3.50%，同时在总收益、Sharpe 和最大回撤上
均优于主候选。因此本次留出期不能证明 LightGBM 策略相对简单基线存在增量。

## 暴露与生命周期

- 主候选平均持仓 30 只，风格最低覆盖率 96.43%。
- SIZE 平均暴露 -1.377、SHORT_REV +0.370、LIQUIDITY -0.479、VOL -0.391。
- 生命周期状态为 normal 80、reduced 12、rollback 12、paused 38 个信号日。
- 生命周期降低收益和换手，但没有改善主候选的最大回撤，整体仍落后 B4。

## 验收边界

- 四候选均通过资金分配、补回、成本、日期、快照和基准检查。
- 14 个汇总结果文件的 SHA256 全部一致。
- 验证失败是经济效果失败，不是执行结构失败。
- 不在同一留出集上重新调参；若继续研究，应使用新的挑战者和新的未来样本。

## 调用关系

- 数据：`alpha.final_holdout_data`
- 模型与暴露：`alpha.final_holdout_eval`
- 完整账本：隐藏 provider `backtest.strategies._final_holdout`
- 监控与生命周期：`alpha.monitor`、`alpha.lifecycle`
- 入口：`scripts.build_alpha.final_holdout`

## 修改注意

1. 普通接口继续拒绝留出区间；留出调用必须同时提供已注册窗口。
2. 不得在留出期重新训练或按结果改门槛。
3. 不得覆盖 `final_holdout_2ba777e855f5`。
4. provider 输入、预测、数据集或生命周期文件任一哈希不一致时必须失败关闭。
5. 新窗口只有积累至少 100 个完整标签信号日后才能运行。
