# 模型监控

日期：2026-10-22

## 职责

`alpha.monitor` 对已登记模型执行研究期离线监控，覆盖分数分布、缺失、滚动
RankIC、完整账本净超额、换手和回撤，以及风格与行业暴露。监控读取现有不可变
产物，不训练模型、不重跑回测，也不读取 2024-2026 最终测试集。

模块分工：

- `monitor.py`：稳定公开接口；
- `monitor_policy.py`：监控策略、结果类型和默认口径；
- `monitor_inputs.py`：预测、标签、股票池、分数分布和 RankIC；
- `monitor_metrics.py`：账本、暴露、告警和衰减演练；
- `monitor_artifacts.py`：来源校验、不可变产物和断点复用；
- `monitor_report.py`：Markdown/JSON 报告和文件哈希。

默认监控对象：

- 模型：`lightgbm_ensemble_label_rank_6aea5c9299d5`
- P4 完整账本：`p4_4a_23c7e101aee9/LIGHTGBM_SMOOTH_4W`
- P3-4 暴露分析：`p3_4_8152f500bb86`
- 部署分数：原始五种子分数按日 rank 后的 4 周因果平滑

V85 新模型正式监控：

- 模型：`sklearn_hist_gbdt_label_rank_83e1200e554f`
- 候选：`evaluation_d57dd9e6e750/SKLEARN_HIST_RAW`
- 完整账本：`acceptance_2c80425c2cfe`
- 监控：`p5_2_5494d7f2b910`
- 部署分数：原始模型分数，1 周平滑

通用账本模型使用评估 ID 和候选名调用：

```bash
.venv/bin/python -m scripts.build_alpha.monitor \
  --model-id <model_id> \
  --acceptance-id <acceptance_id> \
  --evaluation-id <evaluation_id> \
  --candidate <candidate>
```

此模式从评估风险时序读取目标组合暴露，平滑周数取候选评分规则。旧 P4/P3 调用
不传 `--evaluation-id` 时继续使用历史来源和 4 周默认值。

## 接口

```python
from alpha.monitor import (
    MonitorPolicy,
    evaluate_monitor_frames,
    simulate_score_decay,
    run_model_monitor,
)

policy = MonitorPolicy()
manifest = run_model_monitor()
```

- `MonitorPolicy`：固定滚动窗口和告警阈值。
- `evaluate_monitor_frames()`：计算已对齐输入的监控时间序列和告警。
- `simulate_score_decay()`：反转最近若干周的截面排名，验证 RankIC 告警。
- `run_model_monitor()`：校验模型、数据集、账本、策略和暴露来源，原子写入完整
  监控包。

命令行：

```bash
.venv/bin/python -m scripts.build_alpha.monitor
.venv/bin/python -m scripts.build_alpha.monitor --force
```

`--force` 只用于同一输入下重建报告。来源文件、阈值或代码变化会生成新的
`monitor_id`，不会覆盖历史监控。

## 告警口径

滚动窗口为 13 个信号周，至少 8 个有效观测。

| 指标 | Warning | Alert |
|---|---|---|
| 部署分数滚动 RankIC | 低于 0 | 连续 2 周低于 0 |
| 滚动净超额 | 低于 0 | 不高于 -10% |
| 当前回撤 | 不高于 -28% | 不高于 -35% |
| 滚动双边年化换手 | 账本基准的 1.5 倍 | 账本基准的 2.0 倍 |
| 可交易股票缺失率 | 高于 1% | 高于 5% |
| 最大绝对风格暴露 | 高于 1.5 | 高于 2.0 |
| 最大绝对行业主动权重 | 高于 25% | 高于 35% |

每期只输出实际触发的规则；同一期有多条规则时取最高严重度作为当期状态。滚动
指标未满最小观测数时标为 `insufficient_data`。

## 产物

`data/models/models/<model_id>/monitoring/<monitor_id>/` 包含：

- `timeseries.csv`：逐信号日的分数、缺失、RankIC、账本、回撤和暴露；
- `alerts.csv`：逐条告警及实际值、阈值和原因；
- `decay_simulation.csv`：最后 13 周的原始/部署 RankIC 和衰减后 RankIC；
- `report.json`、`report.md`：机器和人工可读报告；
- `manifest.json`：来源、策略、代码和输出哈希；
- `complete.json`：结果文件哈希完成标记。

## 来源校验

运行前校验：

1. `model_id` 已登记且模型 manifest 与注册记录一致；
2. P4 验收状态为 `pass_p4_4a`，候选完成标记内全部文件哈希一致；
3. 策略 provider 使用同一 `model_id`，回测信号日等于预测日期；
4. P3-4 manifest 的模型、预测哈希和状态一致，风格文件哈希未变；
5. 数据集 manifest ID 与模型一致。

任一检查失败时整次运行失败，不写入部分结果。已有完整监控包默认复用；重建时
先写入临时目录，全部完成后再替换目标目录。

## 口径与限制

- 净超额按相邻信号日之间的完整账本收益计算：组合累计收益减基准累计收益。
- 滚动换手按区间内每日 `turnover` 之和年化，沿用完整账本的双边换手口径。
- 风格暴露复用评估候选的原始目标持仓，不是完整账本成交持仓；通用来源直接读取
  评估风险时序，旧来源仍读取 P3-4 暴露包。
- 行业暴露使用完整账本实际持仓，并以同一信号日可交易股票池为基准计算主动权重。
- 衰减演练反转最后 13 周原始分数排名后重新平滑，只验证告警链路，不构成策略回测。
- 本模块不提供生产实时推送、自动降权或交易操作；这些属于后续 P5-3。

## 调用关系

- 入口：`scripts.build_alpha.monitor`
- 依赖：`alpha.registry`、`alpha.predict`、`alpha.evaluate`、`alpha.portfolio`
- 产物：`data/models/models/<model_id>/monitoring/`
- 日志：`logs/research/modeling/alpha_monitor/`

## 修改注意

1. 阈值变化必须同步 `MonitorPolicy`、本文档和测试。
2. 不得把 P3-4 风格暴露描述成完整账本成交暴露。
3. 不得在监控中读取封存的 2024-2026 数据。
4. 不得覆盖已有监控 ID；代码或输入变化应生成新 ID。
5. 新增指标时同步时间序列、告警、报告和来源哈希。
