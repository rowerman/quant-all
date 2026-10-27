# Alpha 模型层

日期：2026-10-23

本目录记录 ML 模型层的研究口径、特征审计和 P0～P5-4 使用方式，以及
2024-2026 最终留出集验证结果。
P4-1/P4-2/P4-3/P4-4A/P4-4B1/P4-4B2、P5-1、P5-2、P5-3 和 P5-4 已完成，
模型、预测、完整账本、容量曲线、历史模型索引、监控、生命周期规则和正式模型卡
已验收。2024-2026 最终留出集已按冻结口径运行一次，状态为
`fail_final_holdout`。

V84 起，模型层定位为可替换模型模块：训练、预测、评估和验收不再绑定
LightGBM/Ridge。当前三个历史模型仅作为实验、回退和归档产物保留。

## 当前范围

- V84：建立 `ModelBackend`、通用训练流水线、标准 v2 manifest、通用评估计划和
  分层产物目录；新增模型只扩展后端，不修改共享折循环。
- P0：研究契约、四条线性基线、47 因子覆盖审计、实验记录。
- P1：`research/modeling/alpha/` 包、周频数据集、截面预处理、purged walk-forward 切分。
- P2：Ridge 线性模型、六折样本外预测、模型评估和同口径基线对比。
- P3-1/P3-2：LightGBM 单种子、RankIC 早停、逐折重拟合和逐年模型清单。
- P3-3：五种子 LightGBM rank 集成、逐种子稳定性和固定 50/50 Ridge 融合对照。
- P3-4：gain/TreeSHAP、12 组分族重训、50 个单因子置换、相关性和 Top-30 组合暴露。
- P3-5：4 周 rank 平滑、Top-30 buffer、固定决策门和候选排序。
- P4-1：已落盘预测的统一读取、覆盖校验和哈希校验。
- P4-2：`ml_rank` 回测提供者与 4 周因果 rank 平滑。
- P4-3：模型、预测、训练数据、特征、折和种子的可复现快照。
- P4-4A：选中的 4 周平滑模型与 B4 已使用同一组 300 个预测日通过完整账本验收；
  资金分配和现金补回均收敛，无账本现金裁剪。
- P4-4B1：Ridge、单 LightGBM、集成原始分数、选中平滑模型和 B1-B4 已在 100 万
  资金下完成同一组 300 日的完整账本对照；正式分析 ID
  `p4_4b1_e9c029453c2d`。
- P4-4B2：选中平滑模型在 100 万、1000 万和 1 亿资金下完成容量曲线；
  正式分析 ID `p4_4b2_52b6a7b661ff`。
- P5-1：三个正式模型已登记到 `data/models/index.json`，可按 `model_id` 定位
  模型、预测、指标、分析产物和完整哈希。
- P5-2：正式选中模型已生成离线监控包，覆盖分数、缺失、滚动 RankIC、净超额、
  换手、回撤、风格和行业暴露，并通过模型衰减告警演练。
- P5-3：固定重训、降权、回滚 Ridge、暂停和恢复规则已接入 `ml_rank` v3，
  并完成完整账本执行验收。正式生命周期 ID `p5_3_dd7c9b4bf235`，验收 ID
  `p5_3_4f91d5b68fba`；年化净收益 20.40%、净超额 21.73%、最大回撤 -27.87%、
  双边年换手 31.68。
- P5-4：正式主模型已生成面向新读者的模型卡，集中说明用途、数据、特征、训练验证、
  历史表现、容量、监控、生命周期、复现方法和失效风险。模型卡见
  [正式主模型卡](model_card_lightgbm_ensemble_label_rank_6aea5c9299d5.md)。
- V83：冻结五种子 LightGBM、Ridge 和 B4，完成 2024-2026 一次性最终验证。
  ML 平滑候选未通过预注册 P3 门，且完整账本收益、Sharpe 和回撤均落后 B4。
  详见 [最终留出集验证](final_holdout.md)。
- V85-ML-1～3：新增直方图 GBDT 后端并训练登记；历史八候选迁移到通用评估计划；
  新窗口 2026-10-16～2027-10-15 已登记，旧窗口只读封存。
- V85-ML-4～6：增加组合风险门槛和计划驱动完整账本入口；十候选评估
  `evaluation_d57dd9e6e750` 已完成，新模型完整账本、监控和生命周期分别为
  `acceptance_2c80425c2cfe`、`p5_2_5494d7f2b910`、`p5_3_65282c575784`。
- V86：后端目录自动发现，训练命令自动登记；模型索引 v3 聚合评估、账本、监控、
  生命周期和最终留出；训练恢复与评估复用补齐完整哈希校验。
- V120：模型特征 rank、标签 rank、1%/99% 分位裁剪和总体标准差 z-score 接入统一
  C++ 核；python/native/auto 输出逐列哈希一致，真实预处理明显加速。

研究口径见 `research_contract.md`，特征审计结论见 `feature_audit.md`，
训练后端和新增模型方式见 `training_backends.md`，模型模块边界见
`model_layer_architecture.md`，现有产出见 `model_inventory.md`。
P3-4 方法和结果见 `importance_ablation.md`，P3-5 决策见 `p3_decision.md`。
预测读取接口见 `prediction_artifacts.md`，模型登记见 `model_registry.md`。
模型监控见 `model_monitoring.md`。
模型生命周期见 `model_lifecycle.md`，正式主模型说明见
`model_card_lightgbm_ensemble_label_rank_6aea5c9299d5.md`。通用完整账本入口见
`ledger_acceptance.md`。截面预处理、日期分组和原生核接入见
`preprocessing.md`。

## 常用命令

```bash
cd /home/hanwen/quant

# P0 审计
.venv/bin/python -m scripts.build_alpha.audit_features \
  --start 2012-01-01 --end 2023-12-31

# P0 基线（一条一条跑，避免占满内存）
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B1_MOM20 --start 2012-01-01 --end 2023-12-31 --engine weekly

# P1 数据集与切分
.venv/bin/python -m scripts.build_alpha.build_dataset \
  --start 2012-01-01 --end 2023-12-31
.venv/bin/python -m scripts.build_alpha.build_folds

# 通用训练（规格文件指定 backend）
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id 6a59680fd71c50a8 \
  --spec configs/alpha/train_ridge.json
.venv/bin/python -m scripts.build_alpha.predict \
  --model-id ridge_label_rank_aaff2cb2fbcf \
  --start 2018-01-12 --end 2023-12-29
.venv/bin/python -m scripts.build_alpha.evaluate \
  --model-id ridge_label_rank_aaff2cb2fbcf

# LightGBM 集成训练
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id 6a59680fd71c50a8 \
  --spec configs/alpha/train_lightgbm_ensemble.json
.venv/bin/python -m scripts.build_alpha.predict \
  --model-id lightgbm_label_rank_aaae94773921 \
  --start 2018-01-12 --end 2023-12-29
.venv/bin/python -m scripts.build_alpha.evaluate \
  --model-id lightgbm_label_rank_aaae94773921

# 五种子集成预测重放
.venv/bin/python -m scripts.build_alpha.predict \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --start 2018-01-12 --end 2023-12-29
.venv/bin/python -m scripts.build_alpha.evaluate \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --ridge-model-id ridge_label_rank_aaff2cb2fbcf --force

# Scikit-learn 直方图 GBDT
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id 6a59680fd71c50a8 \
  --spec configs/alpha/train_sklearn_hist_gbdt.json

# P3-4 重要性与消融
.venv/bin/python -m scripts.build_alpha.explain_model \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --top-n 30 --sample-rows 5000 \
  --permutation-seed 20261020 --ablation-seed 0 --num-threads 4

# P5-1 模型登记、P5-2 监控和 P5-3 生命周期
.venv/bin/python -m scripts.build_alpha.registry --rebuild
.venv/bin/python -m scripts.build_alpha.monitor
.venv/bin/python -m scripts.build_alpha.lifecycle
.venv/bin/python -m scripts.build_alpha.lifecycle_accept \
  --lifecycle-id p5_3_dd7c9b4bf235

# 通用评估计划
.venv/bin/python -m scripts.build_alpha.evaluate_plan \
  --plan configs/alpha/development_evaluation_v85.json

# 按评估计划和候选执行通用完整账本
.venv/bin/python -m scripts.build_alpha.accept_plan \
  --evaluation-id <evaluation_id> --candidate selected --resume

# P4-1 读取并校验正式模型预测
.venv/bin/python - <<'PY'
from alpha.predict import load_predictions

_, manifest, predictions = load_predictions(
    "lightgbm_ensemble_label_rank_6aea5c9299d5"
)
print(manifest["prediction_sha256"], len(predictions))
PY

# P4-4A 完整账本验收（选中模型 + B4）
.venv/bin/python -m scripts.build_alpha.backtest_accept \
  --round p4-4a --candidate all --resume

# P4-4B1 固定 100 万资金八候选完整账本
.venv/bin/python -m scripts.build_alpha.backtest_accept_b1 \
  --candidate all --resume

# 续跑最终正式分析（复用之前已通过的五候选）
.venv/bin/python -m scripts.build_alpha.backtest_accept_b1 \
  --candidate all --resume \
  --resume-from-analysis p4_4b1_5ed76ca9ce9d

# P4-4B2 选中策略三档容量曲线
.venv/bin/python -m scripts.build_alpha.backtest_accept_b2 --resume

# P5-1 完整校验并重建历史模型索引
.venv/bin/python -m scripts.build_alpha.registry --rebuild

# P5-2 模型监控
.venv/bin/python -m scripts.build_alpha.monitor

# V83 一次性最终留出集验证
.venv/bin/python -m scripts.build_alpha.final_holdout
.venv/bin/python -m scripts.build_alpha.final_holdout \
  --load final_holdout_2ba777e855f5
```

训练使用验证集选择 alpha，随后用 train+valid 重拟合并预测当年 test。
LightGBM 使用验证集 RankIC 早停，再按最佳轮数用 train+valid 重拟合。
线性模型目录包含 `model.npz`；LightGBM 目录包含 `models/wf_YYYY.txt` 和
`fold_models.csv`。五种子集成目录包含 30 个成员模型、逐种子预测、
`seed_stability.csv/json` 和集成预测。三者都保留配置、特征、manifest、指标和分年结果。

## 目录

| 路径 | 内容 |
|---|---|
| `research/modeling/alpha/` | 研究契约、数据集、预处理、切分与日志 |
| `data/models/audit/` | 覆盖率、缺失、方向、相关簇和基线 IC |
| `data/models/baselines/` | 四条基线的配置、快照、指标、日结果和订单 |
| `data/models/datasets/<dataset_id>/` | 周频数据集与 manifest |
| `data/models/splits/` | 2018-2023 purged walk-forward 切分表 |
| `data/models/models/<model_id>/` | 活跃模型、预测、指标和模型专属分析 |
| `data/models/archive/models/<model_id>/` | 归档但仍可审计的历史模型 |
| `data/models/runs/acceptance/` | 完整账本和生命周期验收运行 |
| `data/models/runs/final_holdout/` | 独立最终留出运行 |
| `data/models/index.json` | 模型索引 v3，可由模型和运行 manifest 重建，包含下游运行链接 |
| `data/models/models/<model_id>/monitoring/<monitor_id>/` | P5-2 不可变监控报告、时序和告警 |
| `data/models/experiments.csv` | 基线实验索引 |
| `trading/backtest/strategies/ml_rank.py` | 已落盘预测的周频 Top-N 提供者 |

基线默认使用 `--engine weekly` 的快速研究模拟器，按周频标签收益和 `CostSpec` 估算成本；
`--engine ledger` 使用完整回测账本，适合短区间校验。两者不能混在同一张结果表里比较。

## 数据口径

- 标签 `fwd_ret_open_5`，周最后交易日收盘生成信号，次一交易日开盘执行。
- 周频信号日按完整开发期交易日历取每周最后交易日，不能按自然年分别取年末最后一天。
- 数据集同时保留 `label_raw` 和当日可交易股票内的 `label_rank`。
- 特征保留原始值；预处理默认按日 rank、缺失中位数、缺失指示列，可选行业/市值中性化。
- 默认切分为 2018-2023 六折扩展窗口，purge 和 embargo 各 6 个交易日。
- 2024-01-01 ~ 2026-10-15 最终测试集只允许通过显式最终验证入口读取；
  普通训练、预测和监控脚本仍会拒绝跨入该区间。

## 修改时注意

1. 先改 `research_contract.md` 和 `research/modeling/alpha/config.py`，再改脚本或模型；
2. 任何全样本统计都不允许进入截面预处理；
3. 历史数据集、预测和模型不覆盖，使用新的 `dataset_id` 或 `model_id`；
4. 新增依赖必须同步 `requirements.txt`、`requirements.lock` 和依赖测试。
5. 回测读取预测统一使用 `load_predictions()`，不要另写分片读取逻辑。
6. 训练命令会自动重建索引；单独更新外部模型或运行产物后，用
   `rebuild_registry()` 重建。`index.json` 不能替代模型目录中的 manifest 和 metrics。
7. 新后端放入 `research/modeling/alpha/modeling/backends/` 并注册，不得在通用流水线、评估和回测中加入
   具体模型类型分支。
8. 模型、验收和最终留出的路径统一通过 `alpha.config` 与 `model_dir()` 解析。

## P2 结果

模型 `ridge_label_rank_aaff2cb2fbcf` 使用 2018-2023 六折样本外预测：

- RankIC 均值 0.0911，ICIR 0.781，胜率 79.0%；
- Top-30 等权年化净收益 14.07%，年化净超额 15.81%；
- 最大回撤 -32.47%，双边年换手 88.63 倍；
- 同区间 B4 全因子 rank 基线 RankIC 0.0562、年化净收益 11.03%、换手 65.57 倍。

RankIC、ICIR 和净收益均优于 B4，P2 决策门通过；但换手比 B4 高 35%，
P3 需要在保留信息的同时加入分数平滑或换手上限的对照。

## P3-1/P3-2 结果

模型 `lightgbm_label_rank_aaae94773921` 使用与 Ridge 相同的 2018-2023 六折：

- RankIC 均值 0.0971，ICIR 0.831，胜率 79.0%；
- Top-30 等权年化净收益 14.50%，最大回撤 -29.08%；
- 双边年换手 81.13 倍，低于 Ridge 的 88.63 倍；
- 六折最佳轮数为 567、403、148、970、24、52，分年 RankIC 均为正。

P3-1/P3-2 当时只形成单种子候选模型；P3-3 已补充多种子稳定性与 Ridge 融合对照，
P3-5 随后完成正式决策。

## P3-3 结果

模型 `lightgbm_ensemble_label_rank_6aea5c9299d5` 固定使用 `seed 0,1,2,3,4`，
在每个信号日内把五个成员分数转为截面 rank 后等权平均。2018-2023 预测为
1,194,198 行、300 个信号日。

| 指标 | 五种子集成 | 单种子均值 | Ridge | 50/50 Ridge 融合 |
|---|---:|---:|---:|---:|
| RankIC | 0.0973 | 0.0967 | 0.0911 | 0.0977 |
| ICIR | 0.829 | 0.828 | 0.781 | 0.817 |
| Top-30 年化净收益 | 17.42% | 13.90% | 14.07% | 20.33% |
| 最大回撤 | -26.53% | -30.11% | -32.47% | -33.07% |
| 双边年换手 | 80.70x | 81.12x | 88.63x | 85.01x |

五个种子的 RankIC 标准差为 0.0007，稳定性报告保留全部成员，没有事后挑选种子。
集成的净收益和回撤优于单种子均值；50/50 Ridge 融合收益更高，但回撤和换手恶化，
P3-3 当轮只作为 P3-5 的候选证据；正式决策由 P3-5 执行。

Ridge 融合当前没有可重放 `model_id`。若 P3-5 确认融合有效，应在 P4 前增加
`rank_blend` 模型产物和重放测试；这不是放弃项。

`H00300.CSI` 当前指数开盘数据只能形成 2021 年以后的有效周收益，因此 2018-2020
分年基准收益为 0；这不影响模型与基线的排序比较，但 P4 使用完整账本前必须补齐基准数据。

## P3-4 结果

分析 ID `p3_4_8152f500bb86`，共生成 100 个转换特征、50 个根特征、12 组分族重训、
50 个单因子置换、18 对高相关特征和完整的风格/行业暴露序列。

- Gain 前列：`LIMIT_UP_CNT_20`、`LN_AMT_20`、`turnover_rate`、`VWAP_BIAS_5`；
- TreeSHAP 前列：`LN_AMT_20`、`VWAP_BIAS_5`、`REV_5`、`turnover_rate`；
- 分族重训中，删除量价关系、财务质量、上下文、Alpha101 和流动性后 RankIC 均下降；
- 单因子置换中，`LN_AMT_20` 的 RankIC 下降 0.01935，是最明显的单点依赖；
- Top-30 组合明显偏小市值、短期反转、低流动性、低波动和负动量，
  机械设备、基础化工、银行、医药生物和环保的平均权重居前。

消融使用 `seed=0`，五种子集成的正式指标仍以 `metrics.json` 为准。方法与边界见
`importance_ablation.md`，完整报告位于对应模型的 `analysis/<analysis_id>/`。

## P3-5 结果

决策分析 ID 为 `p3_5_5c02d3520f25`，2018-2023 固定候选比较通过 P3 决策门。
选中的组合规则为 LightGBM 五种子集成的 4 周历史 rank 平滑，Top-30 等权，
不启用 buffer。

| 指标 | 选中平滑候选 | Ridge |
|---|---:|---:|
| RankIC | 0.0851 | 0.0911 |
| ICIR | 0.6815 | 0.7808 |
| Top-30 年化净收益 | 30.92% | 14.07% |
| 最大回撤 | -28.13% | -32.47% |
| 双边年换手 | 43.00x | 88.63x |

平滑候选靠年化净收益和最大回撤两项优于 Ridge 通过决策门，RankIC 和 ICIR 低于
Ridge。buffer=45 将原始 LightGBM 换手从 80.70 倍降到 73.15 倍，但收益低于平滑候选。
50/50 Ridge 融合 2022 年净超额为负，未通过分年稳定检查，因此没有生成
`rank_blend` 模型。完整方法和限制见 `p3_decision.md`。
