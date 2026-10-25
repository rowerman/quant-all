# ML 模型层第一版研究契约

日期：2026-10-20

本文件是 alpha 模型层 P0～P3 的口径来源。代码中的
`alpha.config.ResearchContract` 必须与本文件保持一致；改口径就是新版本，不能覆盖旧结果。

## 1. 研究问题

给定股票在 t 日收盘时可见的因子和上下文特征，预测它在未来 5 个交易日的收益截面排名。
模型只输出分数或排序，不决定买卖、仓位、行业约束或交易规则；这些仍由回测层负责。

## 2. 固定口径

| 项目 | 第一版口径 |
|---|---|
| 标签 | `fwd_ret_open_5` |
| 决策时点 | 周最后交易日收盘 |
| 执行时点 | 次一交易日开盘 |
| 调仓 | 周频 |
| 股票池 | 全 A 可交易：`can_buy_close`、非 ST、非停牌、非退市整理期、上市满 60 个自然日 |
| 组合 | Top-30 等权，只做多 |
| 基准 | `H00300.CSI`（沪深 300 全收益） |
| 成本 | 回测默认成本：佣金 0.03%、最低 5 元、滑点 0.05%、平方根冲击、历史印花税/过户费/红利税 |
| 开发期 | 2012-01-01 ~ 2023-12-31 |
| 最终测试集 | 2024-01-01 ~ 2026-10-15，未冻结模型前不运行 |
| purge / embargo | 各 6 个交易日，即 5 日标签的 `h+1` |

股票池保留退市股票；上市前、退市后和停牌行不进入信号池。标签使用后复权价格计算，
停牌段由上游标签表按最后有效价格填平。

## 3. P0 基线

四条基线统一使用上述日期、股票池、组合、成本和基准。

| ID | 分数定义 | 用途 |
|---|---|---|
| `B1_MOM20` | `MOM_20` | 单因子参照 |
| `B2_MOM_EP_VOL` | `MOM_20 + EP_TTM - VOL_20` | 现成手工多因子组合的周频版本 |
| `B3_ALL_ZSCORE` | 47 个已实现因子按方向调整后每日截面 z-score 等权 | 线性等权基线 |
| `B4_ALL_RANK` | 47 个因子按方向调整后每日截面分位等权 | 对极值和分布更稳健的简单打分 |

B3 的 z-score 只在截面内计算，缺失因子不贡献分数；B4 在可交易池内计算分位，缺失因子
按当日有效因子数量平均。两者都不使用全样本统计。

## 4. 特征审计

审计覆盖 47 个已实现因子，逐因子、逐年度统计：

- 可交易池覆盖率、缺失集中度、分位数、3×MAD 极端值占比、非有限值数量；
- 注册表方向与周频 RankIC 的符号一致性；
- 2012-2023 月度截面 Spearman 相关，报告 `|rho| >= 0.8` 的因子对和 `>= 0.9` 的相关簇。

第一版特征清单固定为全部 47 个因子加 `log_mv`、`turnover_rate`、`listed_days`。
只有 2021-2023 年平均覆盖率低于 50% 的因子会剔出第一版；低覆盖但近期可用的因子保留，
由缺失指示列处理。高相关组只报告，不在 v1 自动删除，留给 P3 消融决定。

## 5. 评估口径

模型层报告：

- RankIC 均值、ICIR（RankIC 均值/标准差）、t 值、胜率；
- 分 10 组的分位收益与单调性；
- 分数自相关和 Top-N 换手；
- 分年度表现。

组合层报告：

- 年化收益、净超额、最大回撤、双边换手、总成本；
- 分年度收益与超额；
- 行业、市值、波动暴露。

模型层 IC 高不等于组合可用；最终判断以扣费后的 Top-N 净收益和稳定性为准。

P2 实现补充：

- 线性模型目标为当日可交易股票内的 `label_rank`，使用 `sample_weight`；
- 每折只用训练集拟合候选正则参数，以验证集 RankIC 选参，再用 train+valid 重拟合；
- 2018-2023 六折测试预测拼接后，与 B1～B4 和验证集选出的单因子基线使用同一日期、
  股票池、Top-30、周频模拟器和成本口径；
- 模型产物只写 `data/models/models/<model_id>/`，不写回 `data/factors/`。

P3-1/P3-2 实现补充：

- LightGBM 目标仍为当日可交易股票内的 `label_rank`，使用与 Ridge 相同的 100 个特征；
- 每折先用训练集拟合、验证集 RankIC 早停，再用 train+valid 按最佳轮数重拟合；
- LightGBM 关闭默认 metric，只允许验证集 RankIC 决定最佳轮数；
- 逐折保存原生文本模型和逐年模型清单，预测仍按 `model_id` 和 SHA256 定位；
- P3-1/P3-2 固定单一种子；P3-3 另起模型验证多种子和固定权重融合。

P3-3 实现补充：

- 集成固定使用 `seed 0,1,2,3,4`，五个成员共享同一数据集、折、特征和超参数；
- 每个成员独立按验证集 RankIC 早停，再用 train+valid 按各自最佳轮数重拟合；
- 集成在每个日期截面内把成员分数转为百分比 rank，再按固定等权平均；
- 稳定性报告保留全部五个种子，不允许按样本外表现挑选或剔除成员；
- Ridge 融合固定为集成 rank 与 Ridge rank 的 50/50 平均，当前只写对照结果；
- P3-5 已确认 50/50 融合未通过分年稳定检查，不生成 `rank_blend` 模型；
- 历史上 P3-5 前，集成模型的正式决策状态保持 `pending_p3`。

P3-4 实现补充：

- gain 和 TreeSHAP 汇总五个固定种子的 30 个折内模型；TreeSHAP 使用 LightGBM 原生
  `pred_contrib`，每折每种子确定性抽样 5,000 条样本外记录；
- 分族消融固定 `seed=0`，对 11 个因子族和 1 个上下文组分别删除全部转换特征后，
  沿用原六折和超参数重训；删除因子族时同步删除其 `__missing` 指示列；
- 单因子消融固定 `seed=0`，在交易日截面内置换根特征及其缺失指示列，不重新训练；
- 相关性使用开发期 100 个确定性截面计算根特征逐日 Spearman；
- 组合暴露按周频研究模拟器的 Top-30 等权目标持仓计算，风格为七类 Barra 近似项，
  行业使用时点申万一级；该结果不能替代 P4 的完整回测账本；
- 解释分析不改写正式模型，不触发 P3-5 决策。

P3-5 实现补充：

- 平滑固定为先计算每日可交易池内百分比 rank，再取当前及过去 4 个信号日的等权均值；
- buffer 固定为 Top-30 持仓、保留当日前 45 名的旧持仓，再按分数补足；
- 决策门要求 RankIC、ICIR、年化净收益、最大回撤至少两项严格优于 Ridge，
  双边年换手不超过 Ridge 两倍，2018-2023 每年 RankIC 和净超额均为正；
- 通过候选按年化净收益、Sharpe、最大回撤、换手排序；全部失败时保留 Ridge；
- 正式决策分析 ID 为 `p3_5_5c02d3520f25`，选中 LightGBM 平滑 4 周候选；
- 决策结果只来自周频研究模拟器，P4 仍须用完整回测账本重新验收。

P4-4A 实现补充：

- 模型样本外预测覆盖 300 个信号日，自然周历有 305 个；五个缺失日期位于逐年
  walk-forward 测试区间之前，不能补造预测。
- P4-4A 新增 `ml_rank` v2 和隐藏的 `factor_rank` v2，以显式
  `rebalance_dates` 让 ML 候选和 B4 在同一组 300 个日期调仓。
- v1 保持原有严格行为。显式日期清单及哈希进入策略参数、快照、分析 ID 和验收 manifest。
- 完整账本在 2021-11-15 前排除北交所代码，与研究模拟器历史口径一致。

P4-4B1 实现补充：

- `ml_rank` v2 新增可选 `smoothing_weeks`，默认 4；原始 Ridge、单 LightGBM
  和五种子集成对照使用 1，只做当日可交易池内 rank。
- 新增隐藏的 `factor_linear` v1，在相同的 300 个显式日期复用 B1-B3 的原有
  因子权重、截面标准化和 Top-30 等权规则；B4 继续使用全因子 rank。
- P4-4B1 固定初始资金 100 万，八个候选使用同一完整账本口径；多资金档位容量曲线
  属于 P4-4B2，不能把单个资金档位的结果外推为全规模容量结论。

P4-4B2 实现补充：

- 只检验 P3 选中的 `LIGHTGBM_SMOOTH_4W`，资金档位固定为 100 万、1000 万和 1 亿；
  100 万档复用 P4-4B1，新增两档按 1000 万、1 亿顺序串行运行。
- 三档继续使用同一组 300 个显式调仓日、Top-30 等权、10% 成交额参与率和默认真实成本。
- 执行结构验收沿用 P4-4A/B1 的收敛、日期、快照、指标和成本检查；经济容量仅报告
  年化净超额仍为正的最大已测档位，不新增事后收益淘汰线。
- 正式分析 ID 为 `p4_4b2_52b6a7b661ff`。1 亿档保持正净超额，但容量受限格数、
  累计限制金额、补单和权重偏差均明显增加，不能把结构通过解释为容量无影响。

V83 最终留出集补充：

- 2024-2026 在模型和 P5 规则冻结后只运行一次，验证 ID 为
  `final_holdout_2ba777e855f5`。
- LightGBM 和 Ridge 固定使用 `wf_2023`，不重新训练或调参；普通接口仍拒绝
  跨入 2024 年，只有最终验证入口显式开放。
- 模型评估使用 142 个 5 日标签完整的周频日期；完整账本运行到 2026-10-15。
- 预注册 P3 门失败：主候选相对 Ridge 只有年化净收益改善，2024 年净超额为
  -31.46%，且总收益、Sharpe 和最大回撤均落后 B4。
- 失败结果不得在同一留出集上调参重试。后续挑战者需要新的未来样本。

## 6. 复现命令

```bash
cd /home/hanwen/quant

# P0 特征审计与基线 IC
.venv/bin/python -m scripts.build_alpha.audit_features \
  --start 2012-01-01 --end 2023-12-31

# P0 四条基线；按 B1→B2→B3→B4 串行运行，避免占满内存
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B1_MOM20 --start 2012-01-01 --end 2023-12-31 --engine weekly
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B2_MOM_EP_VOL --start 2012-01-01 --end 2023-12-31 --engine weekly
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B3_ALL_ZSCORE --start 2012-01-01 --end 2023-12-31 --engine weekly
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B4_ALL_RANK --start 2012-01-01 --end 2023-12-31 --engine weekly

# P3 LightGBM 六折训练与重放
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id 6a59680fd71c50a8 \
  --spec configs/alpha/train_lightgbm_ensemble.json
.venv/bin/python -m scripts.build_alpha.predict \
  --model-id lightgbm_label_rank_aaae94773921 \
  --start 2018-01-12 --end 2023-12-29

# P3-3 五种子 rank 集成、重放和 Ridge 融合对照
.venv/bin/python -m scripts.build_alpha.predict \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --start 2018-01-12 --end 2023-12-29
.venv/bin/python -m scripts.build_alpha.evaluate \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --ridge-model-id ridge_label_rank_aaff2cb2fbcf --force

# P3-4 重要性、消融、相关性与组合暴露
.venv/bin/python -m scripts.build_alpha.explain_model \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --top-n 30 --sample-rows 5000 \
  --permutation-seed 20261020 --ablation-seed 0 --num-threads 4

# P3-5 决策门
.venv/bin/python -m scripts.build_alpha.evaluate_plan \
  --plan configs/alpha/development_evaluation_v85.json

# V83 最终留出集
.venv/bin/python -m scripts.build_alpha.final_holdout \
  --load final_holdout_2ba777e855f5
```

`weekly` 是 P0 快速研究模拟器：用周频标签收益和 `CostSpec` 估算换手成本；
`ledger` 是完整回测账本，逐周分段在 15GiB 机器上耗时很长，主要用于短区间校验。
P4 接入策略时必须使用 `ledger`。

基线结果写入 `data/models/baselines/`，审计结果写入 `data/models/audit/`；每次运行在
`data/models/experiments.csv` 留一条记录。普通脚本会拒绝任何跨入 2024-01-01 的运行；
最终留出集只能使用显式 V83 入口。

## 7. 修改规则

1. 标签、股票池、调仓、组合、成本或基准任一项变化，都形成新的研究版本。
2. 最终测试集只能在模型方案冻结后运行一次；之前的任何调参都不得使用 2024-2026 结果。
3. 不把模型分数写回 `data/factors/` 原始因子库；模型产物只放 `data/models/`。
4. 不覆盖历史模型、预测或实验结果；新版本使用新的 `model_id` 和目录。
