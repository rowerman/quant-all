# P3-4 重要性与消融

日期：2026-10-20

正式分析对象：`lightgbm_ensemble_label_rank_6aea5c9299d5`；
分析 ID：`p3_4_8152f500bb86`；区间：2018-01-12 ~ 2023-12-29。
2024-2026 最终测试集未读取、未训练、未预测、未评估。

完整机器产物位于
`data/models/lightgbm_ensemble_label_rank_6aea5c9299d5/analysis/p3_4_8152f500bb86/`，
其中 `report.md` 是逐项结果，`manifest.json` 记录模型、预测、代码和全部文件哈希。

## 模块职责

| 模块 | 负责功能 | 调用方 | 主要依赖 | 修改注意 |
|---|---|---|---|---|
| `research/modeling/alpha/feature_groups.py` | 根特征、因子族、缺失指示列分组 | 解释入口和消融 | 因子目录、模型 manifest | 新增特征必须被单特征组和分族组完整覆盖 |
| `research/modeling/alpha/explain.py` | gain、TreeSHAP、置换、相关性 | 解释入口 | LightGBM、pandas | TreeSHAP 必须校验贡献加基值等于模型分数 |
| `research/modeling/alpha/ablation.py` | 分族重训、单因子置换 | 解释入口 | 预处理、六折、LightGBM | 不得使用测试标签训练或早停 |
| `research/modeling/alpha/exposure.py` | Top-N 风格和行业暴露 | 解释入口 | 周频持仓口径、风格数据 | 组合暴露不是完整账本成交持仓 |
| `core/data/scripts/build_alpha/explain_model.py` | P3-4 编排与产物写入 | 命令行 | 上述模块、周频模拟器 | 不覆盖正式模型，不跨入最终测试集 |

## 口径

- gain 和 TreeSHAP 汇总五个固定种子的 30 个折内模型。
- 分族重训只使用 `seed=0`，沿用原数据集、六折、purge/embargo、超参数和验证集
  RankIC 早停；删除分组时同时删除其 `__missing` 指示列。
- 11 个因子族和 1 个上下文组共 12 组重训；单因子置换覆盖 47 个因子和 3 个上下文特征。
- 单因子置换在交易日截面内打乱整组列，组内列共用同一置换；置换种子为 `20261020`。
- TreeSHAP 每折每种子确定性抽取 5,000 条样本外记录，不使用新增 SHAP 依赖。
- 相关性使用开发期 100 个确定性截面，根特征逐日 Spearman 后取均值。
- 组合暴露对应周频模拟器的 Top-30 等权目标持仓，不是完整账本成交持仓。

## 主要结果

### 特征重要性

| 方法 | 前列特征 |
|---|---|
| Gain | `LIMIT_UP_CNT_20` 13.60%、`LN_AMT_20` 10.86%、`turnover_rate` 10.66%、`VWAP_BIAS_5` 10.01% |
| TreeSHAP | `LN_AMT_20`、`VWAP_BIAS_5`、`REV_5`、`turnover_rate`、`BOLL_POS_20` |

Gain 最高的是 `LIMIT_UP_CNT_20`；TreeSHAP 最高的是 `LN_AMT_20`。两套排序不完全一致：
树分裂收益会偏向频繁参与分裂的特征，TreeSHAP 更接近当前样本上的边际贡献。

### 分族重训

相对 `seed=0` 完整模型（RankIC 0.09708），删除后 RankIC 下降最多的分组为：

| 分组 | 删除后 RankIC | RankIC 变化 | 年化净收益变化 |
|---|---:|---:|---:|
| `volume_price` 量价关系 | 0.09427 | -0.00281 | -4.67 个百分点 |
| `fundamental` 财务与质量 | 0.09483 | -0.00225 | +1.94 个百分点 |
| `context` 上下文特征 | 0.09548 | -0.00160 | +1.00 个百分点 |
| `alpha101` | 0.09551 | -0.00157 | +1.88 个百分点 |
| `liquidity` 流动性 | 0.09553 | -0.00155 | +1.58 个百分点 |

`technical` 删除后 RankIC 基本不变；`cn_special` 删除后 RankIC 上升 0.00186，
但年化净收益从 14.50% 降到 12.83%、ICIR 从 0.831 升到 0.870，
不同指标方向相反，说明该族不是稳定的增量来源，不能只按单项指标判断。

### 单因子置换

影响最大的单因子依次为 `LN_AMT_20`、`VWAP_BIAS_5`、`turnover_rate`、
`BOLL_POS_20`。置换 `LN_AMT_20` 后 RankIC 从 0.09708 降到 0.07773，
下降 0.01935，是唯一明显高于其余因子的依赖。

### 相关性

100 个截面共报告 18 对 `|rho|>=0.8` 的根特征。主要冗余包括：

- `LN_MV` 与 `log_mv` 完全同向；
- `MOM_20`、`REV_20`、`RS_20` 构成近似镜像/同向簇；
- `ROA_YEARLY`、`ROE_YEARLY`、`ROIC` 高度重合；
- `CURRENT_RATIO` 与 `QUICK_RATIO`、`NP_GROWTH` 与 `OP_GROWTH` 高度重合。

因此分族与单因子贡献不可直接相加；置换某个特征时，模型仍可能从同族或高相关特征补足。

### 组合暴露

七类风格的时序均值为：

| 风格 | 均值 | t 值 | 覆盖 |
|---|---:|---:|---:|
| SIZE 规模 | -1.202 | -39.94 | 98.8% |
| SHORT_REV 短期反转 | +0.448 | +20.21 | 100.0% |
| VOL 波动 | -0.332 | -11.78 | 99.9% |
| LIQUIDITY 流动性 | -0.360 | -16.37 | 98.7% |
| MOMENTUM 动量 | -0.319 | -16.41 | 87.7% |
| BETA_60 贝塔 | -0.046 | -1.96 | 99.9% |
| VALUE 估值 | -0.054 | -2.98 | 98.3% |

组合明显偏小市值、短期反转、低流动性、低波动和负动量。行业平均权重最高的是
机械设备 20.5%（股票池等权 9.8%）、基础化工 12.6%（8.6%）、银行 9.0%、
医药生物 8.4%、环保 8.0%。银行只出现在 32 个信号日，属于阶段性集中，不宜理解为长期稳定行业偏好。

## 复现命令

```bash
cd /home/hanwen/quant
.venv/bin/python -m scripts.build_alpha.explain_model \
  --model-id lightgbm_ensemble_label_rank_6aea5c9299d5 \
  --top-n 30 --sample-rows 5000 \
  --permutation-seed 20261020 --ablation-seed 0 --num-threads 4
```

相同模型、数据和代码会复用已有的已完成分析；需要重算时显式传 `--force`。

## 解释边界

- 重要性描述当前模型如何使用特征，不是因果结论，也不代表删除后模型一定会失效。
- 消融统一基于 `seed=0`；五种子集成的正式 RankIC 为 0.09727、年化净收益为 17.42%。
- 组合暴露来自研究模拟器目标持仓，P4 仍须用完整回测账本重新验收。
- P3-5 决策门已单独执行，结论见 `p3_decision.md`；本解释分析不改写模型状态。
