# 模型训练后端

日期：2026-10-23

## 训练入口

所有模型通过同一入口训练：

```bash
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id <dataset_id> \
  --spec configs/alpha/<spec>.json
```

规格由 `TrainingSpec` 解析。`backend` 选择已注册后端，`seeds` 非空时由通用流水线
生成多种子 rank 集成。模型 ID、折循环、重拟合、预测、manifest 和实验索引均由
共享流水线处理。命令行在模型写完后自动重建模型索引；登记失败时命令非零退出。

## 新增后端

在 `research/modeling/alpha/modeling/backends/` 新增模块，实现 `ModelBackend` 的以下能力并在模块
导入时注册：

- `build_config`、`config_dict`、`with_seed`
- `fit_fold`、`refit`
- `predict`
- `save_model`、`load_model`
- `manifest_metadata`

后端只处理单折算法；不得处理股票池、组合、交易或最终验收。模型专属分析应留在
后端或独立分析模块。后端包会自动导入目录内的公开模块，不需要修改
`research/modeling/alpha/modeling/__init__.py` 或 `backends/__init__.py` 的固定清单。

## Manifest

- 新模型写 `alpha_model_v2`，记录后端名称、后端源码哈希、共享代码哈希、配置、
  特征、折、模型文件和标准预测哈希。
- 历史 `alpha_model_v1` 在读取时归一化，不修改原文件。
- 预测重放按 manifest 后端名称从注册表加载，不判断模型类型。
- 已有训练产物恢复时会校验全部折模型文件、预测分片、行数和日期范围；
  文件损坏时拒绝恢复，需显式使用 `--force` 重建。

## 内置后端

| 后端 | 模型 |
|---|---|
| `linear` | Ridge、ElasticNet |
| `lightgbm` | LightGBM 单模型；多 `seeds` 时形成集成 |
| `sklearn_hist_gbdt` | Scikit-learn 直方图梯度提升单模型 |

`sklearn_hist_gbdt` 使用固定超参数、训练集内部早停和 train+valid 固定轮数重拟合；
模型文件为 `.joblib`。实际训练规格见
`configs/alpha/train_sklearn_hist_gbdt.json`。

`sklearn_hist_gbdt_label_rank_83e1200e554f` 的 manifest 保持训练状态 `trained`；
它参与的十候选评估、完整账本、监控和生命周期通过模型索引 v3 的 `downstream`
关联，不回写模型 manifest。
