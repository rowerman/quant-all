# 模型登记与历史模型索引

日期：2026-10-23

## 职责

`alpha.registry` 从模型目录中的 manifest、config、features、metrics、实际模型产物
和运行 manifest 生成统一索引 `data/models/index.json`。索引用于按 `model_id`
定位历史模型及其下游运行，不复制模型文件，也不修改模型 manifest 或指标。

事实来源始终保持为各模型目录；`index.json` 是可删除、可重建的派生结果。
`data/models/experiments.csv` 继续记录基线和训练实验，不替代模型索引。

## 接口

```python
from alpha.registry import (
    rebuild_registry,
    load_registry,
    get_registered_model,
)

payload = rebuild_registry()
registry = load_registry()
model = get_registered_model("lightgbm_ensemble_label_rank_6aea5c9299d5")
```

- `rebuild_registry()`：扫描、完整校验并原子重建索引。
- `load_registry()`：读取并校验索引结构，不重新扫描模型产物。
- `get_registered_model()`：按模型 ID 返回单条记录。

命令行：

```bash
.venv/bin/python -m scripts.build_alpha.registry --rebuild
```

## 登记内容

每条记录包含：

- 模型标识：`model_id`、`model_type`、`member_type`；
- 状态与决策：manifest 中的 `status`、`decision`，以及 metrics 中的决策摘要；
- 研究口径：标签、目标、股票池、基准、数据集和创建/评估时间；
- 特征：原始特征、上下文特征、转换特征及数量；
- 窗口与折：训练和测试起止日、折数、逐折日期及 purge/embargo；
- 参数与指标：完整 config 和 RankIC、ICIR、收益、回撤、Sharpe、换手等摘要；
- 产物：模型目录、分析目录、manifest/config/features/metrics 哈希、全部模型文件、
  预测分片、数据集摘要和 `artifact_sha256`。
- 下游：`downstream.evaluations`、`acceptances`、`monitoring`、`lifecycles` 和
  `final_holdouts`，每项记录运行 ID、状态、区间和相关 manifest 哈希。

索引 schema 为 `alpha_model_registry_v3`，记录按 `created_at`、`model_id` 排序。
除 `generated_at` 外，相同产物重复重建应得到相同内容。

顶层 `status` 是模型 manifest 的训练状态，`status_scope` 固定为 `training`。
模型是否完成评估、账本、监控或生命周期应读取 `downstream`，不能从训练状态推断。

## 校验流程

1. 扫描 `data/models/models/` 和 `data/models/archive/models/`，跳过运行数据、
   数据集、基线和切分目录。
2. 只处理含 `model_id` 和 `model_type` 的 manifest，并校验目录名与 ID 一致。
3. 校验 features.json 与 manifest 特征清单一致，metrics 中的模型和数据 ID 一致。
4. 若 manifest 声明 `evaluation_sha256`，核对实际 metrics 文件哈希。
5. 调用 `model_artifact_snapshot()` 校验模型文件、预测分片和训练数据集。
6. 扫描评估、账本、监控、生命周期和最终留出 manifest，按模型 ID 建立运行链接。
7. 全部模型通过后，原子写入 `data/models/index.json`。

## 错误语义

- `ALPHA-401`：模型文件、预测、数据集或元数据缺失、损坏、内容不一致。
- `ALPHA-402`：目录与 ID 不一致、没有可登记模型、索引 schema/计数错误或记录冲突。

重建失败时保留上一次成功索引，不写入部分结果。

## 调用关系

- 训练命令成功后自动调用；手工更新模型或运行产物时也可通过
  `scripts.build_alpha.registry --rebuild` 显式调用。
- P5-2 监控运行前使用登记索引按 `model_id` 获取模型、指标和分析产物；
  模型或预测更新后仍应先重建索引。
- 登记模块依赖 `alpha.predict.model_artifact_snapshot()` 和
  `alpha.artifacts` 的 JSON/哈希工具。

## 修改注意

- 不修改历史模型目录来适配索引；需要新增信息时扩展登记提取逻辑。
- `status` 和 `decision` 保留模型原值，不在登记层自行解释为运行生命周期。
- 下游运行只读扫描 manifest，不回写模型 manifest。
- 新增模型类型时同步检查 `fold_models`、预测文件和 config 的归一化逻辑。
- 索引只是发现入口，完整指标仍以模型目录下的 `metrics.json` 为准。
- 登记不得读取 2024-2026 最终测试集。
