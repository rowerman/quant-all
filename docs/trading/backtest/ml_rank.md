# ml_rank 目标权重提供者

## 职责

`ml_rank` 把已落盘 ML 模型预测转换为回测目标权重。它只读取预测，不训练模型、
不改写预测，也不自行模拟现金、持仓或订单。

## 参数

### v1：固定周最后交易日

```json
{
  "name": "ml_rank",
  "version": "1",
  "params": {
    "model_id": "lightgbm_ensemble_label_rank_6aea5c9299d5",
    "top_n": 30
  }
}
```

- `model_id`：必需的模型产物 ID。
- `top_n`：每周等权持有的股票数，默认 30，必须是正整数。
- 平滑窗口固定为 4 个信号日，不开放参数，避免回测和 P3-5 研究口径分叉。
- v1 要求回测中的每个自然周最后交易日都有预测；缺失直接报错。

### v2：显式调仓日

```json
{
  "name": "ml_rank",
  "version": "2",
  "params": {
    "model_id": "lightgbm_ensemble_label_rank_6aea5c9299d5",
    "top_n": 30,
    "smoothing_weeks": 4,
    "rebalance_dates": ["2023-01-13", "2023-01-20"]
  }
}
```

- `rebalance_dates` 可选；省略时行为和 v1 一致。
- `smoothing_weeks` 可选，默认 4；使用 1 表示只做当日截面 rank，不做跨周平滑。
- 提供显式日期时，只在这些日期生成目标，并要求每个日期都有预测。
- 日期必须排序去重、使用 `YYYY-MM-DD`，且不能超出本次行情面板。
- P4-4A 使用模型实际存在的 300 个预测日，而不是 305 个自然周最后交易日。

### v3：生命周期执行

```json
{
  "name": "ml_rank",
  "version": "3",
  "params": {
    "lifecycle_id": "p5_3_dd7c9b4bf235",
    "model_id": "lightgbm_ensemble_label_rank_6aea5c9299d5",
    "fallback_model_id": "ridge_label_rank_aaff2cb2fbcf",
    "top_n": 30,
    "smoothing_weeks": 4,
    "rebalance_dates": ["2023-01-13", "2023-01-20"]
  }
}
```

- `lifecycle_id` 必须指向完整且哈希校验通过的 P5-3 生命周期包。
- `normal` 使用主模型满仓；`reduced` 使用主模型并将目标权重缩为 0.5。
- `rollback` 使用 Ridge 预测和相同的因果 rank 平滑；`paused` 返回全 0 行。
- 其他日期保持整行 `NaN`。
- v1、v2 的行为和源码语义保持不变。

## 信号流程

1. 通过 `alpha.predict.load_predictions()` 读取并完整校验模型预测。
2. v1 在行情面板的周最后交易日生成目标；v2/v3 在显式 `rebalance_dates` 或
   默认周最后交易日生成目标，其他交易日返回整行 `NaN`。
3. 每个预测日对存储分数做截面百分比 rank。
4. 按 `smoothing_weeks` 对当前及此前信号日的 rank 做因果等权平均；
   默认 4 周，值为 1 时保持原始当日 rank。
5. v3 按生命周期状态选择主模型、Ridge 或空仓，再应用仓位乘数。
6. 用回测信号层的 `eligible` 矩阵过滤停牌、ST、退市整理期、上市不足和指数池限制。
7. 在可用股票中选平滑分数最高的 `top_n` 只，等权持有；没有股票时返回全 0 行。

平滑只使用当前和过去预测。回测分块只改变行情面板长度，不改变同一信号日的
历史预测窗口，因此单块与分块应得到相同目标权重。

## 失败行为

- 参数错误：`SpecError`（`BT-101`）。
- 模型或预测缺失、哈希不一致、预测结构损坏：转换为 `DataAlignmentError`（`BT-201`）。
- 回测信号日超出模型预测起止范围，或缺少某个信号日预测：直接报错。
- 不允许静默跳过缺失预测后继续交易。

## 调用关系

- 上游：`backtest.providers.discover_providers()` 自动发现并注册。
- 输入：`MarketData`、信号日 `eligible` 矩阵、provider 参数。
- 下游：`backtest.signals.build_signal_targets()` 校验权重矩阵，再交给执行账本。
- 模型读取：`alpha.predict.load_predictions()`。

## 输入快照

`MlRankProvider.artifact_snapshot()` 和 v2 继承的同名方法调用
`alpha.model_artifact_snapshot()`，
把以下内容写入 `rules.provider.artifacts`：

- 模型 manifest、全部实际模型文件和声明哈希，包括集成成员模型。
- 模型参数、预处理、标签、目标、股票池、折、随机种子和全部特征。
- 训练数据集及数据集 manifest 的实际 SHA256、版本、规模和覆盖。
- 主预测和成员预测的哈希、行数、日期数及覆盖区间。

产物规范化后的 `artifact_sha256` 参与最终 `snapshot_id`。模型文件、数据集或同步更新
manifest 后的预测内容变化都会产生新快照。模型或数据集产物缺失、不可读时在创建
运行记录前报 `BT-201`。

v3 的快照同时包含主模型、Ridge 回滚模型、生命周期 manifest、完成文件哈希和
生命周期时序哈希。任何一项变化都会改变回测 `snapshot_id`。

## 版本与审计

- `backtest.strategies.ml_rank` 注册 v1，源码与参数语义保持不变。
- `backtest.strategies.ml_rank_v2` 注册 v2；同名不同版本可由 provider 注册表并存，
  显式版本请求会取得对应实现。
- v2 的 `smoothing_weeks=1` 用于 P4-4B1 原始分数对照，默认 4 周行为保持不变。
- 显式调仓日清单会原样写入策略参数和输入快照。P4-4A 另外记录日期数量、
  起止日期和清单 SHA256。
- `backtest.strategies.ml_rank_v3` 注册 v3，在回测中执行 P5-3 状态切换。
- P5-3 正式验收使用生命周期 `p5_3_dd7c9b4bf235`，完整账本 ID
  `p5_3_4f91d5b68fba`；300 个信号日和 299 个执行段全部通过。
