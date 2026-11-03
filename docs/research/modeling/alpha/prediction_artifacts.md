# 已落盘预测读取与校验

## 职责

`alpha.predict.load_predictions()` 按 `model_id` 读取训练阶段已经落盘的样本外预测，
供评估、分析和回测提供者共用。它不重新加载数据集，也不重新运行模型。

重新计算预测仍使用 `alpha.predict.predict()`；两者语义不同：

- `predict()`：读取数据集和模型文件，重算预测，并在完整区间核对重放哈希。
- `load_predictions()`：读取已经生成的年度分片，校验产物后用于下游消费。

## 接口

```python
load_predictions(
    model_id: str,
    start: str | None = None,
    end: str | None = None,
    *,
    root: Path | None = None,
) -> tuple[Path, dict, pd.DataFrame]

model_artifact_snapshot(
    model_id: str,
    root: Path | None = None,
) -> dict
```

- 返回模型目录、已校验 manifest 和排好序的预测表。
- `start`、`end` 为闭区间；任一为 `None` 时使用 manifest 对应端点。
- `root` 只用于测试或指定模型产物根目录，默认是 `data/models/`。
- `model_artifact_snapshot()` 返回模型、预测和训练数据集的稳定摘要，供回测输入快照使用。

预测表固定为：

| 列 | 类型 | 含义 |
|---|---|---|
| `trade_date` | datetime | 周最后交易信号日 |
| `stock_code` | str | 股票代码 |
| `score` | float64 | 模型分数 |

## 校验顺序

1. 模型目录名和 manifest 中的 `model_id` 一致。
2. 每个年度分片存在，年度键为四位数字，分片日期属于对应年度。
3. 必需列齐全，日期和股票代码非空，分数全部有限。
4. 分片内及合并后都不存在重复的 `(trade_date, stock_code)`。
5. 每个分片 SHA256 和合并预测 SHA256 与 manifest 一致。
6. 行数、日期数、起止日期与 manifest 一致。
7. 完整产物通过后，才按请求区间截取。

任何一步失败都抛出 `AlphaArtifactError`（`ALPHA-401`），不会返回部分结果。

## 产物快照

`model_artifact_snapshot()` 先调用 `load_predictions()` 完整校验主预测，再读取：

- 模型 manifest 的实际 SHA256，以及模型 ID、类型、参数、预处理、种子、折和特征。
- `fold_models` 指向的全部实际模型文件 SHA256，并保留 manifest 声明值及是否一致。
- 训练数据集和数据集 manifest 的实际 SHA256、版本、规模、覆盖和特征。
- 主预测及成员预测的哈希、行数、日期数和覆盖区间。

这些内容规范化后得到 `artifact_sha256`。摘要不含创建时间或绝对路径，相同产物重复
构建结果一致。模型文件内容变化会改变摘要；预测文件内容被修改但 manifest 未同步时，
仍由 `load_predictions()` 直接报错。

## 调用关系

- `scripts.build_alpha.evaluate`：模型评估和成员预测读取。
- `scripts.build_alpha.evaluation_executor`：通用评估计划的预测输入。
- `scripts.build_alpha.analysis_common`：P3-4 解释分析。
- `backtest.strategies.ml_rank`：回测目标权重提供者。
- 回测输入快照：`ml_rank` 通过 `artifact_snapshot()` 写入 `rules.provider.artifacts`。

## 修改注意

- 不修改任何历史预测文件；模型更新应生成新的 `model_id`。
- manifest 字段变化时，同步检查训练写出逻辑和全部调用方。
- 回测只允许使用开发期预测，不能借读取接口载入 2024-2026 最终测试集。
