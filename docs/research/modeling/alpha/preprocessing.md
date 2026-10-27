# Alpha 模型截面预处理

日期：2026-10-27（V120）

本文说明周频模型数据集和训练前的截面预处理。输入是带 `trade_date`、可交易标记、标签和
原始因子的 DataFrame；输出仍是同一行顺序的 DataFrame、模型特征清单、缺失列和诊断报告。

## 负责什么

- 只在 `eligible=True` 的样本上保留特征值。
- 可选按交易日做 1%/99% 线性插值分位裁剪。
- 按交易日计算平均并列名次 rank，缺失值在后续步骤填 0.5。
- 或按交易日计算总体标准差口径的 z-score，缺失和零方差结果填 0。
- 可选按行业、市值做中性化，再对 z-score 结果执行 `clip_z`。
- 为每个根特征生成可选的缺失指示列，并保持 float32 输出和固定列顺序。
- 数据集构建阶段用同一 rank 核计算可交易样本内的 `label_rank`。

不在本模块负责：原始数据读取、股票池定义、标签生成、折切分、模型训练和预测组合。

## 调用关系

```text
alpha.dataset.assemble_dataset
  -> alpha.kernel_ops.rank_pct_by_date

alpha.preprocess.transform
  -> alpha.kernel_ops.date_groups
  -> alpha.kernel_ops.rank_pct_by_date
  -> alpha.kernel_ops.winsorize_quantile_by_date
  -> alpha.kernel_ops.zscore_population_by_date
  -> kernels.get_backend
```

`alpha.kernel_ops` 只负责日期稳定分组、后端选择和原行序回填。数值算法由 `kernels` 的
Python/C++ 后端提供；中性化和市值残差仍由 `alpha.preprocess` 的 Python 代码完成。

## 数据和执行契约

- 每次 `transform()` 只选择一次后端，并用同一组日期边界处理全部特征。
- 日期先稳定分组，即使输入行序变化也按原行序返回；缺失日期行不进入核。
- 正负无穷在进入核前转为 NaN，NaN 不参与排名、分位或均值统计。
- `rank` 使用平均并列名次，输出范围为 0 到 1。
- 1%/99% 裁剪使用 pandas 默认的线性插值分位；分位在当日有效样本内计算。
- 总体 z-score 使用 `ddof=0`；均值保持补偿求和顺序，方差沿用 pandas groupby 的逐样本
  更新顺序，避免 float32 落值漂移。
- 中性化发生在缺失填充之后；只有 z-score 模式最后执行 `clip_z`。
- 任何 C++ 运行期错误转换为 `KERNEL-004`，不会静默切换后端。
- 模型代码哈希包含 `alpha.kernel_ops`、统一核协议、Python fallback 和对应 C++ 源码，
  因此数值实现变化会生成新的模型 ID。

## C++ 接入

| 模型路径 | 原实现 | 统一核 | 默认后端 |
|---|---|---|---|
| 特征和标签截面排名 | `groupby().rank(pct=True, method="average")` | `daily_rank_pct` | `auto` |
| 1%/99% 分位裁剪 | `groupby().transform(quantile)` + `clip` | `daily_winsorize_quantile` | `auto` |
| 总体标准差 z-score | `groupby().transform("std", ddof=0)` | `daily_zscore_population` | `auto` |

后端由 `QUANT_KERNEL_BACKEND=auto|python|native` 控制，线程由
`QUANT_NATIVE_THREADS` 控制并默认 1。扩展不可导入时 `auto` 回退 Python；显式 `native`
缺少扩展时失败。

V120 将“模型截面排名”“模型分位去极值”“模型总体标准差 z-score”分别登记为实际
C++ 替换模块，因此项目替换模块数由 10 增至 13。

## 性能证据

固定开发集 `6a59680fd71c50a8`：2,138,239 行、613 个交易日、50 个根特征。每个后端在
独立进程运行 3 次取中位数，单线程：

| 模式 | Python | native | auto | native 加速 | 峰值 RSS 比例 |
|---|---:|---:|---:|---:|---:|
| rank | 26.095s | 7.951s | 7.996s | 3.282x | 0.988x |
| zscore | 6.607s | 3.509s | 3.475s | 1.883x | 1.007x |
| rank + winsorize | 58.726s | 17.095s | 16.959s | 3.435x | 0.995x |
| zscore + winsorize | 38.841s | 12.345s | 12.604s | 3.146x | 0.999x |

四个模式的 50 列 float32 列级 SHA-256 和组合 SHA-256 在三后端完全一致；正确性门禁和
“native/auto 不慢于 Python”门禁均通过。证据位于
`AiRecord/benchmarks/V120/p2-01-20261027/`，复现命令：

```bash
.venv/bin/python -m tools.monitor.modeling_preprocess_baseline \
  --output AiRecord/benchmarks/V120/p2-01-20261027
```

## 修改注意

- 不要用评估层的样本标准差或 MAD 核替代模型总体标准差和 1%/99% 分位裁剪。
- 不要为每个特征重新做日期因子化；应复用一次 `DateGroups`。
- 不要改变 rank 后的缺失填充、中性化顺序或 z-score 的 `clip_z` 顺序。
- 新核必须同时覆盖空组、单样本、常数、NaN、无穷、并列和打乱行序。
- 不读取最终留出集来验证优化；模型性能验收使用固定开发集和合成训练数据。
