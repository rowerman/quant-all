# C++ 批量计算扩展

`native` 负责构建顶层扩展 `quant_native`，当前提供完整矩阵和目标行两种成对 Spearman
核，以及逐日排名、Pearson、MAD、z-score、融合标准化、分位裁剪、总体标准差
z-score、分位分组、多标签分位收益、分位换手、逐行 Spearman 和逐日 OLS 核。
融合标准化同时支持单列和多列批量入口。评估路径、模型预处理和挖掘候选相关性均已
接入；有扩展时默认 `auto` 选择 native，无扩展时回退 Python。

## 负责什么

- 接收按日期或批次分组的股票×因子 C 连续 `float64` 矩阵。
- 一次计算所有组的完整 Spearman 相关矩阵，或只计算一个目标因子的行、列和对角线。
- 按实际配对样本剔除非有限值，使用并列平均名次，常数截面返回 `NaN`。
- 按组计算平均名次分位和 Pearson 相关，保持原有效计数与最小样本门槛。
- 按组计算中位数、MAD、裁剪和 z-score；z-score 使用与 pandas 相同的补偿求和顺序。
- 按组使用 pandas 线性插值口径计算 1%/99% 分位裁剪。
- 按组计算总体标准差口径 `ddof=0` 的 z-score。
- 一次处理风格暴露的多列标准化，共享日分组和 offsets。
- 按组计算平均名次分位；批量计算多标签分位收益、池内均值和分位换手。
- 对每个 lag 的共同有限股票逐行重新排名并计算 Spearman。
- 按日计算截面 OLS、active mask、t 值、R² 和残差；病态矩阵按日回退 Python。
- 计算期间释放 GIL，按组使用受控线程数。

不负责输入数据读取、因子目录、评估 payload、日志聚合、业务回退选择或并发阶段锁。

## 被谁调用

- `kernels.native.CppKernelBackend` 是唯一 Python 适配入口。
- 正式完整矩阵和候选目标行经 `calculate.correlation` 调用 `kernels` 适配层。
- IC、RankIC、多周期衰减和增量 IC 经 `calculate.daily` 调用同一适配层。
- 风格暴露经 `calculate.exposure` 调用融合标准化核。
- 风格暴露经 `calculate.exposure` 调用逐日 OLS。
- 分位分组、多标签收益、换手和自相关经 `calculate.binning` 调用同一适配层。
- 模型特征和标签排名、分位裁剪、总体 z-score 经 `alpha.kernel_ops` 调用同一适配层。
- 挖掘候选对正式因子的抽样相关性经 `mining.evaluation_correlation` 调用逐行
  Spearman。
- 核测试和 P1-02/P1-03 基准工具直接验证适配层或扩展。
- 模型训练、挖掘表达式执行和回测尚未调用这些核。

业务模块不得直接导入 `quant_native`。后续接入必须经过 `kernels.get_backend()` 和统一
`NativeKernelBackend` 协议。

## 构建

构建链使用：

| 组件 | 固定版本 | 用途 |
|---|---|---|
| C++ | C++20 | 编译标准 |
| CMake | 4.4.4 | 原生工程配置 |
| Ninja | 1.13.2 | 构建执行 |
| pybind11 | 3.1.0 | Python 绑定、NumPy 数组和异常转换 |
| scikit-build-core | 1.1.1 | setuptools 兼容构建后端 |
| Eigen | 3.4.0 | OLS QR 和协方差计算；源码内置，MPL2 |

安装主环境后执行：

```bash
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -c "import quant_native"
```

普通 wheel：

```bash
.venv/bin/python -m pip wheel . --no-deps --wheel-dir /tmp/quant-wheel
```

当前验收环境为 Python 3.12.3、x86_64 Linux、GCC 13.3.0。其他平台需要重新构建并留下
对应 wheel 和测试证据。

## 接口

```python
pairwise_spearman(values, group_offsets, min_n=5, target_index=None)
daily_rank_pct(values, group_offsets)
daily_pearson(left, right, group_offsets, min_n=5)
daily_winsorize_mad(values, group_offsets, n_mad=3.0)
daily_zscore(values, group_offsets)
daily_winsorize_zscore(values, group_offsets, n_mad=3.0)
daily_winsorize_zscore_columns(values, group_offsets, n_mad=3.0)
daily_winsorize_quantile(
    values, group_offsets, lower_quantile=0.01, upper_quantile=0.99
)
daily_zscore_population(values, group_offsets)
daily_quantile_bins(values, group_offsets, quantiles)
bin_returns_multi(
    values, day_codes, bin_codes, weights, day_count, bin_count
)
bin_turnover(values, quantiles)
daily_row_spearman(left, right, min_n=5)
daily_ols(design, response, group_offsets, min_rows=30)
```

- `values`：C 连续 `float64`，形状为 `[样本, 因子]`。
- `group_offsets`：C 连续 `int64`，长度等于组数加一，首项 0、末项样本数、单调不减。
- `min_n`：每个因子对参与相关计算的最少有限配对数，不得小于 5。
- `target_index`：留空计算完整矩阵；给出矩阵内整数列号时，只填充目标行、目标列和
  目标对角线，其余相关值为 `NaN`、计数为 0。
- 返回：`(correlation[组, 因子, 因子], count[组, 因子, 因子])`。

逐日接口约定：

- `daily_rank_pct.values`：C 连续一维 `float64`；NaN 原位保留，其他值按组计算平均并列
  名次分位。
- `daily_pearson.left/right`：长度相同的一维 C 连续 `float64`；只使用两侧都有限的配对。
- `daily_winsorize_mad`：返回与输入等长的裁剪数组；MAD 为零或无法裁剪时保持原值。
- `daily_zscore`：样本标准差 `ddof=1`；有效样本不足 2 或标准差为零时返回 `NaN`。
- `daily_winsorize_quantile`：默认使用当日有效样本的 1%/99% 线性插值分位裁剪；
  NaN 保持缺失。
- `daily_zscore_population`：总体标准差 `ddof=0`；均值保持 pandas groupby 的补偿
  求和顺序，方差保持逐样本更新顺序，使 float32 落值一致。
- `daily_winsorize_zscore`：对裁剪结果继续计算 z-score，供风格暴露一次完成标准化。
- `daily_winsorize_zscore_columns`：输入二维 C 连续矩阵，对每一列执行与
  `daily_winsorize_zscore` 相同的融合标准化；用于风格列和因子共享一次分组。
- 两个日度接口的 `group_offsets` 都与 Spearman 相同，返回一维结果；Pearson 额外返回
  每组有效配对数。
- 业务评估的最小截面继续固定为 5；底层 Pearson 接口接受不小于 2 的门槛。

分位与稳定性接口约定：

- `daily_quantile_bins` 返回 0 起始组号；NaN 返回 -1，正负无穷与其他有限值一起排序。
- `bin_returns_multi` 的 `values` 为 `[样本, 标签]`，`day_codes` 与 `bin_codes` 为
  非负 `int64`；返回 `[交易日, 分位, 标签]` 收益和 `[交易日, 标签]` 池内均值。
- `bin_turnover` 的 `values` 为 `[交易日, 股票]` 的 0 起始组号矩阵，NaN 表示当日不在池；
  返回逐日各分位换手和全池换手。
- `daily_row_spearman` 的输入为两张同形矩阵；每行只使用两侧有限的配对，少于 5 个或
  任一侧为常数时返回 NaN，并同时返回有效配对数。
- `daily_ols` 返回有效日期、完整列系数和 t、R²、样本数、active mask、分类跳过计数，
  以及成功回归日的残差行号和值。正常矩阵走 QR，病态矩阵按日回退 Python。

Python 适配层和 C++ 绑定层都会检查 dtype、维度、连续内存、offsets 和参数。原生错误
转换为 Python 异常。显式请求 `native` 但扩展缺失时返回 `KERNEL-004`；`auto` 只在扩展
不可导入时回退 Python，运行期计算错误不静默回退。

## 线程与语义

- `QUANT_NATIVE_THREADS` 每次调用读取，默认 1，必须是正整数。
- 线程按组并行；单日单组输入不会在组内自行扩线程。
- 评估请求仍由 `calculate.native` 的阶段锁串行隔离，默认单线程避免与 DuckDB 叠加。
- 不使用全局可变状态或跨请求缓存。
- `calculate.native` 仍是评估并发锁；它不是本扩展，也不负责计算后端选择。

## 正确性与性能证据

P1-02 在固定 2026-08-03 的 5,219×47 截面上，以单组、单线程、独立子进程对照现有
Python 实现：

| 后端 | 热运行中位 | 峰值 RSS | 与 Python 对照 |
|---|---:|---:|---|
| Python | 0.172836s | 282.820MB | 金标准 |
| C++ | 0.073312s | 283.129MB | 相关系数 `1e-10` 内一致，计数完全一致 |

C++ 核热运行加速 `2.358x`，峰值 RSS 为 Python 的 `1.001x`。证据位于
`AiRecord/benchmarks/V113/p1-02-20261027/`。

## P1-03 评估相关性接入

P1-03 增加目标行模式，正式完整矩阵和候选目标行分别经统一后端计算。固定三年区间结果：

| 路径 | Python | native | auto | 内存比例 |
|---|---:|---:|---:|---:|
| 正式完整矩阵 | 217.770s | 140.460s | 135.788s | 1.0001x |
| 候选目标行 | 137.599s | 118.680s | 120.995s | 1.0001x |

正式 native 下降 `35.5%`，默认 auto 下降 `37.6%`；候选 native 加速 `1.159x`，auto
加速 `1.137x`。固定 5,219×47 核微基准中，完整矩阵 `0.167855s → 0.060035s`
（`2.796x`），目标行 `0.042791s → 0.021802s`（`1.963x`）。Python/native 相关值在
`1e-10` 内一致，有效计数和 `NaN` 位置完全一致。证据位于
`AiRecord/benchmarks/V114/p1-03-20261027/`。

## P1-04 逐日排名与 Pearson 接入

`calculate.daily` 在每次 IC、RankIC、多周期衰减或增量 IC 调用中只选择一次后端，先按
交易日稳定整理为连续分组，再复用同一 offsets。多标签衰减仍对每个标签独立取有限配对，
不会复用其他标签的排名或缺失掩码。

固定三年真实面板为 3,724,919 行、722 组、单线程，热运行 5 次取中位数：

| 核 | Python | C++ | 加速 |
|---|---:|---:|---:|
| 逐日排名 | 1.114445s | 0.238150s | 4.680x |
| 逐日 Pearson | 0.210444s | 0.011909s | 17.670x |

| stage | Python | native | 加速 |
|---|---:|---:|---:|
| `overview` | 4.661s | 2.239s | 2.082x |
| `decay` | 15.145s | 5.371s | 2.820x |
| `incremental_ic` | 21.727s | 16.298s | 1.333x |

Python/native/auto 的 payload 在 `1e-10` 内一致，有效计数和 `NaN` 位置一致；native/auto
峰值 RSS 比例最高 `1.037x`。`incremental_ic` 的主要耗时仍是风格回归，因此该 stage 以不退化
为门槛。证据位于 `AiRecord/benchmarks/V115/p1-04-20261027/`。

## P1-05 逐日 MAD 与 z-score 接入

原生 `daily_winsorize_mad`、`daily_zscore` 和 `daily_winsorize_zscore` 保持现有 pandas
缺失、无穷、常数、MAD 为零和最小样本语义。z-score 的和与平方和使用 Kahan 补偿，
并保持 `count * (mean * mean)` 的结合顺序，以匹配 pandas groupby 的逐值结果。

固定 3,724,919 行、722 组、单线程、热运行 5 次：

| 核 | Python | C++ | 加速 |
|---|---:|---:|---:|
| MAD 去极值 | 0.195444s | 0.084487s | 2.313x |
| z-score | 0.158660s | 0.014637s | 10.840x |
| 融合标准化 | 0.342608s | 0.096023s | 3.568x |

真实 `overview` 由 4.516 秒降至 1.782 秒（2.535x），`exposure` 由 15.922 秒降至
13.237 秒（1.203x）。三后端 payload 在 `1e-10` 内一致，最高 RSS 比例 1.0195x。
证据位于 `AiRecord/benchmarks/V116/p1-05-20261027/`。

## P1-06 分位收益、换手和自相关接入

`calculate.binning` 的分位分组、秩宽表、多标签分位收益、换手和自相关分别接入统一后端。
`bin_daily_returns` 继续使用 pandas 分组求和，以保持分位图市值加权的既有逐值语义；分位图
stage 的收益主要来自原生分位分组和秩宽表。

固定 3,751,348 行、727 个交易日、5,606 只股票，单线程热运行 5 次取中位数：

| 核 | Python | C++ | 加速 |
|---|---:|---:|---:|
| 分位分组 | 1.113666s | 0.251838s | 4.422x |
| 等权多标签收益 | 0.236963s | 0.046971s | 5.045x |
| 市值加权多标签收益 | 0.240939s | 0.048037s | 5.016x |
| 分位换手 | 0.068774s | 0.039926s | 1.723x |
| 三档自相关 | 2.454977s | 1.423984s | 1.724x |

| stage | Python | native | 加速 |
|---|---:|---:|---:|
| `bins` | 2.353s | 1.452s | 1.621x |
| `bin_curves` | 2.366s | 1.503s | 1.575x |
| `spread` | 2.320s | 1.456s | 1.594x |
| `bin_dist` | 2.591s | 1.717s | 1.509x |
| `turnover` | 5.787s | 3.121s | 1.854x |
| `decay` | 15.482s | 4.756s | 3.255x |

`bins + turnover` 合计改善 `43.82%`；Python/native/auto 数组和 payload 在 `1e-10`
内一致，分位组号与有效计数逐值一致，最高 RSS 比例 `1.0060x`。证据位于
`AiRecord/benchmarks/V117/p1-06-20261027/`。

## C++ 替换台账

| 项目 | 当前状态 |
|---|---|
| 扩展 | `quant_native` 已构建 |
| 原生核 | `pairwise_spearman`、`daily_rank_pct`、`daily_pearson`、`daily_winsorize_mad`、`daily_zscore`、`daily_winsorize_zscore`、`daily_winsorize_zscore_columns`、`daily_winsorize_quantile`、`daily_zscore_population`、`daily_quantile_bins`、`bin_returns_multi`、`bin_turnover`、`daily_row_spearman`、`daily_ols` |
| 可选后端 | `native`、`python`、`auto` |
| 默认后端 | `auto` |
| 业务接入 | 相关性、IC/RankIC、多周期衰减、增量 IC、风格暴露、OLS、分位图、换手、自相关、模型预处理和挖掘候选相关性 |
| 替换原实现 | `_daily_spearman`；`rank_pct_by_day`；`_daily_pearson`；`winsorize_by_day`；`zscore_by_day`；分位分组；多标签分位收益；分位换手；因子自相关；`_daily_regression`；模型 groupby rank/quantile/std；挖掘候选逐日逐 peer pandas 排名 |
| 实际 C++ 替换模块数 | 14 |
| 真实业务 stage 收益 | 相关性下降 35.5%；overview 2.535x；bins 1.621x；turnover 1.854x；decay 3.255x；OLS 链路改善 46.26%；模型 rank 3.282x；挖掘候选相关性 1.360x |

## P2-02 挖掘候选相关性接入

`mining.evaluation_correlation` 把候选目标和正式 peer 对齐成连续日期股票矩阵，复用
`daily_row_spearman`。单次筛选的正式 peer 网格由 `CandidateCorrelationCache` 复用，
不跨运行或请求保存。

- 固定四候选、批量物化后筛选：未缓存 Python 技术阶段 `40.941s`，native `33.055s`，
  改善 `19.66%`；auto 总耗时由未缓存 Python `44.863s` 降至 `36.882s`（`1.216x`）。
- 缓存后相关性阶段 Python `4.411s`、native `3.243s`，加速 `1.360x`。
- native/auto 稳定 payload 与 Python 一致，峰值 RSS 为缓存后 Python 的
  `1.001x/1.000x`。
- 轻量 RankIC 的 C++ 对照中位 `0.755s`，慢于 DuckDB SQL `0.658s`，因此继续保留
  SQL 路径，不登记为 C++ 替换。
- 证据位于 `AiRecord/benchmarks/V121/p2-02-20261028/`。

## 修改注意

- 不得为了通过 POC 改变现有缺失、并列、常数和最少样本语义。
- 不得在业务模块中直接导入 `quant_native`。
- `auto` 只处理扩展不可用，不得吞掉已加载原生核的运行期异常。
- 候选路径必须使用 `target_index`，不能为单个候选重算完整矩阵。
- 逐日核调用前必须保证各组样本连续；不能把带分组编号但未排序的面板直接传入。
- 多标签 RankIC 必须逐档重新取有限配对，不能复用其他标签的排名结果。
- 分位分组只把 NaN 视为缺失，正负无穷仍参与排名；秩宽表写入 `float32` 的既有语义
  由 Python 宽表层保持。
- 因子自相关必须对每个 lag 的共同股票重新排名，不能复用完整截面秩。
- 新增原生核必须同时提供 Python 金标准、异常、并发、子进程压力和资源证据。

## P1-07 OLS 与暴露链路优化

V118 曾按 `1e-10` 口径停止原生 OLS。V119 用户确认 OLS 业务一致性门槛调整为
`1e-5`，因此恢复 Eigen 3.4.0 和逐日 OLS。正常条件数使用列主元 Householder QR，
协方差由 SVD 计算；病态矩阵按日显式回退 Python。真实矩阵核耗时由 `0.749s` 降至
`0.322s`（`2.33x`），系数、R²、残差和业务消费的风格 t 在 `1e-5` 内一致。

保留的暴露链路优化不改变公式：

- 风格取数在 DuckDB 内生成整数股票编码，pandas 再还原为类别列，避免把 390 万行字符串
  直接转换为 object。
- 风格窗口查询使用 4 个 DuckDB 线程；该阶段结束后才进入 native 计算，不与
  `QUANT_NATIVE_THREADS` 叠加。
- 8 个原始风格列和因子一次取成连续矩阵，经 `daily_winsorize_zscore_columns` 标准化，
  共用稳定日序和 offsets。

固定三年 `exposure + incremental_ic`：Python `24.375s`、native `12.935s`、auto
`13.100s`；native 相对 Python 改善 `46.26%`。三后端 payload 在 `1e-5` 内一致，
native/auto 峰值 RSS 为 Python 的 `0.998x/0.992x`。证据位于
`AiRecord/benchmarks/V119/p1-07-20261027/`。

## P2-01 模型预处理

模型预处理通过 `kernels.get_backend()` 使用同一批原生接口。`daily_rank_pct` 同时服务
特征和标签排名；新增 `daily_winsorize_quantile` 与 `daily_zscore_population` 分别匹配
模型现有的 1%/99% 分位裁剪和总体标准差 z-score。

固定 2,138,239 行、50 列、单线程、三次独立进程：

| 模式 | Python | native | 加速 |
|---|---:|---:|---:|
| rank | 26.095s | 7.951s | 3.282x |
| zscore | 6.607s | 3.509s | 1.883x |
| rank+winsorize | 58.726s | 17.095s | 3.435x |
| zscore+winsorize | 38.841s | 12.345s | 3.146x |

四个模式的 50 列 float32 哈希在三后端完全一致，native/auto 均不慢于 Python。证据位于
`AiRecord/benchmarks/V120/p2-01-20261027/`。
