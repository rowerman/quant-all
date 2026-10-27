# C++ 批量计算扩展

`native` 负责构建顶层扩展 `quant_native`，当前提供完整矩阵和目标行两种成对 Spearman
核，以及逐日排名和逐日 Pearson 核。三类核都已接入评估业务路径；有扩展时默认 `auto`
选择 native，无扩展时回退 Python。

## 负责什么

- 接收按日期或批次分组的股票×因子 C 连续 `float64` 矩阵。
- 一次计算所有组的完整 Spearman 相关矩阵，或只计算一个目标因子的行、列和对角线。
- 按实际配对样本剔除非有限值，使用并列平均名次，常数截面返回 `NaN`。
- 按组计算平均名次分位和 Pearson 相关，保持原有效计数与最小样本门槛。
- 计算期间释放 GIL，按组使用受控线程数。

不负责输入数据读取、因子目录、评估 payload、日志聚合、业务回退选择或并发阶段锁。

## 被谁调用

- `kernels.native.CppKernelBackend` 是唯一 Python 适配入口。
- 正式完整矩阵和候选目标行经 `calculate.correlation` 调用 `kernels` 适配层。
- IC、RankIC、多周期衰减和增量 IC 经 `calculate.daily` 调用同一适配层。
- 核测试和 P1-02/P1-03 基准工具直接验证适配层或扩展。
- 模型、MAD/z-score、OLS、挖掘表达式和回测尚未调用这些核。

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
- 两个日度接口的 `group_offsets` 都与 Spearman 相同，返回一维结果；Pearson 额外返回
  每组有效配对数。
- 业务评估的最小截面继续固定为 5；底层 Pearson 接口接受不小于 2 的门槛。

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

## C++ 替换台账

| 项目 | 当前状态 |
|---|---|
| 扩展 | `quant_native` 已构建 |
| 原生核 | `pairwise_spearman`、`daily_rank_pct`、`daily_pearson` |
| 可选后端 | `native`、`python`、`auto` |
| 默认后端 | `auto` |
| 业务接入 | 相关性、IC/RankIC、多周期衰减、增量 IC |
| 替换原实现 | `_daily_spearman`；`rank_pct_by_day`；`_daily_pearson` 的业务调用路径 |
| 实际 C++ 替换模块数 | 3 |
| 真实业务 stage 收益 | 相关性 native 下降 35.5%；overview 加速 2.082x；decay 加速 2.820x |

## 修改注意

- 不得为了通过 POC 改变现有缺失、并列、常数和最少样本语义。
- 不得在业务模块中直接导入 `quant_native`。
- `auto` 只处理扩展不可用，不得吞掉已加载原生核的运行期异常。
- 候选路径必须使用 `target_index`，不能为单个候选重算完整矩阵。
- 逐日核调用前必须保证各组样本连续；不能把带分组编号但未排序的面板直接传入。
- 多标签 RankIC 必须逐档重新取有限配对，不能复用其他标签的排名结果。
- 新增原生核必须同时提供 Python 金标准、异常、并发、子进程压力和资源证据。
