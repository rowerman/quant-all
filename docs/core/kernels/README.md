# 批量计算核

`core/kernels` 以顶层包名 `kernels` 提供与具体加速后端无关的批量数组契约。当前包含
Python 回退、Numba 研究原型和 C++ Spearman、逐日排名、逐日 Pearson 后端；评估相关性、
IC/RankIC、多周期衰减和增量 IC 已默认走 `auto`，模型、OLS、MAD/z-score、挖掘和回测
尚未接入加速后端。

## 负责什么

- 校验 C 连续 `float64` 数组和 C 连续 `int64` 分组边界。
- 为逐日排名、MAD 去极值、Pearson、成对 Spearman 和逐日 OLS 固定输入输出口径。
- 提供直接复用现有 `calculate` 实现的 Python 回退。
- 提供逐日排名和 Spearman 的 Numba 研究原型，用于决定是否值得进入 C++。
- 提供 C++ Spearman 完整矩阵和目标行后端。
- 提供 C++ 逐日排名和逐日 Pearson 后端。
- 通过 `KERNEL-001`～`KERNEL-004` 区分数组、offset、参数和后端错误。

## 被谁调用

当前主要调用方：

- `calculate.correlation`：正式完整矩阵和候选目标行。
- `calculate.daily`：IC、RankIC、多周期衰减和增量 IC。
- `tests/test_native_kernels.py`
- `tests/test_cpp_native.py`
- `tools/monitor/kernel_baseline.py`
- `tools/monitor/native_kernel_baseline.py`
- `tools/monitor/correlation_native_baseline.py`

业务模块不得直接导入 `quant_native` 或 Numba 原型。后续接入统一从此包选择后端。

## 调用了谁

- Python 回退调用 `calculate.daily`、`calculate.normalize`、
  `calculate.correlation._daily_spearman` 和 `calculate.exposure` 的现有实现。
- Numba 原型的未验证核显式委托 Python 回退。
- `CppKernelBackend` 把 `pairwise_spearman`、`daily_rank_pct` 和 `daily_pearson`
  交给 `quant_native`；MAD 和 OLS 继续显式委托 Python 回退。
- 基准工具读取正式因子 Parquet 和评估面板，只读，不写业务数据。

## 数据契约

- 一维核的 `values` 与多维核的第一维必须按分组稳定排序。
- `group_offsets` 长度等于组数加一，首项为 0、末项为样本数、单调不减。
- 空组可以保留，但整体空输入明确失败。
- 允许通过 `float64` 中的 NaN/inf 表达非有限值，各核按现有评估语义过滤。
- `pairwise_spearman` 返回 `(相关系数[组, 因子, 因子], 有效计数[组, 因子, 因子])`。
- `daily_rank_pct` 返回与输入等长的一维分位排名；NaN 保持原位。
- `daily_pearson` 返回 `(相关系数[组], 有效配对数[组])`，只使用两侧有限的配对。
- 可选 `target_index` 只返回目标行、目标列和对角线；其他位置相关值为 `NaN`、计数为 0。
- Spearman 的最小样本参数不得小于当前评估口径 5。
- `daily_ols` 返回日期索引、系数、t 值、R²、样本数、残差行号和残差值。

## 后端选择

- `QUANT_KERNEL_BACKEND` 生产支持 `auto`、`python`、`native`，缺省为 `auto`。
- `auto` 仅在扩展不可导入时回退 Python，原因记为 `cpp_extension_unavailable`。
- 已加载原生核的运行期错误转换为 `KERNEL-004`，不静默切换后端。
- `numba` 保留为研究和基准入口，不作为生产默认三态之一。
- Numba 未验证的方法委托 Python 回退，并在调用方保留显式后端选择信息。
- 显式请求 `native` 但扩展缺失时返回 `KERNEL-004`，不静默切换 Python。
- `QUANT_NATIVE_THREADS` 默认 1。Numba 原型和 C++ 扩展沿用同一预算。

## P1-01 门禁

- V112 起，P1-01 以 `daily_rank_pct` 的正确性和热运行中位加速 `1.5x` 作为放行门禁。
- V112 复测逐日排名由 Python 0.016966 秒降至 Numba 0.008388 秒，加速 2.023x。
- Spearman Numba 继续做正确性和耗时诊断，但 `0.179x` 未达标，不作为 P1-01 放行依据；
  其性能和真实 stage 收益由 P1-02 的 C++ POC 重新验证。
- 基准 schema 为 `kernel_baseline_v2`，证据位于
  `AiRecord/benchmarks/V112/p1-01-20261027-recheck/`。

## P1-02 C++ Spearman POC

- 扩展 `quant_native` 使用 C++20、pybind11、CMake 和 scikit-build-core 构建。
- 在固定 5,219×47 截面上，Python 热运行中位 `0.172836s`，C++ `0.073312s`，
  加速 `2.358x`；峰值 RSS 分别为 282.820MB 和 283.129MB。
- C++ 相关系数与 Python 在 `1e-10` 内一致，有效计数和 `NaN` 位置完全一致。
- 覆盖缺失、无穷、空组、并列、常数、最少样本、GIL、并发和子进程压力。
- 证据位于 `AiRecord/benchmarks/V113/p1-02-20261027/`。

## P1-03 相关性业务接入

- 完整矩阵与候选目标行均通过统一协议选择 `python`、`native` 或 `auto`。
- 每请求只选择一次后端，避免逐月或逐日重新读取配置。
- 固定三年正式 stage 由 `217.770s` 降至 native `140.460s`，下降 `35.5%`；默认
  auto 为 `135.788s`，下降 `37.6%`。
- 候选目标行由 `137.599s` 降至 native `118.680s`；auto `120.995s`，没有退化。
- 固定截面核加速：完整矩阵 `2.796x`，目标行 `1.963x`；RSS 比例不超过 `1.0017x`。
- 证据位于 `AiRecord/benchmarks/V114/p1-03-20261027/`。

## P1-04 逐日相关业务接入

- `calculate.daily` 每次调用只选择一次后端，并把按交易日稳定排序后的数组和 offsets
  交给统一核；不能把仅有分组编号、但组内样本不连续的数组直接传入。
- `daily_ic` 复用同一 offsets 计算 Pearson IC 和 RankIC；RankIC 始终对原始有限配对
  排名。多标签衰减逐档独立取样本和排名，不跨标签复用错误掩码。
- 3,724,919 行、722 组固定面板上，排名由 1.114445 秒降至 0.238150 秒（4.680x），
  Pearson 由 0.210444 秒降至 0.011909 秒（17.670x）。
- `overview`、`decay`、`incremental_ic` 分别加速 2.082x、2.820x、1.333x；
  Python/native/auto payload 在 `1e-10` 内一致，native/auto 峰值 RSS 比例最高 1.037x。
- 证据位于 `AiRecord/benchmarks/V115/p1-04-20261027/`。

## C++ 替换台账

只有真实替换、接入并留下正确性和性能证据的模块才计入替换数。

| 项目 | 当前状态 |
|---|---|
| 实际 C++ 替换模块数 | 3 |
| C++ 扩展 | `quant_native` 已构建 |
| 默认计算后端 | `auto`，扩展可用时选择 native |
| P1-01 门禁 | 逐日排名门禁已通过 |
| Spearman 模块 | 评估相关性完整矩阵与候选目标行已接入 |
| 逐日排名模块 | IC、RankIC、多周期衰减、增量 IC 已接入 |
| 逐日 Pearson 模块 | Pearson IC 与通用日度相关已接入 |

后续登记每个 C++ 模块时必须写明：C++ 模块名、替换的原 Python 函数、默认/可选后端、
Python 回退路径、真实 stage 耗时、峰值 RSS、正确性证据和对应 Tracker 任务。

## 修改注意

- `calculate.native` 是评估并发锁，不是计算后端；不要合并两套职责。
- 不复制排名、MAD、Pearson 或 OLS 公式，回退必须继续调用现有实现。
- 不让 `auto` 吞掉已加载原生核的计算错误。
- 矩阵第一维始终是组；不要按股票或单元格逐次调用。
- 新增核必须同时覆盖空组、NaN、正负无穷、并列、常数和最小样本边界。
