# 批量计算核

`core/kernels` 以顶层包名 `kernels` 提供与具体加速后端无关的批量数组契约。当前包含
Python 回退、Numba 研究原型和 C++ Spearman POC；评估、模型、挖掘和回测都没有把加速
后端接入默认业务路径。

## 负责什么

- 校验 C 连续 `float64` 数组和 C 连续 `int64` 分组边界。
- 为逐日排名、MAD 去极值、Pearson、成对 Spearman 和逐日 OLS 固定输入输出口径。
- 提供直接复用现有 `calculate` 实现的 Python 回退。
- 提供逐日排名和 Spearman 的 Numba 研究原型，用于决定是否值得进入 C++。
- 提供显式 `native` 后端的 C++ Spearman POC。
- 通过 `KERNEL-001`～`KERNEL-004` 区分数组、offset、参数和后端错误。

## 被谁调用

当前调用方只有：

- `tests/test_native_kernels.py`
- `tests/test_cpp_native.py`
- `tools/monitor/kernel_baseline.py`
- `tools/monitor/native_kernel_baseline.py`

业务模块不得直接导入 `quant_native` 或 Numba 原型。后续接入统一从此包选择后端。

## 调用了谁

- Python 回退调用 `calculate.daily`、`calculate.normalize`、
  `calculate.correlation` 和 `calculate.exposure` 的现有实现。
- Numba 原型的未验证核显式委托 Python 回退。
- `CppKernelBackend` 只把 `pairwise_spearman` 交给 `quant_native`，其他核显式委托
  Python 回退。
- 基准工具读取正式因子 Parquet 和评估面板，只读，不写业务数据。

## 数据契约

- 一维核的 `values` 与多维核的第一维必须按分组稳定排序。
- `group_offsets` 长度等于组数加一，首项为 0、末项为样本数、单调不减。
- 空组可以保留，但整体空输入明确失败。
- 允许通过 `float64` 中的 NaN/inf 表达非有限值，各核按现有评估语义过滤。
- `pairwise_spearman` 返回 `(相关系数[组, 因子, 因子], 有效计数[组, 因子, 因子])`。
- Spearman 的最小样本参数不得小于当前评估口径 5。
- `daily_ols` 返回日期索引、系数、t 值、R²、样本数、残差行号和残差值。

## 后端选择

- `QUANT_KERNEL_BACKEND` 支持 `python`、`numba`、`native`，缺省为 `python`。
- P1-01 不开放 `auto`；非法值返回 `KERNEL-004`。
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

## C++ 替换台账

只有真实替换、接入并留下正确性和性能证据的模块才计入替换数。

| 项目 | 当前状态 |
|---|---|
| 实际 C++ 替换模块数 | 0 |
| C++ 扩展 | `quant_native` 已构建 |
| 默认计算后端 | Python 回退 |
| P1-01 门禁 | 逐日排名门禁已通过 |
| Spearman POC | C++ 加速 `2.358x`，业务未接入 |

后续登记每个 C++ 模块时必须写明：C++ 模块名、替换的原 Python 函数、默认/可选后端、
Python 回退路径、真实 stage 耗时、峰值 RSS、正确性证据和对应 Tracker 任务。

## 修改注意

- `calculate.native` 是评估并发锁，不是计算后端；不要合并两套职责。
- 不复制排名、MAD、Pearson 或 OLS 公式，回退必须继续调用现有实现。
- 不把未验证的 Numba 或 C++ 后端设为默认，也不静默回退。
- 矩阵第一维始终是组；不要按股票或单元格逐次调用。
- 新增核必须同时覆盖空组、NaN、正负无穷、并列、常数和最小样本边界。
