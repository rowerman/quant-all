# C++ 批量计算扩展

`native` 负责构建顶层扩展 `quant_native`，当前只提供成对 Spearman 批量核。它是 P1-02
单核 POC，不接入评估默认路径，业务结果仍由 Python 实现产生。

## 负责什么

- 接收按日期或批次分组的股票×因子 C 连续 `float64` 矩阵。
- 一次计算所有组的成对 Spearman 相关矩阵和有效配对数。
- 按实际配对样本剔除非有限值，使用并列平均名次，常数截面返回 `NaN`。
- 计算期间释放 GIL，按组使用受控线程数。

不负责输入数据读取、因子目录、评估 payload、日志聚合、业务回退选择或并发阶段锁。

## 被谁调用

- `kernels.native.CppKernelBackend` 是唯一 Python 适配入口。
- 当前调用方只有核测试和 `tools/monitor/native_kernel_baseline.py`。
- 评估、模型、挖掘和回测尚未调用该核。

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
pairwise_spearman(values, group_offsets, min_n=5)
```

- `values`：C 连续 `float64`，形状为 `[样本, 因子]`。
- `group_offsets`：C 连续 `int64`，长度等于组数加一，首项 0、末项样本数、单调不减。
- `min_n`：每个因子对参与相关计算的最少有限配对数，不得小于 5。
- 返回：`(correlation[组, 因子, 因子], count[组, 因子, 因子])`。

Python 适配层和 C++ 绑定层都会检查 dtype、维度、连续内存、offsets 和参数。原生错误
转换为 Python 异常；显式请求 `native` 但扩展缺失时返回 `KERNEL-004`，不静默回退。

## 线程与语义

- `QUANT_NATIVE_THREADS` 每次调用读取，默认 1，必须是正整数。
- 线程按组并行；单日单组输入不会在组内自行扩线程。
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

## C++ 替换台账

| 项目 | 当前状态 |
|---|---|
| 扩展 | `quant_native` 已构建 |
| 原生核 | `pairwise_spearman` |
| 可选后端 | `native`，仅显式请求 |
| 默认后端 | `python` |
| 业务接入 | 未接入 |
| 实际 C++ 替换模块数 | 0 |
| 真实业务 stage 收益 | 尚不适用，留待 P1-03 |

只有 P1-03 将 C++ 核接入评估相关性，并通过真实三年 stage 的 `20%` 收益、正确性和
内存门槛后，才把相关性核计入实际 C++ 替换模块。

## 修改注意

- 不得为了通过 POC 改变现有缺失、并列、常数和最少样本语义。
- 不得在业务模块中直接导入 `quant_native`。
- 不得把 `native` 设为默认或新增静默 `auto` 回退。
- 新增原生核必须同时提供 Python 金标准、异常、并发、子进程压力和资源证据。
