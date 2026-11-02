# 算子与 C++ 实施跟踪

- 更新日期：2026-10-28
- 来源报告：[OPERATOR_CPP_ANALYSIS_REPORT.md](TODO_docs/OPERATOR_CPP_ANALYSIS_REPORT.md)
- 当前状态：P0～P3 原登记主任务及后续 `P1-01-F01` 均已结束，当前没有待修改问题。
- 默认计算后端：`auto`，扩展可用时选择 C++，不可用时回退 Python。
- 实际 C++ 替换模块数：14。

## 1. 当前状态

- 评估层的相关性、逐日排名、Pearson、MAD、z-score、分位分组、多标签分位收益、
  分位换手、因子自相关和逐日 OLS 已接入统一核。
- 模型特征排名、标签排名、分位去极值和总体标准差 z-score 已复用统一核。
- 挖掘候选与正式因子的相关性已复用统一核，轻量 RankIC 保留更快的 DuckDB SQL。
- 回测的高耗时来源是重复段级报告和快照，已完成去重；清洗层 C++ 和整体回测账本
  C++ 化均已明确否决。
- Numba 实验后端的逐日排名无穷值语义已与 Python/C++ 对齐，不影响默认路径。

## 2. 已完成修改摘要

### P0：基线、数据流和重复计算

| ID | 已完成内容 | 主要结果 | 证据 |
|---|---|---|---|
| P0-01 | 固定评估、相关性、轻量 IC、因子构建和挖掘基准；保存输入指纹、耗时、内存与 Python 金标准 | 建立后续统一对照基线 | `AiRecord/benchmarks/V105/p0-01-20261027/` |
| P0-02 | 正式相关性保留完整矩阵，候选相关性只计算目标行、目标列和对角线 | 候选相关性从 233.5s 降至 132.9s，下降 43.1% | `AiRecord/benchmarks/V106/p0-02-20261027/` |
| P0-03 | 缓存单日缺失掩码、稳定排序和成对重排名工作区 | 单日矩阵下降 15.7%，三年正式 stage 下降 8.5% | `AiRecord/benchmarks/V107/p0-03-20261027/` |
| P0-04 | 建立稳定表达式结构哈希、候选 DAG、根映射和共享节点诊断 | 支持确定性公共子表达式规划 | `AiRecord/progress/V108.md` |
| P0-05 | 同轮候选合并为共享 DuckDB CTE，拆分到独立候选目录并保留失败隔离与单候选回退 | 4 候选物化由 11.395s 降至 3.517s，下降 69.1% | `AiRecord/benchmarks/V109/p0-05-20261027/` |
| P0-06 | 因子族改为一次全区间计算并按年原子发布，保留 `yearly` 回退 | 两年构建下降 41.1%，全历史累计扫描显著减少 | `AiRecord/benchmarks/V110/p0-06-20261027/` |

### P1：统一批量核和 C++ 接入

| ID | 已完成内容 | 主要结果 | 证据 |
|---|---|---|---|
| P1-01 | 建立统一批量核协议、Python 回退、后端选择和 Numba 原型 | 逐日排名 Numba 加速 2.023x，P1-02 门禁通过 | `AiRecord/benchmarks/V112/p1-01-20261027-recheck/` |
| P1-01-F01 | 修正 Numba 逐日排名将正负无穷误判为缺失的问题 | Python、Numba、native 对 `NaN`、正负无穷和并列值逐值一致 | `AiRecord/progress/V127.md` |
| P1-02 | 建立 C++20、pybind11、CMake 构建链和 Spearman 单核 POC | C++ Spearman 加速 2.358x，RSS 比例 1.001x | `AiRecord/benchmarks/V113/p1-02-20261027/` |
| P1-03 | Spearman 接入完整矩阵和候选目标行，默认后端改为 `auto` | 正式相关性下降 35.5%，auto 下降 37.6% | `AiRecord/benchmarks/V114/p1-03-20261027/` |
| P1-04 | 接入逐日排名、Pearson 和 IC/RankIC/decay 共享路径 | 排名加速 4.680x，Pearson 加速 17.670x | `AiRecord/benchmarks/V115/p1-04-20261027/` |
| P1-05 | 接入逐日 MAD、z-score 和多列融合标准化 | MAD 2.313x，z-score 10.840x，overview 2.535x | `AiRecord/benchmarks/V116/p1-05-20261027/` |
| P1-06 | 接入分位分组、多标签分位收益、分位换手和因子自相关 | `bins + turnover` 合计改善 43.82% | `AiRecord/benchmarks/V117/p1-06-20261027/` |
| P1-07 | 接入逐日 OLS、active mask、跳过原因和病态矩阵按日回退 | OLS 加速 2.33x，暴露与增量 IC 链路改善 46.26% | `AiRecord/benchmarks/V119/p1-07-20261027/` |

### P2：模型和挖掘复用

| ID | 已完成内容 | 主要结果 | 证据 |
|---|---|---|---|
| P2-01 | 模型特征和标签排名、分位裁剪、总体 z-score 复用统一核 | 三种接入模式均不慢于 Python，rank 加速 3.282x | `AiRecord/benchmarks/V120/p2-01-20261027/` |
| P2-02 | 挖掘候选相关性复用逐行 Spearman，单轮复用正式 peer 网格 | 相关性加速 1.360x，技术阶段改善 19.66% | `AiRecord/benchmarks/V121/p2-02-20261028/` |
| P2-03 | 评估 DuckDB SQL、表函数和外部原生 DSL 执行器 | 原生替换端到端收益上限 5.92%，结论为不实施 | `AiRecord/benchmarks/V122/p2-03-20261028/` |

### P3：回测和清洗专项

| ID | 已完成内容 | 主要结果 | 证据 |
|---|---|---|---|
| P3-01 | 分段只生成拼接明细，最终报表、归因和输入快照在拼接后生成一次 | 全 A 三年改善 74.89%，指数十年改善 57.67% | `AiRecord/benchmarks/V123/p3-01-20261028/` |
| P3-02 | 建立临时目录隔离的清洗年度剖析和纯数组规则决策门 | 最大纯数组规则仅占 0.17%，结论为不实施清洗层 C++ | `AiRecord/benchmarks/V124/p3-02-20261028/` |
| P3-03 | 复核回测账本、分配、订单和公司行动的阶段占比与重写成本 | 不支持整体 C++ 账本，继续使用 `LedgerPortfolio` | `AiRecord/benchmarks/V125/p3-03-20261028/` |

### 当前 C++ 替换范围

评估相关性 Spearman、逐日排名、逐日 Pearson、逐日 MAD、逐日 z-score、分位分组、
多标签分位收益、分位换手、因子自相关、逐日 OLS、模型截面排名、模型分位去极值、
模型总体标准差 z-score、挖掘候选正式因子相关性，共 14 个模块。

## 3. 维护约定

- 本文件只保留当前状态、已完成功能摘要和维护约定。
- 已完成任务的详细过程继续由 `AiRecord/version.md`、`AiRecord/progress/`、
  `AiRecord/changes/`、`AiRecord/diffs/` 和 `AiRecord/benchmarks/` 追踪。
- 新任务完成后，只把可复用的能力、收益、结论和证据入口更新到本文件，不再复制
  全量执行记录。
- 全局约束继续保持：Python 回退可用、批量数组计算、默认单线程、禁止未来数据、
  保持外部契约、C++ 错误转换为 Python 异常，并且不以微基准替代真实业务阶段证据。

## 4. 明确不实施事项

- 不整体替换 `core/factors/ops.py`。
- 不把 DuckDB 查询改写为独立 C++ 服务。
- 不使用逐股票、逐日或逐单元格 C++ UDF。
- 不自研 LightGBM、sklearn 或 GBDT 训练内核。
- 不把 Web、SSE、审批和状态管理迁移到 C++。
- 不建设原生挖掘表达式 VM。
- 不实施清洗层 C++。
- 不整体重写 Python 回测账本。
