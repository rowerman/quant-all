# 算子与 C++ 后续工作进度追踪

- 来源报告：[OPERATOR_CPP_ANALYSIS_REPORT.md](TODO_docs/OPERATOR_CPP_ANALYSIS_REPORT.md)
- 创建日期：2026-10-27
- 代码基线：`main` 分支提交 `fe10c55`
- 当前状态：全部任务待开始，本文档不表示任何后续任务已经实施。

## 1. 使用规则

本文件用于后续 session 逐项执行。默认规则如下：

1. 一次 session 只执行一个 `P?-??` 主任务；除非任务明确拆成同一 session 内可验证的子项。
2. 开始任务前，按项目要求创建 `AiRecord/plans/YYYYMMDD-X.md`。
3. 开始执行时把任务状态改为 `进行中`，记录开始日期和 session 编号。
4. 完成后记录实际修改、测试、性能证据、结论和 commit；未通过验收时改为 `已停止`，
   不保留无效 C++ 路径。
5. 每个 session 结束前更新 `AiRecord/changes/`、`AiRecord/diffs/` 和对应 progress。
6. 任务存在阻塞时保持 `阻塞`，写明缺失条件，不跳到依赖未完成的任务。
7. 不用全量测试代替针对性分析；修改了哪些模块，就至少运行这些模块及上下游联动测试。

状态统一使用：

| 状态 | 含义 |
|---|---|
| 待开始 | 前置条件已满足，可以执行 |
| 等待依赖 | 前置任务尚未完成 |
| 进行中 | 当前 session 正在执行 |
| 待验收 | 实现和定向测试完成，等待证据复核 |
| 已完成 | 验收标准全部满足并记录证据 |
| 已停止 | 基准或收益门槛未达到，取消后续扩展 |
| 阻塞 | 缺环境、数据或用户选择，当前无法继续 |

## 2. 全局约束

以下约束适用于全部任务：

- 普通因子窗口和聚合继续由 DuckDB 执行，不整体改写 `core/factors/ops.py`。
- C++ 只处理已证明有收益的批量数组或矩阵计算，不做逐股票、逐日、逐单元格调用。
- Python 回退必须始终可用；任何自动回退都要记录原因。
- 因子值、评估 payload、stage 顺序和前端契约不得无迁移变化。
- 新 C++ 路径必须通过 Python 金标准对照，排名类输出要求完全一致。
- 不得读取未来数据；所有截断重算和前视检查继续通过。
- 默认线程数为 1，统一由 `QUANT_NATIVE_THREADS` 控制，避免超额订阅。
- C++ 错误必须转换为 Python 异常，不允许 `abort`、静默吞错或返回未初始化数组。
- 不以微基准替代端到端基准；目标业务阶段必须有真实区间数据。
- 不以“引入 C++”为完成标准，只以正确性、性能、内存和维护成本门槛为完成标准。
- 不修改生产数据库、正式因子、模型、候选、日志或回测数据来迁就实现。

## 3. 总览

| ID | 优先级 | 任务 | 依赖 | 建议 session |
|---|---|---|---|---|
| P0-01 | P0 | 固定基准、金标准和性能证据 | 无 | 1 |
| P0-02 | P0 | 审计相关性消费方并裁掉非必要矩阵计算 | P0-01 | 2 |
| P0-03 | P0 | 优化日度排名和缺失掩码缓存 | P0-02 | 3 |
| P0-04 | P0 | 建立挖掘候选表达式 DAG 和公共子表达式 | P0-01 | 4 |
| P0-05 | P0 | 实现挖掘候选批量物化和复用 | P0-04 | 5 |
| P0-06 | P0 | 评估并改造因子族全历史重复扫描 | P0-01 | 6 |
| P1-01 | P1 | 建立批量核适配层并完成 Numba 算法验证 | P0-01、P0-02、P0-03 | 7 |
| P1-02 | P1 | 建立 C++ 构建骨架和 Spearman 单核 POC | P1-01 | 8 |
| P1-03 | P1 | 将 Spearman 批量核接入评估相关性 | P1-02 | 9 |
| P1-04 | P1 | 接入逐日排名和 Pearson 核 | P1-03 | 10 |
| P1-05 | P1 | 接入逐日 MAD 去极值和 z-score 核 | P1-04 | 11 |
| P1-06 | P1 | 接入分位收益、换手和自相关核 | P1-04 | 12 |
| P1-07 | P1 | 接入逐日 OLS 暴露核 | P1-04 | 13 |
| P2-01 | P2 | 将排名和标准化核共享给模型预处理 | P1-04、P1-05 | 14 |
| P2-02 | P2 | 将共享核用于挖掘门禁、轻量 IC 和相关性 | P1-03、P1-04、P0-05 | 15 |
| P2-03 | P2 | 评估受控 DSL 原生执行器是否需要建设 | P0-05、P2-02 | 16 |
| P3-01 | P3 | 回测 I/O、分块和内存专项优化 | P0-01 | 17 |
| P3-02 | P3 | 清洗层条件化剖析与共享核评估 | P0-01 | 18 |
| P3-03 | P3 | 回测账本 C++ 化专项决策 | P3-01 | 仅在触发后 |

推荐严格按 ID 顺序执行。P0-04 和 P0-06 在 P0-01 完成后可独立开展，但为简化 session
管理，默认仍按表中顺序。

## 4. P0：先消除重复计算和确认基准

P0 不引入 C++。目标是把后续所有优化放到同一可复现基线上，并优先消除算法和数据流上的
重复工作。

### P0-01 固定基准、金标准和性能证据

- 状态：已完成（V105，2026-10-27）
- 依赖：无
- 建议 session 数：1

目标：在任何代码优化前，固定评估、挖掘和因子构建的代表性输入、当前输出、耗时和内存，
作为后续 C++/算法改动的唯一对照。

涉及文件：

- `tools/monitor/benchmark.py`
- `tools/monitor/backtest_profile.py`
- `tools/monitor/operator_baseline.py`
- `tools/monitor/baseline_contract.py`
- `tools/monitor/baseline_cases.py`
- `research/evaluation/calculate/tests/`
- `tests/test_mining_expression.py`
- `tests/test_factor_build.py`
- `tests/test_operator_baseline.py`
- `AiRecord/benchmarks/`

详细工作：

- [x] 记录当前 commit、Python、DuckDB、NumPy、pandas、CPU、内存和数据版本。
- [x] 固定默认评估案例：`MOM_20`、全 A、三年、5 日 close、10 分位、winsorize。
- [x] 固定轻量 IC 案例和相关性单 stage 案例。
- [x] 固定一个小型因子构建案例，至少覆盖一个族的一个完整年份。
- [x] 固定一个候选表达式技术门禁和快速筛选案例。
- [x] 记录输入文件 SHA-256、行数、股票数、交易日数和因子列数。
- [x] 输出逐 stage 耗时、总耗时、峰值 RSS 和中间数据规模。
- [x] 保存 Python 当前输出为金标准，包括 IC、RankIC、相关矩阵、分位收益和暴露系数。
- [x] 增加边界用例：并列值、NaN、正负无穷、空日期、单样本日期、常数截面。
- [x] 规定完整评估只做一次基准，核函数微基准至少 3 次取中位数。
- [x] 将证据路径和复现命令写入本任务完成记录。

测试：

- [x] 金标准结果重复运行稳定。
- [x] JSON、NumPy 和 Parquet 输出均可比较。
- [x] 缺失值和并列值的比较规则明确。

验收：

- 后续任务无需重新选择评估区间和基础参数。
- 能区分取数、面板构建、各指标 stage、相关性、JSON 转换的耗时。
- 金标准可保存且可被后续测试直接读取。

完成记录：

- 证据：`AiRecord/benchmarks/V105/p0-01-20261027/`。
- 复现：`.venv/bin/python -m tools.monitor.operator_baseline --suite all`。
- 完整评估：293.4 秒，峰值 RSS 3152.9MB，相关性 236.0 秒。
- 相关性单 stage：234.2 秒，峰值 RSS 3410.3MB。
- 全历史轻量 IC：12.8 秒；momentum 2024 构建：11.8 秒；挖掘门禁+快速筛选：13.6 秒。
- 完整评估和单 stage 相关性在 `1e-10` 容差内一致；边界案例重复输出完全一致。
- 输入树 SHA-256：`543bb728c7c279fc3c460a67d9c1e0a538a625d743b88debb2401d73da323865`。

停止/回退：

- 如果现有基准工具无法输出所需阶段，先补基准工具，不进入任何性能优化。

### P0-02 审计相关性消费方并裁掉非必要矩阵计算

- 状态：已完成（V106，2026-10-27）
- 依赖：P0-01
- 建议 session 数：1

目标：确认页面和 API 是否真的使用完整因子相关矩阵；如果不是，只计算目标因子所在行，
避免为 Top-N 构造全部成对结果。

涉及文件：

- `research/evaluation/calculate/correlation.py`
- `research/evaluation/calculate/metrics/correlation.py`
- `apps/monitor/backend/service.py`
- `apps/monitor/frontend/app.js`
- `research/evaluation/calculate/tests/test_eval_correlation.py`
- `apps/monitor/backend/tests/`

详细工作：

- [x] 搜索 `correlation.matrix`、`result["matrix"]` 和前端图表对矩阵的读取。
- [x] 明确正式因子和候选因子两条路径分别消费哪些字段。
- [x] 确认 `matrix`、`days`、`names` 是否是外部 API 契约。
- [x] 若完整矩阵无消费方，新增目标行结果路径，同时保持现有公共字段兼容。
- [x] 若完整矩阵确有消费方，保留矩阵，只增加目标行预计算和避免重复索引。
- [x] 让 `compute_correlation` 和 `compute_target_correlation` 共用目标行计算逻辑。
- [x] 避免同一目标因子对同一 peer 重复计算和重复构造 DataFrame。
- [x] 记录裁剪前后的因子数、行数、目标行耗时和总阶段耗时。

测试：

- [x] Top-N 因子、相关系数和有效天数与 P0-01 金标准一致。
- [x] 候选因子路径只与正式因子对照，不与其他候选互算；自身仅保留对角线。
- [x] API JSON 字段保持兼容。
- [x] 空 peer、样本不足和常量因子行为不变。

验收：

- 不改变现有 stage payload 的外部字段语义。
- 相关性单 stage 有可量化下降；若算法未变，至少证明完整矩阵浪费被消除。
- 前端和 HTTP 测试通过。

停止/回退：

- 如果完整矩阵是明确契约且无法低成本兼容，停止裁剪，转入 P0-03 仅优化计算内部。

完成记录：

- 证据：`AiRecord/benchmarks/V106/p0-02-20261027/`。
- 正式热力图消费完整 `matrix` 和 `days`，正式路径保持完整矩阵；候选路径只填充目标行、
  目标列和对角线。
- 47 个相关性列的候选矩阵有限单元格从 2,209 个降至 93 个。
- 三年候选相关性阶段从 233,534.6ms 降至 132,933.0ms，下降 43.1%；峰值 RSS 从
  2788.2MB 降至 2682.3MB。
- 正式相关性阶段复测 228.6 秒，与 P0-01 金标准完全一致。
- 本轮为 Python 算法优化，未采用 C++。

### P0-03 优化日度排名和缺失掩码缓存

- 状态：已完成（V107，2026-10-27）
- 依赖：P0-02
- 建议 session 数：1

目标：减少 `daily_spearman_matrix` 中重复排序、重复缺失掩码和重复名次计算。

涉及文件：

- `research/evaluation/calculate/correlation.py`
- `research/evaluation/calculate/daily.py`
- `research/evaluation/calculate/tests/test_eval_correlation.py`
- `research/evaluation/calculate/tests/test_eval_regressions.py`

详细工作：

- [x] 记录当前每日循环中每个因子的排序次数、掩码分组数和排名缓存命中情况。
- [x] 把缺失掩码编码为稳定键，避免重复构造 `packbits`。
- [x] 复用单日完整排序索引和可安全复用的完整排名，成对交集仍按实际配对样本重新排名。
- [x] 对相同缺失掩码的因子组复用分组和样本位置。
- [x] 缓存生命周期限制在单日工作区，不做跨日或跨请求全局缓存。
- [x] 保证数据顺序、日期顺序和并列平均名次与 Python 金标准一致。
- [x] 记录矩阵模式与目标行模式的内存和性能表现。

测试：

- [x] 与 SciPy `spearmanr` 的逐对结果一致。
- [x] 不同缺失掩码、常数序列、单样本和少于 5 个样本的日期一致。
- [x] 对称性、有效计数和 Top-N 排序一致。
- [x] 缓存没有跨日或跨请求污染，重复调用结果稳定。

验收：

- 相关性结果在 P0-01 容差内一致。
- 目标相关性或默认完整评估至少有一个可重复的正收益。
- 峰值 RSS 不超过基线 `1.2x`。

完成记录：

- 证据：`AiRecord/benchmarks/V107/p0-03-20261027/`。
- 单日 5,219×47 矩阵 5 次中位耗时由 0.1976 秒降至 0.1666 秒，下降 15.7%；
  修改前后最大差异为 0，`NaN` 位置一致。
- 三年正式相关性 stage 由 228.6 秒降至 209.1 秒，下降 8.5%；峰值 RSS 由
  2681.7MB 降至 2680.0MB。
- 三年候选目标行 stage 由 132.9 秒降至 127.8 秒；峰值 RSS 为 V106 的 1.054 倍，
  未超过 1.2 倍门槛。
- 正式三年结果与 V105、V106 金标准一致。
- 本轮为 Python 算法优化，未采用 C++，不计入 C++ 模块替换收益。

停止/回退：

- 若缓存维护成本高于直接计算，保留结果和证据，回退到基线实现。

### P0-04 建立挖掘候选表达式 DAG 和公共子表达式

- 状态：等待依赖 P0-01
- 依赖：P0-01
- 建议 session 数：1

目标：在 SQL 生成前建立稳定表达式 DAG，识别同轮候选之间重复字段、窗口、标量和子树，
为批量计算提供依据。本任务不改变候选评估结果。

涉及文件：

- `research/mining/expression_ast.py`
- `research/mining/expression_parser.py`
- `research/mining/expression_compiler.py`
- `research/mining/expression_validation.py`
- `research/mining/contract_candidates.py`
- `tests/test_mining_expression.py`
- `tests/test_mining_contracts.py`

详细工作：

- [ ] 为 AST 节点定义稳定的结构哈希，不能依赖 Python `id()` 或对象插入顺序。
- [ ] 将规范化表达式转换为 DAG，保留输出节点和每个候选的根节点映射。
- [ ] 识别重复叶子字段、常量、标量表达式和相同窗口子树。
- [ ] 保留每个候选独立的 canonical expression、semantic hash 和 artifacts 身份。
- [ ] 增加“候选组计划”数据结构，记录共享节点、候选节点和预计可复用次数。
- [ ] 输出只读诊断：候选数、AST 节点数、去重节点数、共享节点比例。
- [ ] 禁止把不同冻结目录或不同数据上下文中的节点混入同一计划。

测试：

- [ ] 交换律规范化前后哈希符合现有规则。
- [ ] 相同表达式跨候选去重，不同窗口和不同缺失参数不复用。
- [ ] 候选名称、内容哈希、semantic hash 和验证结果不变。
- [ ] 非受控表达式仍被拒绝。

验收：

- 同一候选组可重复生成完全相同的 DAG 计划和诊断。
- 不改变候选物化结果和评估报告。
- 能产出共享节点比例和潜在扫描减少量。

停止/回退：

- 如果现有 AST 无法稳定表达共享节点，先补齐 AST 契约，不进行 SQL 批处理。

### P0-05 实现挖掘候选批量物化和复用

- 状态：等待依赖 P0-04
- 依赖：P0-04
- 建议 session 数：1～2

目标：使用 P0-04 的 DAG 计划，把同轮多个候选合并为共享中间层的批量查询，并复用公共
子树结果；保持每个候选独立目录、独立哈希和失败隔离。

涉及文件：

- `research/mining/candidate_materialization.py`
- `research/mining/expression_compiler.py`
- `research/mining/orchestrator_screening.py`
- `research/mining/candidate_evaluator.py`
- `tests/test_mining_orchestrator.py`
- `tests/test_mining_expression.py`

详细工作：

- [ ] 明确批量边界：同一 run、同一 FrozenContext、同一日期区间、同一字段目录。
- [ ] 为共享中间结果定义临时关系或临时 Parquet，不写入正式因子目录。
- [ ] 批量查询只物化最终候选列和必要的共享中间层。
- [ ] 每个候选继续写入独立 staging 和原子目录。
- [ ] 保留单候选物化兼容入口，供调试和故障恢复使用。
- [ ] 单候选失败只影响该候选；共享层失败则整批返回明确错误。
- [ ] 批量完成后重新校验每个候选的 artifact hash 和 materialization manifest。
- [ ] 记录批量前后扫描次数、SQL 数、临时文件大小、总耗时和峰值 RSS。

测试：

- [ ] 批量物化与逐候选物化的行集、列、非空数和数值一致。
- [ ] 候选内容哈希、artifact hash 和可复用身份不被批量路径改变。
- [ ] 相同批量的重复执行保持幂等。
- [ ] 一个候选失败不会发布半成品，也不会污染正式因子目录。
- [ ] 公共子树和不同数据窗口不能错误复用。

验收：

- 多候选轮次的端到端耗时或扫描次数有可量化下降。
- 任意候选仍可单独重算并得到一致结果。
- 临时目录可清理，崩溃恢复不会复用不完整共享层。

停止/回退：

- 如果共享层物化成本高于重复扫描，记录适用候选数和阈值后关闭批量路径，保留 DAG 诊断。

### P0-06 评估并改造因子族全历史重复扫描

- 状态：等待依赖 P0-01
- 依赖：P0-01
- 建议 session 数：1～2

目标：评估“每个族一次扫全历史、按年切分输出”替代“族 × 年重复扫到目标年末”的收益、
内存和原子性，避免普通 C++ 重写。

涉及文件：

- `core/data/scripts/build_factors/sql.py`
- `core/data/scripts/build_factors/build.py`
- `core/data/scripts/build_factors/validate.py`
- `tests/test_factor_build.py`
- `tests/test_factor_ops.py`

详细工作：

- [ ] 记录当前单族全年、单族单年、全族全年的扫描量、耗时和峰值内存。
- [ ] 证明当前 `base_cte` 的完整历史是截面算子和停牌窗口正确性所必需。
- [ ] 设计全历史一次计算的输出策略，例如临时 Parquet 后按年分区，避免长期留在内存。
- [ ] 保持年度 `part.parquet` 的行集、列和主键契约。
- [ ] 保持 float32 落盘、有限值检查和缺失口径。
- [ ] 评估中途失败、部分年文件已写和目录库未刷新的恢复行为。
- [ ] 优先减少重复扫描，不改普通窗口 SQL 的数值语义。

测试：

- [ ] 新路径与金标准逐行比较，浮点差异在项目既有容差内。
- [ ] 截面算子跨年、停牌复牌、次新股回看和前视检查通过。
- [ ] 年度文件键集和 `md.daily_hfq` 一致。
- [ ] 中途失败可定位，不产生目录库与文件状态混淆。

验收：

- 代表性全族或单族基准至少有 `20%` 墙钟改善，或扫描量显著下降且内存不超 `1.2x`。
- 因子数值和自检结果保持一致。
- 若内存或原子性不可接受，保留现有实现并记录否决证据。

停止/回退：

- 不以牺牲横截面完整性、停牌窗口或年度文件完整性换取扫描减少。

## 5. P1：统一批量核、Numba 验证和首批 C++

P1 只在 P0 证明优化目标和基准后开展。先做统一接口和算法验证，再引入编译链。

### P1-01 建立批量核适配层并完成 Numba 算法验证

- 状态：等待依赖 P0-01、P0-02、P0-03
- 依赖：P0-01、P0-02、P0-03
- 建议 session 数：1～2

目标：建立与 C++ 实现无关的 Python 批量核契约，并先用已有 Numba 验证相关性和逐日
排名的算法、内存布局和线程模型。

涉及文件：

- 新增 `core/kernels/backend.py`
- 新增 `core/kernels/fallback.py`
- 新增 `core/kernels/__init__.py`
- `pyproject.toml`
- `tests/test_native_kernels.py`
- `research/evaluation/calculate/`

详细工作：

- [ ] 定义 `NativeKernelBackend` 协议和 Python 回退适配器。
- [ ] 接口至少覆盖 `daily_rank_pct`、`daily_winsorize_mad`、`daily_pearson`、
  `pairwise_spearman`、`daily_ols`。
- [ ] 规定输入为 C 连续 `float64` 数组、C 连续 `int64` group offsets。
- [ ] 在 Python 适配层校验 dtype、连续性、维度、offset 单调性和最小样本数。
- [ ] 回退适配器直接调用现有 Python/NumPy 实现，不复制业务语义。
- [ ] 为 Spearman 和逐日排名编写 Numba 原型，不接入生产默认路径。
- [ ] 对比逐日循环、矩阵布局、排序和缺失掩码策略。
- [ ] 记录 Numba 首次编译成本，基准时区分冷启动和热运行。
- [ ] 增加环境开关和显式 fallback reason。

测试：

- [ ] 协议与 Python 回退在边界输入上与现有实现一致。
- [ ] 错误 dtype、非连续数组、offsets 倒置和空输入明确报错。
- [ ] Numba 原型与 SciPy/pandas 金标准一致。
- [ ] 并列值、NaN 和常数截面不漂移。

验收：

- 至少一个目标核热运行中位耗时改善 `1.5x`，且真实阶段改善 `20%`。
- 若没有核达到门槛，停止 P1-02 的 C++ 建设并回到算法优化。
- 适配层默认仍使用 Python 回退，不影响现有调用。

停止/回退：

- 若 Numba 原型没有收益，保留接口和测试，不引入 C++ 编译链。

### P1-02 建立 C++ 构建骨架和 Spearman 单核 POC

- 状态：等待依赖 P1-01
- 依赖：P1-01
- 建议 session 数：1

目标：只实现一个 C++ Spearman 批量核，验证构建、绑定、异常、GIL、内存和性能，不接入
默认业务路径。

涉及文件：

- 新增 `native/CMakeLists.txt`
- 新增 `native/include/quant_native/`
- 新增 `native/src/`
- 新增 `core/kernels/native.py`
- `pyproject.toml`
- `requirements.txt`
- `requirements.lock`
- `tests/test_native_kernels.py`

详细工作：

- [ ] 采用 C++20、`pybind11`、`scikit-build-core` 和 CMake。
- [ ] 产物模块名固定为 `quant_native`。
- [ ] 只暴露 `pairwise_spearman(values, group_offsets, min_n)`。
- [ ] 输入校验和缺失语义由 Python 适配层与 C++ 层共同防守。
- [ ] 计算期间释放 GIL，线程数受 `QUANT_NATIVE_THREADS` 控制，默认 1。
- [ ] C++ 异常转换为 Python 异常；禁止 `abort`、静默失败和全局可变状态。
- [ ] 生成可安装 wheel，并记录 Python 3.12、x86_64 构建命令。
- [ ] 未安装扩展时 `import` 明确失败，业务仍可使用 Python 回退。

测试：

- [ ] C++ 与 Python 金标准逐值一致。
- [ ] 并列、缺失、空组、常数序列和最少样本一致。
- [ ] 错误形状、dtype、offset 和 min_n 明确报错。
- [ ] 重复并发调用没有崩溃、泄漏或结果污染。
- [ ] 子进程压力测试通过。

验收：

- 核函数中位耗时至少改善 `1.5x`。
- 峰值 RSS 不超过 Python 实现 `1.2x`。
- 构建可在目标环境重复完成。
- 未接入业务路径，现有测试完全不受影响。

停止/回退：

- 任一正确性、崩溃或构建门槛不通过，停止 P1-03 并删除未接入的原生路径。

### P1-03 将 Spearman 批量核接入评估相关性

- 状态：等待依赖 P1-02
- 依赖：P1-02
- 建议 session 数：1

目标：在相关性 stage 增加可回退的 native/Python 后端选择，并证明真实三年区间端到端
收益。

涉及文件：

- `research/evaluation/calculate/correlation.py`
- `research/evaluation/calculate/metrics/correlation.py`
- `research/evaluation/calculate/native.py`
- `apps/monitor/backend/service.py`
- `research/evaluation/calculate/tests/test_eval_correlation.py`
- `apps/monitor/backend/tests/test_native_concurrency.py`

详细工作：

- [ ] 接入统一 `kernels` 适配层，不直接调用 `quant_native`。
- [ ] 增加 `native`、`python`、`auto` 三种配置；初始默认仍为 `python`。
- [ ] 保持 `calculate.native` 的现有锁职责，不把 C++ 核绑定塞进该模块。
- [ ] 保证现有评估后端锁和 C++ 线程预算不互相叠加。
- [ ] 记录 backend、fallback reason、线程数、行数、组数和 elapsed_ms。
- [ ] 保持 `top`、`matrix`、`days`、`names`、`basis` 字段语义。
- [ ] 对目标行和完整矩阵两种范围分别测量。

测试：

- [ ] native/Python 的完整评估相关性 payload 一致。
- [ ] HTTP、SSE、取消、慢消费者和并发测试通过。
- [ ] native 缺失时显式回退。
- [ ] 目标因子、候选因子和无正式 peer 路径一致。

验收：

- 真实默认三年相关性 stage 至少改善 `20%`，默认核 `1.5x`。
- 峰值 RSS 不超过基线 `1.2x`。
- 端到端没有 SIGSEGV、死锁或未释放原生对象。

停止/回退：

- 未达门槛则保持默认 Python，量化核收益留档，不扩展其他评估核。

### P1-04 接入逐日排名和 Pearson 核

- 状态：等待依赖 P1-03
- 依赖：P1-03
- 建议 session 数：1

目标：将已验证的批量排名和逐日 Pearson 核接入 IC、RankIC、多周期衰减和增量 IC 的
共享路径。

涉及文件：

- `research/evaluation/calculate/daily.py`
- `research/evaluation/calculate/metrics/ic.py`
- `research/evaluation/calculate/metrics/decay.py`
- `research/evaluation/calculate/metrics/exposure.py`
- `research/evaluation/calculate/tests/test_eval_ic.py`
- `research/evaluation/calculate/tests/test_eval_decay.py`
- `research/evaluation/calculate/tests/test_eval_regressions.py`

详细工作：

- [ ] 将 `rank_pct_by_day` 和 `_daily_pearson` 路由到统一后端。
- [ ] 保持最小截面 5、并列平均名次和有效计数语义。
- [ ] 避免在 IC、decay、incremental IC 中重复转换同一数组。
- [ ] 对多标签样本各自取有效配对，不跨标签复用错误掩码。
- [ ] 保持日期索引、序列长度和 stage payload 不变。

测试：

- [ ] 与 pandas rank、SciPy/Pearson 金标准一致。
- [ ] IC、RankIC、多周期衰减、增量 IC payload 一致。
- [ ] 因子或标签常数、缺失、尾部样本不足时行为一致。
- [ ] 截断重算无前视。

验收：

- 目标阶段真实基准至少改善 `20%`。
- 核函数中位耗时至少改善 `1.5x`。
- 内存不超基线 `1.2x`。

停止/回退：

- 若评估总收益低于门槛，保留 Python 路径，不继续将相同核扩散到更多指标。

### P1-05 接入逐日 MAD 去极值和 z-score 核

- 状态：等待依赖 P1-04
- 依赖：P1-04
- 建议 session 数：1

目标：将逐日中位数、MAD、裁剪和 z-score 融合为一个批量核，减少多次 pandas 分组和
中间 DataFrame。

涉及文件：

- `research/evaluation/calculate/normalize.py`
- `research/evaluation/calculate/daily.py`
- `research/evaluation/calculate/exposure.py`
- `research/evaluation/calculate/tests/test_eval_ic.py`
- `research/evaluation/calculate/tests/test_eval_exposure.py`

详细工作：

- [ ] 路由 `winsorize_by_day` 和 `zscore_by_day` 到统一后端。
- [ ] 保留 MAD 为 0、样本不足、非常数约束等现有分支。
- [ ] 保留 Pearson 去极值对 RankIC 无影响的口径。
- [ ] 评估是否将 winsorize 与 daily Pearson 合并为一次遍历。
- [ ] 避免复制未参与目标的整张面板。

测试：

- [ ] 与 P0-01 金标准逐值对照。
- [ ] 单样本、常数、NaN 和极小 MAD 行为一致。
- [ ] native/Python payload 一致。

验收：

- 对 IC 或暴露阶段带来可量化收益。
- 不改变现有浮点结果容差和内存等级。
- 若只减少局部耗时但没有端到端收益，停止继续融合。

停止/回退：

- 保留更简单的 NumPy 实现，记录未达门槛原因。

### P1-06 接入分位收益、换手和自相关核

- 状态：等待依赖 P1-04
- 依赖：P1-04
- 建议 session 数：1～2

目标：把分位收益、市值加权、换手和因子自相关的中间过程批量化，减少 pandas unstack、
逐行排名和临时宽表。

涉及文件：

- `research/evaluation/calculate/binning.py`
- `research/evaluation/calculate/metrics/bins.py`
- `research/evaluation/calculate/metrics/turnover.py`
- `research/evaluation/calculate/metrics/decay.py`
- `research/evaluation/calculate/tests/test_eval_binning.py`
- `research/evaluation/calculate/tests/test_eval_decay.py`

详细工作：

- [ ] 明确 `assign_bins`、多标签收益、换手和自相关的 Kernel 边界。
- [ ] 保持组内等权、市值加权、缺市值剔除和并列分位语义。
- [ ] 保持净值和多空曲线使用 1 日标签。
- [ ] 保持自相关按逐行有效配对重新排名。
- [ ] 逐个核接入，避免一次替换整个 binning 模块。
- [ ] 记录每个 stage 的中间矩阵大小和峰值 RSS。

测试：

- [ ] 分位成员、收益、换手、自相关与金标准一致。
- [ ] 并列因子、缺失市值、空组和单日样本不足行为一致。
- [ ] decay 与 bin 的共享中间结果不能改变各档独立取样本口径。

验收：

- 至少两个相关 stage 合计改善 `20%`。
- 峰值 RSS 不超过基线 `1.2x`。
- 前端图表所消费的 payload 不变。

停止/回退：

- 若某个核单独无收益，只保留已通过的部分，不强行完成全部核。

### P1-07 接入逐日 OLS 暴露核

- 状态：等待依赖 P1-04
- 依赖：P1-04
- 建议 session 数：1～2

目标：原生实现逐日截面 OLS、截距、风格系数、t 值、R²、残差和跳过原因，减少 Python
逐日循环和重复矩阵构造。

涉及文件：

- `research/evaluation/calculate/exposure.py`
- `research/evaluation/calculate/metrics/exposure.py`
- `research/evaluation/calculate/styles.py`
- `research/evaluation/calculate/tests/test_eval_exposure.py`

详细工作：

- [ ] 保持至少 30 个回归样本和设计矩阵自由度检查。
- [ ] 保持整日常数风格列删除、截距保留和共线跳过语义。
- [ ] 返回逐日 active mask，供系数、t 值和残差正确回填。
- [ ] 保持残差样本、原始 IC 对照和增量 IC 的同样本口径。
- [ ] 行业哑变量和未知行业处理不变。
- [ ] 记录每日矩阵规模、跳过早数和峰值内存。

测试：

- [ ] 与现有 `np.linalg.lstsq` 金标准系数、标准误、t 值、R²、残差一致。
- [ ] 共线、常数、少样本、空行业和单个有效风格行为一致。
- [ ] 增量 IC 与原始 IC 对照一致。

验收：

- 暴露和增量 IC 合计阶段至少改善 `20%`。
- 核 `1.5x`，内存不超过基线 `1.2x`。
- 无法估计的日期和列仍明确跳过。

停止/回退：

- 若 LAPACK 调度或数据搬运使 C++ 无收益，保留 Python 实现并关闭该核。

## 6. P2：共享到模型和挖掘

P2 只复用 P1 已验收的稳定核，不新增未经评估的算法。

### P2-01 将排名和标准化核共享给模型预处理

- 状态：等待依赖 P1-04、P1-05
- 依赖：P1-04、P1-05
- 建议 session 数：1

目标：让周频模型数据集的逐日 rank、z-score、去极值和中性化复用评估层已验证的批量核，
避免重复实现两套口径。

涉及文件：

- `research/modeling/alpha/preprocess.py`
- `research/modeling/alpha/dataset.py`
- `core/kernels/`
- `tests/test_alpha_preprocess.py`
- `tests/test_alpha_dataset.py`
- `tests/test_alpha_train.py`

详细工作：

- [ ] 识别预处理中逐日 rank、z-score、winsorize、均值/标准差和标签排名。
- [ ] 只复用输入输出完全匹配的核，不为模型强行重排数据。
- [ ] 保持缺失指示、中性化、clip_z、sample_weight 和 label_rank 语义。
- [ ] 保持 float32 输出和列顺序。
- [ ] 记录数据集构建和预处理阶段收益。
- [ ] 训练后端保持 sklearn/LightGBM，不引入自研训练核。

测试：

- [ ] 预处理 DataFrame 逐列对照，允许的浮点差异符合既有测试。
- [ ] 标签排名的并列和缺失处理一致。
- [ ] 模型训练输入 hash、模型指标和预测在容差内一致。

验收：

- 预处理或数据集构建有明显改善。
- 训练和预测结果不出现实质漂移。
- 未降低模型产物可复现性。

停止/回退：

- 若收益主要在训练而非预处理，停止该任务，不做无收益共享。

### P2-02 将共享核用于挖掘门禁、轻量 IC 和相关性

- 状态：等待依赖 P1-03、P1-04、P0-05
- 依赖：P1-03、P1-04、P0-05
- 建议 session 数：1～2

目标：把正式评估已验证的批量核应用到挖掘技术门禁、快速筛选和候选相关性，降低候选
数量较大时的重复计算。

涉及文件：

- `research/mining/candidate_evaluator.py`
- `research/mining/evaluation_gates.py`
- `research/mining/candidate_stage_evaluator.py`
- `research/evaluation/calculate/history.py`
- `tests/test_mining_evaluation_consistency.py`
- `tests/test_mining_orchestrator.py`

详细工作：

- [ ] 复用覆盖率、常数日、轻量 RankIC 和目标因子相关性核。
- [ ] 保持 100 个最大相关性样本日、单日至少 5 只股票和至少 20 个成对有效日。
- [ ] 保持前视检查的首、中、末三个截断交易日。
- [ ] 不把候选与同批候选互算相关。
- [ ] 保持候选失败关闭、证据哈希和输入哈希不变。
- [ ] 将批量评估与 P0-05 批量物化联调。

测试：

- [ ] 技术门禁每项检查与 Python 金标准一致。
- [ ] 快速筛选 RankIC、ICIR、年度同号率和耗时字段一致。
- [ ] full/holdout 报告状态和证据哈希一致。
- [ ] 候选失败、空正式 peer、样本不足和快照失效行为不变。

验收：

- 多候选运行的总耗时或重复读取下降。
- 候选状态机、审批资格和发布资格不变。
- 不具备相关性样本时仍 fail-closed。

停止/回退：

- 若候选批量路径没有收益，保留原评估流程，只保留 DAG 诊断和单候选兼容入口。

### P2-03 评估受控 DSL 原生执行器是否需要建设

- 状态：等待依赖 P0-05、P2-02
- 依赖：P0-05、P2-02
- 建议 session 数：1，默认只做决策

目标：在 DAG/CSE、批处理和共享核都已实施后，判断剩余瓶颈是否足以支持原生表达式 VM。
默认结论是“不建设”，只有明确证据才进入实现。

涉及文件：

- `research/mining/expression_ast.py`
- `research/mining/expression_compiler.py`
- `research/mining/candidate_materialization.py`
- `docs/research/mining/`
- `AiRecord/benchmarks/`

详细工作：

- [ ] 测量 DAG/CSE 后扫描、表达式求值、物化和评估的剩余占比。
- [ ] 统计不同算子在候选中的使用频率、平均深度和复杂度。
- [ ] 评估 DuckDB 表函数、外部 C++ 表达式 VM 和继续 SQL 三条路线。
- [ ] 计算版本锁定、构建、调试、崩溃隔离和测试成本。
- [ ] 形成明确的实施或否决决定，并记录阈值证据。

测试：

- [ ] 只做基准和可行性验证，不进入生产代码路径。

验收：

- 如果表达式求值占多候选总耗时不足主要比例，明确标记“不实施”。
- 如果达到门槛，另建详细设计和单独任务，不在本任务中直接实现。

停止/回退：

- 默认不建设原生 VM；不以 C++ 重写 PostgreSQL/DuckDB 已知能力作为目标。

## 7. P3：回测和清洗专项

P3 当前只做剖析和 I/O 优化。C++ 回测账本不是既定交付，必须有明确触发条件。

### P3-01 回测 I/O、分块和内存专项优化

- 状态：等待依赖 P0-01
- 依赖：P0-01
- 建议 session 数：1～2

目标：确认回测真实瓶颈，优先减少数据装载、重复读取、宽表复制和跨块状态开销，不先
重写 Python 账本。

涉及文件：

- `trading/backtest/data.py`
- `trading/backtest/chunked.py`
- `trading/backtest/pipeline.py`
- `tools/monitor/backtest_profile.py`
- `trading/backtest/tests/test_data.py`
- `trading/backtest/tests/test_chunked.py`

详细工作：

- [ ] 重新记录 load、chunk、signals、probe、baseline、orders、allocation、execute 和 done。
- [ ] 区分 `chunk` 外层时段与其内部子事件，避免重复累计。
- [ ] 记录各宽表的 dtype、复制次数和峰值 RSS。
- [ ] 检查重复 DuckDB 查询、无必要 DataFrame copy 和全历史读取。
- [ ] 评估块大小、股票集合和缓存复用对内存与耗时的影响。
- [ ] 优先优化 I/O 和块规划，不改变交易规则和账本状态。

测试：

- [ ] 全 A 三年和指数十年回测结果逐字段一致。
- [ ] 订单、拒单、分配、现金、持仓、公司行动、费用和 NAV 一致。
- [ ] 跨块交接和恢复一致。

验收：

- 至少一个真实场景墙钟改善 `20%`，或峰值 RSS 改善 `20%`，且无行为漂移。
- 若无收益，记录真实瓶颈，不进入 P3-03。

停止/回退：

- 不因 C++ 目标而改变回测规则或牺牲审计字段。

### P3-02 清洗层条件化剖析与共享核评估

- 状态：等待依赖 P0-01
- 依赖：P0-01
- 建议 session 数：1，默认只做剖析

目标：仅在剖析证明某个 NumPy 规则占主要时间时，考虑共享核；否则明确关闭该方向。

涉及文件：

- `core/data/scripts/clean/`
- `tests/test_adjust.py`
- `tests/test_limits.py`
- `tests/test_lifecycle.py`
- `AiRecord/benchmarks/`

详细工作：

- [ ] 记录复权、昨收、涨跌停、ST、生命周期和交易标记的耗时。
- [ ] 区分 Parquet I/O、pandas 组装、NumPy 规则和 DuckDB 建表。
- [ ] 找出可复用的纯数组函数。
- [ ] 如果单一规则不足清洗总耗时 `20%`，标记“不实施 C++”。
- [ ] 达到门槛时另建实现任务，本任务不直接改业务路径。

测试：

- [ ] 只做只读剖析和基准夹具，不改变清洗结果。

验收：

- 给出明确“实施/不实施”结论和证据。
- 不因评估层已有共享核而强行迁移清洗代码。

停止/回退：

- 默认不实施；I/O 或数据源占主体时关闭本方向。

### P3-03 回测账本 C++ 化专项决策

- 状态：等待依赖 P3-01
- 依赖：P3-01
- 建议 session 数：未排期，仅在触发后

目标：在 I/O 和分块优化后，判断 Python 执行状态机是否仍为回测主瓶颈，并评估重写成本。

触发条件：

- P3-01 完成后，I/O 和分块优化已实施；
- Python 账本、分配、订单和公司行动处理合计仍占回测墙钟至少 `25%`；
- 有稳定逐笔金标准和并发压力测试；
- 预期收益足以覆盖 C++ 构建、调试和维护成本。

潜在涉及文件：

- `trading/backtest/ledger.py`
- `trading/backtest/allocation.py`
- `trading/backtest/orders.py`
- `trading/backtest/risk.py`
- `trading/backtest/tests/`

决策内容：

- [ ] 逐项比较 Python 状态机与 C++ 事件驱动账本。
- [ ] 评估公司行动、FIFO 税费、跨块状态、拒单和审计输出的实现复杂度。
- [ ] 定义逐笔成交、现金、持仓、费用和 NAV 金标准。
- [ ] 计算构建、发布、排障和维护成本。
- [ ] 未满足任一触发条件时标记“不实施”。

验收：

- 只产出决策和设计，不直接重写账本。
- 如决定实施，另建详细计划和独立 session。

## 8. 明确不实施事项

以下事项从报告中明确排除，不应作为后续 session 自行扩展：

- 不整体把 `core/factors/ops.py` 替换为 C++。
- 不把 DuckDB 查询改写为独立 C++ 服务。
- 不使用逐股票、逐日或逐单元格 C++ UDF。
- 不自研 LightGBM、sklearn 或 GBDT 训练内核。
- 不把 Web、SSE、审批和状态管理迁移到 C++。
- 不在 P0 的 DAG/CSE 和批量物化完成前开发挖掘原生表达式 VM。
- 不为清洗层普通 NumPy 规则单独引入 C++，除非 P3-02 达到触发门槛。
- 不在 P3-01 证明状态机为主瓶颈前重写回测账本。

## 9. Session 完成模板

后续 session 完成一个任务后，将以下内容写入对应 changes 和 progress：

```text
任务 ID：
开始/完成日期：
代码基线：
实际修改文件：
公共接口/数据契约变化：
新增测试：
执行的定向测试：
基准场景与结果：
峰值 RSS：
正确性结论：
是否达到验收：
回退方式：
遗留风险：
commit：
```

本文件更新时只改相应任务的状态、结论、日期、证据路径和 commit，不删除历史停止原因。
如果后续决定改变优先级，新增决策记录，不静默覆盖原顺序。

## 10. 整体完成定义

算子与 C++ 优化方向只有在以下条件全部满足时才算完成：

- P0 基准和重复计算专项已结案；
- 至少一个 C++ 批量核通过正确性、性能、内存、并发和回退验收；
- 评估层端到端有真实收益；
- 未通过门槛的候选路径已明确停止；
- 模型、挖掘、回测和清洗各自有明确的实施或否决结论；
- 所有证据、计划、changes、diffs、progress 和 commit 可追溯。

如果 C++ 本身没有达到收益门槛，项目可以以“Python/DuckDB 已足够、C++ 路径已否决”
正常结案，这不视为任务失败。
