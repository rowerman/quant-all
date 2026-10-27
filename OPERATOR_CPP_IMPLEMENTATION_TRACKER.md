# 算子与 C++ 后续工作进度追踪

- 来源报告：[OPERATOR_CPP_ANALYSIS_REPORT.md](TODO_docs/OPERATOR_CPP_ANALYSIS_REPORT.md)
- 创建日期：2026-10-27
- 代码基线：`main` 分支 V112 工作区（P1-01 V111 提交 `4d3943a` 之后）
- 当前状态：P0-01～P0-06、P1-01～P1-06 已完成；Spearman、逐日排名、逐日 Pearson、
  逐日 MAD、逐日 z-score、分位分组、多标签分位收益、分位换手和因子自相关已接入评估
  路径；P1-07 已按用户确认的 `1e-5` 一致性门槛完成，逐日 OLS 已接入 native 默认路径。
  模型预处理和标签排名已接入统一核；默认后端为 `auto`，实际 C++ 替换模块数为 13；
  P2-01 已完成，P2-02 等待开始。

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

- 状态：已完成（V108，2026-10-27）
- 依赖：P0-01
- 建议 session 数：1

目标：在 SQL 生成前建立稳定表达式 DAG，识别同轮候选之间重复字段、窗口、标量和子树，
为批量计算提供依据。本任务不改变候选评估结果。

涉及文件：

- `research/mining/expression_ast.py`
- `research/mining/expression_validation.py`
- `research/mining/expression_dag.py`
- `research/mining/candidate_materialization.py`
- `research/mining/candidate_batch.py`
- `tests/test_mining_expression_dag.py`

详细工作：

- [x] 为 AST 节点定义稳定的结构哈希，不能依赖 Python `id()` 或对象插入顺序。
- [x] 将规范化表达式转换为 DAG，保留输出节点和每个候选的根节点映射。
- [x] 识别重复叶子字段、常量、标量表达式和相同窗口子树。
- [x] 保留每个候选独立的 canonical expression、semantic hash 和 artifacts 身份。
- [x] 增加“候选组计划”数据结构，记录共享节点、候选节点和预计可复用次数。
- [x] 输出只读诊断：候选数、AST 节点数、去重节点数、共享节点比例。
- [x] 禁止把不同冻结目录或不同数据上下文中的节点混入同一计划。

测试：

- [x] 交换律规范化前后哈希符合现有规则。
- [x] 相同表达式跨候选去重，不同窗口和算子参数不复用；当前 DSL 没有缺失策略参数，
  所有字面量参数均进入结构键。
- [x] 候选名称、内容哈希、semantic hash 和验证结果不变。
- [x] 非受控表达式仍被拒绝。

验收：

- 同一候选组可重复生成完全相同的 DAG 计划和诊断。
- 不改变候选物化结果和评估报告。
- 能产出共享节点比例和潜在扫描减少量。

停止/回退：

- 如果现有 AST 无法稳定表达共享节点，先补齐 AST 契约，不进行 SQL 批处理。

完成记录：

- 新增 `mining.expression_dag`，提供 `build_candidate_expression_plan()`、
  `CandidateExpressionPlan`、`ExpressionDagNode`、`CandidateExpressionRoot` 和
  `ExpressionDagDiagnostics`。
- 结构哈希使用 DAG schema 版本和规范化 AST 结构键；候选按 `candidate_id` 排序，
  节点按后序输出，因此候选输入换序和重复规划得到相同计划。
- 抽取 `validate_candidate_expression()`，候选物化和 DAG 规划共用同一语义一致性规则。
- 示例：两个候选共 11 个 AST 节点，去重后 7 个节点，重复 4 个，共享节点 2 个；
  `potential_evaluation_reduction_ratio=0.363636...`。该指标仅为潜在节点求值减少，
  不表示已减少 SQL 或 DuckDB 扫描。
- 定向测试 99 passed；本轮未真实采用 C++，不增加 C++ 模块收益记录。
- Worker 客户端、发布和存储下游测试 33 passed。
- 回退：删除 `research/mining/expression_dag.py` 和对应测试，撤销 AST/校验复用改动，
  恢复文档与 Tracker；不涉及生产数据或候选产物。

### P0-05 实现挖掘候选批量物化和复用

- 状态：已完成（V109，2026-10-27）
- 依赖：P0-04
- 建议 session 数：1～2

目标：使用 P0-04 的 DAG 计划，把同轮多个候选合并为共享中间层的批量查询，并复用公共
子树结果；保持每个候选独立目录、独立哈希和失败隔离。

涉及文件：

- `research/mining/candidate_materialization.py`
- `research/mining/expression_batch.py`
- `research/mining/expression_compiler.py`
- `research/mining/orchestrator_screening.py`
- `research/mining/orchestrator_holdout.py`
- `research/mining/candidate_evaluator.py`
- `tests/test_mining_orchestrator.py`
- `tests/test_mining_expression_batch.py`
- `tests/test_mining_batch_materialization.py`

详细工作：

- [x] 明确批量边界：同一 run、同一 FrozenContext、同一日期区间、同一字段目录。
- [x] 为共享中间结果定义临时关系或临时 Parquet，不写入正式因子目录。
- [x] 批量查询只物化最终候选列和必要的共享中间层。
- [x] 每个候选继续写入独立 staging 和原子目录。
- [x] 保留单候选物化兼容入口，供调试和故障恢复使用。
- [x] 单候选失败只影响该候选；共享层失败则整批返回明确错误。
- [x] 批量完成后重新校验每个候选的 artifact hash 和 materialization manifest。
- [x] 记录批量前后扫描次数、SQL 数、临时文件大小、总耗时和峰值 RSS。

测试：

- [x] 批量物化与逐候选物化的行集、列、非空数和数值一致。
- [x] 候选内容哈希、artifact hash 和可复用身份不被批量路径改变。
- [x] 相同批量的重复执行保持幂等。
- [x] 一个候选失败不会发布半成品，也不会污染正式因子目录。
- [x] 公共子树和不同数据窗口不能错误复用。

验收：

- 多候选轮次的端到端耗时或扫描次数有可量化下降。
- 任意候选仍可单独重算并得到一致结果。
- 临时目录可清理，崩溃恢复不会复用不完整共享层。

停止/回退：

- 如果共享层物化成本高于重复扫描，记录适用候选数和阈值后关闭批量路径，保留 DAG 诊断。

完成记录：

- 证据：`AiRecord/benchmarks/V109/p0-05-20261027/`。
- 新增 `mining.expression_batch`，把候选 DAG 编译为一条共享 CTE 查询；筛选和测试段
  按发现轮次自动批量物化，候选级失败隔离，共享层失败返回 `MINING-124` 并尝试逐候选回退。
- 4 候选三年示例物化中位耗时由 11.395 秒降至 3.517 秒，下降 69.1%，加速 3.24 倍；
  峰值 RSS 由 1224.914MB 降至 1217.789MB，为串行路径的 0.994 倍。
- 分析 SQL 由 4 条降为 1 条，`md.daily_hfq` 引用由 8 次降为 2 次；临时宽 Parquet
  为 16,461,881 字节，完成后删除。
- 串行与批量 4 个候选的年度 Parquet 字节、artifact tree hash 和文件清单完全一致。
- 本轮采用 Python 编排和 DuckDB DAG 批量执行，未采用 C++，不计入 C++ 模块替换收益。
- 复现：
  `.venv/bin/python -m tools.monitor.operator_baseline --case mining_batch_serial`
  和 `--case mining_batch_dag`，证据输出目录使用
  `AiRecord/benchmarks/V109/p0-05-20261027`。

### P0-06 评估并改造因子族全历史重复扫描

- 状态：已完成（V110，2026-10-27）
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

- [x] 记录当前单族全区间、单族单年、全族全年的扫描量、耗时和峰值内存。
- [x] 证明当前 `base_cte` 的完整历史是截面算子和停牌窗口正确性所必需。
- [x] 设计全历史一次计算的输出策略，例如临时 Parquet 后按年分区，避免长期留在内存。
- [x] 保持年度 `part.parquet` 的行集、列和主键契约。
- [x] 保持 float32 落盘、有限值检查和缺失口径。
- [x] 评估中途失败、部分年文件已写和目录库未刷新的恢复行为。
- [x] 优先减少重复扫描，不改普通窗口 SQL 的数值语义。

测试：

- [x] 新路径与金标准逐行比较，浮点差异在项目既有容差内。
- [x] 截面算子跨年、停牌复牌、次新股回看和前视检查通过。
- [x] 年度文件键集和 `md.daily_hfq` 一致。
- [x] 中途失败可定位，不产生目录库与文件状态混淆。

验收：

- 代表性全族或单族基准至少有 `20%` 墙钟改善，或扫描量显著下降且内存不超 `1.2x`。
- 因子数值和自检结果保持一致。
- 若内存或原子性不可接受，保留现有实现并记录否决证据。

停止/回退：

- 不以牺牲横截面完整性、停牌窗口或年度文件完整性换取扫描减少。

完成记录：

- 证据：`AiRecord/benchmarks/V110/p0-06-20261027/`。
- 新增 `full_history` 构建模式：每族一次执行全区间查询，DuckDB 按年分区，staging 内
  校验列、主键和行数后原子发布；发布异常恢复已有年度文件。原 `yearly` 路径保留。
- 动量 2023～2024 年：SQL 由 2 条降为 1 条，输入扫描由 28,846,271 行降为
  15,069,961 行；耗时由 22.303 秒降为 13.140 秒，下降 41.1%，峰值 RSS 由
  3491.6MB 降为 3236.6MB。
- 动量全历史：27 条 SQL 降为 1 条，理论累计扫描由 173,510,525 行降为
  17,413,423 行；耗时 34.672 秒，峰值 RSS 3508.8MB，输出 27 个年度文件、
  17,999,882 行。
- 两年新旧路径的主键、空值和全部因子值逐行一致。全族 2024 年构建通过，共 11 族、
  47 个因子、14,264,745 行，耗时 191.6 秒，峰值 RSS 3330.7MB。
- 全历史路径默认 `memory_limit=3GB`，可用 `QUANT_FACTOR_BUILD_MEMORY_LIMIT` 覆盖；
  `yearly` 模式保留为回退。本轮未采用 C++，不计入 C++ 模块替换收益。

## 5. P1：统一批量核、Numba 验证和首批 C++

P1 只在 P0 证明优化目标和基准后开展。先做统一接口和算法验证，再引入编译链。

### P1-01 建立批量核适配层并完成 Numba 算法验证

- 状态：已完成（V112，2026-10-27；按逐日排名门禁）
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

- [x] 定义 `NativeKernelBackend` 协议和 Python 回退适配器。
- [x] 接口至少覆盖 `daily_rank_pct`、`daily_winsorize_mad`、`daily_pearson`、
  `pairwise_spearman`、`daily_ols`。
- [x] 规定输入为 C 连续 `float64` 数组、C 连续 `int64` group offsets。
- [x] 在 Python 适配层校验 dtype、连续性、维度、offset 单调性和最小样本数。
- [x] 回退适配器直接调用现有 Python/NumPy 实现，不复制业务语义。
- [x] 为 Spearman 和逐日排名编写 Numba 原型，不接入生产默认路径。
- [x] 对比逐日循环、矩阵布局、排序和缺失掩码策略。
- [x] 记录 Numba 首次编译成本，基准时区分冷启动和热运行。
- [x] 增加环境开关和显式 fallback reason。

测试：

- [x] 协议与 Python 回退在边界输入上与现有实现一致。
- [x] 错误 dtype、非连续数组、offsets 倒置和空输入明确报错。
- [x] Numba 原型与 SciPy/pandas 金标准一致。
- [x] 并列值、NaN 和常数截面不漂移。

验收：

- 逐日排名核结果正确，热运行中位耗时至少改善 `1.5x`。
- Spearman 继续保留正确性、冷热耗时和加速比证据，但不作为本轮放行依据。
- 真实 stage 改善留到 P1-02 C++ POC 及后续业务接入任务验证。
- 适配层默认仍使用 Python 回退，不影响现有调用。

停止/回退：

- 若逐日排名结果错误或未达到 `1.5x`，保留接口和测试，不重新开放 P1-02。

完成记录：

- 新增 `kernels` 包，提供 `NativeKernelBackend`、`PythonKernelBackend`、
  `BackendSelection`、`get_backend()` 和 `KERNEL-001`～`KERNEL-004`。
- Python 回退覆盖五个核；Numba 只实现 `daily_rank_pct` 和 `pairwise_spearman`，
  其余方法显式委托回退。Numba 和未来的 C++ 均不进入评估默认路径。
- 固定微基准使用 2026-08-03、5,219×47 真实因子截面和 2026-08 共 21 个交易日、
  109,652 行排名输入，每项热运行 5 次取中位数。
- Spearman：Python 0.155752 秒，Numba 0.872288 秒，加速 0.179x，冷启动
  20.104 秒；结果与 Python 金标准在 `1e-10` 内一致，计数完全一致。
- 逐日排名：Python 0.017885 秒，Numba 0.008859 秒，加速 2.019x，冷启动
  0.585 秒；结果逐值一致。
- 核、IC、相关性、暴露和回归定向测试 88 passed；既有基准契约 6 passed；
  后端原生锁并发测试 11 passed；`pip check`、editable 导入和 `git diff --check`
  通过。
- 用户确认采用 Spearman 专项门禁。微基准未达到 `1.5x`，因此没有运行完整三年
  stage，P1-02 停止。逐日排名的局部收益不改变该结论。
- 证据：`AiRecord/benchmarks/V111/p1-01-20261027/`。
- 本轮实际 C++ 替换模块数为 0，不计入 C++ 模块替换收益。
- V112 用户确认改用逐日排名门禁。新证据中逐日排名 Python 0.016966 秒、Numba
  0.008388 秒，加速 2.023x，结果逐值一致；Spearman 诊断复测为 0.179x，仍明确
  标记为未达标。门禁 schema 升级为 `kernel_baseline_v2`，P1-02 恢复为待开始。
- V112 证据：`AiRecord/benchmarks/V112/p1-01-20261027-recheck/`；本轮仍未采用
  C++，实际 C++ 替换模块数为 0。
- 回退：删除 `core/kernels`、`kernels` 包映射、核测试和基准工具，恢复文档、
  Tracker 和 V111 记录；不涉及生产数据或业务调用路径。

### P1-02 建立 C++ 构建骨架和 Spearman 单核 POC

- 状态：已完成（V113，2026-10-27）
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

- [x] 采用 C++20、`pybind11`、`scikit-build-core` 和 CMake。
- [x] 产物模块名固定为 `quant_native`。
- [x] 只暴露 `pairwise_spearman(values, group_offsets, min_n)`。
- [x] 输入校验和缺失语义由 Python 适配层与 C++ 层共同防守。
- [x] 计算期间释放 GIL，线程数受 `QUANT_NATIVE_THREADS` 控制，默认 1。
- [x] C++ 异常转换为 Python 异常；禁止 `abort`、静默失败和全局可变状态。
- [x] 生成可安装 wheel，并记录 Python 3.12、x86_64 构建命令。
- [x] 未安装扩展时 `import` 明确失败，业务仍可使用 Python 回退。

测试：

- [x] C++ 与 Python 金标准逐值一致。
- [x] 并列、缺失、空组、常数序列和最少样本一致。
- [x] 错误形状、dtype、offset 和 min_n 明确报错。
- [x] 重复并发调用没有崩溃、泄漏或结果污染。
- [x] 子进程压力测试通过。

验收：

- 核函数中位耗时至少改善 `1.5x`。
- 峰值 RSS 不超过 Python 实现 `1.2x`。
- 构建可在目标环境重复完成。
- 未接入业务路径，现有测试完全不受影响。

停止/回退：

- 任一正确性、崩溃或构建门槛不通过，停止 P1-03 并删除未接入的原生路径。

停止记录：

- P1-01 已按用户确认的 Spearman 专项门禁完成验证；微基准和真实 stage 需同时达标，
  但微基准本身已失败，故未进入构建链和 C++ 代码。
- 后续只有在 Spearman 算法或 Python 实现发生实质变化并重新通过完整门禁后，才重新
  评估 P1-02；当前不创建 `native/`、不安装 CMake/pybind11，也不修改构建依赖。
- V112 更新：用户确认把 P1-01 放行口径改为逐日排名核 `1.5x`，Spearman 直接交给本
  任务 C++ POC 验证。以上 V111 Spearman 停止记录作为历史保留；V112 未创建 C++ 工程。

完成记录：

- 新增 `quant_native` C++20 扩展，使用 pybind11、CMake 4.4.4、Ninja 1.13.2 和
  scikit-build-core 1.1.1 的 setuptools 兼容后端；现有包目录映射和动态依赖保持不变。
- 新增长期可复用的 C++ Spearman 算法：每个因子只排序一次，按相同缺失掩码复用实际样本
  内的并列平均名次，再批量计算组对点积。最初逐因子对重复排序的 POC 仅 `0.161x`，
  该无效路径未保留。
- 固定 2026-08-03、5,219×47 真实截面、单组单线程：Python 热运行中位 0.172836 秒，
  C++ 0.073312 秒，加速 2.358x；峰值 RSS 282.820MB 对 283.129MB，比例为 1.001x。
- C++ 与 Python 相关系数在 `1e-10` 内一致，有效计数和 `NaN` 位置完全一致；覆盖空组、
  并列、缺失、正负无穷、常数、自定义 `min_n`、GIL、并发和子进程压力。
- 新增 `kernels.CppKernelBackend`，只允许显式请求 `native`；默认仍为 Python，不提供
  `auto`。未实现核显式回退 Python，扩展缺失时返回 `KERNEL-004`。
- 证据：`AiRecord/benchmarks/V113/p1-02-20261027/`；构建环境为 Python 3.12.3、
  x86_64 Linux、GCC 13.3.0。
- 本轮没有修改评估默认路径，实际 C++ 替换模块数为 0，不计入 C++ 模块替换收益。
- 回退：删除 `native/`、`core/kernels/native.py`、C++ 测试和基准工具，恢复构建依赖、
  Tracker 与文档；不涉及生产数据或业务结果。

### P1-03 将 Spearman 批量核接入评估相关性

- 状态：已完成（V114，2026-10-27）
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

- [x] 接入统一 `kernels` 适配层，不直接调用 `quant_native`。
- [x] 增加 `native`、`python`、`auto` 三种配置；达标后默认 `auto`。
- [x] 保持 `calculate.native` 的现有锁职责，不把 C++ 核绑定塞进该模块。
- [x] 保证现有评估后端锁和 C++ 线程预算不互相叠加。
- [x] 记录 backend、fallback reason、线程数、行数、组数和 elapsed_ms。
- [x] 保持 `top`、`matrix`、`days`、`names`、`basis` 字段语义。
- [x] 对目标行和完整矩阵两种范围分别测量。

测试：

- [x] native/Python 的完整评估相关性 payload 一致。
- [x] HTTP、SSE、取消、慢消费者和并发测试通过。
- [x] native 缺失时显式回退。
- [x] 目标因子、候选因子和无正式 peer 路径一致。

验收：

- 真实默认三年相关性 stage 至少改善 `20%`，默认核 `1.5x`。
- 峰值 RSS 不超过基线 `1.2x`。
- 端到端没有 SIGSEGV、死锁或未释放原生对象。

停止/回退：

- 未达门槛则保持默认 Python，量化核收益留档，不扩展其他评估核。

完成记录：

- 新增 `pairwise_spearman(..., target_index=None)` 目标行模式；完整矩阵仍保持原接口和
  语义，目标模式只填充目标行、目标列和对角线。
- `QUANT_KERNEL_BACKEND` 默认 `auto`。扩展不可导入时回退 Python并记录
  `cpp_extension_unavailable`；运行期原生错误不静默回退。
- 正式三年 stage：Python 217.770 秒，native 140.460 秒，auto 135.788 秒；native
  下降 35.5%，auto 下降 37.6%。
- 候选目标行 stage：Python 137.599 秒，native 118.680 秒，auto 120.995 秒；native
  加速 1.159x，auto 加速 1.137x。
- 固定 5,219×47 核微基准：完整矩阵 0.167855 秒降至 0.060035 秒（2.796x）；目标行
  0.042791 秒降至 0.021802 秒（1.963x）。相关系数 `1e-10` 内一致，有效计数和
  `NaN` 位置完全一致。
- 两条路径 RSS 比例为 Python 的 1.0001x 左右，未超过 1.2x 门槛；门禁全部通过。
- 证据：`AiRecord/benchmarks/V114/p1-03-20261027/`。
- 实际 C++ 替换模块数由 0 更新为 1，登记模块为“评估相关性 Spearman”。
- 回退：把默认后端改回 `python`，恢复 `_daily_spearman` 直接路由并删除目标行参数；
  保留 Python 金标准、业务数据和历史证据。

### P1-04 接入逐日排名和 Pearson 核

- 状态：已完成（V115，2026-10-27）
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

- [x] 将 `rank_pct_by_day` 和 `_daily_pearson` 路由到统一后端。
- [x] 保持最小截面 5、并列平均名次和有效计数语义。
- [x] 避免在 IC、decay、incremental IC 中重复转换同一数组。
- [x] 对多标签样本各自取有效配对，不跨标签复用错误掩码。
- [x] 保持日期索引、序列长度和 stage payload 不变。

测试：

- [x] 与 pandas rank、SciPy/Pearson 金标准一致。
- [x] IC、RankIC、多周期衰减、增量 IC payload 一致。
- [x] 因子或标签常数、缺失、尾部样本不足时行为一致。
- [x] 截断重算无前视。

验收：

- 目标阶段真实基准至少改善 `20%`。
- 核函数中位耗时至少改善 `1.5x`。
- 内存不超基线 `1.2x`。

停止/回退：

- 若评估总收益低于门槛，保留 Python 路径，不继续将相同核扩散到更多指标。

完成记录：

- 新增 `quant_native.daily_rank_pct` 和 `quant_native.daily_pearson`，按组并行、释放
  GIL，并沿用 `QUANT_NATIVE_THREADS` 单线程默认预算。
- `calculate.daily` 在每次 IC、RankIC、多标签衰减或通用日度相关调用中只选择一次后端，
  先按交易日稳定分组，再复用同一 offsets 调用批量核；多标签仍逐档独立取有限配对。
- 固定 3,724,919 行、722 组真实面板：逐日排名 Python 1.114445 秒降至 C++ 0.238150 秒
  （4.680x），Pearson 0.210444 秒降至 0.011909 秒（17.670x）。
- 真实 stage：`overview` 由 4.661 秒降至 2.239 秒（2.082x），`decay` 由 15.145 秒降至
  5.371 秒（2.820x），`incremental_ic` 由 21.727 秒降至 16.298 秒（1.333x）。
- Python/native/auto 三个 stage payload 在 `1e-10` 内一致；native/auto 峰值 RSS 比例
  最高 1.037x，未超过 1.2x。证据：`AiRecord/benchmarks/V115/p1-04-20261027/`。
- 实际 C++ 替换模块数由 1 更新为 3，新增登记“评估逐日排名”和“评估逐日 Pearson”。
  Pearson 的独立业务占比很小，其价值以核级 17.670x 和 `overview` 的组合收益为准。
- 回退：删除两个原生核的业务路由与台账增量，保留 Spearman 相关性接入、Python 金标准
  和本任务证据。

### P1-05 接入逐日 MAD 去极值和 z-score 核

- 状态：已完成（V116，2026-10-27）
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

- [x] 路由 `winsorize_by_day` 和 `zscore_by_day` 到统一后端。
- [x] 保留 MAD 为 0、样本不足、非常数约束等现有分支。
- [x] 保留 Pearson 去极值对 RankIC 无影响的口径。
- [x] 评估是否将 winsorize 与 daily Pearson 合并为一次遍历。
- [x] 避免复制未参与目标的整张面板。

测试：

- [x] 与 P0-01 金标准逐值对照。
- [x] 单样本、常数、NaN 和极小 MAD 行为一致。
- [x] native/Python payload 一致。

验收：

- 对 IC 或暴露阶段带来可量化收益。
- 不改变现有浮点结果容差和内存等级。
- 若只减少局部耗时但没有端到端收益，停止继续融合。

停止/回退：

- 保留更简单的 NumPy 实现，记录未达门槛原因。

完成记录：

- 新增 `quant_native.daily_winsorize_mad`、`daily_zscore` 和
  `daily_winsorize_zscore`；z-score 使用与 pandas groupby 一致的 Kahan 补偿求和和
  运算结合顺序，固定真实数组上两个 z-score 路径均逐值一致。
- `calculate.daily` 一次按日稳定分组，复用 offsets 完成 Pearson IC 的因子/标签 MAD
  和 Pearson；RankIC 继续对原始有限配对排名。`calculate.exposure` 对风格列和因子
  使用融合标准化核，并按原行序写回。
- 固定 3,724,919 行、722 组、单线程、热运行 5 次：MAD `0.195444s → 0.084487s`
  （`2.313x`），z-score `0.158660s → 0.014637s`（`10.840x`），融合标准化
  `0.342608s → 0.096023s`（`3.568x`）。
- 真实三年 stage 各重复 3 次：`overview` 由 `4.516s` 降至 `1.782s`（`2.535x`），
  `exposure` 由 `15.922s` 降至 `13.237s`（`1.203x`）；Python/native/auto payload
  在 `1e-10` 内一致，native/auto 最高 RSS 比例分别为 `1.0172x`、`1.0195x`。
- 本轮实际 C++ 替换模块数由 3 更新为 5，新增登记“评估逐日 MAD 去极值”和
  “评估逐日 z-score”。证据：`AiRecord/benchmarks/V116/p1-05-20261027/`。
- 回退：删除三个标准化原生核和业务路由，`calculate.daily`、`normalize`、`exposure`
  恢复 Python 路径，台账减回 3；保留 Tracker 证据和 Python 金标准。

### P1-06 接入分位收益、换手和自相关核

- 状态：已完成（V117，2026-10-27）
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

- [x] 明确 `assign_bins`、多标签收益、换手和自相关的 Kernel 边界。
- [x] 保持组内等权、市值加权、缺市值剔除和并列分位语义。
- [x] 保持净值和多空曲线使用 1 日标签。
- [x] 保持自相关按逐行有效配对重新排名。
- [x] 逐个核接入，避免一次替换整个 binning 模块。
- [x] 记录每个 stage 的中间矩阵大小和峰值 RSS。

测试：

- [x] 分位成员、收益、换手、自相关与金标准一致。
- [x] 并列因子、缺失市值、空组和单日样本不足行为一致。
- [x] decay 与 bin 的共享中间结果不能改变各档独立取样本口径。

验收：

- 至少两个相关 stage 合计改善 `20%`。
- 峰值 RSS 不超过基线 `1.2x`。
- 前端图表所消费的 payload 不变。

停止/回退：

- 若某个核单独无收益，只保留已通过的部分，不强行完成全部核。

完成记录：

- 新增 `daily_quantile_bins`、`bin_returns_multi`、`bin_turnover` 和
  `daily_row_spearman` 四个原生核；Python 回退保留现有 pandas/NumPy 实现。
- `assign_bins` 使用原生平均名次分位分组；`rank_matrix` 复用已验收的
  `daily_rank_pct`；多标签衰减、换手和因子自相关分别走对应原生核。
- 单标签 `bin_daily_returns` 继续使用 pandas 分组求和，以保持市值加权和
  `bin_curves`、`spread`、`bin_dist` 的既有逐值语义；这些 stage 的收益来自分位分组。
- 固定 3,751,348 行分位样本、727 个交易日和 5,606 只股票的单线程微基准：
  分位分组 `1.114s → 0.252s`（4.422x），等权收益 `0.237s → 0.047s`（5.045x），
  市值加权收益 `0.241s → 0.048s`（5.016x），换手 `0.0688s → 0.0399s`（1.723x），
  三个 lag 自相关 `2.455s → 1.424s`（1.724x）。
- 真实 stage：`bins` 1.621x、`bin_curves` 1.575x、`spread` 1.594x、
  `bin_dist` 1.509x、`turnover` 1.854x、`decay` 3.255x；`bins + turnover`
  合计改善 43.82%，native/auto 最高 RSS 比例为 1.0060x。
- Python/native/auto 微基准和业务输出在 `1e-10` 内一致；分位组号和有效计数逐值一致。
  V105 的 `turnover`、`decay` 仍与金标准一致；`bins` 系列中市值加权统计存在与当前
  Python 基线相同的既有小漂移，native 未劣于 Python。
- 实际 C++ 替换模块数由 5 更新为 9，新增登记“评估分位分组”“评估多标签分位收益”
  “评估分位换手”和“评估因子自相关”。证据：
  `AiRecord/benchmarks/V117/p1-06-20261027/`。
- 回退：删除四个原生核和对应业务路由，恢复 `calculate.binning` 的 Python 路径，
  台账减回 5；保留 Tracker、V117 证据和当前 Python 金标准。

### P1-07 接入逐日 OLS 暴露核

- 状态：已完成（V119，2026-10-27；用户确认业务一致性门槛为 `1e-5`）
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

- [x] 保持至少 30 个回归样本和设计矩阵自由度检查。
- [x] 保持整日常数风格列删除、截距保留和共线跳过语义。
- [x] 返回逐日 active mask，供系数、t 值和残差正确回填。
- [x] 保持残差样本、原始 IC 对照和增量 IC 的同样本口径。
- [x] 行业哑变量和未知行业处理不变。
- [x] 记录每日矩阵规模、跳过早数和峰值内存。

测试：

- [x] 与现有 `np.linalg.lstsq` 金标准系数、标准误、t 值、R²、残差在 `1e-5` 内一致。
- [x] 共线、常数、少样本、空行业和单个有效风格行为一致。
- [x] 增量 IC 与原始 IC 对照一致。

验收：

- [x] 暴露和增量 IC 合计阶段 native 相对 Python 改善 `46.26%`。
- [x] 核加速 `2.33x`，内存不超过基线 `1.2x`。
- [x] 无法估计的日期和列仍明确跳过。

停止/回退：

- 若 LAPACK 调度或数据搬运使 C++ 无收益，保留 Python 实现并关闭该核。

历史停止记录（V118，`1e-10` 口径）：

- 尝试用内置 Eigen 3.4.0 实现逐日 OLS；真实回归矩阵上核耗时由 Python 约 `0.748s`
  降至约 `0.231s`，但系数约 `1e-14`、残差约 `1e-14` 的算法差异会改变近似并列样本的
  排名。逐日 t 统计的相对差异最高约 `6.5%`，增量 IC 均值差异约 `3e-6`，未达到
  `1e-10` 的功能一致性要求。
- 因正确性优先，已删除 Eigen、原生 OLS 和对应测试，不保留无效 C++ 路径；`daily_ols`
  继续显式回退 Python，实际 C++ 替换模块数保持 9。
- 暴露链路保留三项已验证优化：风格取数改为整数股票编码并在 pandas 中还原为类别列；
  风格 DuckDB 窗口查询由 2 线程调整为 4 线程；风格列和因子改为一次多列融合
  MAD/z-score 调用并复用同一稳定日序。
- 固定三年 `exposure + incremental_ic`：改动前 auto `14.523s`，保留优化后 auto
  `9.820s`，改善 `32.39%`；Python/native/auto payload 在 `1e-10` 内一致，native/auto
  峰值 RSS 约 `1.001x`，OLS 仍走 Python。
- 证据：`AiRecord/benchmarks/V118/p1-07-20261027/prechange/` 和
  `.../retained/`；OLS 停止证据为 `.../final/stop.md`。

V119 复核与完成记录：

- 用户确认 OLS 业务一致性门槛调整为 `1e-5`，接受此前删除的 Eigen/C++ OLS。
- 正常条件数使用列主元 Householder QR；病态矩阵按日显式回退 Python。
- 真实矩阵 OLS 热运行中位由 `0.749s` 降至 `0.322s`，加速 `2.33x`；系数、R²、残差和
  业务消费的风格 t 在 `1e-5` 内一致。
- `exposure + incremental_ic`：Python `24.375s`、native `12.935s`、auto `13.100s`；
  native 相对 Python 改善 `46.26%`。V118 auto 基线仅作补充观测，当前波动下改善 9.80%。
- Python/native/auto payload 在 `1e-5` 内一致；native/auto 峰值 RSS 为 Python 的
  `0.998x/0.992x`，auto 实际选择 native。
- 实际 C++ 替换模块数由 9 更新为 10，登记“评估逐日 OLS”；证据：
  `AiRecord/benchmarks/V119/p1-07-20261027/`。

## 6. P2：共享到模型和挖掘

P2 只复用 P1 已验收的稳定核，不新增未经评估的算法。

### P2-01 将排名和标准化核共享给模型预处理

- 状态：已完成（V120，2026-10-27）
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

- [x] 识别预处理中逐日 rank、z-score、winsorize、均值/标准差和标签排名。
- [x] 模型 rank 复用 `daily_rank_pct`；新增分位裁剪和总体标准差 z-score 核。
- [x] 每次预处理只做一次日期稳定分组，并按原行序回填。
- [x] 保持缺失指示、中性化、clip_z、sample_weight 和 label_rank 语义。
- [x] 保持 float32 输出和列顺序。
- [x] 记录预处理阶段收益。
- [x] 训练后端保持 sklearn/LightGBM，不引入自研训练核。

测试：

- [x] 预处理 DataFrame 在 python/native/auto 下逐列和报告一致。
- [x] 标签排名的并列和缺失处理一致。
- [x] 合成模型指标和预测在 `1e-10` 内一致。

验收：

- 三种实际 C++ 接入模块的 native 均不慢于 Python，门禁通过。
- 训练和预测结果不出现实质漂移。
- 未降低模型产物可复现性。

停止/回退：

- 若收益主要在训练而非预处理，停止该任务，不做无收益共享。

完成记录：

- 新增 `daily_winsorize_quantile` 和 `daily_zscore_population`，模型特征/标签 rank
  复用 `daily_rank_pct`。
- 固定 2,138,239 行、50 列、613 日开发集，每种模式三后端各运行 3 次：rank
  `26.095s → 7.951s`（`3.282x`），zscore `6.607s → 3.509s`（`1.883x`），
  rank+winsorize `58.726s → 17.095s`（`3.435x`），zscore+winsorize
  `38.841s → 12.345s`（`3.146x`）。
- python/native/auto 的 50 列 float32 列哈希与组合哈希完全一致，native/auto 均不慢于
  Python；训练和预测联动测试通过。
- 实际 C++ 替换模块数由 10 增至 13，新增“模型截面排名”“模型分位去极值”
  “模型总体标准差 z-score”。
- 证据：`AiRecord/benchmarks/V120/p2-01-20261027/`；复现：
  `.venv/bin/python -m tools.monitor.modeling_preprocess_baseline`。
- 定向测试 154 passed；普通 wheel、`pip check`、`compileall` 和
  `git diff --check` 通过。wheel SHA-256：
  `81d6868ecdf9ce0e37b974a3869ce0ac0db3782cdc4788934dee400e9452e673`。
- commit：`522cfd6`。

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

- 状态：待开始
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

- 状态：待开始
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
