# 隔离代码研究沙箱

## 职责

沙箱只用于产生代码因子研究提案。它不能批准、发布或直接写入正式候选与因子目录。
沙箱默认关闭，只有显式启用 `mining_sandbox_v1` 政策后才能启动。

```text
Host 研究协调器
  -> DockerSandboxRunner
       -> 无网络、非 root、只读根容器
       -> /snapshot 只读 Qlib 开发期快照
       -> /workspace 独立临时工作区
  -> 只读产物清单和人工审核
```

## 威胁边界

沙箱默认阻断网络，根文件系统只读，删除全部 capability，禁止提权，并限制 CPU、内存、
进程数、文件描述符、单文件大小、工作区容量、运行时间和输出大小。实际执行固定到已
解析的不可变 Docker 镜像 ID，避免运行期间标签漂移。容器只挂载只读快照和独立工作区，
不挂载正式因子目录、候选库、主行情库、项目根目录、Docker socket、Token 或模型 Key。

命令入口只允许 `python`/`python3`，不经过 shell。超时后按容器名执行强制终止和清理；
宿主侧持续检查标准输出、错误输出和工作区总容量，超限后终止容器。路径必须解析到政策
允许的快照根或工作区根，符号链接或 `..` 越界会被拒绝。容器退出后还会比较执行前后
`proposal.py` 哈希，研究代码不得修改或删除自身。

## Qlib 快照

快照从 `md.calendar`、`md.daily_hfq` 和 `md.fwd_return` 导出，只使用冻结上下文的
开发期 `2012-01-01～2019-12-31`，股票池为全 A。字段固定为后复权 OHLC、VWAP、
成交量、成交额、日收益、换手率、流通市值、总市值和 `fwd_ret_open_5`。

输出采用 Qlib 文件格式：

```text
calendars/day.txt
instruments/all.txt
features/<股票小写代码>/<字段>.day.bin
manifest.json
```

清单记录上下文哈希、数据版本、日期、股票池、字段、行数、每个文件的大小和 SHA-256，
以及日历、股票池和总清单哈希。导出前检查磁盘容量；发布时原子替换目录并封存为只读。
相同上下文重复构建会校验文件并复用，内容漂移时拒绝覆盖。每次研究启动前调用
`verify_qlib_snapshot()` 校验清单、完整文件集合、普通文件类型和全部文件哈希，额外
文件、缺失文件或符号链接都会失败。

构建命令：

```bash
.venv/bin/python tools/mining/build_sandbox_snapshot.py
```

镜像：

```bash
docker build -t quant-mining-sandbox:s5 \
  -f research/mining_sandbox/Dockerfile .
```

运行配置位于 `configs/mining/sandbox.json`，Compose 安全参数见
`compose.mining-sandbox.yaml`。

## 代码研究闭环

Worker v4 使用 `mining_worker_code_v2`，由固定 RD-Agent 模型生成 Qlib 研究脚本；脚本
只作为未信任提案返回主服务。Host 研究循环最多执行 3 轮，每轮把代码写入独立工作区，
先做静态编译和代码限制检查，再由断网容器执行。提示版本
`mining_qlib_code_prompt_v3` 要求脚本同时写出 `factor.parquet` 和
`rank_ic / icir / coverage / valid_days` 指标，并依据上一轮结构化反馈和当前运行知识
证据修正下一轮提案。

代码模型不直接运行在 Worker 容器。沙箱只挂载冻结快照和工作区，不能访问主数据库、正式
候选、正式因子、Token 或 Docker socket。沙箱指标只用于产生研究提案，不是正式质量结论。

## 确定性评估

`mining_sandbox.evaluator` 不执行提案代码，只按冻结输入验证输出；代码策略和指标契约
分别由 `mining_sandbox.code_policy`、`mining_sandbox.metrics` 维护：

- `factor.parquet` 必须且只能包含 `date`、`instrument`、`value`。
- 日期必须在冻结交易日历内，股票代码必须在冻结索引内，且日期位于该股票有效区间。
- `(date, instrument)` 不得重复，`value` 必须全部为有限数值；缺失因子直接省略行。
- `coverage` 由有限因子行数除以快照 `row_count` 重算，`valid_days` 由输出中的不同
  交易日重算，并与 `metrics.json` 完全一致。
- 评估结果写入 `evaluation.json`，记录快照哈希、因子文件哈希、行数、股票数、有效
  交易日、覆盖率和稳定 `evaluation_hash`，不包含宿主路径或当前时间。

执行前静态检查只允许 `json`、`math`、`statistics`、`pathlib`、`numpy`、`pandas`、
`scipy` 和 `qlib` 等研究导入，拒绝 shell、网络、进程、环境变量、动态执行、动态导入、
内部属性和固定路径之外的绝对路径。该检查负责尽早反馈；容器断网、只读挂载、非 root
和资源限制仍是强制安全边界。

## 结构化反馈

每个失败尝试除原 `error_code` 外，还保存 `mining_sandbox_feedback_v1` 反馈对象：

```json
{
  "schema_version": "mining_sandbox_feedback_v1",
  "error_code": "MINING-140",
  "category": "output_format",
  "stage": "output",
  "summary": "沙箱指标字段不符合契约",
  "repair_hint": "metrics.json 必须且只能包含四个研究指标",
  "retryable": true
}
```

分类固定为 `compile`、`execution`、`output_format`、`index`、`coverage`、`numeric`、
`code_policy` 和 `resource`。反馈摘要和修复提示最多 500 字符，不写原始代码、完整异常、
Prompt 或研究假设，并遮蔽 API Key、Bearer Token、已知密钥和宿主机绝对路径。

业务日志位于 `logs/research/mining_sandbox/mining_sandbox_research/`，只记录运行 ID、
轮次、代码哈希、错误码、分类、阶段、`retryable` 和反馈哈希。原始 `stdout.log` 与
`stderr.log` 仍属于受访问控制的沙箱研究产物，不是业务事件日志；反馈对象本身不复制这些
文件内容。

## 运行内知识库

`mining_sandbox.knowledge` 只读取当前 `sandbox_run_id` 的尝试和产物，生成
`knowledge.json`。知识条目分为三类：

- `failure`：按轮次记录错误码、分类、脱敏摘要、修复提示、代码哈希和来源产物。
- `success`：记录成功代码、指标、评估哈希及代码/因子来源。
- `repair`：失败后首次成功时关联失败轮次与成功轮次，给出最近错误和成功代码哈希。

索引绑定 `sandbox_run_id`、完整 `request_hash` 和不含运行 ID 的 `task_hash`。只有
`task_hash` 相同的研究任务可以检索当前运行的知识，不读取全局数据库或其他运行。
条目、来源和索引分别保存版本、稳定哈希和文件 SHA-256；来源路径必须是当前运行内的
相对路径。知识文件只保存脱敏摘要和来源引用，不复制原始代码或完整异常。

研究循环每完成一个尝试就原子刷新 `knowledge.json`，因此下一轮可以在运行封存前读取
已完成历史。恢复时从 `run-state.json` 重建索引，最终知识文件进入 `manifest.json`
并随产物目录封存。旧运行没有知识文件时，可用原研究请求只读派生，不会修改已封存
产物。

## Worker 知识证据

`mining_sandbox.knowledge_evidence` 从当前运行的 `SandboxKnowledgeBase` 生成
`mining_worker_code_knowledge_v1` 证据。首轮不携带历史知识；后续轮次必须携带同一任务
索引、检索结果及上一尝试反馈。每条证据保留知识条目哈希、代码/评估哈希、轮次、固定
四项指标和脱敏摘要，最多 4 条。

证据不包含 `sandbox_run_id`、来源产物路径、原始代码、stdout/stderr 或完整知识来源。
每个失败反馈还包含稳定 `feedback_hash`。研究循环把知识索引哈希、检索哈希和实际 Worker
请求哈希写入尝试记录，随后随整个运行封存。协议、Worker 服务和提示层都拒绝未知字段、
跨任务证据、当前或未来轮次证据以及非法指标。

## 产物与人工提升

每次研究把所有代码提案、运行状态、stdout/stderr、因子明细、评估结果、指标和错误写入
`data/mining_sandbox/work/<sandbox_run_id>/`，并写入 `knowledge.json`。`manifest.json`
记录全部尝试、代码哈希、镜像 ID、Python/Qlib 版本、依赖锁哈希、模型和提示版本，以及
每个产物的 SHA-256。运行中自改代码失败会保存原始提案和实际执行哈希；工作区符号链接
或其他特殊文件会被拒绝。完成后目录封存为只读；`verify_artifact_tree()` 要求实际文件
集合与清单完全一致，并检测文件缺失、新增、符号链接、大小变化和哈希篡改。

`record_manual_promotion()` 只记录审核人、原因、来源沙箱运行、产物清单哈希和目标
DSL/算子引用，并明确标记 `formal_candidate_created=false`。它不调用候选存储、审批或
发布接口。人工转写后的正式候选必须在新搜索上下文中重新经过 DSL、门禁、选择期、完整
评估和人工审批。

## 安全验收

P1-05 的自动验收覆盖以下故障和攻击路径：

| 场景 | 预期结果 |
|---|---|
| Worker 或 Docker 启动故障 | 返回 `MINING-134`，不把尝试写成成功，新运行可独立恢复 |
| 网络、写根目录、写快照 | 真实锁定镜像中的操作全部失败，工作区仍可写 |
| 超时、OOM、输出或工作区超限 | 终止并清理容器，尝试返回 `MINING-145` |
| 运行中修改、删除提案代码 | 保存原始提案与执行哈希，尝试返回 `MINING-144` |
| 工作区符号链接或特殊文件 | 删除不安全项并记录相对路径，尝试返回 `MINING-144` |
| 跨运行知识、来源哈希或封存知识污染 | 返回 `MINING-146`，不进入 Worker 下一轮证据 |
| 快照或产物新增、缺失、链接或哈希变化 | 返回 `MINING-135` 或 `MINING-137` |

稳定错误码：

| 错误码 | 含义 |
|---|---|
| `MINING-133` | 沙箱政策非法 |
| `MINING-134` | 容器或研究循环执行失败 |
| `MINING-135` | Qlib 快照生成或校验失败 |
| `MINING-136` | 路径、挂载或隔离边界破坏 |
| `MINING-137` | 沙箱产物或人工提升记录非法 |
| `MINING-138` | 提案代码无法通过静态编译 |
| `MINING-139` | 沙箱进程非零退出 |
| `MINING-140` | 指标或输出格式不符合契约 |
| `MINING-141` | 交易日或股票索引没有有效观察 |
| `MINING-142` | 覆盖率越界 |
| `MINING-143` | 指标不是有限数值或 RankIC 越界 |
| `MINING-144` | 代码违反沙箱研究限制 |
| `MINING-145` | 沙箱资源限制或执行超时 |
| `MINING-146` | 运行内知识索引、来源或任务绑定非法 |
