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
进程数、文件描述符、运行时间和输出大小。容器只挂载只读快照和独立工作区，不挂载正式
因子目录、候选库、主行情库、项目根目录、Docker socket、Token 或模型 Key。

命令入口只允许 `python`/`python3`，不经过 shell。超时后按容器名执行强制终止和清理；
路径必须解析到政策允许的快照根或工作区根，符号链接或 `..` 越界会被拒绝。

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
相同上下文重复构建会校验文件并复用，内容漂移时拒绝覆盖。

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

Worker v4 使用 `mining_worker_code_v1`，由固定 RD-Agent 模型生成 Qlib 研究脚本；脚本
只作为未信任提案返回主服务。Host 研究循环最多执行 3 轮，每轮把代码写入独立工作区，
由断网容器执行，再读取严格的 `rank_ic / icir / coverage / valid_days` 指标，并把上一轮
失败或指标反馈给下一轮。

代码模型不直接运行在 Worker 容器。沙箱只挂载冻结快照和工作区，不能访问主数据库、正式
候选、正式因子、Token 或 Docker socket。沙箱指标只用于产生研究提案，不是正式质量结论。

## 产物与人工提升

每次研究把所有代码提案、运行状态、stdout/stderr、指标和错误写入
`data/mining_sandbox/work/<sandbox_run_id>/`。`manifest.json` 记录全部尝试、代码哈希、
镜像 ID、Python/Qlib 版本、依赖锁哈希、模型和提示版本，以及每个产物的 SHA-256。完成后
目录封存为只读；`verify_artifact_tree()` 可检测文件缺失、大小变化和哈希篡改。

`record_manual_promotion()` 只记录审核人、原因、来源沙箱运行、产物清单哈希和目标
DSL/算子引用，并明确标记 `formal_candidate_created=false`。它不调用候选存储、审批或
发布接口。人工转写后的正式候选必须在新搜索上下文中重新经过 DSL、门禁、选择期、完整
评估和人工审批。

稳定错误码：

| 错误码 | 含义 |
|---|---|
| `MINING-133` | 沙箱政策非法 |
| `MINING-134` | 容器或研究循环执行失败 |
| `MINING-135` | Qlib 快照生成或校验失败 |
| `MINING-136` | 路径、挂载或隔离边界破坏 |
| `MINING-137` | 沙箱产物或人工提升记录非法 |
