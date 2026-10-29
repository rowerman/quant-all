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
