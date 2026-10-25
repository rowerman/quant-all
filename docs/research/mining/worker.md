# 独立 RD-Agent Worker

## 职责与隔离

Worker 固定使用 Python 3.11 和 RD-Agent 1.0.0，只通过 DeepSeek 兼容的 LiteLLM 后端
生成受控候选 JSON。它不读取行情数据库、候选库或正式因子目录，也不运行 RD-Agent 的
Qlib coder、workspace 或任何模型生成代码。

主服务客户端位于 `mining.worker_client`，共享协议与 Worker 服务位于独立
`mining_worker` 包。Worker 返回的表达式全部重新经过 S3；主服务不信任 Worker 给出的
候选名称、依赖、回看窗口、方向哈希或候选 ID。

## 作业协议

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 返回 Worker 版本、模型与 Key 配置状态，不返回 Key |
| POST | `/v1/jobs` | 提交一轮有界生成请求 |
| GET | `/v1/jobs/<job_id>` | 查询作业状态 |
| POST | `/v1/jobs/<job_id>/cancel` | 取消排队或运行中的作业 |

作业状态为 `queued/running/succeeded/failed/cancelled/interrupted`。同一 `request_id`
和相同内容幂等，内容变化返回 `MINING-105`；Worker 同时只运行一个作业，占用时返回
`MINING-102`。

作业记录保存在 `data/mining_worker/jobs.sqlite`。Worker 重启时只把旧的
`queued/running` 作业标记为 `interrupted`，不会自动重放，以免重复消耗 LLM 预算。
调用方需要自行决定是否以新请求重试。

Worker 的创建、开始、完成、失败和取消事件写入
`logs/research/mining/mining_worker_service/`，只包含作业 ID、轮次、候选数量和错误码，
不记录研究假设、表达式、Prompt、模型响应或 Key。

## 启动

主服务首次需要 Token 时会在 `data/mining_worker/token` 生成 384 位随机值并设置
`0600` 权限。容器通过 Compose secret 读取同一文件。真实 Key 只通过
`FACTOR_DEEPSEEK_API` 映射到 Worker 容器的 `DEEPSEEK_API_KEY`，不会传给主服务。

```bash
.venv/bin/python -c \
  'from common.paths import DATA_DIR; from mining.worker_client import ensure_worker_token; ensure_worker_token(DATA_DIR / "mining_worker" / "token")'

docker compose --env-file .env -f compose.mining.yaml up -d --build
curl http://127.0.0.1:8790/health
```

生产配置不提供静默 Mock。Mock 只允许在测试中通过 `MockWorkerClient` 显式注入。

## 超时与取消

固定配置位于 `configs/mining/worker.json`。默认 Worker 生成时限 300 秒，主服务等待
上限 330 秒。每轮生成在独立进程组中执行；超时或取消会终止该进程组。调用方先到达
等待上限时会主动请求 Worker 取消作业。

## 修改注意

1. 修改请求字段、作业状态或响应结构时必须升级 Worker schema 并同步主服务测试。
2. 不要把行情库、候选库、正式因子目录或 `.env` 挂载进 Worker 容器。
3. 不得恢复 Mock 自动回退，也不得在 Worker 内执行候选 SQL 或 Python。
4. 升级 RD-Agent、Python 或模型时必须同步固定配置、依赖锁、镜像和验收证据。
