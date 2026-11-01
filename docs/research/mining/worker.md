# 独立 RD-Agent Worker

## 职责与隔离

Worker 固定使用 Python 3.11 和 RD-Agent 1.0.0，通过 DeepSeek 兼容的 LiteLLM 后端
生成 V1 候选表达式、V2 搜索建议、V3 工具补充响应和 V4 未信任 Qlib 研究代码。
它不读取行情数据库、候选库或正式因子目录，也不在 Worker 容器内执行模型生成代码。
代码只由宿主侧沙箱在断网容器中运行。

主服务客户端位于 `mining.worker_client`，共享协议与 Worker 服务位于独立
`mining_worker` 包。Worker 返回的表达式全部重新经过 S3；主服务不信任 Worker 给出的
候选名称、依赖、回看窗口、方向哈希或候选 ID。

## 作业协议

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 返回 Worker 版本、模型与 Key 配置状态，不返回 Key |
| POST | `/v1/jobs` | 提交一轮 V1 有界生成请求 |
| POST | `/v2/jobs` | 提交一轮 V2 证据驱动搜索请求 |
| POST | `/v3/jobs` | 提交一轮工具感知搜索请求，返回工具批次或最终决策 |
| POST | `/v4/jobs` | 提交一轮隔离沙箱代码提案请求 |
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

## V2 搜索协议

`mining_worker_v2` 在 V1 候选生成之外增加当前假设、候选谱系、开发期证据、冻结预算、
允许动作和下一假设建议。响应只允许以下两种形态：

- `refine/branch`：必须包含下一假设和 1～4 个尚未信任的 DSL 候选草稿。
- `stop/promote`：不得包含下一假设或候选，最终动作仍由主服务控制器决定。

请求和响应都拒绝未知字段、非法动作、超限内容以及选择期、最终研究期、测试段字段和
日期。主服务收到草稿后重新解析、校验、规范化并计算候选名称、依赖、语义哈希和 ID。
V2 与 V1 共用同一个单槽位作业库；Worker 同时只执行一个作业。

## V3 工具协议

`mining_worker_v3` 在 V2 请求外增加固定工具目录和工具结果。首轮只能返回最多 4 个
工具调用；主服务在开发期内执行后，通过同一 turn 的 `finalize` 请求返回脱敏结果，
Worker 随后必须给出最终 `refine/branch/stop/promote` 决策。工具失败会阻断当前 turn，
不会转换成空结果或扩大查询范围。

## V4 代码研究协议

`mining_worker_code_v2` 接收开发期快照标识、任务哈希、上一轮代码和完整结构化反馈，
以及经过白名单转换的当前运行知识证据，返回最多 3 轮迭代中的一轮未信任 Python 代码
提案。知识证据只包含失败、成功和修复条目的轮次、脱敏摘要、固定指标、反馈和稳定哈希，
不包含原始代码、stdout/stderr、宿主路径或来源文件路径。宿主服务把代码写入独立工作区，
由 `DockerSandboxRunner` 执行。提示版本 `mining_qlib_code_prompt_v3` 要求脚本写出
`factor.parquet` 和 `rank_ic / icir / coverage / valid_days`，宿主再用
`mining_sandbox.evaluator` 校验冻结索引、覆盖率、非有限值和代码限制。该流程是项目
自定义的 CoSTEER 兼容循环，不是直接运行上游 `FactorCoSTEER`，代码产物不能直接进入
候选、审批或发布。

研究循环在每轮提案前读取当前运行已验证的 `knowledge.json`，校验任务、索引和检索哈希
后注入请求。第一轮不带历史知识，后续轮次必须带上一尝试和证据。恢复复用已经持久化的
代码与请求哈希，不会重复调用已完成的付费步骤。`mining_worker_code_v1` 请求仍可只读
解析，以兼容旧作业库。

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

生产配置不提供静默 Mock。Mock 只允许在测试中通过 `MockWorkerClient` 或
`MockSearchWorkerClient` 显式注入。

## 超时与取消

固定配置位于 `configs/mining/worker.json`。默认 Worker 生成时限 300 秒，主服务等待
上限 330 秒。每轮生成在独立进程组中执行；超时或取消会终止该进程组。调用方先到达
等待上限时会主动请求 Worker 取消作业。

## 修改注意

1. 修改请求字段、作业状态或响应结构时必须升级 Worker schema 并同步主服务测试。
2. 不要把行情库、候选库、正式因子目录或 `.env` 挂载进 Worker 容器。
3. 不得恢复 Mock 自动回退，也不得在 Worker 内执行候选 SQL；模型生成代码只能进入
   隔离沙箱。
4. 升级 RD-Agent、Python 或模型时必须同步固定配置、依赖锁、镜像和验收证据。
