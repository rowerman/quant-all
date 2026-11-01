# 接口与流事件

服务仅绑定 localhost。因子评估与回测保持只读。

| 接口 | 返回/输入 |
|---|---|
| `/api/options` | 日期范围、默认参数、股票池/标签/分组枚举、指标声明、资源上限 |
| `/api/factors` | 已实现且实际落盘的因子元数据列表：定义、公式、步骤与覆盖率 |
| `/api/factors/{name}` | 指定因子的元数据；未知或未落盘返回404 |
| `/api/evaluate/stream` | 单因子SSE，必传factor |
| `/api/compare/stream` | 同参数顺序对比SSE，必传factors，以逗号分隔的2～4个不同名称 |
| `/api/mining/options` | 挖掘默认预算、限制、冻结窗口、默认审核人和 Worker 配置状态 |
| `/api/mining/runs` | `GET` 列表；`POST` 提交研究假设、幂等键和可选预算 |
| `/api/mining/research-runs` | `POST` 从固定资料根目录下的相对 PDF 文件或目录创建 V2 运行 |
| `/api/mining/runs/{run_id}` | 运行、预算进度和候选摘要 |
| `/api/mining/runs/{run_id}/events` | 按 `after` 序号读取不可变事件 |
| `/api/mining/runs/{run_id}/stream` | 从 `after` 回放并持续推送 SSE 事件 |
| `/api/mining/runs/{run_id}/cancel` | 在安全阶段边界取消运行 |
| `/api/mining/runs/{run_id}/resume` | 仅恢复 `interrupted` 运行 |
| `/api/mining/runs/{run_id}/candidates/{candidate_id}` | 候选定义、四阶段证据和审批资格 |
| `/api/mining/runs/{run_id}/candidates/{candidate_id}/decision` | 记录人工批准或拒绝 |
| `/api/mining/runs/{run_id}/publication-batches` | `POST` 整批构建并发布全部已批准候选；`GET` 列出批次 |
| `/api/mining/runs/{run_id}/publication-batches/{batch_id}` | 读取发布状态、候选状态和产物哈希 |

流查询公共参数：start/end（YYYY-MM-DD，默认库尾最近3年）、universe（all_a或
index:指数代码）、horizon（1/3/5/10/15/20）、price（close/open）、quantiles（5/10/20）、
winsorize（true/false）、mode（full/ic，默认full）、stages（逗号分隔阶段名，可省略）、
metric_options（指标key到参数对象的JSON）。重复参数、未知参数与非法值报400。

行业默认off。开启静态行业时应同时设置：
`{"exposure":{"industry":"stock_basic"},"incremental_ic":{"industry":"stock_basic"}}`。
相关性参数 `{"correlation":{"top_n":10}}`，范围1～100。
轻量模式仅可选overview/ic_series/ic_dist，返回ic_stats/ic_series/ic_dist，不含覆盖率。

每条SSE为 `event: <stage>` 和 `data: <JSON>`，空行分隔，非有限数值为null。
首先queued；任务状态变化时插入status，包含排队、原生锁等待、运行、取消与耗时；
接着每个因子的meta与选择的阶段，再factor_done，所有因子最后一次done。
结果保留原计算包字段，补充factor与request_id。阶段内同stage指标放metrics映射，
elapsed_ms为该阶段耗时；轻量factor_done还给ic_load_elapsed_ms。
meta发生在完整面板建好后、轻量IC取数前；等待期间每10秒心跳注释。

HTTP错误格式：`{code,message,request_id}`，参数400、因子404、队列满429、
缺数据503、内部失败500。已开流的错误以error事件发送相同结构，并结束流。
收到done/error后关闭连接，禁止自动重连。连接取消会在当前阶段结束后停止计算；
等待共享原生锁时可立即取消，不会进入计算阶段。
服务同时允许两个运行请求与四个等待请求；原生计算阶段互斥，两个请求在阶段边界
交替推进，慢消费者的背压等待不占用计算锁。

```bash
curl -N 'http://127.0.0.1:8800/api/evaluate/stream?factor=MOM_20&mode=ic&start=2000-01-04&end=2026-09-28'
curl -N 'http://127.0.0.1:8800/api/compare/stream?factors=MOM_20,MOM_60&mode=ic'
```

## 挖掘接口

`POST /api/mining/runs` 请求体：

```json
{
  "hypothesis": "研究假设，至少 8 个字符",
  "idempotency_key": "web-...",
  "constraints": {
    "max_rounds": 3,
    "max_candidates": 12,
    "full_eval_top_k": 4,
    "max_llm_calls": 30
  }
}
```

`constraints` 可省略；服务端把部分覆盖合并到默认预算后校验并冻结。固定股票池、标签和
价格通道不接受页面修改。相同幂等键和相同请求返回原运行，内容不同返回 `MINING-105`。

挖掘 SSE 使用 `event: mining`、`data: <事件 JSON>` 和递增 `id`。事件 JSON 包含
`sequence`、`stage`、`event_type`、`payload` 与 `created_at`。客户端可从
`?after=<最后序号>` 重连；运行进入待审核、中断或终态后发送 `event: done`。断开挖掘
事件流不会取消运行。

`decision` 请求体：

```json
{
  "decision": "approved",
  "reviewer": "本机用户名",
  "note": "审核依据",
  "candidate_content_hash": "<候选详情返回值>",
  "evidence_hash": "<候选详情返回值>"
}
```

批准和拒绝都只写候选库。候选内容或证据变化返回 `MINING-108`，数据、目录或正式因子
快照变化返回 `MINING-120`。

正式发布请求体：

```json
{
  "candidate_ids": ["<运行内全部已批准候选>"],
  "expected_context_hash": "<运行上下文哈希>"
}
```

受理返回 `202`，后台完成 `mined` 全量构建、校验和原子提交。漏选已批准候选、并发批次
或快照失效返回 `MINING-121`；全量校验失败返回 `MINING-122`；提交、回滚或恢复失败
返回 `MINING-123`。页面轮询批次接口，不把断网当作取消发布。

设置 `MINING_SEARCH_V2_ENABLED=true` 时，`POST /api/mining/runs` 仍使用同一路径，
但请求体中的预算字段改为 V2 `budget`，运行 ID 以 `searchrun_` 开头。
`GET /api/mining/options` 会增加 `search_v2`，列出四段窗口、搜索预算、控制策略和
分析工具目录。V2 的创建、列表、详情、事件、SSE、取消和恢复复用上述运行接口。

资料驱动请求体：

```json
{
  "material_path": "2026-10",
  "idempotency_key": "web-research-...",
  "budget": {}
}
```

`material_path` 只能是 `data/source/reports` 下的相对文件或目录，`.` 表示根目录。
服务端在创建时冻结 PDF 指纹；后台解析、抽取、转换和自动准入后进入 V2 状态机。
`GET /api/mining/options` 的 `search_v2.research` 返回资料根目录标签、文件/草稿上限和
提示版本。运行详情的 `research.summary` 返回脱敏资料与筛选摘要；候选详情的
`research_source` 返回页码、偏移和短引用，不返回完整报告正文。

V2 运行复用候选详情、审批和发布批次接口。`GET .../candidates/{candidate_id}` 返回
五阶段证据、评分卡、审批资格、阻塞原因和最近批次；`POST .../decision` 使用与 V1
相同的决定、审核人、备注和两个证据哈希字段；`POST .../publication-batches` 仍要求
提交本运行全部 `approved/build_failed` 候选和 V2 顶层 `context_hash`。

`review_ready` 只表示候选已具备人工审核条件，不表示已经批准或发布。批准后候选进入
`approved/rejected`；整批发布成功后进入 `published`，失败进入 `build_failed`，
可由页面重试。
