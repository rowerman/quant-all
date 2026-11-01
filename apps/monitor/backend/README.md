# 因子监测后端

从项目根目录执行 `bash tools/monitor/serve.sh`，浏览器打开 http://127.0.0.1:8800。
环境缺少接口依赖时执行
`.venv/bin/pip install -r apps/monitor/backend/requirements.txt`。
服务仅使用本地产物，不自动取数或建库；必须单进程单 worker。

| 模块 | 职责与调用关系 | 修改注意事项 |
|---|---|---|
| app | FastAPI 路由、SSE；调用 service/scheduler/mining_service；提供 frontend 静态文件 | HTTP 错误须在开流前返回；流内错误使用 error 事件 |
| mining_service | 挖掘预算、运行视图、候选证据、人工审批和发布批次入口 | 只调用 mining 公开接口；批准和发布必须分开，发布不得绕过整批资格 |
| mining_search_service | V2 搜索视图、五阶段审核证据、审批和 V2 发布服务适配 | 运行级接口继续复用现有路径；V2 事实只从 `MiningV2Store` 读取 |
| mining_routes | `/api/mining/*` REST 与可续传 SSE | 断流不能取消运行；事件序号必须可重放 |
| service | 目录与参数预检；调用 quant 元数据读取及 calculate 指标 | 不在这里重写指标公式；保留只读与已有资源限制 |
| scheduler | 调度名额、背压、状态、锁等待与取消；由 app 持有 | Future 真正结束才释放容量，不共享数据库连接；原生计算阶段由 `calculate.native` 串行保护，同步和流式回测共用同一调度路径 |
| errors | MONITOR 错误和 EVAL 错误映射 | 不把内部堆栈发送页面 |
| logging_conf | 模块日志、请求编号、128MB 轮转 | 日志写入失败必须向标准错误报告 |
| preflight | 检查运行依赖与本地产物；由启动脚本调用 | 不触发下载和重建 |

所有结果仅在请求内存在；服务重启不会恢复请求。排队取消立即释放，运行取消在
当前计算阶段结束后生效；不能强制杀死线程中的 DuckDB/Numpy 计算。
单因子完整分析与最多4因子顺序对比都占一个任务槽。线程事件队列容量2，慢客户端
会使计算线程等待，避免缓存无限增长。完整请求仍可能运行数分钟。任务仍保留两个
运行线程和四个等待名额，名额在预检前预留。原生阶段互斥，两个请求可在阶段
边界交替推进；JSON转换和清理在锁内，队列背压在锁外。
因子评估与回测的 SSE 都会发送 `status` 事件，显示原生锁等待和取消等待时间；
同步回测则通过服务日志记录同一组状态和耗时。
2026-10-13 已验收：双任务完整分析重复5轮（真实调度器+SSE）与真实HTTP 1轮均完整结束、
退出码0、无SIGSEGV，取消与慢消费者也通过；原生崩溃根因仍未知，该保护只覆盖后端入口，
直接并行调用底层 calculate 不在此列。实测见
[性能记录](../../../docs/apps/monitor/performance.md)。

验证：`.venv/bin/python -m pytest -q`。
（前后端都在 `apps/monitor/` 下，该目录的 `conftest.py` 已把它加进导入路径）。
接口详解见 [接口文档](../../../docs/apps/monitor/api.md)；性能实测见
[性能记录](../../../docs/apps/monitor/performance.md)。

回测接口：

- `GET /api/backtest/options`：回测页可选策略、股票池、基准、因子与资源上限；
- `POST /api/strategies`：创建策略版本；`GET /api/strategies[/{id}]` 读取策略；
- `POST /api/providers/validate`：校验提供者参数与必需因子并返回规范化参数；
- `GET /api/backtests`：历史运行列表；`POST /api/backtests` 同步运行一次回测；
- `GET /api/backtests/compare?run_ids=`：按共同交易日对比 2～4 次成功运行；
- `GET /api/backtests/stream`：流式运行，按阶段推送进度（独立调度器，默认 1 运行 / 2 等待）；
- `DELETE /api/backtests/{run_id}`：事务删除已结束运行及全部结果，运行中返回 409；
- `GET /api/backtests/{run_id}`：运行状态与元数据；`/metrics` 绩效指标；
- `GET /api/backtests/{run_id}/daily|holding|order|rejection|liquidation|target_weight`：标准化结果。

同步 `POST /api/backtests` 会等待调度器完成，但不绕过任务容量、排队和共享原生锁；请求编号通过
`X-Request-ID` 响应头返回。回测任务状态会区分排队、预检、等待原生锁、运行、取消和终态，
取消不会强杀原生计算线程，只在阶段或调仓执行段边界生效。

回测结果保存到独立的 `data/databases/backtest.duckdb`，不会修改主行情库。内置策略在
`GET /api/strategies` 时幂等写入策略表，与用户新建的策略共用同一条运行路径。
接口详解见 [回测接口](../../../docs/trading/backtest/api.md)；浏览器验收脚本为 `tools/monitor/backtest_check.py`。

挖掘 Web 使用独立候选库 `data/databases/factor_mining.duckdb`。监测站启动时只把遗留活动
运行标为中断，不自动恢复；关闭时等待编排器退出。Worker 未显式配置时创建任务返回
`MINING-119`，不启用 Mock 回退。路由、事件流和审批边界见
[挖掘 Web](../../../docs/research/mining/web.md)。审批后发布由
`MiningPublicationService` 在后台执行，失败会恢复旧正式版本；规则见
[正式入库](../../../docs/research/mining/publication.md)。
