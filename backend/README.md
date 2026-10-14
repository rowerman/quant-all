# 因子监测后端

从项目根目录执行 `bash tools/monitor/serve.sh`，浏览器打开 http://127.0.0.1:8800。
环境缺少接口依赖时执行 `.venv/bin/pip install -r backend/requirements.txt`。
服务仅使用本地产物，不自动取数或建库；必须单进程单 worker。

| 模块 | 职责与调用关系 | 修改注意事项 |
|---|---|---|
| app | FastAPI 路由、SSE；调用 service/scheduler；提供 frontend 静态文件 | HTTP 错误须在开流前返回；流内错误使用 error 事件 |
| service | 目录与参数预检；调用 quant 元数据读取及 calculate 指标 | 不在这里重写指标公式；保留只读与已有资源限制 |
| scheduler | 两线程、四等待名额、背压与取消；由 app 持有 | Future 真正结束才释放容量，不共享数据库连接；原生计算阶段由 `calculate.native` 串行保护 |
| errors | MONITOR 错误和 EVAL 错误映射 | 不把内部堆栈发送页面 |
| logging_conf | 模块日志、请求编号、128MB 轮转 | 日志写入失败必须向标准错误报告 |
| preflight | 检查运行依赖与本地产物；由启动脚本调用 | 不触发下载和重建 |

所有结果仅在请求内存在；服务重启不会恢复请求。排队取消立即释放，运行取消在
当前计算阶段结束后生效；不能强制杀死线程中的 DuckDB/Numpy 计算。
单因子完整分析与最多4因子顺序对比都占一个任务槽。线程事件队列容量2，慢客户端
会使计算线程等待，避免缓存无限增长。完整请求仍可能运行数分钟。任务仍保留两个
运行线程和四个等待名额，名额在预检前预留。原生阶段互斥，两个请求可在阶段
边界交替推进；JSON转换和清理在锁内，队列背压在锁外。
2026-10-13 已验收：双任务完整分析重复5轮（真实调度器+SSE）与真实HTTP 1轮均完整结束、
退出码0、无SIGSEGV，取消与慢消费者也通过；原生崩溃根因仍未知，该保护只覆盖后端入口，
直接并行调用底层calculate不在此列。实测见 docs/monitor/performance.md。

验证：`.venv/bin/python -m pytest backend/tests calculate/tests tests -q`。
接口详解见 `docs/monitor/api.md`；性能实测见 `docs/monitor/performance.md`。

回测接口：

- `GET /api/backtest/options`：回测页可选策略、股票池、因子与资源上限；
- `POST /api/strategies`：创建策略版本；`GET /api/strategies[/{id}]` 读取策略；
- `GET /api/backtests`：历史运行列表；`POST /api/backtests` 同步运行一次回测；
- `GET /api/backtests/stream`：流式运行，按阶段推送进度（独立调度器，默认 1 运行 / 2 等待）；
- `GET /api/backtests/{run_id}`：运行状态与元数据；`/metrics` 绩效指标；
- `GET /api/backtests/{run_id}/daily|holding|order|rejection|target_weight`：标准化结果。

回测结果保存到独立的 `db/backtest.duckdb`，不会修改主行情库。内置策略在
`GET /api/strategies` 时幂等写入策略表，与用户新建的策略共用同一条运行路径。
接口详解见 `docs/backtest/api.md`；浏览器验收脚本为 `tools/monitor/backtest_check.py`。
