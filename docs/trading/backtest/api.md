# 回测接口与流事件

服务仅绑定 localhost。策略与回测结果保存在独立的 `data/databases/backtest.duckdb`，不影响主行情库。

| 接口 | 返回/输入 |
|---|---|
| `GET /api/backtest/options` | 可选策略（含内置）、目标权重提供者、股票池、因子清单、默认参数（含无风险利率）、成本与冲击口径、资源上限 |
| `GET /api/strategies` | 策略列表，内置策略与用户策略在同一个列表里 |
| `GET /api/strategies/{strategy_id}?version=` | 指定策略版本的配置与哈希 |
| `POST /api/strategies` | 新建策略第一版，body 为 `StrategySpec`（可带 `description`）；支持因子 Top-N 或 `signal.provider` 提供者配置，风险项可选单股、时点申万一级行业与单边换手上限，成交额参与率默认 10% |
| `POST /api/providers/validate` | 校验提供者名称、版本、参数和必需因子，返回规范化参数与提供者元数据 |
| `GET /api/backtests?limit=` | 历史运行列表，含状态、区间、股票池、基准、`snapshot_id`、累计收益/年化/最大回撤与耗时；股票池与基准落 `bt.run` 的列，可直接 SQL 筛 |
| `POST /api/backtests` | 同步跑完一次回测，body 为运行请求 |
| `GET /api/backtests/stream` | 流式回测，查询参数即运行请求参数 |
| `GET /api/backtests/compare?run_ids=` | 对比 2～4 次成功运行；使用共同交易日，返回归一净值与累计超额曲线 |
| `GET /api/backtests/{run_id}` | 运行状态、`snapshot_id`、元数据（含 `snapshot`、`risk`、`calibration`）与绩效指标 |
| `DELETE /api/backtests/{run_id}` | 物理删除一次已结束运行及其全部结果表记录；运行中返回 409 |
| `GET /api/backtests/{run_id}/metrics` | 按已落库日结果重算的绩效指标，并带成本、换手和行业暴露归因 |
| `GET /api/backtests/{run_id}/daily\|holding\|order\|rejection\|liquidation\|target_weight\|allocation\|industry_exposure` | 标准化结果表；`order.forced_exit` 标记退市强制退出，`industry_exposure` 为逐日行业、现金和未分类权重 |

`daily` 同时返回 `rights_subscription_cash`、`rights_subscribed_shares` 与
`rights_unsubscribed_shares`；公司行动明细通过运行详情中的结果表接口读取。

运行请求参数：`strategy_id`（必填）、`version`、`start`、`end`、`init_cash`、`risk_free_rate`，
可选运行级覆盖 `universe`（如 `index:000905.SH`）与 `benchmark`。
`risk_free_rate` 是年化小数，默认 `0.02`，只影响夏普、Sortino、Alpha 等指标。
参数非法在开流前返回 400；`version` 必须存在，否则 400。

同步 `POST /api/backtests` 与流式接口共用回测调度器、任务容量和原生计算锁。
同步接口会等待任务完成并保持原有响应格式；请求编号通过 `X-Request-ID` 响应头返回。
任务进入队列、等待原生锁或运行阶段时，不会绕过调度器直接执行。

运行对比按 `run_ids` 给出的顺序返回结果，只接受 2～4 个互不重复且状态为 `success` 的运行。
后端取共同交易日，策略净值在共同首日归一为 1，累计超额由已落库的日度 `excess_return`
复合计算并在共同首日归零。原始区间、股票池和基准可以不同，响应会逐项返回供页面标明。

删除没有回收站：`success`、`failed`、`cancelled` 均会在同一事务中清理 `bt.run` 和所有
`bt.*` 结果行；运行中记录返回 409，缺失记录返回 404。

`GET /api/backtest/options` 的 `benchmarks` 当前提供沪深300/中证500的价格与全收益指数。
默认 `H00300.CSI`；使用中证500成分池时可覆盖为 `H00905.CSI`。每个基准返回
`start`、`end`、`default_end`，`defaults.end` 取默认基准与行情的共同最后交易日。
开跑前会检查请求区间内的基准行情；超出覆盖范围返回 `BT-201` 并列出开始日下限、
结束日上限、换基准和显式关闭基准等选项。
运行请求的 `benchmark` 不传时沿用策略版本；传 `null` 或空字符串表示本次关闭基准。
`limits` 同时返回单块自动切换阈值 `panel_cells`、流式块预算 `chunk_cells` 与块数上限 `max_chunks`。
`costs` 返回可编辑的佣金、最低佣金、滑点、冲击模型/系数和红利税模式，并附带历史印花税、
过户费、红利税分段摘要；印花税和过户费不再作为前端可编辑字段。
新策略默认 `impact_model=sqrt`、系数 0.5、20 日波动、单笔冲击上限 5%；
旧策略 JSON 缺少冲击字段时按关闭解释。

`providers` 返回 `trading/backtest/strategies/` 中已注册的提供者，字段包括 `name`、`version`、
`description`、`required_factors`、`uses_position_state`、`module`、`code_sha256` 和
`param_schema`。`param_schema` 是扁平字段说明，包含 `name`、`label`、`type`、`default`、
`min`、`max`、`step`、`options` 和 `description`；类型支持 `integer`、`number`、
`percent`、`boolean`、`string`、`enum`，其中 `percent` 的后端值仍是小数。
提供者策略的 `signal` 必须使用：

```json
{
  "provider": {"name": "momentum_timing", "version": "1", "params": {"top_n": 30}},
  "factors": ["MOM_20"],
  "weights": []
}
```

同时 `portfolio.selection` 必须为 `provider`，`rebalance.frequency` 必须为 `custom`。
创建策略时未知提供者、版本不匹配、缺少必需因子或参数非法返回 `BT-101`；
提供者返回权重形状不一致、负权重、部分缺失、合计超过 1 或给不可买入股票正权重返回 `BT-201`。
前端新建策略页可编辑提供者字段并直接调用本接口创建；所有创建调用都走同一条版本存储路径。

`POST /api/providers/validate` 的请求体为：

```json
{
  "provider": {"name": "momentum_timing", "version": "1", "params": {}},
  "factors": ["MOM_20"]
}
```

成功返回 `{provider: {...}, params: {...}}`；`params` 已由提供者 `validate_params`
补全默认值并完成类型与范围校验。参数错误、缺少必需因子或版本不匹配返回 HTTP 400 / `BT-101`。

`provider.params` 必须是可 JSON 序列化的对象，不能含 `NaN`；`version` 必须与当前注册版本一致，
代码改动应提升版本。提供者元数据包含 `uses_position_state`；为 true 时 `build` 额外接收
执行层生成的逐日 `ProviderState`，为 false 时仍只接收行情、因子和 eligible 矩阵。
状态提供者按“上一遍最终账本状态 → 新目标”重复执行，目标权重稳定即收敛，最多 20 遍；
未收敛的运行标记失败。运行元数据记录模块和源码 SHA-256，完整源码与数据分项摘要写入
`metadata.snapshot`。创建策略时策略版本与 `bt.provider_revision` 在同一事务中写入，
列表只会看到完整提交的策略；同一提供者名称与版本出现不同源码时返回 `BT-701`，策略写入回滚。
跨资源块滚动历史必须使用已落盘因子。

`POST /api/strategies` 的响应在原有 `strategy_id`、`version`、`config_hash` 之外增加
`provider`：因子策略为 `null`，提供者策略返回当前名称、版本、源码摘要与参数结构。

## 流事件

每条 SSE 为 `event: <stage>` 与 `data: <JSON>`，空行分隔：

```text
queued  -> 已进入回测队列
status  -> 任务状态、排队等待时间、原生锁等待时间和运行时间
meta    -> run_id、snapshot_id、策略名与版本、配置哈希、股票池、基准、因子/提供者、区间与初始资金
load    -> 交易日数、股票数、因子、是否有成分矩阵与基准、snapshot_id
chunk   -> 资源准备或调仓段阶段、当前序号/总数与日期窗口
signals -> 调仓日数、入选格数、单股与行业上限统计
probe   -> 探针成交笔数
orders  -> 计划下单、再平衡格、规则拒单
baseline-> 基准遍完成（现金缓冲、调仓日数）
allocation -> 当前调仓段的资金分配结果（迭代轮次、风险约束、建仓/补最小申报量/未建仓格数）
provider_state -> 状态提供者的反馈轮次、最大轮次、目标变化格数和是否收敛
execute -> 最终成交、成本模型逐笔偏差、现金成本/滑点合计、被现金裁剪的格数
done    -> run_id、snapshot_id、成交与拒单笔数、calibration、绩效指标、总耗时
```

错误以 `error` 事件发送 `{code,message,request_id}` 并结束流；收到 `done`/`error` 后要关闭连接，
不要自动重连。连接断开会在当前阶段结束后取消运行，并把 `bt.run.status` 记为 `cancelled`。

任务状态包括 `queued`、`preflight`、`waiting_native`、`running`、`cancelling`、
`success`、`failed` 和 `cancelled`。`waiting_native` 表示等待回测与因子评估共享的原生锁，
原生计算已经开始后不会强杀线程，取消在当前阶段或调仓执行段边界生效。

`success`、`failed`、`cancelled` 是终态：同一运行并发完成时只有一个请求能提交结果，
其余请求返回 `BT-602`；迟到的失败或取消清理不会覆盖终态。结果表与成功状态在同一事务内提交，
竞争失败方不会留下重复结果行。

## 并发

回测用独立的调度器，默认 1 个运行名额与 2 个等待名额（`BACKTEST_WORKERS` / `BACKTEST_WAITING` 可调）。
回测与因子评估共用一把原生计算锁，回测执行阶段会阻塞因子评估；大区间回测建议单独跑。

```bash
curl -N 'http://127.0.0.1:8800/api/backtests/stream?strategy_id=builtin-mom20-top30&start=2026-08-03&end=2026-09-28'
curl -s 'http://127.0.0.1:8800/api/backtests' | python -m json.tool | head
```

