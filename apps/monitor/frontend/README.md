# 因子监测前端

纯 HTML/JS + ECharts 5.6.0，由后端同源托管，无 Node/Vite 构建步骤。
图表库及 Apache 2.0 许可证保存在 vendor，浏览器不访问 CDN。

| 模块 | 功能与调用 | 修改注意事项 |
|---|---|---|
| index/style | 页面布局、固定口径说明 | 手机端必须保留图表与错误可读性 |
| app | hash路由、参数表单、按注册顺序建卡片；调用 stream/charts | 切换页面须取消旧请求，不持久化结果 |
| stream | fetch读取SSE、HTTP错误与流意外结束检查 | 不自动重连，避免重复计算 |
| charts | 消费计算结果绘图及通用数据表；由 app 调用 | 不重算收益/IC/回归；空值显示“—”，未知指标用通用表 |
| mining | 挖掘列表、新建、运行详情、预算与事件轨迹 | 切换路由必须中止 SSE；批准只记录决定 |
| mining_candidate | 候选公式、依赖、门禁、四阶段图表、审批和整批发布 | 图表复用 renderMetric；页面不能自行判定审批或发布资格 |
| mining_api | 挖掘 JSON 请求、可续传 SSE 和状态文案 | 事件按 sequence 去重，断流不得触发取消 |

回测相关页面新增 `#/backtest/new` 和 `#/backtest/compare?run_ids=...`；完整路由包括
`#/factors`、`#/factor/MOM_20`、`#/compare`、`#/backtest`、`#/backtest/new`、
`#/backtest/compare`、`#/backtest/{run_id}`。
详情支持完整分析与轻量IC；对比选择2～4个因子。完整分析显示两套权重并列的
分位均值（±1标准误、t值）、净值、多空价差、箱线图，以及风格和相关性。
箱线须在提示中区分每日分位均值和分位内个股分布。

`#/backtest` 选择内置策略、股票池、区间与初始资金，用 `/api/backtests/stream` 的 SSE
逐阶段显示进度，并在下方列出历史运行；列表支持物理删除与选择 2～4 次成功运行对比。
`#/backtest/{run_id}` 展示运行信息、绩效汇总、净值/回撤/日超额、换手与成本、规则拒单、
执行校准、成交明细与最新持仓。`#/backtest/new` 创建当前 `StrategySpec` 支持的完整配置，
可在因子 Top-N 和目标权重提供者之间切换；提供者字段由后端 `param_schema` 生成，
提交前调用 `/api/providers/validate`，创建成功后才返回并选中新策略。
运行流程与内置策略完全一致。

端到端浏览器脚本：`tools/monitor/browser_check.py`，需要独立安装 Playwright/Chromium，
它们仅用于验收，不属于服务运行依赖。脚本使用真实数据与服务端结果；
回测页面的验收脚本为 `tools/monitor/backtest_check.py`（跑一次短区间回测并打开详情页）。

挖掘路由包括 `#/mining`、`#/mining/new`、`#/mining/{run_id}` 和候选详情页。新建页默认
使用已验收预算，高级参数折叠。候选页展示公式、步骤、字段依赖、技术门禁、快速筛选、
完整评估、模块测试段和审批表；批准按钮必须同时呈现候选哈希与关键质量结论。
`tools/monitor/mining_check.py` 使用临时夹具检查桌面、390px 移动端、批准、拒绝和失败
候选禁用状态，并在两个候选批准后完成整批发布、状态轮询和发布截图，不需要 Worker Key。
S8 截图输出到 `AiRecord/evidence/mining_s8/`。`tools/monitor/mining_search_check.py`
使用独立 V2 夹具验证五阶段证据、批准、拒绝、首次发布失败、重试发布和移动端，截图
输出到 `AiRecord/evidence/mining_v2_web/`。
