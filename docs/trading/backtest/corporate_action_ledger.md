# 公司行动与原始价执行账本

## 口径

- 行情使用 `md.daily_bar` 的原始开收盘价，成交价按开盘价加滑点。
- 除权日按 `ref.corporate_action.share_multiplier` 调整持股；不足 1 股的零头按除权日
  开盘价折成现金，记入 `bt.corporate_action.fractional_cash`。
- 现金分红在除权日按税前每股金额确认应收，在 `pay_date` 进入现金。
- 配股在除权日按 `floor(持股 × rights_ratio)` 计算可配股数，只在现有现金范围内
  `floor(现金 / rights_price)` 认购；不融资、不强制卖出其他股票，不足部分记入
  `bt.corporate_action.rights_unsubscribed_shares`。认购股份形成新的 FIFO 批次。
- 红利税采用递延 FIFO：卖出时按买入批次、持有期和卖出日生效政策结算。
  2005-06-13 前统一 20%，之后统一 10%；2013-01-01 起持有不超过 1 个月 20%、
  1 个月至 1 年 10%、超过 1 年 5%；2015-09-08 起超过 1 年免税。
  未卖出的批次不提前扣现金。
- 买入和卖出按股票所属板块解析申报单位；公司行动形成的零股可以在清仓时一次卖出。

## 账本流程

每个交易日按以下顺序执行：

1. 支付到期的现金分红应收；
2. 施加除权日送转/拆并，处理零头现金，并按可用现金认购配股；
3. 先卖后买，按板块申报单位校验订单，再按原始开盘价、滑点、佣金、最低佣金、
   历史印花税和历史过户费成交；
4. 更新 FIFO 批次并结算卖出对应的红利税；
5. 按收盘价估值，停牌沿用上一有效收盘价。

组合价值 = 现金 + 应收分红 + 持仓市值。`bt.daily.cost` 为佣金、印花税、过户费和
已结算红利税之和；`bt.daily.total_cost` 再加上滑点和冲击归因。滑点与冲击已经体现在成交价里，
不能再次从现金扣除。

## 数据边界

`ref.corporate_action` 以 `stock_code + ex_date` 为唯一键，字段包括送转倍数、税前
现金分红、每股配股比例和配股价。数据优先级：

1. `source=gbbq`：固定版本 gbbq 类别 1，提供送转、现金和配股的完整条款；
2. `source=dividend`：补 gbbq 未覆盖或 2026-04-13 之后的 Tushare 分红事件；
3. `source=share_count`：两边都缺失时，仅当总股本和复权因子在同一除权日按相同比例变化，
   才作为送转/拆并兜底。

Tushare 与 gbbq 重叠事件逐项比较现金和送转，冲突采用 gbbq 并把数量和样例写入
`meta.corporate_action_manifest`。并购重组、换股吸收合并尚无可靠结构化映射，
P2-2B 已按用户决定跳过，不猜测入账；若回测在事件退市日仍持有目标股票，现有逻辑会按
零价值核销，该限制见 `docs/trading/backtest/completeness.md`。北交所转板同样未建模。

增量建表命令：

```bash
.venv/bin/python -m scripts.build_db.corporate_actions
.venv/bin/python tools/data_audit/audit_corporate_actions.py
```

首次使用先运行 `.venv/bin/python -m scripts.sources.fetch_all gbbq`。下载器固定提交和
SHA-256，只下载 3.4MB 单文件；摘要不符时停止建库。

该命令在事务内替换 `ref.corporate_action` 和 `meta.corporate_action_manifest`，不重建
其它表；完整 `build_db.build` 也会调用同一个构建函数。
