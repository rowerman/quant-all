# 因子评估指标

本页逐项说明因子详情页和对比页当前使用的 13 个指标：每个指标读取哪些面板字段、按什么
门槛取样本、公式如何计算、什么情况下返回空值，以及结论在量化研究中能说明什么、不能说明
什么。评估面板、stage 调度、SSE 和并发控制由
[因子评估](03-factor-evaluation.md)介绍，本页从指标注册表开始。

## 1. 职责与位置

指标计算集中在
[research/evaluation/calculate/metrics](../../research/evaluation/calculate/metrics)，
公共逐日相关、分箱和序列统计分别位于
[daily.py](../../research/evaluation/calculate/daily.py)、
[binning.py](../../research/evaluation/calculate/binning.py)和
[stats.py](../../research/evaluation/calculate/stats.py)。风格回归与最终相关性读取分别由
[exposure.py](../../research/evaluation/calculate/exposure.py)、
[styles.py](../../research/evaluation/calculate/styles.py)和
[correlation.py](../../research/evaluation/calculate/correlation.py)完成。

指标层只做三件事：

1. 从一次评估请求共享的 `EvalContext` 读取面板、参数、因子方向和缓存结果。
2. 按指标声明计算一个可 JSON 序列化的 `dict`。
3. 把结果交回 stage 调度；指标层不负责数据库持久化、页面渲染或交易执行。

当前注册表有 13 个指标、12 个业务 stage。`overview` 同时承载 `coverage` 和 `ic_stats`，
因此指标数比业务 stage 多一个。完整 stage 顺序为：

```text
meta
overview
ic_series
ic_dist
bins
bin_curves
spread
bin_dist
turnover
decay
exposure
incremental_ic
correlation
done
```

`meta` 和 `done` 不是指标 stage。普通 Python 和 Web 请求都按这份顺序执行；调用方传入的
stage 顺序不会改变最终顺序。轻量 IC 是另一条路径，只提供 `overview`、`ic_series` 和
`ic_dist` 中的三个 IC 指标，不运行分层、风格或相关性。

## 2. 指标注册与返回契约

### 2.1 `Metric` 声明

[metrics/base.py](../../research/evaluation/calculate/metrics/base.py)把“算一个指标”
定义为不可变声明：

| 字段 | 含义 |
|---|---|
| `key` | 唯一指标标识，也是 payload 中的指标键 |
| `label` | 页面卡片标题 |
| `stage` | 所属 SSE stage 和前端卡片分组 |
| `compute(context, options)` | 实际计算函数 |
| `params` | 可选参数声明，目前支持 `int`、`bool`、`enum` |
| `requires_quantiles` | 是否依赖按 `params.quantiles` 生成的分位表 |

`metrics/__init__.py` 自动发现同目录指标模块，模块导入时调用 `register_metric()`。
同 stage 内按注册顺序展示，跨 stage 按公共 `STAGE_ORDER` 排序。重复 key、非法 stage、
未知 key、未知参数和参数类型错误都报 `EVAL-201`，且参数校验发生在读面板之前。

### 2.2 当前 13 个指标

| stage | key | 页面标题 | 计算入口 | 可选参数 |
|---|---|---|---|---|
| `overview` | `coverage` | 覆盖率与取值分布 | `metrics.coverage.compute` | 无 |
| `overview` | `ic_stats` | IC / RankIC 汇总 | `metrics.ic.compute_stats` | 无 |
| `ic_series` | `ic_series` | IC 时序 | `metrics.ic.compute_series` | 无 |
| `ic_dist` | `ic_dist` | IC 分布 | `metrics.ic.compute_dist` | 无 |
| `bins` | `bin_returns` | 分位平均收益与单调性 | `metrics.bins.compute_bin_returns` | 无 |
| `bin_curves` | `bin_curves` | 分位累计净值 | `metrics.bins.compute_bin_curves` | 无 |
| `spread` | `spread` | 多空价差 | `metrics.bins.compute_spread` | 无 |
| `bin_dist` | `bin_dist` | 分位收益分布 | `metrics.bins.compute_bin_dist` | 无 |
| `turnover` | `turnover` | 分位换手与因子自相关 | `metrics.turnover.compute_turnover` | 无 |
| `decay` | `decay` | 多周期衰减 | `metrics.decay.compute_decay` | 无 |
| `exposure` | `exposure` | 风格暴露 | `metrics.exposure.compute_exposure` | `industry` |
| `incremental_ic` | `incremental_ic` | 剔除风格后的增量 IC | `metrics.exposure.compute_incremental_ic` | `industry` |
| `correlation` | `correlation` | 与已有因子的相关性 | `correlation.compute_correlation` | `top_n` |

`bin_returns`、`bin_curves`、`spread`、`bin_dist` 和 `turnover` 声明
`requires_quantiles=True`。它们都通过 `EvalContext.bins()` 或
`EvalContext.bin_matrices()` 复用请求内缓存，不应各自重新透视因子值。

`exposure` 和 `incremental_ic` 的 `industry` 默认 `off`，当前还可选 `stock_basic`。
两项参数独立传入，但假设相同，则共用 `EvalContext.exposures(industry)` 的缓存结果。
`correlation.top_n` 默认 `10`，只允许 1～100 的整数；它限制 Top-N 明细数量，不改变完整
相关矩阵。

### 2.3 统一返回位置

调度事件的外层形状是：

```python
stage, {
    "stage": stage,
    "metrics": {
        metric_key: metric_payload,
        ...
    },
    ...
}
```

每个 `metric_payload` 都由对应 `compute()` 返回。指标内部可能使用 NumPy 数组、`NaN`
和 NumPy 标量；统一入口会在发送 SSE 前转成标准 JSON 类型。直接调用 `compute()` 做调试时，
需要自行处理这些类型，或使用 `calculate.jsonutil.json_safe()`。

## 3. 公共输入、统计量与样本门槛

### 3.1 面板字段

所有指标共享同一次请求构建的评估面板。主要字段如下，完整面板契约见
[评估面板说明](../research/evaluation/panel.md)：

| 字段 | 指标用途 |
|---|---|
| `stock_code`、`trade_date` | 股票和交易日主键 |
| `factor` | 因子值 |
| `label` | 页面选择的 `horizon + price` 主标签 |
| `label_1d` | 同价格通道的 1 日前瞻收益，净值和多空价差使用 |
| `label_h{h}` | 衰减阶段额外需要的 3/10/20 日标签 |
| `in_universe` | 当日是否属于所选股票池 |
| `total_mv` | 分位内市值加权 |
| 日线原料字段 | 风格计算另行使用，不直接参与大多数指标 |

标签周期可选 1/3/5/10/15/20 个交易日，价格通道可选 `close` 或 `open`：

```text
close：t 日收盘 -> t+H 日收盘
open ：t+1 日开盘 -> t+1+H 日开盘
```

标签来自 `md.fwd_return`，价格是后复权口径。`t` 日停牌、未来价格不足或标签不可得时为空。
不同周期分别取有效样本，不用其他周期填补。

全 A 池要求 `can_buy_close`、非 ST，并上市满 60 个自然日。指数池使用上一个已结束月份的
月末权重快照，再叠加相同的可交易过滤。股票池本身的构造和限制由
[03-factor-evaluation.md](03-factor-evaluation.md)说明。

### 3.2 非空与有限的区别

代码对两种状态处理不同：

- `NaN` 表示字段为空。覆盖率分子使用 `notna()`，因此 `NaN` 不计入非空覆盖。
- `+inf` / `-inf` 不属于 `NaN`。覆盖率会先把它们计入“非空”，再通过 `missing` 和
  `non_finite` 分开报告；IC 和成对相关会通过 `isfinite()` 排除它们，序列统计也会忽略
  非有限值。
- `assign_bins()` 只按 `factor.notna()` 筛池内因子，没有额外删除无穷值。正常构建的因子
  应为有限值；若覆盖率显示 `non_finite > 0`，应先修数据，不应忽略。
- 公共逐日相关先同时删除因子和标签的空值，再要求两侧都有限。分母是该日真实有效配对数，
  不是原始股票池人数。

### 3.3 共用统计量

对一条剔除空值和无穷值后的序列 `x_1...x_n`：

```text
mean      = average(x)
std       = sample_std(x)，ddof=1
stderr    = std / sqrt(n)
IR        = mean / std
t         = mean / stderr = IR * sqrt(n)
win_rate  = count(x > 0) / n
Sharpe    = mean / std * sqrt(252)
NAV_t     = NAV_(t-1) * (1 + x_t)
drawdown_t = NAV_t / max(NAV_0...NAV_t) - 1
```

单样本日序列的 `std`、`IR`、`stderr` 和 `t` 为空；标准差为零时 `IR`、`t` 和年化 Sharpe
为空。最大回撤把初始资本 1 纳入峰值计算，因此首日亏损也会计入回撤。

逐日 IC、RankIC 和大部分收益序列按日重叠。普通 t 值没有 HAC 修正，Sharpe 也不处理
重叠持有期、自相关、交易成本和无风险利率，只能用于研究比较。

### 3.4 各指标的实际门槛

不同指标不能共享一套“少于 20 天全部隐藏”的规则：

| 指标或序列 | 单日/单截面门槛 | 时间序列门槛 | 不足时行为 |
|---|---|---|---|
| 日度 IC、RankIC、因子自相关、相关矩阵 | 至少 5 个有限配对 | 无 | 当日相关为空 |
| `ic_stats` | 继承 5 个配对 | 至少 20 个有效日 | 日数保留，其余统计量置空 |
| `ic_dist` | 继承 5 个配对 | 无 20 日隐藏 | 有有效日就出分布 |
| `bin_returns`、`bin_curves`、`spread`、`bin_dist` | 不设最少股票数 | 不设 20 日门禁 | 组或日期可为空；样本少时仍可能出值 |
| 分位换手 | 不设额外最少样本 | 不设 20 日门禁 | 分母为零的日期为空 |
| `decay` 的 RankIC 曲线 | 每档至少 5 个配对 | 每档至少 20 个有效日 | 只保留天数和总配对数 |
| `decay` 的多头超额、多空价差 | 继承分位收益 | 不设 20 日门禁 | 有多少有限值就统计多少 |
| `exposure` 的单日回归 | 至少 30 行 | 至少 20 个有效回归日 | 少于 20 日返回样本不足结构 |
| `incremental_ic` | 残差日度相关至少 5 个配对 | 至少 20 个有效日 | 不返回逐日序列和差值 |
| `correlation` | 每对因子每日至少 5 只 | 每对均值至少 20 日 | 该对均值保持空值 |

“不设门禁”不等于结果可靠。短区间、少量股票、常数截面或高缺失仍会产生空值、零天数和
不稳定的均值。前端把比例字段保留为 `0～1` 的小数，不负责替研究判断正确性。

## 4. 覆盖率与取值分布

注册位置：`coverage`，stage 为 `overview`。实现见
[metrics/coverage.py](../../research/evaluation/calculate/metrics/coverage.py)。

### 4.1 输入与公式

读取 `factor`、`label` 和 `in_universe`，在全部面板行上计算：

```text
coverage_all      = 因子非空行数 / 面板总行数
coverage_universe = 池内因子非空行数 / 池内总行数
pair_coverage     = 池内因子与主标签同时非空行数 / 池内总行数
pool_share        = 池内行数 / 面板总行数
```

这些“非空”判断按当前的 `notna()` 语义执行。池内行数为零时，后三项中的池内比率为空。
`missing` 只统计 `NaN`，`non_finite` 统计正负无穷；取值分布先删除全部非有限值。

### 4.2 返回内容

主要字段：

- `rows`、`pool_rows`、`stocks`、`days` 和 `pool_share`。
- 三种覆盖率和缺失、无穷值计数。
- `daily_pairs`：每日有效配对数序列的均值、中位数、最小值和最大值。
- `distribution`：池内非空且有限因子值的数量、均值、样本标准差、极值、偏度、超额峰度，
  以及 1/5/25/50/75/95/99 分位点。

分布没有有效值时只返回 `{"count": 0}`；只有一个值时标准差为空；偏度需要至少 3 个值，
超额峰度至少 4 个值且标准差大于零。

### 4.3 阅读方式

- 池内覆盖低说明因子只覆盖部分股票，IC 和分层不能自动外推到全市场。
- 覆盖高但有效配对率低，通常问题在标签缺失、停牌或区间末端未来价格不足。
- 每日有效配对数波动大时，不同日期的 IC 可比性下降。
- 标准差接近零说明截面区分度弱；长尾明显时，Pearson IC 更容易受极端值影响。

列表页的“平均覆盖率”来自因子目录库，是历史落盘统计；详情页覆盖率来自当前日期、股票池和
标签。两者不能混用。列表页把平均覆盖率乘 100 加 `%`，详情页表格仍显示原始小数。

## 5. IC / RankIC 汇总

注册位置：`ic_stats`，stage 为 `overview`。实现见
[metrics/ic.py](../../research/evaluation/calculate/metrics/ic.py)和
[daily.py](../../research/evaluation/calculate/daily.py)。

### 5.1 逐日相关

在池内、因子和主标签同时有限的当日集合 `S_t` 上：

```text
IC_t     = Pearson(F(i,t), Y(i,t))
RankIC_t = Pearson(rank(F(i,t)), rank(Y(i,t)))
```

RankIC 等价于平均并列名次的 Spearman，直接对原始值排名。Pearson IC 默认先对因子和标签
分别做当日截面的 3 倍 MAD 裁剪；`winsorize=False` 时使用原始值。MAD 为零或有效值不足
3 个时不做裁剪。

当日有效配对少于 5 个、任一侧方差为零时，相关系数为空。限制来自
`MIN_CROSS_SECTION = 5`。

### 5.2 时间汇总与方向

对 IC 和 RankIC 日序列分别计算均值、标准差、标准误、IR、t、正值比例和有效天数，
IC 的 IR 即 ICIR。`valid_days` 取 IC 与 RankIC 序列天数的较大值；正常样本下两者相同。

有效日少于 20 天时：

- `insufficient=True`，保留 `valid_days`。
- IC 和 RankIC 的数值统计全部置空。
- `direction_aligned` 为空，只给“拉长评估区间”的提示。

有效日足够且 RankIC 均值可得时，比较符号与 `spec.direction`：方向为 1 期望均值为正，
方向为 -1 期望均值为负。该字段只表示实测方向是否与定义一致，不表示显著或可交易。

### 5.3 阅读边界

- RankIC 更关注排序，对极端值通常比 Pearson IC 稳健。
- ICIR 是平均相关除以日间波动，不等于年化收益与风险的比率。
- 多日标签逐日重叠，日度 IC 不独立，普通 t 值和 ICIR 只能作参考。
- 相关系数大不代表扣费后收益高；横截面方向正确仍可能因换手、容量和交易约束失效。

## 6. IC 时序

注册位置：`ic_series`，stage 为 `ic_series`。

### 6.1 返回内容

每个面板交易日输出：

- `dates`、逐日 `ic`、逐日 `rank_ic` 和有效配对数 `n`。
- 20 日滚动 IC、RankIC 均值，窗口为 20，`min_periods=1`。
- 累计 IC 和累计 RankIC。

累计规则为“遇到空值保留空值，但在后续有效日继续累加此前有效值”，不会把空值当零写入
累计结果。因此无效日显示断点，不会污染后续累计水平。

### 6.2 页面与阅读

payload 同时包含 rolling 和 cumulative 两组序列，当前前端只画累计 IC 和累计 RankIC；
`rolling` 仍供 Python 调用和后续展示使用。

长期在零附近说明方向弱或不稳定；持续在零一侧说明方向较稳定；多次穿越零可能表示状态依赖
或衰减。单日尖峰要结合 `n` 看，少数股票形成的相关更容易异常。有效日少于 20 天时本指标
仍返回时序和累计值，不做隐藏；需要结论时应回到 `ic_stats` 的样本不足标记。

## 7. IC 分布

注册位置：`ic_dist`，stage 为 `ic_dist`。

### 7.1 计算

对逐日 IC 和 RankIC 的有限值分别建立 40 个等宽直方图桶，返回边界 `edges` 和计数
`counts`，并计算数量、均值、样本标准差、偏度、超额峰度及 5/25/50/75/95 分位点。
横轴范围由当前样本最小值和最大值决定，不是固定 `[-1,1]`。

没有有限日度相关时只返回空边界、空计数和 `count=0`。偏度需要至少 3 个值，超额峰度需要
至少 4 个值。

本指标没有 20 个有效日的隐藏门槛。即使 `ic_stats` 报样本不足，只要存在日度相关，
`ic_dist` 仍会返回分布，方便定位数值形态，但不能把短样本分布当稳定结论。

### 7.2 阅读方式

分布重心偏正或偏负表示日度相关总体倾向；分布越宽表示日间预测能力波动越大；均值和中位数
差异大表示少数日期造成偏斜。长尾应回查对应日期、市场状态和有效股票数。

## 8. 分位平均收益与单调性

注册位置：`bin_returns`，stage 为 `bins`。分箱和收益的公共实现见
[binning.py](../../research/evaluation/calculate/binning.py)。

### 8.1 分箱

每天只取 `in_universe=True` 且 `factor.notna()` 的股票。标签为空的行保留在分位表，但不
进入收益计算。

```text
average_rank = 当日因子截面的平均名次
position     = ceil(average_rank / 当日样本数 * Q)
bin          = clip(position, 1, Q)
```

组 1 是最低因子组，组 Q 是最高组。并列值使用平均名次，因此并列股票不会被随机拆组。
`Q` 只能是 5、10 或 20。这里不设“每个截面至少 5 只”的门槛；股票很少时某些组会为空。

### 8.2 两种收益口径

```text
等权：R_equal(t,q) = average(Y(i,t) | bin(i,t)=q)
市值：R_mv(t,q) = sum(M(i,t) * Y(i,t)) / sum(M(i,t))
```

市值加权只使用 `total_mv` 非空且大于零的股票。两种口径分别对每个组返回均值、样本标准差、
标准误、t、正值比例和有效天数。

单调性是分位编号与各组平均收益之间的 Spearman 相关：

```text
monotonicity = corr(rank(1...Q), rank(mean_R_1...mean_R_Q))
```

只使用均值有限的组；可用组少于 2 个时 `spearman=None`。它是描述性秩相关，不是回归斜率。

### 8.3 空值与阅读

池内完全没有非空因子时返回 `empty=True` 和提示。池内有因子但标签全空时，会返回 Q 个
空统计组，而不是统一标成 `empty`。分位收益没有 20 日门禁，短样本仍可能出现均值。

- 高分组高、低分组低且单调性接近 1，支持因子排序与未来收益同向。
- 接近 -1 可能是反向因子，也可能是方向元数据错误。
- 只有最高组异常、中间无梯度，更像尾部信号而非连续单调因子。
- 等权和市值加权差异大，说明结果可能集中于某种规模结构。
- 多日标签下各组收益重叠，t 值不能当独立日检验。

## 9. 分位累计净值

注册位置：`bin_curves`，stage 为 `bin_curves`。

净值固定使用同价格通道的 `label_1d`，不使用页面选择的 3/5/10 等主标签：

```text
NAV(t,q) = NAV(t-1,q) * (1 + R_1d(t,q))
NAV(0,q) = 1
```

等权和分位内市值加权各生成 Q 条曲线，返回日期、曲线和期末净值。`compound()` 把缺失日
收益按零处理，因此当日净值不变；这不等于把缺失标签补进 IC 或均值统计。

固定使用 1 日收益，是因为多日期收益逐日重叠。把 5 日收益每天累乘会重复计算同一持有期，
曲线无法对应清晰的持有组合。

曲线适合比较分位相对路径、交叉和长期方向，但没有模拟 T+1、涨跌停、停牌成交、佣金、
印花税、过户费、滑点、冲击成本和资金约束。它不是回测净值，也不是可实现策略收益。

## 10. 多空价差

注册位置：`spread`，stage 为 `spread`。

### 10.1 计算

同样固定使用 1 日前瞻收益：

```text
Spread_t = R_1d(t,Q) - R_1d(t,1)
```

等权和市值加权分别返回日度价差、复利净值、均值、标准差、t、正值比例、有效天数、年化
Sharpe、累计收益和最大回撤。累计收益等于期末净值减 1；最大回撤为负数，例如 `-0.12`
表示 12% 回撤。

价差序列没有 20 日门禁。年化 Sharpe 需要至少 2 个有限值且标准差大于零；只有 1 天时
仍可有均值和正值比例，但 Sharpe 为空。缺失收益日进入复利时按零处理，统计量则不把缺失值
计入天数。

### 10.2 阅读边界

均值正表示最高因子组平均跑赢最低组；正值比例高表示多数日期同向；最大回撤描述价差曲线
经历过的最大逆风。当前口径忽略融券可得性、做空限制、T+1、涨跌停不可成交和交易成本，
不能视为可执行市场中性策略收益。

## 11. 分位收益分布

注册位置：`bin_dist`，stage 为 `bin_dist`。

提供两种分布：

1. `daily`：每个交易日的组分位收益均值，跨日期汇总成箱线数据；等权和市值加权各一套。
2. `stocks`：池内全部个股的主标签期收益，按当日分位跨股票和日期汇总；权重口径不影响。

每组返回样本数、均值、样本标准差、最小值、5/25/50/75/95 分位点和最大值。当前前端箱线图
只画最小、25%、中位、75% 和最大；payload 中的 5%/95% 分位点没有画入箱体。

中位数与均值差异大说明偏斜；箱体宽说明组内日期或个股差异大；各组箱体重叠严重时，即使
均值有梯度，横截面区分也可能不稳定。该指标仍是研究分布，不包含交易规则和成本。

## 12. 分位换手与因子自相关

注册位置：`turnover`，stage 为 `turnover`。

### 12.1 分位换手

分位表先转成“交易日 × 股票”宽表，只比较今天和上一交易日都有分组的连续股票。

```text
全池换手_t =
今天组号与昨天不同的连续样本数
/
今天和昨天都有组号的样本数
```

单组换手：

```text
单组换手(t,q) =
今天在 q、昨天不在 q 的连续样本数
/
今天在 q、昨天也有组号的样本数
```

首日没有上一日基准，换手为空。新进入或退出股票池、上日为空的股票不进入分母。分子按每个
被比较的股票计数，不使用组合权重。

### 12.2 因子自相关

先按日对因子值做截面秩；比较当前日与 `L` 个交易日前时，只在两日都存在的股票交集内重新
排名并计算相关：

```text
Autocorr(t,L) = corr(rank(F(i,t)), rank(F(i,t-L)))
```

滞后固定为 1、5、20。单日交集少于 5 只或任一侧秩方差为零时，当日相关为空。宽表保留面板
中没有有效池子的日期，不从位置序列中压缩掉，因此 lag 仍按面板交易日位置推进。

本指标没有统一的 20 日隐藏规则；每个自相关滞后分别返回均值、标准差、t、正值比例和有效
天数。滞后超过可用长度时，该阶有效天数为零。

### 12.3 阅读边界

换手低表示分位成员稳定，潜在交易成本压力较小；自相关高表示排序持续。IC 高但自相关低，
信号可能短促，需要更关注调仓频率和成本。这里的分位换手不是账户实际换手，也不包含权重
变化、费用、涨跌停、停牌和成交容量。

## 13. 多周期衰减

注册位置：`decay`，stage 为 `decay`。

### 13.1 五个固定档位

页面主标签可选 1/3/5/10/15/20 日，但衰减曲线固定比较 1/3/5/10/20 日，不包含 15 日。
每个档位使用同价格通道的对应标签，只取因子与该档标签都有限的股票：

```text
RankIC_t^(H) = Spearman(F(i,t), Y(i,t;H,P))
```

每档分别返回 RankIC 均值、RankICIR、t、正值比例、有效天数、总配对数以及是否样本不足。
`samples` 是该档逐日有限配对计数之和；某日少于 5 个配对时，该日的 `n` 仍会进入
`samples`，但不会形成 RankIC，也不会增加 `days`。

每档有效日少于 20 天时，均值和统计量置空，只保留 `days`、`samples` 和 `insufficient`。
各档独立取样本，长周期末端天然少几天，不做填充。

### 13.2 多头超额与多空价差

在相同五档标签上，分别用等权和分位内市值加权计算：

```text
多头超额 = 最高分位收益 - 池内因子非空样本平均收益
多空价差 = 最高分位收益 - 最低分位收益
```

“池内均值”在该档标签的市值权重口径下计算；市值加权只纳入有效的正市值。两项收益差返回
均值、标准差、t、正值比例和有效天数，没有 20 日隐藏门禁。因此 RankIC 曲线显示“样本
不足”时，收益差仍可能带有短样本数值，需要自行克制解读。

### 13.3 主档提示

`primary_horizon` 优先取页面主标签周期。若页面选择 15 日，代码选择最接近的 20 日档作
主档提示，但 `curve` 仍只有固定的五个档位。提示只帮助阅读，不重新定义评估主标签。

短周期强、长周期弱表示信号可能短命；长周期仍稳定可能适合低频策略；多头超额和多空价差
的分化可帮助判断收益来自高分组还是低分组。各档持有区间重叠，跨档 t 值不能当独立实验比较。

## 14. 风格暴露

注册位置：`exposure`，stage 为 `exposure`。风格原始值见
[styles.py](../../research/evaluation/calculate/styles.py)，回归见
[exposure.py](../../research/evaluation/calculate/exposure.py)。

### 14.1 七个近似风格

窗口按 `status='trading'` 的有效交易日推进，停牌日不占窗口。风格原始值只算到评估结束日，
为了获得 `t-250` 价格，取数会从区间起点前回看 260 个市场交易日，并为长期停牌股票补足
此前 250 次有效交易。

| 风格 | 当前近似定义 | 最少观测 |
|---|---|---|
| `SIZE` | `ln(total_mv)` | 当日有正市值 |
| `VALUE` | 标准化 `1/pb` 与 `1/pe_ttm` 的均值，缺一项用另一项 | 至少一项可用 |
| `MOMENTUM` | `close[t-20] / close[t-250] - 1` | 251 个有效交易日 |
| `SHORT_REV` | `1 - close[t] / close[t-20]` | 21 个有效交易日 |
| `VOL` | 最近 60 个有效交易日 `ret` 的样本标准差 | 40 个有效收益 |
| `LIQUIDITY` | 最近 20 个有效交易日平均换手率 | 15 个有效换手 |
| `BETA_60` | `ret` 对沪深 300 日收益的 60 日回归斜率 | 40 个有效收益对 |

基准固定为 `000300.SH`，指数收益使用 `md.index_daily.pct_chg / 100`。缺少
`md.index_daily` 时报 `EVAL-302`。`pb` 或 `pe_ttm` 非正视为没有估值可用。原始风格值不
直接进入面板，而是在进入暴露 stage 时懒加载并缓存。

### 14.2 回归

回归样本是“池内且因子非空”，不要求标签非空，使暴露与主标签周期解耦：

```text
z(F(i,t)) =
alpha_t
+ beta_SIZE,t * z(SIZE(i,t))
+ ...
+ beta_BETA60,t * z(BETA60(i,t))
+ error(i,t)
```

因子与风格在同一样本上逐日做 3 倍 MAD 裁剪和 z-score。标准化后缺失的风格填零，代表
“中性假设”，不代表真实暴露为零。填零会改变列方差，VALUE 又是两个 z 值的均值，因此各
系数只适合近似比较，不能解释成严格的同单位边际效应。

每日截面 OLS 的规则：

1. 当日样本少于 30 时跳过并计数。
2. 响应非有限或整日因子恒定时跳过。
3. 某个风格整日恒定或全部缺失时，只删除该风格列；该系数为空，其他风格继续回归。
4. 删除不可估计列后，剩余设计矩阵仍秩亏或自由度不足时，整日跳过。
5. 回归成功时保存截距、7 个风格位置上的系数、日度 t、R²、样本数和残差。

返回的日度系数统计为：系数时序均值、标准差、均值除以标准误的 Fama-MacBeth t、系数为正
日期比例、日度回归 `|t| > 1.96` 的比例和有效天数。另返回平均样本数、平均 R²、
`skipped_days`、逐日系数序列及逐日 R²。普通标准误没有 HAC 修正。

有效回归日少于 20 天时，`insufficient=True`，不生成系数统计和逐日序列。池内完全没有非空
因子、风格表主键重复或所有风格都没有值时报 `EVAL-301`。

### 14.3 行业层

`industry=off` 时不加行业项。当前已注册的 `stock_basic` 读取
`ref.stock_basic.industry` 静态分类，去掉第一类后加入行业哑变量，映射缺失标为“未知”。
返回行业列数、已映射覆盖率和类别数。

静态行业使用今天的分类解释历史，无法反映行业变更，退市股票也可能缺失，因此默认关闭。
行业源不可读或为空时报 `EVAL-303`。行业哑变量可能让原本可估计的截面变得秩亏，此时按
整日跳过规则处理。

绝对暴露大且显著，说明因子可能是该风格的代理；平均 R² 高说明这些近似风格解释较多因子
截面变化。这里的 R²、系数和显著比例都是统计诊断，不是收益归因。

## 15. 剔除风格后的增量 IC

注册位置：`incremental_ic`，stage 为 `incremental_ic`。它与暴露共用同一行业参数和回归
缓存，但单独计算残差与标签的相关。

在风格回归成功的日期和股票样本上：

```text
ResidualRankIC_t = Spearman(error(i,t), Y(i,t))
RawRankIC_t      = Spearman(F(i,t), Y(i,t))
```

原始 RankIC 必须使用残差 IC 的完全相同日期和股票重算，不能直接取 `ic_stats` 的全样本
RankIC。逐日相关仍要求至少 5 个有限配对。

有效残差日相关少于 20 天时，只返回：

- `days`、`insufficient`、`min_days`、`min_cross_section`；
- 行业信息、主标签和计算说明；
- 仅有天数、其余统计量为空的 `raw` 和 `residual`。

此时不返回 `delta`、`survivor_share`、`series` 和完整 note。因子完全被风格解释时残差
会按数值容差置零，残差 RankIC 无定义，最终表现为样本不足，而不是“增量 IC 等于 0”。

有效日足够时，原始与残差各返回均值、IR、t、正值比例和天数，并给出：

```text
delta         = raw - residual
survivor_share = residual_mean / raw_mean
```

`delta` 同时包含均值和 IR 差值；`raw_mean` 为零时 `survivor_share` 为空。比例可小于 0
或大于 1。`delta` 不是各风格可加贡献，`survivor_share` 也不是 R² 或解释率。

## 16. 与已有因子的相关性

注册位置：`correlation`，stage 为 `correlation`。核心实现见
[correlation.py](../../research/evaluation/calculate/correlation.py)。

### 16.1 因子集合

正式因子评估会扫描 `context.factor_dir` 中各族年度 Parquet，并结合因子注册表识别最后
年度文件里仍存在的因子列。只保留已注册、已落盘且有同名列的因子。`mined` 族若存在目录，
按其动态注册表解析。这个集合不是因子定义目录库的全量列表。

每个月的池内 `stock_code + trade_date` 先作为键，再左连接所有因子值。相关性计算不要求
收益标签非空，因为回答的是“两个因子排序是否重复”，不是预测能力。

挖掘候选构建的非正式 `FactorView` 使用目标对正式因子路径：完整矩阵只保留候选与各正式
因子，不构建正式因子之间的全量两两矩阵。目标行、目标列和对角线以外的 `matrix` 为
`null`，对应 `days` 为 `0`；正式因子路径仍返回完整矩阵供热力图展示。

### 16.2 成对日度 Spearman

对因子对 `(a,b)`：

1. 当日只取两列都有限的股票交集。
2. 少于 5 只时该日无效。
3. 在交集内分别重算平均并列名次。
4. 计算当日 Pearson 秩相关，即 Spearman。
5. 对有效日期做等权平均。

```text
Correlation(a,b) = average_t rho(a,b,t), 有效日数 >= 20
```

每对因子独立取交集，所以不同单元格的有效天数和股票样本可以不同。矩阵对角线也按该逻辑
计算；因子当天常数时自相关也可能无定义，因此对角线天数不一定等于评估面板天数。

实现上，每个交易日建立一次私有相关性工作区。因子有限值只打包一次，相同缺失掩码因子
直接成组，组对交集通过打包字节按位与得到；每个因子每天只生成一次稳定排序索引。
成对交集仍会按实际保留样本重新计算并列平均名次，不能复用完整截面的名次。排名矩阵直接
预分配，均值与范数由并列块大小计算，缓存只服务当前交易日，不跨请求复用。

### 16.3 输出与阅读

返回 `names`、完整均值 `matrix`、有效天数 `days`、Top-N `top`、`top_n`、
`min_days=20`、`min_stocks=5`、计算说明和 `insufficient`。

Top-N 排除当前因子自身，只保留均值有限的同侪，按绝对相关降序；绝对相关相同时按因子名
稳定排序。目标因子没有不少于 20 日的同侪时 `top=[]`、`insufficient=True`，但矩阵中其他
因子对仍可能存在。这条“其他因子对仍可能存在”只适用于正式完整矩阵；候选路径只保留
目标相关。

接近 1 表示排序高度重复；接近 -1 表示方向相反的镜像信号；接近 0 只表示截面排序弱相关，
不等于未来收益无关。每对样本不同，时间均值矩阵不保证半正定，不能直接当协方差矩阵或组合
优化输入。是否删除高相关因子还要结合 IC、换手、容量、成本和可解释性。

## 17. 前端字段与单位

页面渲染见 [charts.js](../../apps/monitor/frontend/charts.js)和
[app.js](../../apps/monitor/frontend/app.js)。浏览器不重算指标，只展示服务端 payload。

| 指标 | 当前主要展示 |
|---|---|
| `coverage` | 覆盖率和计数表；展开显示分布与每日样本数 |
| `ic_stats` | IC 与 RankIC 的均值、标准差、标准误、IR、t、正值比例和天数 |
| `ic_series` | 累计 IC、累计 RankIC 折线 |
| `ic_dist` | IC、RankIC 两个直方图和零值参考线 |
| `bin_returns` | 两套权重的均值柱、标准误线和分位明细表 |
| `bin_curves` | 两套权重的 Q 条分位净值曲线 |
| `spread` | 两套权重的多空净值和日度价差曲线及统计表 |
| `bin_dist` | 两组分位箱线和个股收益分布明细表 |
| `turnover` | 全池换手曲线、各滞后自相关表、各分位平均换手表 |
| `decay` | RankIC 均值、RankICIR 曲线，五档统计表，两套权重收益差表 |
| `exposure` | 平均暴露柱、统计表、7 条逐日系数曲线 |
| `incremental_ic` | 原始/残差 RankIC 表，有效日足够时再显示曲线和差值 |
| `correlation` | 目标因子 Top-N 表和完整相关热力图 |

payload 中相关值、覆盖率、正值比例、换手、系数为正比例、显著占比等都是原始小数，不是已经
乘 100 的百分数：

```text
0.05 = 5%
max_drawdown = -0.12 = 最大回撤 12%
```

通用格式化只做约 5 位有效数字和千分位，不稳定地为字段自动换算百分比。例外是因子列表页
把目录库 `avg_coverage` 乘 100 显示；详情页覆盖率以及本页其他比例仍显示原始小数。
累计收益、年化 Sharpe、IC、t 值和暴露系数也是原始数值。`null` 统一显示为 `—`。

## 18. 最小使用示例

只查看注册表而不读正式数据：

```bash
PYTHONPATH="$PWD:$PWD/apps/monitor" .venv/bin/python - <<'PY'
from calculate.metrics import list_metrics

for metric in list_metrics():
    print(metric.key, metric.stage, metric.label, metric.requires_quantiles)
PY
```

只运行覆盖、分层和衰减，验证当前数据与 stage：

```bash
PYTHONPATH="$PWD:$PWD/apps/monitor" .venv/bin/python - <<'PY'
from calculate import EvalParams, evaluate

params = EvalParams(
    factor="MOM_20",
    start="2024-05-06",
    end="2024-05-31",
    universe="all_a",
    horizon=5,
    price="close",
    quantiles=10,
    winsorize=True,
)

for stage, payload in evaluate(
    "MOM_20",
    params,
    stages=("overview", "bins", "decay"),
):
    metrics = payload.get("metrics")
    keys = sorted(metrics) if isinstance(metrics, dict) else []
    print(stage, keys)
PY
```

示例只读主库和因子 Parquet，不修改正式数据。输出数值会随当前数据快照变化；应核对 stage
和 payload 键，而不是把示例日期的结果当成固定基准。

## 19. 错误、日志与测试映射

### 19.1 指标相关错误

| 情况 | 错误码 | 发生位置 |
|---|---|---|
| 指标未知、重复注册、stage 非法、参数非法 | `EVAL-201` | 注册表、`resolve_options()`、相关性参数校验 |
| 空池风格拆分、风格主键重复、样本没有任何风格值 | `EVAL-301` | `exposure.run_exposure()` |
| 主库或指数收益不足，无法计算 BETA_60 | `EVAL-302` | `styles.build_styles()` |
| 行业表缺失、为空或数据源未知 | `EVAL-303` | `styles.py` |

面板日期、行数、因子和股票池错误通常在指标运行前以 `EVAL-101`～`EVAL-104` 返回。
HTTP/SSE 如何映射这些错误见 [03-factor-evaluation.md](03-factor-evaluation.md)。

计算包日志写入 `logs/research/evaluation/<模块名>/`，由
[logging_conf.py](../../research/evaluation/calculate/logging_conf.py)统一配置。大多数纯指标
不单独写日志；风格取数、回归和相关性按月读取会记录样本规模、设计矩阵列数、跳过日和月度块
耗时。排查时先看后端错误码，再看对应模块日志和面板 `meta` 概览。

### 19.2 测试覆盖

| 测试 | 主要证明 |
|---|---|
| [test_eval_registry.py](../../research/evaluation/calculate/tests/test_eval_registry.py) | 13 指标注册顺序、参数声明、重复注册和扩展 |
| [test_eval_evaluate.py](../../research/evaluation/calculate/tests/test_eval_evaluate.py) | 完整 stage 顺序、payload 可序列化、默认口径 |
| [test_eval_ic.py](../../research/evaluation/calculate/tests/test_eval_ic.py) | IC/RankIC、去极值、最小截面、20 日门槛、分布和覆盖率 |
| [test_eval_binning.py](../../research/evaluation/calculate/tests/test_eval_binning.py) | 分箱、两类权重、换手、自相关、净值、价差和分布 |
| [test_eval_decay.py](../../research/evaluation/calculate/tests/test_eval_decay.py) | 五档独立样本、RankIC、收益差、空池和缺列 |
| [test_eval_exposure.py](../../research/evaluation/calculate/tests/test_eval_exposure.py) | 标准化、日度 OLS、缺失风格、行业哑变量和残差 IC |
| [test_eval_correlation.py](../../research/evaluation/calculate/tests/test_eval_correlation.py) | 成对重排名、掩码尾部位、单日缓存、门槛、Top-N、池过滤和标签无关性 |
| [test_eval_regressions.py](../../research/evaluation/calculate/tests/test_eval_regressions.py) | 常数因子、全解释残差、空池、参数和缓存边界 |
| [test_eval_styles.py](../../research/evaluation/calculate/tests/test_eval_styles.py) | 风格 SQL、真实数据覆盖、静态行业源 |
| [test_mining_evaluation_consistency.py](../../tests/test_mining_evaluation_consistency.py) | 候选与正式路径共用全部指标口径 |

新增或修改指标时至少检查：

1. `Metric` 声明和 `STAGE_ORDER` 是否一致。
2. 输入列、空值、无穷值、常数截面和样本不足是否都有明确语义。
3. 返回字典是否能 JSON 序列化，前端未知字段是否仍能通用展示。
4. 指标是否复用 `EvalContext` 缓存，避免重复透视和重复取风格。
5. 多日标签是否保持独立样本，净值和价差是否误用重叠期收益累乘。
6. 风格参数是否在 `exposure` 和 `incremental_ic` 两项独立校验与回显。
7. 相关性是否只扫描池内键、成对删除后重排名且不使用收益标签筛选。
8. 前端标题、字段名、比例单位和空值行为是否与 payload 一致。
9. 定向单元测试和联合测试是否覆盖正常路径、样本不足与失败路径。
10. 文档中的源码链接、测试链接、stage 和门槛是否仍对应当前实现。

## 20. 推荐阅读顺序与系统边界

阅读一个因子结果时，建议按以下顺序：

1. 覆盖率：先确认因子、标签和股票池是否有足够样本。
2. IC/RankIC：确认方向和平均预测关系。
3. IC 时序与分布：确认是否靠少数日期驱动。
4. 分位收益和单调性：确认是否存在完整横截面梯度。
5. 净值、多空和收益分布：观察路径、尾部及规模口径。
6. 换手、自相关和衰减：判断信号持续期和潜在成本压力。
7. 风格暴露、增量 IC 和相关性：判断风格代理和同侪冗余。

系统边界：

- 页面只展示服务端指标，不在浏览器重算因子、收益或回归。
- 完整分析受 2400 个交易日和约 800 万面板行保护；全历史轻量 IC 不证明其他指标也能全历史运行。
- 普通评估结果不落库，刷新页面需要重新计算。
- 分位净值、多空价差和分位收益是研究代理，没有完整 A 股交易规则、容量和执行成本。
- 所有结论依赖日期、股票池、标签周期、价格通道、因子方向和参数；不同口径不能直接横向比较。
