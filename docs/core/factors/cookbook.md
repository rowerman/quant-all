# 因子计算入门：从一只股票算起

这篇写给第一次接触因子的人。目标很简单：**看完能自己把 MOM_20 算出来，并且和库里的数对上**。
文中的每段代码都在本机实测跑过。

## 0. 先建立三个概念

1. **因子就是「某一天、某只股票的一个数字」**。MOM_20 = 它过去 20 个交易日的涨跌幅，
   EP_TTM = 它当前盈利收益率的倒数，RSI_14 = 它最近 14 天的相对强弱。没有更玄的东西。
2. **因子表长这样**：`data/factors/<族>/year=YYYY/part.parquet`，主键是
   `stock_code` + `trade_date`，后面每一列是一个因子。年分区的目的是让「按年重算」变便宜。
3. **「公式」和「实现」是一回事的两种写法**：`docs/core/factors/formulas.md` 里的公式是给人看的
   （`close_t / close_(t-20) - 1`），实现表达式是给机器跑的 SQL
   （`close / lag(close, 20) over w - 1`），两者算出来必须完全一样。

## 1. 取一只股票的数据

```python
import quant

df = quant.load_panel(
    start="2024-01-01", end="2024-03-31", stocks=["SH600000"],
    fields=["close", "ret", "amount_cny", "turnover_rate"],
    adjust="hfq",          # 后复权价，做因子必须用它
    drop_suspended=True,   # 停牌日没有行情，直接从窗口里去掉
)
df.head()
```

两个参数不能省：

- `adjust="hfq"`：用未复权价时，除权日价格会凭空跳一大截，动量/波动全都会算错；
- `drop_suspended=True`：停牌日没有成交，不该占用「20 个交易日」里的位置。

## 2. 用 pandas 自己算 MOM_20

```python
df = df.sort_values("trade_date")
df["close_20d_before"] = df["close"].shift(20)      # 20 个交易日前的收盘价
df["MOM_20_mine"] = df["close"] / df["close_20d_before"] - 1
df[["trade_date", "close", "close_20d_before", "MOM_20_mine"]].head(25)
```

注意 `shift(20)` 是**往前看 20 天**。写成 `shift(-20)` 就变成用未来的价格，是最常见也最致命的错误。

多只股票一起算时，要先分组，否则会把上一只股票的价格错位接上来：

```python
df = df.sort_values(["stock_code", "trade_date"])
df["MOM_20_mine"] = df.groupby("stock_code")["close"].transform(lambda s: s / s.shift(20) - 1)
```

## 3. 和库里的值对照

```python
lib = quant.load_factor_panel(
    start="2024-01-01", end="2024-03-31", names=["MOM_20"], stocks=["SH600000"],
)
merged = df[["trade_date", "MOM_20_mine"]].merge(lib, on="trade_date")
(merged["MOM_20_mine"] - merged["MOM_20"]).abs().max()
```

算出来的差异应该在 `1e-6` 量级，**不是 0 也是对的**：库里存的是 float32，只保留约 7 位有效数字，
而 pandas 用的是 float64。要判对错，看数量级就够，别用 `==` 比较浮点。

本机实测（SH600000，2024-01-01 ~ 2024-03-31）：58 行，最大差 `3.7e-09`。

## 4. 用 SQL 算同一件事（库里就是这么算的）

```sql
-- 第一步：在「全部历史」上算窗口，先把 base 铺好
WITH base AS (
    SELECT * FROM md.daily_hfq
    WHERE status = 'trading' AND trade_date <= DATE '2024-03-31'
)
-- 第二步：算因子，再筛出你要看的股票与区间
SELECT trade_date, MOM_20
FROM (
    SELECT trade_date,
           close / lag(close, 20) OVER (PARTITION BY stock_code ORDER BY trade_date) - 1 AS MOM_20
    FROM base
    WHERE stock_code = 'SH600000'
) t
WHERE trade_date >= DATE '2024-01-01'
ORDER BY trade_date;
```

**顺序很关键**：窗口必须在全部历史上算完，最后才过滤出要看的区间。
如果把 `trade_date BETWEEN '2024-01-01' AND '2024-03-31'` 写进窗口里面，
1 月初就没有「20 天前」可看，那几行会变成空值——这是新手最容易踩的坑，
构建脚本也是按「先算窗口、后写当年」来组织的。

本机实测：与库里逐行一致，最大差 `3.7e-09`。

Python 里跑这段 SQL：

```python
from quant.db import get_conn

sql = """上面那段"""
with get_conn() as conn:      # 只读连接，和别的进程并存
    print(conn.execute(sql).df())
```

## 5. 换个复杂点的：RSI_14 怎么拆

RSI 的公式是 `100 × 平均涨幅 ÷（平均涨幅 + 平均跌幅）`，拆成三步就清楚了：

1. 每天的收益拆成两部分：`greatest(ret, 0)` 是涨的部分，`greatest(-ret, 0)` 是跌的部分；
2. 两部分各取 14 日平均；
3. 相除。

```sql
WITH base AS (
    SELECT * FROM md.daily_hfq WHERE status = 'trading' AND trade_date <= DATE '2024-03-31'
)
SELECT trade_date, RSI_14
FROM (
    SELECT trade_date,
           100 * avg(greatest(ret, 0)) OVER w14
               / nullif(avg(greatest(ret, 0)) OVER w14 + avg(greatest(-ret, 0)) OVER w14, 0) AS RSI_14
    FROM base
    WHERE stock_code = 'SH600000'
    WINDOW w14 AS (PARTITION BY stock_code ORDER BY trade_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
) t
WHERE trade_date >= DATE '2024-01-01'
ORDER BY trade_date;
```

本机实测：58 行，与库里最大差 `3.8e-06`——RSI 取值范围是 0~100，
float32 在这个量级上的精度就是这样，属于正常。

SQL 的关键字：`WINDOW w14 AS (...)` 定义一个可复用的窗口，
`ROWS BETWEEN 13 PRECEDING AND CURRENT ROW` 表示「当前行和往前 13 行」，一共 14 行。
构建脚本里 `core/factors/ops.py` 的函数（`ts_mean`、`ts_std` …）就是把这类窗口拼成字符串，
所以你写因子时只需要写 `ts_mean("close", 20)`，不用手写窗口。

## 6. 新手最容易错的五件事

| 错误 | 写错的样子 | 对的样子 | 为什么 |
|---|---|---|---|
| 用了未来数据 | `s.shift(-20)` | `s.shift(20)` | 因子只能用 t 日及以前的信息，反过来就是作弊 |
| 用未复权价 | `adjust="raw"` 算动量 | `adjust="hfq"` | 除权日价格跳空，动量里会混进一个假涨跌 |
| 停牌当正常日 | 不筛状态直接 `shift` | `drop_suspended=True` 或筛 `status='trading'` | 停牌日没有成交，占窗口会把「20 个交易日」拉长成几个月 |
| 历史不足当 0 | `fillna(0)` | 留空 | 次新股没有 60 日历史时填 0，会变成「动量为 0」的假信号 |
| 先过滤后算窗口 | 日期写在窗口 CTE 里 | 先算窗口，再过滤输出 | 月初没有历史，长窗口因子会集体变空（实测：58 行里有 20 行变空） |

## 7. 想看别的因子怎么算

```bash
# 命令行讲解：公式 + 白话步骤 + 可直接跑的手工复现 SQL
.venv/bin/python -m scripts.build_factors.catalog --explain RSI_14
.venv/bin/python -m scripts.build_factors.catalog --explain BETA_60 --stock SZ000001 --start 2023-06-01 --end 2023-12-31
```

```python
import quant

# 全部因子的公式、白话步骤、实现表达式、方向、覆盖率
quant.load_factor_list(family="technical")

# 只想看还没实现的规划因子
quant.load_factor_list(status="planned")
```

- `docs/core/factors/formulas.md`：47 个已实现因子的「公式 → 步骤 → 真实 SQL」；
- `docs/core/factors/catalog.md`：186 条因子的清单（含待实现项的公式）；
- `docs/core/factors/conventions.md`：每个算子的准确含义（`ts_rank` 为什么不用 `percent_rank` 之类）。

## 8. 想自己写一个因子

看 `docs/core/factors/README.md` 的三步；简言之就是在 `core/factors/` 的族模块里写一条
`define(name=..., formula=..., steps=..., layers=...)`，然后跑构建与自检。
`steps` 就是上面这种白话步骤——写给自己看的，也写给以后复习的自己看。
