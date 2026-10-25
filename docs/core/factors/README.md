# 因子库使用说明

因子库建在现有日线库之上：因子定义写在 `core/factors/` 包里，构建脚本算出因子值落成
Parquet，目录库记下每个因子的定义与覆盖率，读取仍然走 `quant`。

```
core/factors/                    # 因子定义：算子、注册表、十一个内置族和已发布 mined 解析
core/data/scripts/build_factors/      # 构建（build）、自检（validate）、目录与文档（catalog）
data/factors/<族>/year=YYYY/part.parquet   # 因子值，一年一族一个文件
data/databases/factors.duckdb           # 因子目录库：定义、覆盖率、自检结论、构建记录、视图
docs/core/factors/               # 本目录：使用说明、口径、清单、待补数据
```

## 快速上手

```bash
# 1) 构建全部已注册因子（默认全历史 2000-01-04 起，逐族逐年算，可反复重跑）
.venv/bin/python -m scripts.build_factors.build

# 只建某几族，或只建某段年份
.venv/bin/python -m scripts.build_factors.build --families momentum,valuation
.venv/bin/python -m scripts.build_factors.build --families momentum --start 2024-01-01 --end 2024-12-31

# 2) 自检：行集、取值、前视偏差、方向
.venv/bin/python -m scripts.build_factors.validate
.venv/bin/python -m scripts.build_factors.validate --families volatility --samples 20

# 3) 重新生成因子清单文档
.venv/bin/python -m scripts.build_factors.catalog --write-docs
```

```python
import quant

# 长表：键 + 指定因子列
df = quant.load_factor_panel(
    start="2024-01-01", end="2024-12-31",
    names=["MOM_20", "EP_TTM", "PV_CORR_20"],
)

# 按族整批取
df = quant.load_factor_panel(start="2024-01-01", end="2024-12-31", families=["valuation"])

# 宽表：交易日 × 股票，便于做截面运算
mom20 = quant.load_factor_wide("MOM_20", start="2024-01-01", end="2024-12-31")

# 因子清单：族、公式、方向、覆盖率；status=implemented / planned
quant.load_factor_list(family="momentum")
quant.load_factor_list(status="planned")
quant.load_factor_list(family="fundamental")
```

也可以直接查目录库：

```sql
-- 已实现因子的平均覆盖率
SELECT name, avg_coverage FROM meta.factor_coverage c JOIN meta.factor_definition d USING (name)
WHERE d.status = 'implemented' GROUP BY ALL ORDER BY 2;

-- 自检结论
SELECT * FROM meta.factor_check WHERE kind = 'direction';
```

按族的视图已经建好（`v.momentum`、`v.volatility`…），SQL 里可以直接查：

```sql
SELECT * FROM v.momentum WHERE trade_date = DATE '2024-01-02' LIMIT 10;
```

人工审批后批量发布的因子位于 `data/factors/mined/`，目录是指向
`.mined-versions/<batch_id>` 的原子符号链接；对应清单为 `definitions.json`。`v.mined`
由正式目录库维护。`factors.resolve(name, factor_dir)` 统一解析内置和已发布因子，
普通 `build_factors` 只重建源码内置族，不会覆盖 `mined`。

## 怎么新增一个因子

三步，不用改构建脚本：

1. 在对应族模块（例如 `core/factors/technical.py`）里写一行定义：

```python
from factors.ops import ts_mean, ts_std
from factors.registry import define

define(
    name="BOLL_WIDTH_20",
    family="technical",
    formula="4 * std(close,20) / mean(close,20)",
    description="布林带带宽，波动状态的代理。",
    steps=(
        "算 20 日收盘价标准差 std(close, 20) 与 20 日均价 mean(close, 20)",
        "标准差乘 4 再除以均价，得到相对带宽",
    ),
    layers=f"4 * {ts_std('close', 20)} / {ts_mean('close', 20)}",
    direction=-1,          # 值越大越看空
    fields=("close",),
)
```

2. 跑构建（会自动补上跨年窗口，并把定义与覆盖率写进目录库）：

```bash
.venv/bin/python -m scripts.build_factors.build --families technical
```

3. 跑自检并重新生成清单文档：

```bash
.venv/bin/python -m scripts.build_factors.validate --families technical
.venv/bin/python -m scripts.build_factors.catalog --write-docs
```

要点：

- `steps` 是**必填**的白话步骤，会出现在 `formulas.md` 与 `--explain` 里，也是文档可信度的来源；
- `layers` 可以写多个表达式，用 `v1`、`v2`… 引用上一层结果，用于「先算行号再算极值位置」这类嵌套写法；
- `lookback` 留空会按表达式自动推导（从 `lag(x, n)` 与 `rows between n preceding` 里取最大值），
  推导值不够会直接报错；
- `direction` 只影响自检时对 IC 符号的核对，不影响因子值；
- `extra_sources` 用来引用指数等附加数据，见 `core/factors/sources.py`。

## 目录

| 文档 | 内容 |
|---|---|
| `README.md` | 本文件：构建、读取、扩展因子的操作步骤 |
| `cookbook.md` | 入门教程：从一只股票算起，pandas / SQL 手算并与库值对照 |
| `formulas.md` | 47 个已实现因子的「公式 → 白话步骤 → 真实 SQL」（自动生成） |
| `catalog.md` | 全部 186 条因子清单（自动生成）：族、状态、方向、公式、说明 |
| `conventions.md` | 因子口径、算子语义、已知限制与多层写法 |
| `missing_data.md` | 要补哪些数据才能做基本面与资金类因子，含接口名 |

想看某个因子到底怎么算，三种方式都行：

```bash
.venv/bin/python -m scripts.build_factors.catalog --explain MOM_20      # 命令行讲解 + 可跑的手工复现 SQL
```

```python
import quant
quant.load_factor_list(family="momentum")[["name", "formula", "steps", "expr"]]
```

也可以直接翻 `formulas.md`（按族分组）或 `cookbook.md`（手把手教程）。
