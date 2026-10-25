# 受控因子表达式 DSL

## 边界

DSL 只接受数字、冻结字段、冻结算子、括号、逗号和四则运算。解析器不使用 Python
`eval`，也不拼接调用方提供的 SQL 文本。字符串、日期字面量、注释、反引号、字段限定名、
子查询、多语句和未登记函数都会在执行前失败。

## 语法

```text
expression := additive
additive   := multiplicative (("+" | "-") multiplicative)*
multiplicative := unary (("*" | "/") unary)*
unary      := ("+" | "-") unary | primary
primary    := number | field | function "(" expression ("," expression)* ")"
```

函数名不区分大小写，`delay`、`average`、`stddev` 分别规范化为 `lag`、`mean`、`std`。
字段名和冻结目录中的名称一致，区分大小写。

## 算子与窗口

- 四则运算：`+`、`-`、`*`、`/`，除法使用安全分母。
- 标量：`abs`、`ln`、`sqrt`、`sign`、`least`、`greatest`、`signed_power`。
- 时序：`lag`、`delta`、`mean`、`sum`、`min`、`max`、`median`、`std`、`skew`、
  `kurt`、`product`、`count`、`corr`、`cov`、`slope`、`r2`、`downside_std`、
  `ts_rank`。
- 截面：`rank`、`zscore`、`demean`、`scale`。

窗口参数必须写成整数，范围为 1～252。`ts_rank`、`corr`、`cov`、`slope`、`r2` 和
`downside_std` 的最小窗口为 2。`signed_power` 的指数范围是 -4～4。

## 量纲

- 加法和减法要求同量纲，数字字面量作为标量可与任一量纲相加；
- 乘法遇到标量时保留另一侧量纲，其余组合记为混合量纲；
- 同量纲相除得到标量，数字字面量作分母时保留分子量纲；
- 排名、标准化、相关系数和 `r2` 输出标量；
- 布尔字段在执行时转换为 0/1，可用于聚合、算术或截面算子。

## 分层编译

DuckDB 不允许窗口函数嵌套窗口函数。编译器按后序顺序把每个时序或截面窗口写成
`v1`、`v2` 等独立层，外层只引用已有列。例如：

```text
rank(mean(close, 5) + lag(close, 1))
```

编译为三层：五日均值、一日滞后、截面排名。未来查询层逐层执行，并统一提供
`WINDOW w AS (partition by stock_code order by trade_date)`。

## 规范化与稳定命名

解析器统一数字格式、函数别名、空白和可交换参数顺序。语义哈希使用 S2 候选的规范化
输入，稳定因子名为 `MINE_` 加语义哈希前 24 位大写十六进制。相同表达式和方向得到相同
名称；方向变化会得到不同名称。

## 财务时点

财务字段沿用 `factors.sources` 的 ASOF 连接：同一股票先取最新报告期，再要求
`trade_date > available_date`。公告当日不可见，次一交易日才可进入因子。用到的来源
通过 `ValidatedExpression.extra_sources` 显式返回。

## 错误

`MINING-114` 覆盖语法、字段、算子、参数、窗口、深度、长度、节点数和量纲错误。
`MINING-115` 覆盖目录损坏或与实际 schema 漂移。日志只记录错误原因和目录哈希，
不记录完整表达式。
