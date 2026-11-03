# 数据审计工具

`tools/data_audit/` 只提供只读诊断脚本，不是正式构建入口。正式数据链路以根 README
中的 `scripts.sources`、`scripts.clean` 和 `scripts.build_db` 命令为准。

```bash
# 原始分片覆盖
.venv/bin/python tools/data_audit/diag_coverage.py

# 股票代码归一化问题
.venv/bin/python tools/data_audit/scan_codes.py

# 公司行动和行业覆盖
.venv/bin/python tools/data_audit/audit_corporate_actions.py \
  --db data/databases/quant.duckdb
.venv/bin/python tools/data_audit/audit_sw_industry.py \
  --db data/databases/quant.duckdb --start 2016-01-01 --end 2026-10-15
```

这些脚本读取的数据量较大时可能耗时数分钟。它们不应修改正式数据库；若脚本参数要求
输出目录，应使用临时目录。
