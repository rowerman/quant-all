# 运行与审计

## 目录

| 路径 | 内容 |
|---|---|
| `data/source` | 外部原始分片 |
| `data/clean` | 清洗分区 |
| `data/databases/quant.duckdb` | 行情主库 |
| `data/factors` | 因子 Parquet |
| `data/models` | 数据集、模型、预测和验收 |
| `data/databases/backtest.duckdb` | 回测结果 |
| `logs` | 按模块组织的日志 |

## 审计

`tools/data_audit` 提供只读数据诊断：

```bash
.venv/bin/python tools/data_audit/diag_coverage.py
.venv/bin/python tools/data_audit/scan_codes.py
.venv/bin/python tools/data_audit/audit_corporate_actions.py \
  --db data/databases/quant.duckdb
```

## 测试

```bash
make test-unit
make test-integration
make test-e2e
make test-full
make lint
```

默认测试不读真实数据、不访问网络、不调用付费模型。真实数据完整测试可能与本地数据
版本绑定，应先核对数据清单和哈希。

## 发布

GitHub Actions 在 push 和 PR 上运行静态检查、快速测试和 wheel 构建。集成测试使用手工
或定时任务。发布标签 `v*` 会构建 sdist、wheel 并创建 GitHub Release。

