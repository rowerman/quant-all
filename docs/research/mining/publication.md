# 人工审批后的正式入库

## 职责

S8 把同一冻结上下文内全部 `approved` 候选作为一个批次，重建完整 `mined` 因子族，
执行全历史校验，再原子替换正式因子链接和因子目录库。批准本身仍只写候选库；用户必须
在 Web 页面另行确认整批发布。

```text
approved 候选（全部）
  -> MiningPublicationService 资格检查
  -> publication_builder      全量 mined 构建与独立校验
  -> 版本目录 + 原子符号链接   正式 Parquet 可见性切换
  -> factors.duckdb 原子替换   定义、覆盖率、清单和 v.mined 视图
  -> published                候选与发布记录收口
```

## 发布资格

- 批次必须包含运行内全部 `approved` 候选，不能漏选或混入其他状态。
- 候选内容哈希、上下文哈希和审批证据哈希必须与批准记录一致。
- 技术门禁、通过的模块测试段报告和测试段领取批次必须仍然有效。
- 行情快照、字段目录、算子目录和当前正式因子清单必须与冻结上下文一致。
- 同一候选库只允许一个 `staging/validating` 批次。
- 审批没有时间有效期；上述任一内容变化都会阻止发布。过期候选不得沿用旧批准。

## 全量构建与校验

构建范围固定为 `2000-01-04` 至冻结上下文的测试段结束日。每次发布重建“此前已发布
`mined` 定义 + 本批新定义”，避免发布结果依赖逐次拼接的历史文件。

正式写入前必须通过：

1. 与 `md.daily_hfq` 双向前后行集一致，且 `(stock_code, trade_date)` 唯一；
2. 所有因子列均为 float32，落盘值不存在 NaN 或无穷值；
3. 每个因子全历史覆盖率不低于冻结上下文 `min_coverage`；
4. 首、中、末三个截止日重算结果与全量结果一致，容差 `1e-5`；
5. 不与内置因子或其他 `mined` 因子重名，不与其他已发布候选语义重复。

通过后写入 `definitions.json`，其中保留运行、候选、审批、内容哈希和语义哈希来源。

## 原子提交

正式 `mined` 族采用版本目录：

```text
data/factors/.mined-versions/<batch_id>/year=YYYY/part.parquet
data/factors/.mined-versions/<batch_id>/definitions.json
data/factors/mined -> .mined-versions/<batch_id>
```

提交顺序为：写入提交日志、临时符号链接原子替换、因子目录库原子替换、写
`committed` 标记、最后更新候选与批次状态。提交前会保留旧链接目标和旧因子库副本。

任一文件步骤失败时恢复旧链接和旧因子库，把整批候选置为 `build_failed`。服务启动时，
没有 `committed` 标记的在途批次一律回滚；已有标记但状态未写入数据库的批次完成收尾。

## 已发布因子读取

`mined` 只在存在正式清单或落盘文件时出现，不要求它作为源码内置空族存在。

- `factors.all_factors()` 仍只返回源码内置定义；
- `factors.resolve(name, factor_dir)` 同时解析内置与已发布定义；
- `factors.available_specs(factor_dir)` 返回两类定义的合集；
- `quant.load_factor_list()` 从正式因子目录库读取 `mined` 元数据；
- `quant.load_factor_panel/wide()`、评估、模型特征和回测通过统一解析入口读取数值。

普通源码构建命令只重建内置族，不会自动覆盖人工批准的 `mined` 族。

## API 与页面

```text
POST /api/mining/runs/{run_id}/publication-batches
GET  /api/mining/runs/{run_id}/publication-batches
GET  /api/mining/runs/{run_id}/publication-batches/{batch_id}
```

POST 请求体为 `candidate_ids` 和 `expected_context_hash`，受理后返回 `202`。构建在后台
执行，页面轮询批次状态。候选页展示批次、候选状态、定义清单哈希、产物树哈希和错误。

错误码：

| 错误码 | 含义 |
|---|---|
| `MINING-121` | 发布资格、批次组成或并发冲突 |
| `MINING-122` | 全量行集、取值、覆盖率、前视或冲突校验失败 |
| `MINING-123` | 提交、回滚或重启恢复失败 |

## 验收

```bash
.venv/bin/python -m pytest -q tests/test_mining_publication.py
.venv/bin/python -m pytest -q apps/monitor/backend/tests/test_mining_api.py
.venv/bin/python tools/monitor/mining_check.py --output AiRecord/evidence/mining_s8
```

真实全量测试使用临时因子目录、临时因子目录库和临时 `mined` 链接，只读访问真实行情库；
不会修改生产 `data/factors/` 或 `data/databases/factors.duckdb`。
