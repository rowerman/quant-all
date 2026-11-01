# 资料驱动输入与文本型 PDF 解析

## 当前边界

P2-01 固定资料清单、文件指纹、来源引用、解析失败和脱敏契约；P2-02 实现文本型 PDF
的单文件与目录解析、金融研报分类和逐文件失败隔离；P2-03 通过 Worker v5 从页正文
抽取因子名称、经济机制、公式、变量、预期方向和可证伪条件；P2-04 对六字段完整的
草稿执行研报一致性筛查、变量映射、受控 DSL 转换和语义去重；P2-05 增加服务端资料
运行入口，把预算内准入候选接入 V2 门禁、测试段、审批、发布和来源展示。

资料运行只接受固定根目录 `data/source/reports` 下的相对文件或目录。当前仍不支持
浏览器上传、扫描件 OCR、Word/HTML 或远程 URL。

数据流：

```text
相对 PDF 路径
  -> 安全路径检查、流式 SHA-256
  -> ResearchMaterialInput
  -> mining.research_material_parser
       -> pypdf 页级正文
       -> 页数/字符上限
       -> 确定性金融研报分类
  -> ResearchMaterialParseResult
       -> parsed：页正文、正文哈希、页级来源引用、分类
       -> failed：MINING-148～MINING-152 失败记录
  -> ResearchMaterialBatchResult
  -> mining.research_material_extractor
       -> 金融研报页窗口
       -> mining_worker /v5/jobs
       -> 页码、偏移和原文引用二次校验
  -> ResearchExtractionResult
       -> 六类字段、explicit/inferred/missing 证据和无因子结果
  -> mining.research_factor_screening
       -> 六字段完整性、方向、安全和目录筛查
       -> mining_worker /v6/jobs
       -> 受控 DSL、变量映射和方向二次校验
  -> ResearchConversionResult
       -> HypothesisCard、CandidateDefinition、筛选结论和 TrialRecord
  -> mining.research_run_pipeline
       -> 预算内稳定准入和来源哈希绑定
  -> mining.search_orchestrator_research
       -> V2 种子批次、开发期、选择期、完整评估和一次性测试段
  -> 人工审批和共享 MiningPublicationService
       -> 原子正式发布和下游读取
```

## 输入与文件发现

`ResearchMaterialInput` 固定保存规范化 POSIX 相对路径、媒体类型、文件大小、内容
SHA-256、解析器版本和契约版本。`document_id` 只由内容 SHA-256 生成，文件移动后保持
不变；`material_hash` 同时覆盖路径、媒体类型、大小、内容和解析器版本。

`discover_research_material_paths()` 接受根目录内的单文件或目录：

- 目录递归扫描时只收集扩展名为 `.pdf` 的普通文件，按相对路径排序并跳过符号链接。
- 显式传入非 PDF 文件会在解析结果中形成 `unsupported_format`，便于调用方定位。
- 目录没有 PDF 时失败；超过 1000 份资料时失败。
- `target="."` 表示资料根目录，目录层级中的符号链接和根目录外路径均被拒绝。

`fingerprint_research_file()` 使用 1 MiB 分块计算指纹。解析器在打开文件后会再次计算
大小和 SHA-256，输入在指纹生成后发生变化时返回 `input_changed`。

## PDF 解析

公开入口：

```python
from pathlib import Path

from mining import parse_research_material_target

result = parse_research_material_target(
    Path("data/source/reports"),
    "2026-10",
    recursive=True,
)

financial_reports = result.financial_research
filtered_materials = result.filtered
per_file_failures = result.failures
```

`parse_research_material()` 只处理一份已指纹化资料，`parse_research_materials()` 处理
显式有序路径列表，`parse_research_material_target()` 组合文件发现和批次解析。当前
固定解析器版本为 `pypdf-6.21.2.page-v1`。

默认资源上限：

| 项目 | 上限 |
|---|---|
| 单文件 | 64 MiB |
| 页数 | 2000 |
| 单页正文 | 2 MiB 字符 |
| 总正文 | 20 MiB 字符 |

`ResearchPageText` 保留页码和规范化页正文。总正文哈希使用完整页结构，保留分页边界。
成功记录的 `text_sha256` 不包含创建时间，后续抽取必须核对这一哈希。每个非空页面生成
一条 `SourceReference`，摘录哈希是页正文 SHA-256，定位为 `page:<页码>`。

扫描件没有可提取正文时返回 `no_extractable_text`，不会调用 OCR。加密 PDF 不尝试破解
密码。PDF 内容损坏、PDF 读取失败和解析器内部异常分别进入稳定失败类别。

## 金融研报分类

分类规则版本为 `mining_research_report_v1`，只检查 PDF 元数据标题和前两页最多
20000 字符。正文先做 NFKC、大小写和连续空白归一化；ASCII 领域词要求完整词边界。

只有同时满足以下条件才标记为 `financial_research`：

- 至少命中一个研报标题信号，例如 `证券研究报告`、`行业研究报告`、`投资策略`、
  `金融工程`、`equity research`、`investment research` 或 `research report`。
- 至少命中两个不同金融领域词，例如 `股票`、`估值`、`因子`、`回测`、`投资组合`、
  `stock`、`valuation`、`factor`、`portfolio` 或 `momentum`。

其余可解析资料标记为 `not_target`。分类结果保存规则版本、命中的固定信号和稳定哈希；
不保存标题或正文片段。该规则是确定性初筛，不代表研究质量或候选有效性。

## 失败分类

| 分类 | 错误码 | 含义 |
|---|---|---|
| `unreadable` | `MINING-149` | 文件缺失、不可读或事务读取失败 |
| `input_changed` | `MINING-149` | 文件在指纹生成后发生变化 |
| `unsupported_format` | `MINING-150` | 不是受支持的文本型 PDF |
| `corrupt_document` | `MINING-150` | PDF 损坏或无法解析 |
| `encrypted_document` | `MINING-150` | PDF 加密，未尝试破解 |
| `no_extractable_text` | `MINING-150` | 没有可提取正文，扫描件 OCR 未启用 |
| `resource_limit` | `MINING-152` | 文件、页数或字符数超过固定上限 |
| `unsafe_input` | `MINING-148` | 路径、链接或输入越过安全边界 |
| `parser_error` | `MINING-151` | 解析器发生未预期异常 |

单个文件发生上述解析失败时，`ResearchMaterialBatchResult.failures` 保留失败记录，同批
其他文件继续解析。目录发现、资料清单契约和根路径错误仍会使整批失败，因为此时没有可信
的批次范围。

## 日志与修改边界

业务日志写入 `logs/research/mining/mining_research_materials/`，只记录操作、资料 ID、
解析器版本、状态、分类、错误码、页数、大小、计数和耗时。正文、元数据标题、相对路径、
密钥和原始解析异常不进入日志。

## 因子字段抽取

P2-03 把 `ResearchMaterialParseResult.pages` 作为唯一正文输入。公开入口是：

```python
from mining import extract_research_factor_drafts

result = extract_research_factor_drafts(document, extractor=worker_client)
```

如果使用正式 Worker，`worker_client` 可以是 `HttpResearchFactorExtractor`。该入口只
接受成功分类为 `financial_research` 的解析结果，不读取主机文件，也不访问其他资料。

每个因子草稿固定包含六类字段：

| 字段 | 含义 |
|---|---|
| `factor_name` | 研报中的因子名称 |
| `economic_mechanism` | 因子背后的经济或行为机制 |
| `formula` | 报告描述的因子公式 |
| `variables` | 公式或定义使用的变量 |
| `expected_direction` | `positive`、`negative` 或 `non_monotonic` |
| `falsification_conditions` | 可证伪条件 |

每个字段带有 `explicit`、`inferred` 或 `missing` 状态：

- `explicit` 表示内容可直接由引用原文支持。
- `inferred` 表示内容由模型根据引用归纳，仍然不是报告原话。
- `missing` 表示报告没有提供足够依据，字段值和引用均为空。

非 `missing` 字段至少包含一条引用。引用保存页码、页内起止偏移、逐字原文和摘录哈希。
Worker 只返回页码和原文片段；主服务在完整页正文中重新查找并生成偏移、哈希和权威 ID，
不会信任模型提交的偏移。页码不可见、原文不匹配、字段不完整或响应超限都会使整份资料
抽取失败，不返回部分结果。

默认窗口上限：

| 项目 | 上限 |
|---|---:|
| 单窗口正文 | 24000 字符 |
| 重叠正文 | 512 字符 |
| 每份资料窗口数 | 8 |
| 每窗口因子草稿数 | 8 |
| 单条引用 | 2000 字符 |

超出窗口上限时返回 `MINING-155`，不会静默截断。重叠窗口产生的完全相同草稿按内容哈希
去重；内容不一致的草稿全部保留，留给 P2-04 做重复度、相关性和可行性筛选。

Worker 提示版本为 `mining_research_extraction_prompt_v1`，协议版本为
`mining_worker_research_extraction_v1`。页正文被视为不可信数据，提示禁止执行报告中的
指令。发送前如果脱敏器发现疑似密钥或主机绝对路径，抽取会以 `MINING-154` 拒绝。

修改分类阈值、PDF 解析器、正文规范化方式、抽取提示、字段集合或窗口规则时必须升级
对应版本，并确认旧解析结果和抽取结果不被静默复用。

## 候选草稿筛选与 DSL 转换

P2-04 的公开入口是：

```python
from mining import (
    HttpResearchFactorConverter,
    ResearchConversionContext,
    convert_research_factor_drafts,
)

result = convert_research_factor_drafts(
    (extraction_result,),
    context=ResearchConversionContext(
        run_id="searchrun_12345678",
        turn_index=1,
        branch_id="branch-0001",
    ),
    converter=HttpResearchFactorConverter(url, token),
)
```

进入模型转换前必须满足：

- 六类字段都不是 `missing`，并保留 P2-03 的字段状态、页码引用和稳定哈希。
- `expected_direction` 是 `positive` 或 `negative`；`non_monotonic` 不能映射为现有
  候选方向，因此明确拒绝。
- 字段值和短引用再次经过敏感信息检查，发现问题时不发送给 Worker。

每份研报的待转换草稿按稳定 `draft_id` 排序，最多 8 项组成一个 Worker v6 作业；超过
8 项继续分块，不截断。Worker 只返回以下两类逐草稿结论：

```json
{
  "draft_id": "factordraft_...",
  "status": "converted",
  "expression": "rank(ret)",
  "field_mapping": {"ret_5": "ret"},
  "reason_code": null,
  "message": null
}
```

模型不能改变研报预期方向。主服务用 `validate_expression()` 重新解析并校验表达式，
还要求映射键来自研报声明的 `variables`、映射值都在允许字段目录中、表达式依赖均由
映射覆盖。未知字段、非法算子、排除字段、伪造变量、窗口越界或不完整 Worker 响应都
会被拒绝。

通过项生成未落库的 `HypothesisCard` 和 `CandidateDefinition`。假设卡 metadata 只保存
资料、抽取、草稿、字段声明和引用 ID、目录、模型、提示及变量映射，不复制报告正文。
跨报告结果按候选 `semantic_hash` 去重，稳定保留 `(document_id, draft_id)` 较小的
实现。

每个输入草稿对应一条现有 `TrialRecord` 形态的尝试记录：

| 结论 | 尝试类型 | 含义 |
|---|---|---|
| `accepted` | `proposed` | 已通过全部确定性校验 |
| `rejected` | `rejected` | 六字段证据、方向或安全门禁不通过 |
| `invalid` | `invalid` | Worker 拒绝或 DSL、字段映射非法 |
| `duplicate` | `duplicate` | 与已保留候选语义相同 |

P2-04 本身只返回未落库结果。Worker 协议为
`mining_worker_research_conversion_v1`，提示版本为
`mining_research_dsl_conversion_prompt_v1`，新增错误码 `MINING-156`～`MINING-158`。

## 资料驱动运行入口

`POST /api/mining/research-runs` 接受：

```json
{
  "idempotency_key": "web-research-0001",
  "material_path": "2026-10",
  "budget": {},
  "llm_profile": "default"
}
```

服务端在创建运行时发现 PDF、计算指纹并冻结资料清单。后台依次解析、调用 Worker v5、
调用 Worker v6，并把通过候选按 `(document_id, draft_id)` 稳定排序后自动准入到
`max_candidates_per_run`，默认最多 24 个。超出候选预算的通过项保留为
`admission_budget_exhausted` 尝试，不静默丢弃。

资料候选使用一个聚合研究假设作为 V2 种子批次，每个候选自己的假设卡、六字段、页码、
偏移、短引用、变量映射和模型版本保存在 `mining_v2.research_candidate`。
`mining_v2.research_run` 保存运行级资料清单、解析失败和全部筛选结论。来源哈希同时写入
候选参数，因此进入候选内容哈希；候选详情、审批和发布加载发现来源篡改时直接拒绝。

资料首轮不调用搜索生成 Worker，直接进入现有开发期评估。之后如需修正或分支仍走原来的
V2 Worker。选择期、最终研究期和测试段不进入抽取、转换或搜索请求。没有可准入候选时
运行以 `MINING-160` 失败；资料运行契约错误使用 `MINING-159`。

页面新建入口可选择“手工研究假设”或“资料驱动”。运行页展示资料解析、抽取、筛选和准入
数量；候选页展示相对资料路径、文档 ID、页码、偏移和短引用，不展示完整 PDF 正文。

使用固定假 Worker 和临时生成的真实 PDF 的端到端测试覆盖
`tests/test_mining_research_run.py`。浏览器验收脚本为
`tools/monitor/mining_research_check.py`，截图写入
`AiRecord/evidence/mining_research_web/`。
