# 资料驱动输入与文本型 PDF 解析

## 当前边界

P2-01 固定资料清单、文件指纹、来源引用、解析失败和脱敏契约；P2-02 已实现文本型
PDF 的单文件与目录解析、金融研报分类和逐文件失败隔离。P2-03 仍需从页正文抽取因子
名称、经济机制、公式、变量和可证伪条件。

当前不支持扫描件 OCR、Word/HTML 解析、数据库表（`mining.research_material*`）或
候选生成。解析结果中的正文只用于当前调用，不写入资料业务日志。

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

P2-03 应把 `ResearchMaterialParseResult.pages` 作为唯一正文输入，将字段抽取结果绑定到
资料 ID、总正文哈希、页码和规则版本。修改分类阈值、PDF 解析器或正文规范化方式时必须
升级对应版本；修改页字段、正文结构或契约字段时同步升级 schema，并确认旧解析结果不被
静默复用。
