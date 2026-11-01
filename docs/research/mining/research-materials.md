# 资料驱动输入契约

## 当前边界

P2-01 只定义资料进入解析器前的稳定输入、文件指纹、来源引用、解析失败和脱敏契约。
当前不读取 PDF 正文，不判断金融研报类型，不调用模型抽取因子，也不创建候选或正式
因子。文本型 PDF 读取与失败隔离属于 P2-02。

数据流：

```text
相对资料路径
  -> 安全路径检查和流式 SHA-256
  -> ResearchMaterialInput
  -> P2-02 解析器
  -> ResearchDocumentRecord
       -> parsed：页数、正文哈希、SourceReference
       -> failed：MINING-149～MINING-151 失败记录
```

## 资料输入

`ResearchMaterialInput` 固定保存以下字段：

| 字段 | 含义 |
|---|---|
| `relative_path` | 规范化 POSIX 相对路径，不得包含主机根目录 |
| `media_type` | 小写 MIME 类型 |
| `size_bytes` | 文件字节数 |
| `sha256` | 文件内容 SHA-256 |
| `parser_version` | 参与清单哈希的解析器版本 |
| `schema_version` | 固定为 `mining_research_materials_v1` |

`document_id` 只由内容 SHA-256 生成，同一文件更换路径或跨 session 后仍保持相同 ID。
`material_hash` 和清单哈希同时覆盖路径、媒体类型、大小、内容和解析器版本。清单哈希
与输入顺序无关；同一清单包含相同内容的不同路径时以 `MINING-147` 拒绝。

`fingerprint_research_file()` 只接受根目录内的普通文件，并逐段拒绝符号链接。绝对
路径、盘符、反斜杠、`.`、`..`、空路径段和根目录外目标均以 `MINING-148` 拒绝。
错误和持久化字典不包含解析后的主机根路径。

## 解析记录与来源

`ResearchDocumentRecord` 有两种互斥状态：

- `parsed` 必须包含正面页数、正文 SHA-256，可选来源引用，不能包含失败记录。
- `failed` 必须包含 `ResearchMaterialFailure`，不能包含页面、正文哈希或来源引用。

`SourceReference` 保存资料 ID、页码、页内定位和摘录 SHA-256，不保存完整原文。
所有引用必须属于对应资料。创建时间不参与资料、引用、解析记录或失败记录的稳定哈希。

失败分类和错误码固定如下：

| 分类 | 错误码 |
|---|---|
| `unreadable` | `MINING-149` |
| `unsupported_format`、`corrupt_document`、`encrypted_document`、`no_extractable_text` | `MINING-150` |
| `unsafe_input` | `MINING-148` |
| `parser_error` | `MINING-151` |

## 脱敏

`redact_research_material_text()` 遮蔽 API Key、Authorization、Token、密码和密钥赋值，
以及常见 Linux、macOS、Windows 主机绝对路径。`ResearchMaterialFailure.message` 和
`SourceReference.locator` 在构造时自动执行脱敏。

`safe_research_material_log_fields()` 是资料模块的业务日志白名单，只允许资料/清单/
记录 ID、哈希、状态、错误码、分类、页数、大小和可重试标记。资料正文、消息和相对
路径不进入日志。

## 使用示例

```python
from pathlib import Path

from mining import build_research_material_manifest

manifest = build_research_material_manifest(
    Path("data/source/reports"),
    ["momentum/2026-10-momentum.pdf"],
    parser_version="pypdf-6.21.2.page-v1",
)
```

P2-02 应在解析器入口重新核对 `sha256` 和 `parser_version`。文件内容或解析器版本变化
后不得复用旧解析记录。
