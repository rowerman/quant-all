"""沙箱提案代码的静态限制。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from pathlib import PurePosixPath
import re

from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
    SandboxFeedbackError,
)

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _policy_error(summary: str, hint: str) -> SandboxFeedbackError:
    return SandboxFeedbackError(
        SandboxCodeFeedback.create(
            SandboxFailureCategory.CODE_POLICY,
            summary=summary,
            repair_hint=hint,
        )
    )


@dataclass(frozen=True)
class SandboxProposalPolicy:
    """生成代码必须遵守的静态限制。"""

    allowed_import_roots: tuple[str, ...] = (
        "__future__",
        "json",
        "math",
        "numpy",
        "pandas",
        "pathlib",
        "qlib",
        "scipy",
        "statistics",
    )
    allowed_absolute_paths: tuple[str, ...] = (
        "/snapshot",
        "/workspace/factor.parquet",
        "/workspace/metrics.json",
    )
    forbidden_call_names: tuple[str, ...] = (
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    )
    forbidden_names: tuple[str, ...] = (
        "globals",
        "locals",
        "vars",
    )

    def __post_init__(self) -> None:
        for field in fields(self):
            values = getattr(self, field.name)
            allow_empty = field.name in {
                "forbidden_call_names",
                "forbidden_names",
            }
            if (
                not isinstance(values, tuple)
                or (not values and not allow_empty)
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{field.name} 必须是非空且唯一的字符串元组")

    def path_is_allowed(self, value: str) -> bool:
        """判断代码中的绝对路径是否属于冻结输入或固定输出。"""

        if not value.startswith("/"):
            return False
        parts = PurePosixPath(value).parts
        if ".." in parts:
            return False
        for allowed in self.allowed_absolute_paths:
            if value == allowed:
                return True
            if allowed == "/snapshot" and value.startswith("/snapshot/"):
                return True
        return False


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def validate_proposal_code(
    code: str,
    *,
    policy: SandboxProposalPolicy | None = None,
) -> None:
    """静态检查提案可编译且不包含明显越权能力。"""

    selected = policy or SandboxProposalPolicy()
    if not isinstance(code, str) or not code.strip():
        raise _policy_error(
            "提案代码为空",
            "生成非空的独立 Python 研究脚本",
        )
    try:
        tree = ast.parse(code, filename="<proposal>", mode="exec")
        compile(tree, "<proposal>", "exec")
    except (SyntaxError, TypeError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        location = f"第 {line} 行" if isinstance(line, int) and line > 0 else ""
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.COMPILE,
                summary=f"提案代码无法通过静态编译 {location}".strip(),
                repair_hint="修复 Python 语法并保持脚本可独立运行",
            )
        ) from exc

    allowed_roots = set(selected.allowed_import_roots)
    forbidden_calls = set(selected.forbidden_call_names)
    forbidden_names = set(selected.forbidden_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_roots:
                    raise _policy_error(
                        f"代码导入了未授权模块：{root}",
                        "只使用冻结数据和明确允许的研究库，不访问网络、进程或环境",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise _policy_error(
                    "代码包含相对导入",
                    "使用明确的允许模块，不引用宿主项目代码",
                )
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_roots:
                raise _policy_error(
                    f"代码导入了未授权模块：{root or '<relative>'}",
                    "只使用冻结数据和明确允许的研究库，不访问网络、进程或环境",
                )
            if any(alias.name == "*" for alias in node.names):
                raise _policy_error(
                    "代码包含通配导入",
                    "逐项导入需要的名称，避免隐式引入未授权能力",
                )
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in forbidden_calls:
                raise _policy_error(
                    f"代码调用了禁用函数：{name}",
                    "移除动态执行、解释器输入和直接文件句柄调用",
                )
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            raise _policy_error(
                f"代码访问了禁用名称：{node.id}",
                "不要读取局部或全局解释器状态",
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr != "__name__"
        ):
            raise _policy_error(
                f"代码访问了内部属性：{node.attr}",
                "使用公开研究 API，不访问解释器内部属性",
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if _WINDOWS_ABSOLUTE.match(value) or (
                value.startswith("/") and not selected.path_is_allowed(value)
            ):
                raise _policy_error(
                    "代码包含未授权的绝对路径",
                    "只读取 /snapshot，并只写入固定的两个 /workspace 输出文件",
                )


__all__ = ["SandboxProposalPolicy", "validate_proposal_code"]
