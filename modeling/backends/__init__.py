"""内置模型后端自动发现。"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType


def discover_backends(package: ModuleType | None = None) -> tuple[str, ...]:
    """导入包内全部公开模块，让各后端在导入时完成注册。"""

    target = package or sys.modules[__name__]
    paths = getattr(target, "__path__", None)
    if paths is None:
        raise RuntimeError(f"模型后端包不可发现：{target.__name__}")
    discovered: list[str] = []
    prefix = f"{target.__name__}."
    for item in pkgutil.iter_modules(paths, prefix=prefix):
        name = item.name.rsplit(".", 1)[-1]
        if name.startswith("_"):
            continue
        importlib.import_module(item.name)
        discovered.append(name)
    return tuple(sorted(discovered))


discover_backends()

__all__ = ["discover_backends"]
