"""模型后端注册表。"""

from __future__ import annotations


from alpha.errors import AlphaModelError
from alpha.modeling.contracts import ModelBackend

_BACKENDS: dict[str, ModelBackend] = {}


def register_backend(backend: ModelBackend) -> None:
    """注册唯一名称的模型后端。"""

    name = str(getattr(backend, "name", ""))
    if not name:
        raise AlphaModelError("模型后端缺少名称")
    if name in _BACKENDS and _BACKENDS[name] is not backend:
        raise AlphaModelError(f"模型后端重复注册：{name}")
    _BACKENDS[name] = backend


def get_backend(name: str) -> ModelBackend:
    """按名称返回模型后端。"""

    try:
        return _BACKENDS[str(name)]
    except KeyError as error:
        raise AlphaModelError(
            f"未知模型后端：{name}；可用后端：{sorted(_BACKENDS)}"
        ) from error


def available_backends() -> tuple[str, ...]:
    """返回已注册后端名称。"""

    return tuple(sorted(_BACKENDS))


def backend_name_for_manifest(manifest: dict) -> str:
    """兼容 v1 manifest 的后端名称解析。"""

    backend = manifest.get("backend") or {}
    if isinstance(backend, dict) and backend.get("name"):
        return str(backend["name"])
    model_type = str(manifest.get("model_type") or "")
    legacy = {
        "ridge": "linear",
        "elastic_net": "linear",
        "lightgbm": "lightgbm",
        "lightgbm_ensemble": "lightgbm",
    }
    if model_type not in legacy:
        raise AlphaModelError(f"无法识别历史模型类型：{model_type}")
    return legacy[model_type]
