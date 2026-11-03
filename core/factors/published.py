"""从正式 `mined` 清单读取已人工批准并发布的因子定义。"""

from __future__ import annotations

import json
from pathlib import Path

from common.paths import FACTOR_DIR
from factors.registry import FactorSpec
from scripts.errors import FactorRegistryError

MINED_FAMILY = "mined"
MINED_MANIFEST = "definitions.json"
PUBLISHED_SCHEMA_VERSION = "mining_published_factors_v1"


def manifest_path(factor_dir: Path | str = FACTOR_DIR) -> Path:
    """返回已发布因子定义清单的稳定路径。"""

    return Path(factor_dir) / MINED_FAMILY / MINED_MANIFEST


def load_manifest(factor_dir: Path | str = FACTOR_DIR) -> dict:
    """读取发布清单；文件不存在时返回空清单。"""

    path = manifest_path(factor_dir)
    if not path.is_file():
        return {"schema_version": PUBLISHED_SCHEMA_VERSION, "factors": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FactorRegistryError(f"已发布因子清单不可用：{path}") from error
    if not isinstance(payload, dict):
        raise FactorRegistryError("已发布因子清单根节点必须是对象")
    if payload.get("schema_version") != PUBLISHED_SCHEMA_VERSION:
        raise FactorRegistryError("已发布因子清单版本不受支持")
    factors = payload.get("factors")
    if not isinstance(factors, list):
        raise FactorRegistryError("已发布因子清单 factors 必须是数组")
    return payload


def published_specs(factor_dir: Path | str = FACTOR_DIR) -> tuple[FactorSpec, ...]:
    """把发布清单中的定义转换为统一 `FactorSpec`。"""

    specs: list[FactorSpec] = []
    names: set[str] = set()
    for item in load_manifest(factor_dir)["factors"]:
        if not isinstance(item, dict):
            raise FactorRegistryError("已发布因子条目必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or name in names:
            raise FactorRegistryError("已发布因子名称缺失或重复")
        names.add(name)
        specs.append(
            FactorSpec(
                name=name,
                family=MINED_FAMILY,
                formula=str(item.get("formula", "")),
                description=str(item.get("description", "")),
                steps=tuple(item.get("steps", ())),
                layers=tuple(item.get("layers", ())),
                lookback=int(item.get("lookback", 0)),
                direction=int(item.get("direction", 1)),
                fields=tuple(item.get("fields", ())),
                extra_sources=tuple(item.get("extra_sources", ())),
            )
        )
    return tuple(sorted(specs, key=lambda spec: spec.name))


def resolve_spec(
    name: str,
    factor_dir: Path | str = FACTOR_DIR,
) -> FactorSpec | None:
    """按名称读取已发布定义，找不到时返回 None。"""

    return next(
        (spec for spec in published_specs(factor_dir) if spec.name == name),
        None,
    )


def family_specs(
    family: str,
    factor_dir: Path | str = FACTOR_DIR,
) -> tuple[FactorSpec, ...]:
    """返回 `mined` 族定义；其他族由源码注册表负责。"""

    if family != MINED_FAMILY:
        raise FactorRegistryError(f"已发布清单不支持因子族：{family}")
    return published_specs(factor_dir)
