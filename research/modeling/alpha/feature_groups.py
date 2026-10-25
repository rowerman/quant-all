"""模型特征分组：根特征、因子族和缺失指示列。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

import quant
from alpha.errors import AlphaConfigError

MISSING_SUFFIX = "__missing"
CONTEXT_GROUP_KEY = "context"


@dataclass(frozen=True)
class FeatureGroup:
    """一组可以整体删除或置换的模型特征。"""

    key: str
    label: str
    kind: str
    root_features: tuple[str, ...]
    transformed_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.label:
            raise AlphaConfigError("特征组 key 和 label 不能为空")
        if self.kind not in ("family", "context", "single"):
            raise AlphaConfigError(f"未知特征组类型：{self.kind}")
        if not self.root_features or not self.transformed_features:
            raise AlphaConfigError(f"特征组 {self.key} 不能为空")
        if len(set(self.transformed_features)) != len(self.transformed_features):
            raise AlphaConfigError(f"特征组 {self.key} 含重复转换特征")

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "root_features": list(self.root_features),
            "transformed_features": list(self.transformed_features),
        }


@dataclass(frozen=True)
class FeatureGroups:
    """单特征组、分族组及转换列到根特征的映射。"""

    singles: tuple[FeatureGroup, ...]
    families: tuple[FeatureGroup, ...]
    root_by_transformed: Mapping[str, str]
    family_by_root: Mapping[str, str]

    def to_dict(self) -> dict:
        """转换为可写入 manifest 的稳定字典。"""

        return {
            "singles": [group.to_dict() for group in self.singles],
            "families": [group.to_dict() for group in self.families],
            "root_by_transformed": dict(self.root_by_transformed),
            "family_by_root": dict(self.family_by_root),
        }


def _catalog_mapping(catalog: pd.DataFrame | None) -> dict[str, tuple[str, str]]:
    """读取实现因子的族信息。"""

    frame = quant.load_factor_list(status="implemented") if catalog is None else catalog
    required = {"name", "family", "family_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlphaConfigError(f"因子目录缺少列：{missing}")
    return {
        str(name): (str(family), str(family_label))
        for name, family, family_label in frame.loc[
            :, ["name", "family", "family_label"]
        ].itertuples(index=False, name=None)
    }


def _root_feature(name: str) -> str:
    """把缺失指示列还原到父特征。"""

    return name[: -len(MISSING_SUFFIX)] if name.endswith(MISSING_SUFFIX) else name


def build_feature_groups(
    manifest: dict,
    *,
    catalog: pd.DataFrame | None = None,
) -> FeatureGroups:
    """从模型 manifest 构建单特征组和分族组。"""

    raw_features = tuple(str(name) for name in manifest.get("raw_features", ()))
    context_features = tuple(
        str(name) for name in manifest.get("context_features", ())
    )
    transformed = tuple(str(name) for name in manifest.get("features", ()))
    if not raw_features or not transformed:
        raise AlphaConfigError("模型 manifest 缺少原始特征或转换特征")
    if len(set(raw_features)) != len(raw_features):
        raise AlphaConfigError("原始特征清单含重复项")
    if len(set(context_features)) != len(context_features):
        raise AlphaConfigError("上下文特征清单含重复项")
    overlap = sorted(set(raw_features) & set(context_features))
    if overlap:
        raise AlphaConfigError(f"因子和上下文特征重复：{overlap}")
    if len(set(transformed)) != len(transformed):
        raise AlphaConfigError("转换特征清单含重复项")

    roots = raw_features + context_features
    root_set = set(roots)
    transformed_set = set(transformed)
    root_by_transformed: dict[str, str] = {}
    for name in transformed:
        root = _root_feature(name)
        if root not in root_set:
            raise AlphaConfigError(f"转换特征 {name} 找不到父特征")
        root_by_transformed[name] = root

    catalog_map = _catalog_mapping(catalog)
    unknown = [name for name in raw_features if name not in catalog_map]
    if unknown:
        raise AlphaConfigError(f"因子目录缺少模型因子：{unknown}")

    singles: list[FeatureGroup] = []
    family_by_root: dict[str, str] = {}
    for name in roots:
        columns = tuple(
            column
            for column in (name, f"{name}{MISSING_SUFFIX}")
            if column in transformed_set
        )
        if not columns:
            raise AlphaConfigError(f"特征 {name} 没有任何转换列")
        if name in catalog_map:
            family, family_label = catalog_map[name]
            kind = "single"
            label = family_label
        else:
            family = CONTEXT_GROUP_KEY
            family_label = "上下文特征"
            kind = "single"
            label = "上下文特征"
        family_by_root[name] = family
        singles.append(
            FeatureGroup(
                key=name,
                label=label,
                kind=kind,
                root_features=(name,),
                transformed_features=columns,
            )
        )

    families: list[FeatureGroup] = []
    family_order: list[str] = []
    for name in raw_features:
        family, _ = catalog_map[name]
        if family not in family_order:
            family_order.append(family)
    for family in family_order:
        members = tuple(name for name in raw_features if catalog_map[name][0] == family)
        columns = tuple(
            column
            for name in members
            for column in (name, f"{name}{MISSING_SUFFIX}")
            if column in transformed_set
        )
        label = catalog_map[members[0]][1]
        families.append(
            FeatureGroup(
                key=family,
                label=label,
                kind="family",
                root_features=members,
                transformed_features=columns,
            )
        )
    if context_features:
        columns = tuple(
            column
            for name in context_features
            for column in (name, f"{name}{MISSING_SUFFIX}")
            if column in transformed_set
        )
        families.append(
            FeatureGroup(
                key=CONTEXT_GROUP_KEY,
                label="上下文特征",
                kind="context",
                root_features=context_features,
                transformed_features=columns,
            )
        )

    single_columns = [
        column for group in singles for column in group.transformed_features
    ]
    family_columns = [
        column for group in families for column in group.transformed_features
    ]
    if len(single_columns) != len(set(single_columns)):
        raise AlphaConfigError("单特征组之间存在重复转换列")
    if len(family_columns) != len(set(family_columns)):
        raise AlphaConfigError("分族组之间存在重复转换列")
    if set(single_columns) != transformed_set:
        missing = sorted(transformed_set - set(single_columns))
        raise AlphaConfigError(f"单特征组遗漏转换列：{missing}")
    if set(family_columns) != transformed_set:
        missing = sorted(transformed_set - set(family_columns))
        raise AlphaConfigError(f"分族组遗漏转换列：{missing}")

    return FeatureGroups(
        singles=tuple(singles),
        families=tuple(families),
        root_by_transformed=MappingProxyType(root_by_transformed),
        family_by_root=MappingProxyType(family_by_root),
    )


__all__ = [
    "CONTEXT_GROUP_KEY",
    "MISSING_SUFFIX",
    "FeatureGroup",
    "FeatureGroups",
    "build_feature_groups",
]
