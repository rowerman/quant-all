"""模型训练后端的稳定契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from alpha.config import DEFAULT_CONTRACT
from alpha.errors import AlphaModelError
from alpha.preprocess import PreprocessConfig


@dataclass(frozen=True)
class ModelArtifact:
    """一次训练完成的模型索引。"""

    model_id: str
    path: Path
    manifest: dict
    metrics: dict
    row_count: int
    date_count: int
    prediction_sha256: str


@dataclass(frozen=True)
class FoldFitResult:
    """单折训练结果，selection 记录选参或早停信息。"""

    model: Any
    selection: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSpec:
    """与具体算法无关的训练规格。"""

    backend: str
    config: dict
    target: str = "label_rank"
    top_n: int = DEFAULT_CONTRACT.top_n
    quantiles: int = 10
    seeds: tuple[int, ...] = ()
    model_type: str | None = None
    preprocess: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise AlphaModelError("training spec 缺少 backend")
        if not isinstance(self.config, Mapping):
            raise AlphaModelError("training spec.config 必须是对象")
        if self.target != "label_rank":
            raise AlphaModelError("当前统一目标为 label_rank")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise AlphaModelError("top_n 必须是正整数")
        if type(self.quantiles) is not int or self.quantiles < 2:
            raise AlphaModelError("quantiles 必须至少为 2")
        seeds = tuple(self.seeds)
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise AlphaModelError("seeds 必须是非负整数")
        if tuple(sorted(set(seeds))) != seeds:
            raise AlphaModelError("seeds 必须严格递增且不重复")
        if seeds and len(seeds) < 2:
            raise AlphaModelError("集成训练至少需要两个种子")
        if self.model_type is not None and (
            not isinstance(self.model_type, str) or not self.model_type.strip()
        ):
            raise AlphaModelError("model_type 必须是非空字符串")
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "preprocess", dict(self.preprocess))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingSpec":
        """从 JSON 对象构造训练规格。"""

        if not isinstance(payload, Mapping):
            raise AlphaModelError("training spec 必须是 JSON 对象")
        backend = payload.get("backend") or payload.get("model_type")
        if not isinstance(backend, str):
            raise AlphaModelError("training spec 缺少 backend")
        return cls(
            backend=backend.strip(),
            config=dict(payload.get("config") or {}),
            target=str(payload.get("target", "label_rank")),
            top_n=int(payload.get("top_n", DEFAULT_CONTRACT.top_n)),
            quantiles=int(payload.get("quantiles", 10)),
            seeds=tuple(int(seed) for seed in payload.get("seeds") or ()),
            model_type=payload.get("model_type"),
            preprocess=dict(payload.get("preprocess") or {}),
        )

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return {
            "backend": self.backend,
            "model_type": self.model_type,
            "config": dict(self.config),
            "target": self.target,
            "top_n": self.top_n,
            "quantiles": self.quantiles,
            "seeds": list(self.seeds),
            "preprocess": dict(self.preprocess),
        }

    def resolved_model_type(self, config: Any | None = None) -> str:
        """返回写入 manifest 的模型类型。"""

        if self.model_type:
            return self.model_type
        concrete = getattr(config, "model_type", None)
        if not self.seeds and isinstance(concrete, str) and concrete:
            return concrete
        return f"{self.backend}_ensemble" if self.seeds else self.backend


class ModelBackend(Protocol):
    """新增模型族需要实现的最小接口。"""

    name: str
    model_suffix: str

    def build_config(self, payload: Mapping[str, Any]) -> Any:
        """校验并构造后端配置。"""

    def config_dict(self, config: Any) -> dict:
        """返回可序列化配置。"""

    def with_seed(self, config: Any, seed: int) -> Any:
        """替换随机种子。"""

    def fit_fold(
        self,
        train_rows: pd.DataFrame,
        valid_rows: pd.DataFrame,
        features: tuple[str, ...],
        config: Any,
    ) -> FoldFitResult:
        """拟合并返回选参或早停结果。"""

    def refit(
        self,
        rows: pd.DataFrame,
        features: tuple[str, ...],
        config: Any,
        fitted: FoldFitResult,
    ) -> Any:
        """在 train+valid 上重拟合。"""

    def predict(self, model: Any, matrix: np.ndarray) -> np.ndarray:
        """生成有限分数。"""

    def save_model(self, model: Any, path: Path, *, fold_id: str) -> None:
        """保存单折模型。"""

    def load_model(self, path: Path, *, fold_id: str) -> Any:
        """读取单折模型。"""

    def manifest_metadata(self, config: Any) -> dict:
        """返回 manifest 的后端专属字段。"""


def preprocess_from_payload(payload: Mapping[str, Any]) -> PreprocessConfig:
    """把 JSON 配置转换为截面预处理设置。"""

    return PreprocessConfig(
        method=str(payload.get("method", "rank")),
        missing_indicator=bool(payload.get("missing_indicator", True)),
        winsorize=bool(payload.get("winsorize", False)),
        neutralize=tuple(payload.get("neutralize") or ()),
        clip_z=float(payload.get("clip_z", 5.0)),
    )
