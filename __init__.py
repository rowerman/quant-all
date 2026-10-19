"""Alpha 模型层：研究契约、数据集、切分、线性和 LightGBM 模型。

P3 已具备 LightGBM 六折训练；回测提供者和生命周期管理在后续批次加入。
"""

from alpha.config import DEFAULT_CONTRACT, ResearchContract
from alpha.dataset import (
    DatasetArtifact,
    DatasetSpec,
    build_dataset,
    load_dataset,
)
from alpha.errors import (
    AlphaArtifactError,
    AlphaConfigError,
    AlphaDataError,
    AlphaError,
    AlphaEvaluationError,
    AlphaModelError,
    AlphaPreprocessError,
    AlphaRegistryError,
    AlphaSplitError,
)
from alpha.gbdt import GbdtConfig
from alpha.preprocess import PreprocessConfig, PreprocessedFrame, transform
from alpha.split import Fold, SplitConfig, build_folds, fold_masks
from alpha.train import TrainConfig, fit, fit_gbdt

__all__ = [
    "DEFAULT_CONTRACT",
    "ResearchContract",
    "DatasetArtifact",
    "DatasetSpec",
    "build_dataset",
    "load_dataset",
    "GbdtConfig",
    "PreprocessConfig",
    "PreprocessedFrame",
    "transform",
    "Fold",
    "SplitConfig",
    "build_folds",
    "fold_masks",
    "TrainConfig",
    "fit",
    "fit_gbdt",
    "AlphaArtifactError",
    "AlphaConfigError",
    "AlphaDataError",
    "AlphaError",
    "AlphaEvaluationError",
    "AlphaModelError",
    "AlphaPreprocessError",
    "AlphaRegistryError",
    "AlphaSplitError",
]
