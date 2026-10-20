"""Alpha 模型层：研究契约、数据集、切分、线性模型和 LightGBM 集成。

P3-3 已具备五种子 rank 集成；回测提供者和生命周期管理在后续批次加入。
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
from alpha.gbdt import GbdtConfig, GbdtEnsembleConfig
from alpha.preprocess import PreprocessConfig, PreprocessedFrame, transform
from alpha.split import Fold, SplitConfig, build_folds, fold_masks
from alpha.train import TrainConfig, fit, fit_gbdt, fit_gbdt_ensemble

__all__ = [
    "DEFAULT_CONTRACT",
    "ResearchContract",
    "DatasetArtifact",
    "DatasetSpec",
    "build_dataset",
    "load_dataset",
    "GbdtConfig",
    "GbdtEnsembleConfig",
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
    "fit_gbdt_ensemble",
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
