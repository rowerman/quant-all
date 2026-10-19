"""Alpha 模型层：研究契约、数据集、预处理、切分与后续模型训练。

第一批只实现 P0/P1 所需能力；Ridge、LightGBM、回测提供者和生命周期管理
在后续批次加入。
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
from alpha.preprocess import PreprocessConfig, PreprocessedFrame, transform
from alpha.split import Fold, SplitConfig, build_folds, fold_masks

__all__ = [
    "DEFAULT_CONTRACT",
    "ResearchContract",
    "DatasetArtifact",
    "DatasetSpec",
    "build_dataset",
    "load_dataset",
    "PreprocessConfig",
    "PreprocessedFrame",
    "transform",
    "Fold",
    "SplitConfig",
    "build_folds",
    "fold_masks",
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
