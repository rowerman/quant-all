"""Alpha 模型层：研究契约、数据集、预处理、切分与线性模型训练。

P2 已具备 Ridge/ElasticNet、样本外预测和模型评估；LightGBM、回测提供者
和生命周期管理在后续批次加入。
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
