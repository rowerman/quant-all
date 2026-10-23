"""模型无关的训练、预测和产物接口。"""

from alpha.modeling.backends import lightgbm, linear
from alpha.modeling.contracts import (
    FoldFitResult,
    ModelArtifact,
    ModelBackend,
    TrainingSpec,
)
from alpha.modeling.folds import load_folds, validate_folds
from alpha.modeling.pipeline import fit_model
from alpha.modeling.registry import (
    available_backends,
    backend_name_for_manifest,
    get_backend,
    register_backend,
)

__all__ = [
    "FoldFitResult",
    "ModelArtifact",
    "ModelBackend",
    "TrainingSpec",
    "available_backends",
    "backend_name_for_manifest",
    "fit_model",
    "get_backend",
    "load_folds",
    "register_backend",
    "validate_folds",
]
