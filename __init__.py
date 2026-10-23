"""Alpha 模型层：研究契约、数据集、切分、模型训练与解释。"""

from alpha.ablation import (
    FamilyAblationResult,
    fit_family_ablation,
    predict_permuted_group,
)
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
    AlphaFinalHoldoutError,
    AlphaLifecycleError,
    AlphaMonitorError,
    AlphaModelError,
    AlphaPreprocessError,
    AlphaRegistryError,
    AlphaSplitError,
)
from alpha.explain import (
    ModelRef,
    compute_feature_correlation,
    compute_gain_importance,
    compute_tree_shap_importance,
    permute_group_within_dates,
)
from alpha.exposure import PortfolioExposureResult, compute_portfolio_exposure
from alpha.evaluation_plan import (
    EvaluationCandidate,
    EvaluationPlan,
    ScorePolicy,
    SourceSpec,
)
from alpha.feature_groups import FeatureGroup, FeatureGroups, build_feature_groups
from alpha.final_holdout import (
    FinalHoldoutSpec,
    load_final_holdout,
    run_final_holdout,
)
from alpha.gbdt import GbdtConfig, GbdtEnsembleConfig
from alpha.modeling import (
    FoldFitResult,
    ModelArtifact,
    ModelBackend,
    TrainingSpec,
    available_backends,
    fit_model,
)
from alpha.preprocess import PreprocessConfig, PreprocessedFrame, transform
from alpha.monitor import (
    MonitorPolicy,
    MonitorResult,
    evaluate_monitor_frames,
    run_model_monitor,
    simulate_score_decay,
)
from alpha.lifecycle import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    LifecyclePolicy,
    LifecycleResult,
    evaluate_lifecycle,
    load_model_lifecycle,
    run_model_lifecycle,
)
from alpha.portfolio import HoldingPolicy, select_holdings, smooth_rank_scores
from alpha.risk import RiskGatePolicy, RiskGateResult, evaluate_risk_gate
from alpha.predict import (
    load_model_manifest,
    load_prediction_parts,
    load_predictions,
    model_artifact_snapshot,
)
from alpha.registry import (
    get_registered_model,
    load_registry,
    rebuild_registry,
)
from alpha.split import Fold, SplitConfig, build_folds, fold_masks

__all__ = [
    "DEFAULT_CONTRACT",
    "ResearchContract",
    "DatasetArtifact",
    "DatasetSpec",
    "build_dataset",
    "load_dataset",
    "ModelRef",
    "compute_feature_correlation",
    "compute_gain_importance",
    "compute_tree_shap_importance",
    "permute_group_within_dates",
    "FeatureGroup",
    "FeatureGroups",
    "build_feature_groups",
    "FinalHoldoutSpec",
    "load_final_holdout",
    "run_final_holdout",
    "FamilyAblationResult",
    "fit_family_ablation",
    "predict_permuted_group",
    "PortfolioExposureResult",
    "compute_portfolio_exposure",
    "EvaluationCandidate",
    "EvaluationPlan",
    "ScorePolicy",
    "SourceSpec",
    "GbdtConfig",
    "GbdtEnsembleConfig",
    "FoldFitResult",
    "ModelArtifact",
    "ModelBackend",
    "TrainingSpec",
    "available_backends",
    "fit_model",
    "PreprocessConfig",
    "PreprocessedFrame",
    "transform",
    "MonitorPolicy",
    "MonitorResult",
    "evaluate_monitor_frames",
    "run_model_monitor",
    "simulate_score_decay",
    "DEFAULT_FALLBACK_MODEL_ID",
    "DEFAULT_PRIMARY_MODEL_ID",
    "LifecyclePolicy",
    "LifecycleResult",
    "evaluate_lifecycle",
    "load_model_lifecycle",
    "run_model_lifecycle",
    "HoldingPolicy",
    "select_holdings",
    "smooth_rank_scores",
    "RiskGatePolicy",
    "RiskGateResult",
    "evaluate_risk_gate",
    "load_model_manifest",
    "load_prediction_parts",
    "load_predictions",
    "model_artifact_snapshot",
    "get_registered_model",
    "load_registry",
    "rebuild_registry",
    "Fold",
    "SplitConfig",
    "build_folds",
    "fold_masks",
    "AlphaArtifactError",
    "AlphaConfigError",
    "AlphaDataError",
    "AlphaError",
    "AlphaEvaluationError",
    "AlphaFinalHoldoutError",
    "AlphaLifecycleError",
    "AlphaMonitorError",
    "AlphaModelError",
    "AlphaPreprocessError",
    "AlphaRegistryError",
    "AlphaSplitError",
]
