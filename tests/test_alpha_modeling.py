"""通用模型后端、路径和评估计划契约测试。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.artifacts import model_dir
from alpha.evaluation_plan import EvaluationPlan
from alpha.modeling import (
    FoldFitResult,
    TrainingSpec,
    available_backends,
    get_backend,
    register_backend,
)
from alpha.modeling.backends import discover_backends
from alpha.modeling.pipeline import fit_model_loaded
from alpha.run_index import rebuild_run_index
from alpha.split import Fold
from common.paths import CONFIG_DIR


@dataclass(frozen=True)
class MeanConfig:
    """测试后端配置。"""

    seed: int = 0


class MeanBackend:
    """只输出首特征均值的契约测试后端。"""

    name = "test_mean"
    model_suffix = ".txt"

    def build_config(self, payload):
        return MeanConfig(seed=int(payload.get("seed", 0)))

    def config_dict(self, config):
        return {"seed": config.seed}

    def with_seed(self, config, seed):
        return MeanConfig(seed)

    def fit_fold(self, train_rows, valid_rows, features, config):
        return FoldFitResult({"center": float(train_rows["label_rank"].mean())})

    def refit(self, rows, features, config, fitted):
        return fitted.model

    def predict(self, model, matrix):
        return matrix[:, 0] + float(model["center"])

    def save_model(self, model, path, *, fold_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(model["center"]), encoding="utf-8")

    def load_model(self, path, *, fold_id):
        return {"center": float(path.read_text(encoding="utf-8"))}

    def manifest_metadata(self, config):
        return {"config": self.config_dict(config)}


def _dataset():
    dates = pd.date_range("2012-01-06", periods=80, freq="W-FRI")
    rows = []
    for day_index, day in enumerate(dates):
        for stock_index in range(6):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": f"S{stock_index}",
                    "eligible": True,
                    "sample_weight": 1.0,
                    "label_raw": float(stock_index + day_index / 100),
                    "label_rank": (stock_index + 1) / 6,
                    "industry_l1": "I1",
                    "log_mv": 10.0,
                    "turnover_rate": 1.0,
                    "listed_days": 1000.0,
                    "F1": float(stock_index),
                }
            )
    frame = pd.DataFrame(rows)
    manifest = {
        "dataset_id": "synthetic",
        "features": ["F1"],
        "context_features": [],
        "versions": {"data_version": "d1", "factor_version": "f1"},
        "code_sha256": "dataset-code",
    }
    folds = (
        Fold(
            "wf_2013",
            str(dates[0].date()),
            str(dates[34].date()),
            str(dates[35].date()),
            str(dates[44].date()),
            str(dates[45].date()),
            str(dates[54].date()),
            6,
            6,
        ),
        Fold(
            "wf_2014",
            str(dates[0].date()),
            str(dates[44].date()),
            str(dates[45].date()),
            str(dates[54].date()),
            str(dates[55].date()),
            str(dates[65].date()),
            6,
            6,
        ),
    )
    return frame, manifest, folds


def test_new_backend_runs_without_pipeline_changes(tmp_path, monkeypatch):
    """新增后端只需注册即可产出标准 v2 模型。"""

    import alpha.modeling.pipeline as pipeline

    monkeypatch.setattr(pipeline, "append_experiment", lambda *args, **kwargs: None)
    backend = MeanBackend()
    register_backend(backend)
    frame, manifest, folds = _dataset()
    artifact = fit_model_loaded(
        frame,
        manifest,
        "synthetic",
        TrainingSpec(backend="test_mean", config={"seed": 0}),
        folds=folds,
        output_root=tmp_path,
    )
    assert artifact.manifest["schema_version"] == "alpha_model_v2"
    assert artifact.manifest["backend"]["name"] == "test_mean"
    assert (artifact.path / "manifest.json").is_file()
    assert (artifact.path / "predictions" / "year=2012" / "part.parquet").is_file()


def test_backend_module_is_discovered_without_package_imports(
    tmp_path,
    monkeypatch,
):
    """后端模块写入目录后无需修改包导入清单即可注册。"""

    package_folder = tmp_path / "discovered_backends"
    package_folder.mkdir()
    (package_folder / "__init__.py").write_text("", encoding="utf-8")
    (package_folder / "custom.py").write_text(
        "\n".join(
            [
                "from alpha.modeling.registry import register_backend",
                "",
                "class CustomBackend:",
                "    name = 'discovered_custom'",
                "    model_suffix = '.txt'",
                "",
                "register_backend(CustomBackend())",
            ]
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "discovered_backends",
        package_folder / "__init__.py",
        submodule_search_locations=[str(package_folder)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "discovered_backends", package)
    spec.loader.exec_module(package)

    assert {"linear", "lightgbm", "sklearn_hist_gbdt"} <= set(
        available_backends()
    )
    assert discover_backends(package) == ("custom",)
    assert get_backend("discovered_custom").name == "discovered_custom"


def test_train_cli_registers_artifact_and_fails_if_missing(
    tmp_path,
    monkeypatch,
    capsys,
):
    """训练命令成功后重建索引，未登记时不能报告成功。"""

    import scripts.build_alpha.train as train_cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps({"backend": "linear", "config": {"model_type": "ridge"}}),
        encoding="utf-8",
    )
    artifact = SimpleNamespace(
        model_id="model_demo",
        manifest={"model_type": "ridge", "backend": {"name": "linear"}},
        path=tmp_path / "model_demo",
        row_count=10,
        date_count=2,
        prediction_sha256="a" * 64,
    )
    monkeypatch.setattr(train_cli, "fit_model", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(
        train_cli,
        "rebuild_registry",
        lambda root: {"model_count": 1, "models": [{"model_id": "model_demo"}]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--dataset-id", "dataset_demo", "--spec", str(spec_path)],
    )
    train_cli.main()
    assert '"registered": true' in capsys.readouterr().out

    monkeypatch.setattr(
        train_cli,
        "rebuild_registry",
        lambda root: {"model_count": 0, "models": []},
    )
    with pytest.raises(RuntimeError, match="未进入索引"):
        train_cli.main()


def test_active_and_archived_model_paths_are_distinct(tmp_path):
    """活跃模型和归档模型通过统一解析函数定位。"""

    active = tmp_path / "models" / "m1"
    archived = tmp_path / "archive" / "models" / "m2"
    active.mkdir(parents=True)
    archived.mkdir(parents=True)
    assert model_dir("m1", tmp_path) == active
    assert model_dir("m2", tmp_path) == archived


def test_evaluation_plan_is_model_agnostic():
    """当前计划不依赖代码中的具体模型候选名。"""

    plan = EvaluationPlan.from_file(
        CONFIG_DIR / "alpha" / "final_holdout_2024_2026.json"
    )
    assert plan.selected
    assert plan.reference
    assert plan.by_name(plan.selected).source.id
    assert plan.controls[0].source.kind == "baseline"


def test_run_index_marks_consumed_holdout(tmp_path):
    """运行索引同时记录已消耗和待使用的留出窗口。"""

    run = tmp_path / "final_holdout" / "final_holdout_demo"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        '{"holdout_id":"final_holdout_demo","status":"fail_final_holdout",'
        '"scope":{"data_start":"2024-01-01","data_end":"2026-10-15"}}',
        encoding="utf-8",
    )
    payload = rebuild_run_index(tmp_path)
    assert payload["run_count"] == 1
    assert payload["runs"][0]["consumed_as_holdout"] is True
    windows = {
        item["window_id"]: item
        for item in payload["holdout_windows"]
    }
    assert windows["holdout_2024_2026"]["consumed_as_holdout"] is True
    assert (
        windows["holdout_2026_2027"]["consumed_as_holdout"] is False
    )
    assert windows["holdout_2026_2027"]["status"] == "planned"
