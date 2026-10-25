"""P5-1 模型登记索引的单元、错误分支和正式产物测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha.artifacts import file_sha256, prediction_sha256, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaArtifactError, AlphaRegistryError
from alpha.registry import (
    REGISTRY_FILENAME,
    REGISTRY_SCHEMA_VERSION,
    get_registered_model,
    load_registry,
    rebuild_registry,
)
from scripts.build_alpha.registry import main as registry_main


def _dataset(root: Path) -> Path:
    """写入登记测试所需的最小训练数据集。"""

    dataset_id = "dataset_test"
    folder = root / "datasets" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2022-12-23", "2022-12-30"]),
            "stock_code": ["SH600000", "SZ000001"],
            "eligible": [True, True],
            "label_rank": [0.2, 0.8],
        }
    ).to_parquet(folder / "data.parquet", index=False)
    write_json(
        folder / "manifest.json",
        {
            "dataset_id": dataset_id,
            "code_sha256": "a" * 64,
            "spec": {"label": "fwd_ret_open_5"},
            "versions": {"data_version": "test_data", "factor_version": "test_factor"},
            "features": ["F1", "F2"],
            "context_features": ["log_mv"],
            "row_count": 2,
            "date_count": 2,
            "stock_count": 2,
            "date_min": "2022-12-23",
            "date_max": "2022-12-30",
            "eligible_rows": 2,
            "label_coverage": 1.0,
        },
    )
    return folder


def _model(root: Path, model_id: str, *, ensemble: bool = False) -> Path:
    """写入一个可通过完整快照校验的最小模型。"""

    _dataset(root)
    folder = root / model_id
    folder.mkdir()
    prediction_frames = {
        "2022": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2022-12-23", "2022-12-30"]),
                "stock_code": ["SH600000", "SZ000001"],
                "score": [0.1, 0.2],
            }
        ),
        "2023": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2023-01-06", "2023-01-06"]),
                "stock_code": ["SH600000", "SZ000001"],
                "score": [0.3, 0.4],
            }
        ),
    }
    prediction_files = {}
    parts = []
    for year, frame in prediction_frames.items():
        target = folder / "predictions" / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True)
        frame.to_parquet(target, index=False)
        prediction_files[year] = prediction_sha256(frame)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    model_type = "lightgbm_ensemble" if ensemble else "ridge"
    fold = {
        "fold_id": "wf_2023",
        "train_start": "2020-01-01",
        "train_end": "2021-12-31",
        "valid_start": "2022-01-01",
        "valid_end": "2022-12-01",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "purge_days": 6,
        "embargo_days": 6,
    }
    model_files = []
    if ensemble:
        for seed in (0, 1):
            model_path = folder / "models" / f"seed={seed}" / "wf_2023.txt"
            model_path.parent.mkdir(parents=True)
            model_path.write_text(f"model-{seed}\n", encoding="utf-8")
            model_files.append(
                {
                    "fold_id": "wf_2023",
                    "seed": seed,
                    "model_file": model_path.relative_to(folder).as_posix(),
                    "model_sha256": file_sha256(model_path),
                }
            )
        config = {
            "aggregation": {"method": "datewise_rank_mean", "weights": "equal"},
            "member": {"target": "label_rank", "top_n": 30, "quantiles": 10},
            "seeds": [0, 1],
        }
    else:
        model_path = folder / "model.npz"
        model_path.write_bytes(b"ridge-model")
        config = {
            "model_type": "ridge",
            "target": "label_rank",
            "top_n": 30,
            "quantiles": 10,
            "alpha_grid": [0.1, 1.0],
        }
    expected_files = {
        "raw": ["F1", "F2"],
        "context": ["log_mv"],
        "transformed": ["F1", "F2", "log_mv"],
    }
    write_json(folder / "config.json", config)
    write_json(folder / "features.json", expected_files)
    write_json(
        folder / "metrics.json",
        {
            "model_id": model_id,
            "dataset_id": "dataset_test",
            "evaluated": True,
            "decision": {"status": "pass"},
            "test_model": {
                "rank_ic_mean": 0.08,
                "icir": 0.6,
                "rank_ic_win_rate": 0.7,
                "days": 2,
            },
            "portfolio": {
                "annual_return": 0.2,
                "excess_annual_return": 0.1,
                "max_drawdown": -0.1,
                "sharpe": 1.2,
                "turnover_two_way_annual": 20.0,
                "periods": 2,
            },
        },
    )
    manifest = {
        "model_id": model_id,
        "model_type": model_type,
        "target": "label_rank",
        "label": "fwd_ret_open_5",
        "universe": "all_a",
        "benchmark": "H00300.CSI",
        "top_n": 30,
        "quantiles": 10,
        "dataset_id": "dataset_test",
        "raw_features": expected_files["raw"],
        "context_features": expected_files["context"],
        "features": expected_files["transformed"],
        "feature_count": 3,
        "folds": [fold],
        "created_at": "2026-10-22T00:00:00+00:00",
        "status": "evaluated",
        "decision": "pass",
        "prediction_files": prediction_files,
        "prediction_sha256": prediction_sha256(combined),
        "prediction_rows": len(combined),
        "prediction_dates": int(combined["trade_date"].nunique()),
        "prediction_start": "2022-12-23",
        "prediction_end": "2023-01-06",
    }
    if ensemble:
        manifest.update(
            {
                "member_type": "lightgbm",
                "fold_models": model_files,
                "seeds": [0, 1],
                "ensemble": {
                    "aggregation": {"method": "datewise_rank_mean", "weights": "equal"},
                    "seeds": [0, 1],
                },
            }
        )
    else:
        manifest["model_sha256"] = file_sha256(model_path)
    write_json(folder / "manifest.json", manifest)
    return folder


def test_rebuild_load_and_query_registry(tmp_path: Path):
    """重建后可读取索引并按 ID 找回模型及核心产物。"""

    _model(tmp_path, "ridge_test")
    _model(tmp_path, "ensemble_test", ensemble=True)

    payload = rebuild_registry(tmp_path)

    assert payload["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert payload["model_count"] == 2
    assert [item["model_id"] for item in payload["models"]] == [
        "ensemble_test",
        "ridge_test",
    ]
    ensemble = get_registered_model("ensemble_test", tmp_path)
    assert ensemble["features"]["count"] == 3
    assert ensemble["window"] == {
        "train_start": "2020-01-01",
        "train_end": "2021-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "fold_count": 1,
    }
    assert ensemble["metrics"]["rank_ic_mean"] == 0.08
    assert ensemble["metrics"]["annual_return"] == 0.2
    assert len(ensemble["artifacts"]["model_files"]) == 2
    assert ensemble["artifacts"]["predictions"]["rows"] == 4
    assert len(ensemble["artifacts"]["artifact_sha256"]) == 64
    assert ensemble["status_scope"] == "training"
    assert ensemble["downstream"] == {
        "evaluations": [],
        "acceptances": [],
        "monitoring": [],
        "lifecycles": [],
        "final_holdouts": [],
    }
    assert payload["run_link_count"] == 0
    assert load_registry(tmp_path) == payload
    assert (tmp_path / REGISTRY_FILENAME).is_file()


def test_registry_aggregates_model_downstream_runs(tmp_path: Path):
    """模型索引按 ID 汇总评估、账本、监控、生命周期和最终留出。"""

    model_id = "ridge_test"
    folder = _model(tmp_path, model_id)
    evaluation_folder = tmp_path / "runs" / "evaluation" / "evaluation_demo"
    evaluation_folder.mkdir(parents=True)
    write_json(
        evaluation_folder / "manifest.json",
        {
            "analysis_id": "evaluation_demo",
            "status": "pass_evaluation",
            "start": "2022-01-01",
            "end": "2023-12-29",
            "selected_candidate": "DEMO",
            "plan": {
                "candidates": [
                    {
                        "name": "DEMO",
                        "source": {"kind": "model", "id": model_id},
                    }
                ],
                "controls": [],
            },
        },
    )
    acceptance_folder = tmp_path / "runs" / "acceptance" / "acceptance_demo"
    acceptance_folder.mkdir(parents=True)
    write_json(
        acceptance_folder / "manifest.json",
        {
            "analysis_id": "acceptance_demo",
            "status": "pass_acceptance",
            "evaluation_id": "evaluation_demo",
            "scope": {"start": "2022-01-01", "end": "2023-12-29"},
            "candidates": {
                "DEMO": {
                    "source": {"kind": "model", "id": model_id},
                    "complete": True,
                    "passed": True,
                }
            },
        },
    )
    monitor_folder = folder / "monitoring" / "p5_2_demo"
    monitor_folder.mkdir(parents=True)
    write_json(
        monitor_folder / "manifest.json",
        {
            "monitor_id": "p5_2_demo",
            "evaluation_id": "evaluation_demo",
            "candidate": "DEMO",
            "summary": {
                "status": "ok",
                "start": "2022-01-01",
                "end": "2023-12-29",
            },
        },
    )
    lifecycle_folder = folder / "lifecycle" / "p5_3_demo"
    lifecycle_folder.mkdir(parents=True)
    write_json(
        lifecycle_folder / "manifest.json",
        {
            "lifecycle_id": "p5_3_demo",
            "sources": {"monitor_id": "p5_2_demo"},
            "summary": {
                "latest_state": "normal",
                "start": "2022-01-01",
                "end": "2023-12-29",
            },
        },
    )
    write_json(
        lifecycle_folder / "complete.json",
        {"lifecycle_id": "p5_3_demo", "complete": True},
    )
    holdout_folder = tmp_path / "runs" / "final_holdout" / "holdout_demo"
    holdout_folder.mkdir(parents=True)
    write_json(
        holdout_folder / "manifest.json",
        {
            "holdout_id": "holdout_demo",
            "status": "fail_final_holdout",
            "scope": {"data_start": "2024-01-01", "data_end": "2026-10-15"},
            "sources": {"primary": {"model_id": model_id}},
        },
    )

    payload = rebuild_registry(tmp_path)
    record = payload["models"][0]
    downstream = record["downstream"]
    assert [item["id"] for item in downstream["evaluations"]] == [
        "evaluation_demo"
    ]
    assert [item["id"] for item in downstream["acceptances"]] == [
        "acceptance_demo"
    ]
    assert [item["id"] for item in downstream["monitoring"]] == ["p5_2_demo"]
    assert [item["id"] for item in downstream["lifecycles"]] == ["p5_3_demo"]
    assert downstream["lifecycles"][0]["state"] == "normal"
    assert downstream["final_holdouts"][0]["window_id"] == "holdout_2024_2026"
    assert downstream["final_holdouts"][0]["roles"] == ["primary"]
    assert payload["run_link_count"] == 5


def test_rebuild_is_deterministic_except_generated_at(tmp_path: Path):
    """相同模型重复重建除时间戳外内容一致。"""

    _model(tmp_path, "ridge_test")
    first = rebuild_registry(tmp_path)
    second = rebuild_registry(tmp_path)
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second


def test_rebuild_rejects_directory_and_id_mismatch(tmp_path: Path):
    """模型目录必须与 manifest 中的 ID 一致。"""

    folder = _model(tmp_path, "ridge_test")
    folder.rename(tmp_path / "renamed")
    with pytest.raises(AlphaRegistryError, match="目录与 manifest ID 不一致"):
        rebuild_registry(tmp_path)


@pytest.mark.parametrize("filename", ["config.json", "features.json", "metrics.json"])
def test_rebuild_rejects_missing_core_metadata(tmp_path: Path, filename: str):
    """config、features 或 metrics 缺失时整次登记失败。"""

    folder = _model(tmp_path, "ridge_test")
    (folder / filename).unlink()
    with pytest.raises(AlphaArtifactError, match="不存在"):
        rebuild_registry(tmp_path)


def test_rebuild_rejects_corrupt_manifest(tmp_path: Path):
    """manifest 不是合法 JSON 对象时登记失败。"""

    folder = _model(tmp_path, "ridge_test")
    (folder / "manifest.json").write_text("[", encoding="utf-8")
    with pytest.raises(AlphaArtifactError, match="无法读取"):
        rebuild_registry(tmp_path)


def test_rebuild_rejects_model_hash_mismatch(tmp_path: Path):
    """模型文件被修改后不能登记。"""

    folder = _model(tmp_path, "ridge_test")
    (folder / "model.npz").write_bytes(b"changed")
    with pytest.raises(AlphaArtifactError, match="文件哈希"):
        rebuild_registry(tmp_path)


def test_rebuild_rejects_prediction_hash_mismatch(tmp_path: Path):
    """预测文件被修改后不能登记。"""

    folder = _model(tmp_path, "ridge_test")
    part = folder / "predictions" / "year=2022" / "part.parquet"
    frame = pd.read_parquet(part)
    frame["score"] = frame["score"] + 1.0
    frame.to_parquet(part, index=False)
    with pytest.raises(AlphaArtifactError, match="分片哈希不一致"):
        rebuild_registry(tmp_path)


def test_failed_rebuild_preserves_previous_index(tmp_path: Path):
    """完整校验失败时保留上一次成功索引。"""

    folder = _model(tmp_path, "ridge_test")
    rebuild_registry(tmp_path)
    index_path = tmp_path / REGISTRY_FILENAME
    before = index_path.read_bytes()

    (folder / "model.npz").write_bytes(b"changed")
    with pytest.raises(AlphaArtifactError):
        rebuild_registry(tmp_path)
    assert index_path.read_bytes() == before


def test_load_registry_rejects_bad_schema_count_and_duplicates(tmp_path: Path):
    """读取接口拒绝非法 schema、计数和重复 ID。"""

    _model(tmp_path, "ridge_test")
    payload = rebuild_registry(tmp_path)
    path = tmp_path / REGISTRY_FILENAME

    bad_schema = dict(payload, schema_version="old")
    write_json(path, bad_schema)
    with pytest.raises(AlphaRegistryError, match="schema"):
        load_registry(tmp_path)

    bad_count = dict(payload, model_count=2)
    write_json(path, bad_count)
    with pytest.raises(AlphaRegistryError, match="model_count"):
        load_registry(tmp_path)

    duplicate = json.loads(json.dumps(payload))
    duplicate["models"].append(dict(duplicate["models"][0]))
    duplicate["model_count"] = 2
    write_json(path, duplicate)
    with pytest.raises(AlphaRegistryError, match="重复 ID"):
        load_registry(tmp_path)

    missing = json.loads(json.dumps(payload))
    missing["models"][0].pop("model_type")
    write_json(path, missing)
    with pytest.raises(AlphaRegistryError, match="缺少字段"):
        load_registry(tmp_path)

    bad_hash = json.loads(json.dumps(payload))
    bad_hash["models"][0]["artifacts"]["artifact_sha256"] = "short"
    write_json(path, bad_hash)
    with pytest.raises(AlphaRegistryError, match="artifact_sha256"):
        load_registry(tmp_path)

    bad_links = dict(payload, run_link_count=1)
    write_json(path, bad_links)
    with pytest.raises(AlphaRegistryError, match="run_link_count"):
        load_registry(tmp_path)

    bad_downstream = json.loads(json.dumps(payload))
    bad_downstream["models"][0]["downstream"].pop("evaluations")
    write_json(path, bad_downstream)
    with pytest.raises(AlphaRegistryError, match="downstream"):
        load_registry(tmp_path)


def test_get_registered_model_rejects_missing_and_invalid_id(tmp_path: Path):
    """未登记和非法 ID 给出明确错误。"""

    _model(tmp_path, "ridge_test")
    rebuild_registry(tmp_path)
    with pytest.raises(AlphaRegistryError, match="未登记"):
        get_registered_model("missing", tmp_path)
    with pytest.raises(AlphaRegistryError, match="非法模型 ID"):
        get_registered_model("", tmp_path)


def test_rebuild_rejects_empty_root(tmp_path: Path):
    """没有模型时不会静默生成空索引。"""

    with pytest.raises(AlphaRegistryError, match="没有可登记模型"):
        rebuild_registry(tmp_path)


def test_registry_cli_rebuilds_index(tmp_path: Path, capsys):
    """命令行可重建索引并输出登记摘要。"""

    _model(tmp_path, "ridge_test")
    assert registry_main(["--rebuild", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "已登记 1 个模型" in output
    assert "ridge_test" in output


def test_formal_models_are_registered_with_verified_artifacts():
    """四个模型可通过完整产物校验并登记。"""

    payload = rebuild_registry(MODEL_DATA_DIR)
    by_id = {item["model_id"]: item for item in payload["models"]}
    expected_files = {
        "ridge_label_rank_aaff2cb2fbcf": 1,
        "lightgbm_label_rank_aaae94773921": 6,
        "lightgbm_ensemble_label_rank_6aea5c9299d5": 30,
        "sklearn_hist_gbdt_label_rank_83e1200e554f": 6,
    }
    assert set(expected_files).issubset(by_id)
    assert payload["model_count"] == 4
    for model_id, file_count in expected_files.items():
        item = by_id[model_id]
        assert item["window"]["fold_count"] == 6
        assert len(item["artifacts"]["model_files"]) == file_count
        assert item["artifacts"]["predictions"]["start"] == "2018-01-12"
        assert item["artifacts"]["predictions"]["end"] == "2023-12-29"
        assert item["artifacts"]["predictions"]["rows"] == 1_194_198
        assert len(item["artifacts"]["artifact_sha256"]) == 64
    assert (
        by_id["lightgbm_ensemble_label_rank_6aea5c9299d5"]["artifacts"][
            "analysis_dir"
        ]
            == "models/lightgbm_ensemble_label_rank_6aea5c9299d5/analysis"
    )
    new_model = by_id["sklearn_hist_gbdt_label_rank_83e1200e554f"]
    assert new_model["status"] == "trained"
    assert new_model["status_scope"] == "training"
    assert {
        item["id"] for item in new_model["downstream"]["evaluations"]
    } >= {"evaluation_d57dd9e6e750"}
    assert {
        item["id"] for item in new_model["downstream"]["acceptances"]
    } >= {"acceptance_2c80425c2cfe"}
    assert {
        item["id"] for item in new_model["downstream"]["monitoring"]
    } >= {"p5_2_5494d7f2b910"}
    assert {
        item["id"] for item in new_model["downstream"]["lifecycles"]
    } >= {"p5_3_65282c575784"}
