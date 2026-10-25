"""已落盘 Alpha 预测的读取、校验与区间截取。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.artifacts import file_sha256, prediction_sha256
from alpha.errors import AlphaArtifactError
from alpha.predict import load_predictions, model_artifact_snapshot

MODEL_ID = "test_model"


def _write_json(path: Path, payload: dict) -> None:
    """写入测试 manifest。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frames() -> dict[str, pd.DataFrame]:
    """构造两个年度的标准预测分片。"""

    return {
        "2022": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2022-12-23", "2022-12-30"] * 2),
                "stock_code": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
                "score": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        "2023": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2023-01-06"] * 2),
                "stock_code": ["SH600000", "SZ000001"],
                "score": [0.5, 0.6],
            }
        ),
    }


def _model_fixture(
    tmp_path: Path,
    frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[Path, dict]:
    """写入一组预测文件并返回模型根目录和 manifest。"""

    tmp_path.mkdir(parents=True, exist_ok=True)
    folder = tmp_path / MODEL_ID
    folder.mkdir()
    files = {}
    parts = []
    for year, frame in (frames or _frames()).items():
        target = folder / "predictions" / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True)
        frame.to_parquet(target, index=False)
        try:
            files[year] = prediction_sha256(frame)
        except AlphaArtifactError:
            files[year] = "0" * 64
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    try:
        combined_hash = prediction_sha256(combined)
    except AlphaArtifactError:
        combined_hash = "0" * 64
    manifest = {
        "model_id": MODEL_ID,
        "prediction_files": files,
        "prediction_sha256": combined_hash,
        "prediction_rows": len(combined),
        "prediction_dates": int(combined["trade_date"].nunique()),
        "prediction_start": str(combined["trade_date"].min().date()),
        "prediction_end": str(combined["trade_date"].max().date()),
    }
    _write_json(folder / "manifest.json", manifest)
    return tmp_path, manifest


def _artifact_fixture(tmp_path: Path, model_type: str) -> Path:
    """构造带模型和训练数据的最小可复现产物。"""

    root, manifest = _model_fixture(tmp_path)
    folder = root / MODEL_ID
    dataset_id = "dataset_test"
    dataset = root / "datasets" / dataset_id
    dataset.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2022-12-23", "2023-01-06"]),
            "stock_code": ["SH600000", "SZ000001"],
            "eligible": [True, True],
            "label_rank": [0.2, 0.8],
        }
    ).to_parquet(dataset / "data.parquet", index=False)
    _write_json(
        dataset / "manifest.json",
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
            "date_max": "2023-01-06",
            "eligible_rows": 2,
            "label_coverage": 1.0,
        },
    )
    manifest.update(
        {
            "model_type": model_type,
            "label": "fwd_ret_open_5",
            "target": "label_rank",
            "universe": "all_a",
            "benchmark": "H00300.CSI",
            "top_n": 30,
            "quantiles": 10,
            "dataset_id": dataset_id,
            "preprocess": {"method": "rank"},
            "raw_features": ["F1", "F2"],
            "context_features": ["log_mv"],
            "features": ["F1", "F2", "log_mv"],
            "feature_count": 3,
            "folds": [
                {
                    "fold_id": "wf_2023",
                    "train_start": "2020-01-01",
                    "train_end": "2022-01-01",
                    "valid_start": "2022-01-01",
                    "valid_end": "2022-12-01",
                    "test_start": "2023-01-01",
                    "test_end": "2023-12-31",
                    "purge_days": 6,
                    "embargo_days": 6,
                }
            ],
        }
    )
    if model_type == "ridge":
        model_file = folder / "model.npz"
        model_file.write_bytes(b"ridge-model")
        manifest["model_sha256"] = file_sha256(model_file)
    else:
        seeds = (0, 1)
        members = []
        for seed in seeds:
            model_file = folder / "models" / f"seed={seed}" / "wf_2023.txt"
            model_file.parent.mkdir(parents=True)
            model_file.write_text(f"model-{seed}\n", encoding="utf-8")
            members.append(
                {
                    "fold_id": "wf_2023",
                    "seed": seed,
                    "model_file": model_file.relative_to(folder).as_posix(),
                    "model_sha256": file_sha256(model_file),
                }
            )
        manifest["model_params"] = {"max_depth": 6}
        manifest["seeds"] = list(seeds)
        manifest["fold_models"] = members
        if model_type == "lightgbm_ensemble":
            manifest["member_type"] = "lightgbm"
            manifest["ensemble"] = {
                "aggregation": {"method": "datewise_rank_mean", "weights": "equal"},
                "seeds": list(seeds),
            }
            manifest["member_predictions"] = {
                str(seed): {"sha256": f"{seed + 1:064x}"} for seed in seeds
            }
    _write_json(folder / "manifest.json", manifest)
    return root


def test_load_predictions_reads_and_filters_full_artifact(tmp_path):
    """完整校验后按闭区间截取预测。"""

    root, manifest = _model_fixture(tmp_path)
    folder, loaded_manifest, frame = load_predictions(MODEL_ID, root=root)

    assert folder == root / MODEL_ID
    assert loaded_manifest == manifest
    assert len(frame) == manifest["prediction_rows"]
    assert frame["trade_date"].nunique() == manifest["prediction_dates"]
    assert list(frame.columns) == ["trade_date", "stock_code", "score"]

    _, _, selected = load_predictions(
        MODEL_ID,
        "2022-12-30",
        "2022-12-30",
        root=root,
    )
    assert selected["trade_date"].unique().tolist() == [pd.Timestamp("2022-12-30")]
    with pytest.raises(AlphaArtifactError, match="没有已落盘预测"):
        load_predictions(
            MODEL_ID,
            "2023-02-01",
            "2023-02-28",
            root=root,
        )


@pytest.mark.parametrize(
    "model_type",
    ["ridge", "lightgbm", "lightgbm_ensemble"],
)
def test_model_artifact_snapshot_covers_model_dataset_and_predictions(
    tmp_path,
    model_type,
):
    """三类模型都记录模型文件、数据集、特征、种子和预测摘要。"""

    root = _artifact_fixture(tmp_path, model_type)
    payload = model_artifact_snapshot(MODEL_ID, root=root)

    assert payload["schema_version"] == "alpha_model_artifact_snapshot_v1"
    assert len(payload["artifact_sha256"]) == 64
    assert payload["model"]["model_type"] == model_type
    assert payload["model"]["features"] == ["F1", "F2", "log_mv"]
    assert payload["dataset"]["dataset_id"] == "dataset_test"
    assert len(payload["dataset"]["data_sha256"]) == 64
    assert payload["predictions"]["rows"] == 6
    assert payload["predictions"]["dates"] == 3
    assert payload["predictions"]["start"] == "2022-12-23"
    assert payload["predictions"]["end"] == "2023-01-06"
    if model_type == "ridge":
        assert payload["model"]["seeds"] == []
        assert [item["path"] for item in payload["model"]["files"]] == ["model.npz"]
    else:
        assert payload["model"]["seeds"] == [0, 1]
        assert [item["path"] for item in payload["model"]["files"]] == [
            "models/seed=0/wf_2023.txt",
            "models/seed=1/wf_2023.txt",
        ]


def test_model_artifact_snapshot_is_deterministic(tmp_path):
    """相同模型和数据集重复生成的产物摘要一致。"""

    root = _artifact_fixture(tmp_path, "lightgbm_ensemble")
    first = model_artifact_snapshot(MODEL_ID, root=root)
    second = model_artifact_snapshot(MODEL_ID, root=root)
    assert first == second


def test_model_artifact_snapshot_detects_model_and_dataset_changes(tmp_path):
    """模型或训练数据实际内容变化必须改变产物摘要。"""

    root = _artifact_fixture(tmp_path, "lightgbm")
    first = model_artifact_snapshot(MODEL_ID, root=root)
    model_file = root / MODEL_ID / "models" / "seed=0" / "wf_2023.txt"
    model_file.write_text("changed-model\n", encoding="utf-8")
    second = model_artifact_snapshot(MODEL_ID, root=root)

    assert second["artifact_sha256"] != first["artifact_sha256"]
    changed = next(
        item for item in second["model"]["files"] if item["path"].endswith("seed=0/wf_2023.txt")
    )
    assert changed["matches_manifest"] is False

    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2022-12-23", "2023-01-06"]),
            "stock_code": ["SH600000", "SZ000001"],
            "eligible": [True, True],
            "label_rank": [0.1, 0.9],
        }
    ).to_parquet(root / "datasets" / "dataset_test" / "data.parquet", index=False)
    third = model_artifact_snapshot(MODEL_ID, root=root)
    assert third["dataset"]["data_sha256"] != second["dataset"]["data_sha256"]
    assert third["artifact_sha256"] != second["artifact_sha256"]


def test_model_artifact_snapshot_detects_prediction_change(tmp_path):
    """同步更新预测文件和 manifest 后，新预测内容会改变产物摘要。"""

    root = _artifact_fixture(tmp_path, "ridge")
    first = model_artifact_snapshot(MODEL_ID, root=root)
    folder = root / MODEL_ID
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    parts = []
    files = {}
    for year in ("2022", "2023"):
        path = folder / "predictions" / f"year={year}" / "part.parquet"
        frame = pd.read_parquet(path)
        frame["score"] = frame["score"] + 1.0
        frame.to_parquet(path, index=False)
        files[year] = prediction_sha256(frame)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    manifest["prediction_files"] = files
    manifest["prediction_sha256"] = prediction_sha256(combined)
    _write_json(folder / "manifest.json", manifest)

    second = model_artifact_snapshot(MODEL_ID, root=root)
    assert second["predictions"]["sha256"] != first["predictions"]["sha256"]
    assert second["artifact_sha256"] != first["artifact_sha256"]


def test_model_artifact_snapshot_rejects_missing_model_file(tmp_path):
    """模型文件缺失时不能生成不完整快照。"""

    root = _artifact_fixture(tmp_path, "ridge")
    (root / MODEL_ID / "model.npz").unlink()
    with pytest.raises(AlphaArtifactError, match="模型文件|没有可定位"):
        model_artifact_snapshot(MODEL_ID, root=root)


def test_load_predictions_rejects_missing_part(tmp_path):
    """年度分片缺失时直接报错。"""

    root, _ = _model_fixture(tmp_path)
    (root / MODEL_ID / "predictions" / "year=2023" / "part.parquet").unlink()
    with pytest.raises(AlphaArtifactError, match="分片缺失"):
        load_predictions(MODEL_ID, root=root)


def test_load_predictions_rejects_bad_part_hash(tmp_path):
    """分片内容变化必须改变分片哈希。"""

    root, manifest = _model_fixture(tmp_path)
    manifest["prediction_files"]["2023"] = "0" * 64
    _write_json(root / MODEL_ID / "manifest.json", manifest)
    with pytest.raises(AlphaArtifactError, match="分片哈希不一致"):
        load_predictions(MODEL_ID, root=root)


def test_load_predictions_rejects_bad_combined_hash(tmp_path):
    """汇总内容变化必须改变汇总哈希。"""

    root, manifest = _model_fixture(tmp_path)
    manifest["prediction_sha256"] = "0" * 64
    _write_json(root / MODEL_ID / "manifest.json", manifest)
    with pytest.raises(AlphaArtifactError, match="合并.*哈希"):
        load_predictions(MODEL_ID, root=root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prediction_rows", 999, "预测行数"),
        ("prediction_dates", 999, "预测日期数"),
        ("prediction_start", "2022-01-01", "预测起始日"),
        ("prediction_end", "2024-01-01", "预测截止日"),
    ],
)
def test_load_predictions_rejects_manifest_coverage_mismatch(
    tmp_path,
    field,
    value,
    message,
):
    """manifest 覆盖信息必须与预测表一致。"""

    root, manifest = _model_fixture(tmp_path)
    manifest[field] = value
    _write_json(root / MODEL_ID / "manifest.json", manifest)
    with pytest.raises(AlphaArtifactError, match=message):
        load_predictions(MODEL_ID, root=root)


def test_load_predictions_rejects_wrong_year_and_duplicate(tmp_path):
    """分片年度和日期股票唯一性都必须校验。"""

    frames = {
        "2023": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2022-12-30"]),
                "stock_code": ["SH600000"],
                "score": [0.1],
            }
        )
    }
    root, _ = _model_fixture(tmp_path / "year", frames)
    with pytest.raises(AlphaArtifactError, match="其他年度"):
        load_predictions(MODEL_ID, root=root)

    duplicate = _frames()["2023"]
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    root, _ = _model_fixture(tmp_path / "duplicate", {"2023": duplicate})
    with pytest.raises(AlphaArtifactError, match="重复日期和股票"):
        load_predictions(MODEL_ID, root=root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.assign(stock_code=[None] * len(frame)), "股票代码不能为空"),
        (
            lambda frame: frame.assign(
                score=[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "分数必须全部为有限值",
        ),
        (
            lambda frame: frame.assign(
                trade_date=[pd.NaT] + list(frame["trade_date"].iloc[1:])
            ),
            "交易日期不能为空",
        ),
    ],
)
def test_load_predictions_rejects_invalid_values(tmp_path, change, message):
    """空代码、空日期和非有限分数不能进入回测。"""

    frames = _frames()
    combined = pd.concat(frames.values(), ignore_index=True)
    broken = change(combined)
    if message == "交易日期不能为空":
        single = _frames()["2023"].copy()
        single.loc[0, "trade_date"] = pd.NaT
        split = {"2023": single}
    else:
        split = {
            "2022": broken.loc[
                broken["trade_date"].astype(str).str.startswith("2022")
            ],
            "2023": broken.loc[
                broken["trade_date"].astype(str).str.startswith("2023")
            ],
        }
    root, _ = _model_fixture(tmp_path, split)
    with pytest.raises(AlphaArtifactError, match=message):
        load_predictions(MODEL_ID, root=root)


def test_load_predictions_checks_formal_ensemble_artifact():
    """正式五种子集成预测保持 manifest 口径。"""

    model_id = "lightgbm_ensemble_label_rank_6aea5c9299d5"
    _, manifest, frame = load_predictions(model_id)
    assert len(frame) == manifest["prediction_rows"] == 1_194_198
    assert frame["trade_date"].nunique() == manifest["prediction_dates"] == 300
    assert prediction_sha256(frame) == manifest["prediction_sha256"]


def test_formal_ensemble_artifact_snapshot_is_reproducible():
    """正式集成快照覆盖种子、特征、模型文件、数据集和预测。"""

    model_id = "lightgbm_ensemble_label_rank_6aea5c9299d5"
    first = model_artifact_snapshot(model_id)
    second = model_artifact_snapshot(model_id)

    assert first == second
    assert first["model"]["seeds"] == [0, 1, 2, 3, 4]
    assert first["model"]["feature_count"] == 100
    assert len(first["model"]["folds"]) == 6
    assert len(first["model"]["files"]) == 30
    assert all(item["matches_manifest"] for item in first["model"]["files"])
    assert first["dataset"]["dataset_id"] == "6a59680fd71c50a8"
    assert first["dataset"]["row_count"] == 2_138_239
    assert first["predictions"]["rows"] == 1_194_198
    assert first["predictions"]["dates"] == 300
    assert first["predictions"]["sha256"] == (
        "6d8ed420c9876bbe46ce9df27540763c01298653d8dc6a51c7d75d3571951e34"
    )
