"""使用已落盘模型对指定样本生成预测。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256, model_dir
from alpha.ensemble import combine_rank_predictions
from alpha.errors import AlphaModelError
from alpha.modeling.manifests import backend_from_manifest, records_by_fold
from alpha.modeling.registry import get_backend
from alpha.predict import load_model_manifest


def predict_model_rows(
    model_id: str,
    rows: pd.DataFrame,
    features: tuple[str, ...],
    *,
    fold_id: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """用指定折模型为任意样本生成标准预测。"""

    manifest = load_model_manifest(model_id, root)
    folder = model_dir(model_id, root)
    records = records_by_fold(manifest).get(fold_id) or []
    if not records:
        raise AlphaModelError(f"模型 {model_id} 缺少折 {fold_id}")
    backend = get_backend(backend_from_manifest(manifest))
    matrix = rows.loc[:, features].to_numpy(dtype="float32")
    parts: list[pd.DataFrame] = []
    for record in sorted(records, key=lambda item: int(item.get("seed", 0))):
        relative = str(record.get("model_file") or "")
        path = folder / relative
        expected = str(record.get("model_sha256") or "")
        if not relative or not path.is_file() or file_sha256(path) != expected:
            raise AlphaModelError(f"模型文件缺失或哈希不一致：{path}")
        model = backend.load_model(path, fold_id=fold_id)
        scores = np.asarray(backend.predict(model, matrix), dtype="float64")
        frame = rows.loc[:, ["trade_date", "stock_code"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame["score"] = scores
        parts.append(frame)
    result = (
        parts[0]
        if len(parts) == 1
        else combine_rank_predictions(
            parts,
            names=tuple(
                f"seed={record.get('seed', index)}"
                for index, record in enumerate(records)
            ),
        )
    )
    return result.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
