"""验收和最终留出运行索引。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from alpha.artifacts import write_json
from alpha.config import RUNS_DIR
from alpha.errors import AlphaArtifactError

RUN_INDEX_SCHEMA_VERSION = "alpha_run_index_v1"


def _read_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlphaArtifactError(f"无法读取运行 manifest：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaArtifactError(f"运行 manifest 必须是对象：{path}")
    return payload


def rebuild_run_index(root: Path | None = None) -> dict:
    """扫描验收与最终留出目录并记录消耗状态。"""

    base = Path(root or RUNS_DIR)
    records = []
    for run_type, folder_name, consumed in (
        ("acceptance", "acceptance", False),
        ("final_holdout", "final_holdout", True),
    ):
        folder = base / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            manifest_path = path / "manifest.json"
            if not path.is_dir() or not manifest_path.is_file():
                continue
            manifest = _read_manifest(manifest_path)
            scope = manifest.get("scope") or {}
            run_id = (
                manifest.get("analysis_id")
                or manifest.get("holdout_id")
                or path.name
            )
            records.append(
                {
                    "run_id": str(run_id),
                    "run_type": run_type,
                    "status": str(manifest.get("status") or "unknown"),
                    "start": scope.get("start") or scope.get("data_start"),
                    "end": scope.get("end") or scope.get("data_end"),
                    "consumed_as_holdout": consumed,
                    "path": path.relative_to(base).as_posix(),
                }
            )
    payload = {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(records),
        "runs": sorted(records, key=lambda item: item["run_id"]),
    }
    write_json(base / "index.json", payload)
    return payload
