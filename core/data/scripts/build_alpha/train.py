"""使用通用训练规格训练任一已注册模型后端。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha.config import MODEL_DATA_DIR
from alpha.logging_conf import get_logger
from alpha.modeling import TrainingSpec, fit_model
from alpha.registry import rebuild_registry

logger = get_logger("scripts.build_alpha.train")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 Alpha 模型")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--spec", required=True, help="训练规格 JSON 文件")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = TrainingSpec.from_dict(payload)
    artifact = fit_model(args.dataset_id, spec, force=args.force)
    registry = rebuild_registry(MODEL_DATA_DIR)
    registered = any(
        item.get("model_id") == artifact.model_id
        for item in registry.get("models") or ()
    )
    if not registered:
        raise RuntimeError(f"训练完成后模型未进入索引：{artifact.model_id}")
    logger.info("模型完成：%s", artifact.model_id)
    print(
        json.dumps(
            {
                "model_id": artifact.model_id,
                "model_type": artifact.manifest.get("model_type"),
                "backend": artifact.manifest.get("backend"),
                "path": str(artifact.path),
                "predictions": artifact.row_count,
                "dates": artifact.date_count,
                "prediction_sha256": artifact.prediction_sha256,
                "registered": registered,
                "registry_model_count": registry.get("model_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
