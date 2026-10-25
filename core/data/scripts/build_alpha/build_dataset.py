"""构建 P1 周频 alpha 数据集。"""

from __future__ import annotations

import argparse
import json

from alpha.config import MODEL_DATA_DIR
from alpha.dataset import DatasetSpec, build_dataset
from alpha.logging_conf import get_logger

logger = get_logger("scripts.build_alpha.build_dataset")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="构建周频 alpha 数据集")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--features", help="逗号分隔的特征名；默认全部已实现因子")
    parser.add_argument("--force", action="store_true", help="重建已有数据集")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    features = None
    if args.features:
        features = tuple(name.strip() for name in args.features.split(",") if name.strip())
    artifact = build_dataset(
        DatasetSpec(start=args.start, end=args.end, features=features),
        force=args.force,
    )
    logger.info(
        "数据集完成：%s 行=%d 日期=%d 股票=%d",
        artifact.dataset_id,
        artifact.row_count,
        artifact.date_count,
        artifact.stock_count,
    )
    print(
        json.dumps(
            {
                "dataset_id": artifact.dataset_id,
                "path": str(artifact.path),
                "rows": artifact.row_count,
                "dates": artifact.date_count,
                "stocks": artifact.stock_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
