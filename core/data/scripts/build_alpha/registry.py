"""重建 Alpha 模型索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha.registry import rebuild_registry


def main(argv: list[str] | None = None) -> int:
    """解析参数并重建模型索引。"""

    parser = argparse.ArgumentParser(description="重建 Alpha 历史模型索引")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="完整校验模型产物并重建 data/models/index.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="可选的模型根目录，默认使用 data/models/",
    )
    args = parser.parse_args(argv)
    if not args.rebuild:
        parser.error("必须指定 --rebuild")
    payload = rebuild_registry(args.root)
    print(f"已登记 {payload['model_count']} 个模型")
    for item in payload["models"]:
        print(
            f"{item['model_id']}\t{item['model_type']}\t"
            f"{item['status']}\t{item['decision'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
