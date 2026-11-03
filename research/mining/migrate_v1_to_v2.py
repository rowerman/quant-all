"""显式迁移挖掘 V1 候选库的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.paths import MINING_DB_FILE
from mining.migration_v1_v2 import migrate_v1_to_v2


def main(argv: list[str] | None = None) -> int:
    """执行迁移并输出不含敏感内容的 JSON 摘要。"""

    parser = argparse.ArgumentParser(description="迁移 mining_v1 元数据到 mining_v2")
    parser.add_argument("--db", type=Path, default=MINING_DB_FILE)
    args = parser.parse_args(argv)
    result = migrate_v1_to_v2(args.db)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
