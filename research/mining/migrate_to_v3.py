"""显式迁移、检查或回滚挖掘 V3 schema 的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.paths import MINING_DB_FILE
from mining.migration_v1_v2_v3 import (
    migrate_v1_v2_to_v3,
    rollback_v3_migration,
    v3_migration_status,
)


def main(argv: list[str] | None = None) -> int:
    """执行 V3 迁移操作并输出不含敏感正文的 JSON 摘要。"""

    parser = argparse.ArgumentParser(description="管理挖掘 V1/V2 到 V3 迁移")
    parser.add_argument("--db", type=Path, default=MINING_DB_FILE)
    parser.add_argument(
        "command",
        choices=("migrate", "status", "rollback"),
        help="迁移、检查状态或回滚迁移",
    )
    args = parser.parse_args(argv)
    actions = {
        "migrate": migrate_v1_v2_to_v3,
        "status": v3_migration_status,
        "rollback": rollback_v3_migration,
    }
    result = actions[args.command](args.db)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
