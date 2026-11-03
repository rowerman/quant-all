"""为冻结搜索上下文构建只读 Qlib 沙箱快照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calculate.correlation import available_factors
from common.paths import DB_FILE, FACTOR_DIR
from mining.catalog import load_catalog
from mining.search_context import FrozenSearchContext
from mining.snapshot import DEFAULT_DATA_MANIFEST, data_snapshot_fingerprint
from mining_sandbox.policy import load_sandbox_policy
from mining_sandbox.qlib_snapshot import build_qlib_snapshot


def main() -> int:
    """解析参数并构建快照。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_FILE)
    parser.add_argument("--factor-dir", type=Path, default=FACTOR_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    catalog = load_catalog()
    context = FrozenSearchContext(
        data_fingerprint=data_snapshot_fingerprint(args.db, args.manifest),
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        existing_factors=tuple(
            sorted(
                name
                for names in available_factors(args.factor_dir).values()
                for name in names
            )
        ),
    )
    policy = load_sandbox_policy()
    output = args.output or (
        policy.snapshot_root_path / context.context_hash[:16]
    )
    manifest, reused = build_qlib_snapshot(
        context=context,
        db_file=args.db,
        output_dir=output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "reused": reused,
                "manifest_hash": manifest.manifest_hash,
                "context_hash": manifest.context_hash,
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": manifest.row_count,
                "total_bytes": manifest.total_bytes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
