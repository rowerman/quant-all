"""单次清洗年度剖析的独立进程入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tools.monitor.baseline_contract import atomic_write_json


def main() -> int:
    """解析 worker 参数并把单次结果原子写入目标文件。"""

    parser = argparse.ArgumentParser(description="清洗年度剖析 worker")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed-year", type=int)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    from tools.monitor.clean_profile_runtime import profile_year

    result = profile_year(args.year, seed_year=args.seed_year)
    atomic_write_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
