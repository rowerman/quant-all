"""重建验收和最终留出运行索引。"""

from __future__ import annotations

import argparse
import json

from alpha.run_index import rebuild_run_index


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="重建 Alpha 运行索引")
    parser.add_argument("--root")
    args = parser.parse_args()
    payload = rebuild_run_index(args.root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
