#!/usr/bin/env bash
# 只读看板：串联日志 + 最新模块日志
#
# 用法：
#   bash tools/v5/watch.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

echo "=== $(date '+%F %T') chain ==="
tail -n 6 logs/core/data/ops_v5/chain.log
echo "--- newest module log ---"
newest=$(find logs -type f -name '*.log' | head -1)
echo "$newest"
tail -n 5 "$newest"
