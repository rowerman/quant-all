#!/usr/bin/env bash
# 等待若干秒后再打一次只读看板
#
# 用法：
#   bash tools/v5/poll.sh 290
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sleep "${1:-290}"
bash "$ROOT/tools/v5/watch.sh"
