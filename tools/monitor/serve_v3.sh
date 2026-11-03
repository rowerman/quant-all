#!/usr/bin/env bash
set -euo pipefail

export MINING_SEARCH_V3_ENABLED=true
export MINING_SEARCH_V2_ENABLED=false
export MINING_WORKER_URL="${MINING_WORKER_URL:-http://127.0.0.1:8790}"

exec bash "$(dirname "$0")/serve.sh"
