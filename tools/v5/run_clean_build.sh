#!/usr/bin/env bash
# V5 后段串联：清洗 -> 建库 -> 自检（取数与 extra 已完成时用这个）
#
# 用法：
#   bash tools/v5/run_clean_build.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

PY=.venv/bin/python
LOGDIR="$ROOT/logs/core/data/ops_v5"
CHAIN="$LOGDIR/chain.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" >> "$CHAIN"; }

run_stage() {
  local name="$1" logfile="$2"
  shift 2
  log "开始 $name：$*"
  "$@" >> "$logfile" 2>&1
  local code=$?
  log "$name 退出码 $code（日志 $logfile）"
  return $code
}

log "===== 后段开始（清洗 -> 建库） ====="

run_stage clean "$LOGDIR/clean.log" "$PY" -u -m scripts.clean.pipeline \
  || { log "清洗失败，链路中止"; exit 1; }
run_stage build "$LOGDIR/build.log" "$PY" -m scripts.build_db.build \
  || { log "建库失败，链路中止"; exit 1; }
run_stage status "$LOGDIR/status_final.log" "$PY" -m scripts.sources.fetch_all status \
  || log "status 退出码非 0"
run_stage pytest "$LOGDIR/pytest.log" "$PY" -m pytest tests -q \
  || log "pytest 退出码非 0"

log "===== ALL DONE ====="
