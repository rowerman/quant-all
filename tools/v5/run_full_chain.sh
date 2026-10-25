#!/usr/bin/env bash
# V5 全链路串联：预检 -> 全量取数(失败补齐) -> extra -> 清洗 -> 建库 -> 自检
#
# 用法：
#   bash tools/v5/run_full_chain.sh
#
# 日志：logs/core/data/ops_v5/*.log，模块日志按业务域分目录。
# 全程幂等：任何中断后重跑同一脚本即可，已完成的交易日会自动跳过。
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

log "===== V5 全链路开始 ====="

# 阶段 0：预检（约 20 个交易日 x 四接口，验证 token 与权限）
if ! run_stage preflight "$LOGDIR/preflight.log" \
     "$PY" -u -m scripts.sources.fetch_all daily --start 20260901 --end 20260928; then
  log "预检退出码非 0，链路中止"
  exit 1
fi
if grep -q "'failed': [1-9]" "$LOGDIR/preflight.log"; then
  log "预检存在失败交易日，链路中止"
  exit 1
fi
log "预检通过"

# 阶段 1：全量取数，出现失败交易日则补拉，最多 3 轮
for round in 1 2 3; do
  fetch_log="$LOGDIR/fetch_r${round}.log"
  if ! run_stage "fetch(round=${round})" "$fetch_log" \
       "$PY" -u -m scripts.sources.fetch_all daily --start 20000101 --end 20260928; then
    log "取数退出码非 0，链路中止"
    exit 1
  fi
  if ! grep -q "'failed': [1-9]" "$fetch_log"; then
    log "取数完成，无失败交易日"
    break
  fi
  log "第 ${round} 轮存在失败交易日，准备补拉"
done

"$PY" -m scripts.sources.fetch_all status > "$LOGDIR/status_before_clean.log" 2>&1
log "status 已写入 $LOGDIR/status_before_clean.log"

# 阶段 2：辅助数据
run_stage extra "$LOGDIR/extra.log" "$PY" -u -m scripts.sources.fetch_all extra \
  || { log "extra 失败，链路中止"; exit 1; }

# 阶段 3：清洗
run_stage clean "$LOGDIR/clean.log" "$PY" -u -m scripts.clean.pipeline \
  || { log "清洗失败，链路中止"; exit 1; }

# 阶段 4：建库
run_stage build "$LOGDIR/build.log" "$PY" -m scripts.build_db.build \
  || { log "建库失败，链路中止"; exit 1; }

# 阶段 5：自检
run_stage status "$LOGDIR/status_final.log" "$PY" -m scripts.sources.fetch_all status \
  || log "status 退出码非 0"
run_stage pytest "$LOGDIR/pytest.log" "$PY" -m pytest tests -q \
  || log "pytest 退出码非 0"

log "===== ALL DONE ====="
