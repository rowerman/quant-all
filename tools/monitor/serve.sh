#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${MONITOR_PYTHON:-.venv/bin/python}"
# 前后端在 apps/monitor 下，仓库根与它都要在导入路径上
export PYTHONPATH="$PWD:$PWD/apps/monitor${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m backend.preflight
exec "$python_bin" -m uvicorn backend.app:app --host 127.0.0.1 --port "${MONITOR_PORT:-8800}" --workers 1
