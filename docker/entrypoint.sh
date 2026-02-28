#!/bin/bash
set -e

# ── Model DESIGNER — Container Entrypoint ───────────────────────────────────────
#
# Python deps and frontend dist are baked into the image at /opt/venv and
# /app/frontend/dist respectively. This script only:
#   1. Creates persistent data directories on the mounted volume
#   2. Runs run.sh (which calls run.py → uvicorn) as PID 1
#
# Environment variables:
#   APP_DIR  - project root inside the image   (default: /app)
#   DATA_DIR - persistent data on host/volume  (default: /app/backend/data)
# ──────────────────────────────────────────────────────────────────────────────

APP_DIR="${APP_DIR:-/app}"
DATA_DIR="${DATA_DIR:-/app/backend/data}"

echo "=================================================="
echo " Model DESIGNER — Starting"
echo " APP_DIR  : ${APP_DIR}"
echo " DATA_DIR : ${DATA_DIR}"
echo " PYTHON   : ${MODEL_DESIGNER_PYTHON:-$(which python3)}"
echo "=================================================="

# ── 1. Create persistent data directories (idempotent) ─────────────────────────
for dir in \
    "${DATA_DIR}/datasets" \
    "${DATA_DIR}/models" \
    "${DATA_DIR}/modules" \
    "${DATA_DIR}/jobs" \
    "${DATA_DIR}/weights" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/exports" \
    "${DATA_DIR}/splits"; do
    mkdir -p "$dir"
done
echo "[entrypoint] Data directories ready at ${DATA_DIR}"

# ── 2. Launch server via run.sh (which selects python + starts uvicorn) ────────
exec bash "${APP_DIR}/run.sh"
