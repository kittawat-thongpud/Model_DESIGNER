#!/bin/bash
set -e

# ── RunPod entrypoint ─────────────────────────────────────────────────────────
# DATA_DIR defaults to /workspace/data (RunPod network volume)
DATA_DIR="${DATA_DIR:-/workspace/data}"

echo "=================================================="
echo " Model DESIGNER — RunPod Entrypoint"
echo " DATA_DIR : ${DATA_DIR}"
echo " APP DIR  : /workspace/Model_DESIGNER"
echo "=================================================="

# Create persistent data subdirectories on the volume (idempotent)
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

# ── Start server ──────────────────────────────────────────────────────────────
cd /workspace/Model_DESIGNER/backend

exec venv/bin/python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
