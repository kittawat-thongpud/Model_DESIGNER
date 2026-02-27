#!/bin/bash
set -e

# ── RunPod entrypoint (project-from-volume mode) ──────────────────────────────
#
# Layout on Network Volume (/workspace):
#   /workspace/Model_DESIGNER/   ← git clone of this repo (persists)
#   /workspace/Model_DESIGNER/venv/  ← pip install (persists)
#   /workspace/data/             ← datasets, weights, jobs (persists)
#
# Environment variables (all optional):
#   APP_DIR     - path to cloned repo          (default: /workspace/Model_DESIGNER)
#   DATA_DIR    - path to persistent data dir  (default: /workspace/data)
#   GIT_REPO    - repo URL to clone on first boot
#                 (default: https://github.com/kittawat-thongpud/Model_DESIGNER)
#   GIT_BRANCH  - branch/tag to checkout       (default: main)
# ─────────────────────────────────────────────────────────────────────────────

APP_DIR="${APP_DIR:-/workspace/Model_DESIGNER}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
GIT_REPO="${GIT_REPO:-https://github.com/kittawat-thongpud/Model_DESIGNER}"
GIT_BRANCH="${GIT_BRANCH:-main}"

echo "=================================================="
echo " Model DESIGNER — RunPod Entrypoint"
echo " APP_DIR  : ${APP_DIR}"
echo " DATA_DIR : ${DATA_DIR}"
echo "=================================================="

# ── 1. Clone repo if not present ─────────────────────────────────────────────
if [ ! -f "${APP_DIR}/run.py" ]; then
    echo "[entrypoint] Project not found at ${APP_DIR} — cloning from ${GIT_REPO}@${GIT_BRANCH}..."
    git clone --branch "${GIT_BRANCH}" --depth 1 "${GIT_REPO}" "${APP_DIR}"
    echo "[entrypoint] Clone complete."
else
    echo "[entrypoint] Project found at ${APP_DIR}."
fi

# ── 2. Create venv + install dependencies if not present ─────────────────────
VENV_DIR="${APP_DIR}/venv"
if [ ! -f "${VENV_DIR}/bin/python3" ]; then
    echo "[entrypoint] venv not found — creating at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    echo "[entrypoint] Installing dependencies from requirements.txt..."
    "${VENV_DIR}/bin/pip" install --upgrade pip -q
    "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt" -q
    echo "[entrypoint] Dependencies installed."
else
    echo "[entrypoint] venv found at ${VENV_DIR} — skipping install."
fi

# ── 3. Create persistent data directories (idempotent) ───────────────────────
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

# ── 5. Start server ───────────────────────────────────────────────────────────
cd "${APP_DIR}/backend"

exec "${VENV_DIR}/bin/python3" -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
