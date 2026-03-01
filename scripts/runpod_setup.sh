#!/usr/bin/env bash
# runpod_setup.sh — RunPod "Start Command" / pre-cmd script
#
# Paste this path (or its contents) into RunPod Pod Template → "Start Command":
#   bash /workspace/Model_DESIGNER/scripts/runpod_setup.sh
#
# What it does (idempotent — safe to run every pod start):
#   1. Install / upgrade Python deps from requirements.txt into the system Python
#   2. Ensure /workspace symlink layout is sane
#   3. Start Model DESIGNER backend server
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/Model_DESIGNER}"
REQ="${REPO_DIR}/requirements.txt"
PYTHON="${MODEL_DESIGNER_PYTHON:-$(which python3)}"
LOG_PREFIX="[runpod_setup]"

echo "=================================================="
echo " Model DESIGNER — RunPod Setup"
echo " REPO_DIR : ${REPO_DIR}"
echo " PYTHON   : ${PYTHON}"
echo " DATE     : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=================================================="

# ── 1. Sanity check ───────────────────────────────────────────────────────────
if [ ! -f "${REQ}" ]; then
    echo "${LOG_PREFIX} ERROR: requirements.txt not found at ${REQ}"
    echo "${LOG_PREFIX} Make sure the repo is cloned to ${REPO_DIR}"
    exit 1
fi

# ── 2. Install / upgrade Python dependencies ──────────────────────────────────
echo "${LOG_PREFIX} Installing Python dependencies from ${REQ} ..."

# Use --no-deps-check equivalent: only re-install if something is missing/outdated.
# --quiet suppresses per-package noise but shows errors.
"${PYTHON}" -m pip install --upgrade pip --quiet

"${PYTHON}" -m pip install \
    --requirement "${REQ}" \
    --upgrade \
    --quiet \
    --no-warn-script-location

echo "${LOG_PREFIX} Python dependencies ready."

# ── 3. Verify critical imports ────────────────────────────────────────────────
echo "${LOG_PREFIX} Verifying critical imports ..."
"${PYTHON}" - <<'PYCHECK'
import sys
critical = ["fastapi", "uvicorn", "torch", "ultralytics", "yaml", "psutil"]
missing = []
import importlib.util
for pkg in critical:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)
if missing:
    print(f"[runpod_setup] MISSING packages: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
import torch
print(f"[runpod_setup] torch={torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"gpu_count={torch.cuda.device_count()}")
PYCHECK

# ── 4. Ensure persistent data directories on /workspace ───────────────────────
DATA_DIR="${REPO_DIR}/backend/data"
for dir in \
    "${DATA_DIR}/datasets" \
    "${DATA_DIR}/models" \
    "${DATA_DIR}/modules" \
    "${DATA_DIR}/jobs" \
    "${DATA_DIR}/weights" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/exports" \
    "${DATA_DIR}/splits"; do
    mkdir -p "${dir}"
done
echo "${LOG_PREFIX} Data directories ready at ${DATA_DIR}"

# ── 5. Start server ───────────────────────────────────────────────────────────
echo "${LOG_PREFIX} Starting Model DESIGNER server ..."
export MODEL_DESIGNER_PYTHON="${PYTHON}"
exec bash "${REPO_DIR}/run.sh"
