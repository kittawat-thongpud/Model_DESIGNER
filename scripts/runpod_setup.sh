#!/usr/bin/env bash
# runpod_setup.sh — RunPod "Start Command" / pre-cmd script
#
# Paste this path (or its contents) into RunPod Pod Template → "Start Command":
#   bash /Model_DESIGNER/scripts/runpod_setup.sh
# or set REPO_DIR explicitly if you keep the repo somewhere else.
#
# What it does (idempotent — safe to run every pod start):
#   1. Install / upgrade Python deps from requirements.txt into the system Python
#   2. Ensure project-local data directories exist
#   3. Start Model DESIGNER backend server
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [ -z "${REPO_DIR:-}" ]; then
    if [ -f "/Model_DESIGNER/requirements.txt" ]; then
        REPO_DIR="/Model_DESIGNER"
    elif [ -f "/workspace/Model_DESIGNER/requirements.txt" ]; then
        REPO_DIR="/workspace/Model_DESIGNER"
    else
        REPO_DIR="/Model_DESIGNER"
    fi
fi
REQ="${REPO_DIR}/requirements.txt"
PYTHON="${MODEL_DESIGNER_PYTHON:-$(which python3)}"
LOG_PREFIX="[runpod_setup]"
CUDA_WAIT_SECONDS="${RUNPOD_CUDA_WAIT_SECONDS:-120}"
CUDA_WAIT_STEP_SECONDS="${RUNPOD_CUDA_WAIT_STEP_SECONDS:-5}"
GPU_KEEPALIVE="${RUNPOD_GPU_KEEPALIVE:-0}"
GPU_KEEPALIVE_INTERVAL="${RUNPOD_GPU_KEEPALIVE_INTERVAL:-60}"

echo "=================================================="
echo " Model DESIGNER — RunPod Setup"
echo " REPO_DIR : ${REPO_DIR}"
echo " PYTHON   : ${PYTHON}"
echo " DATE     : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=================================================="

# RunPod sometimes leaves NVIDIA_VISIBLE_DEVICES=void even when nvidia-smi can
# see the GPU. Normalize this before importing torch so the backend worker
# inherits a CUDA-visible environment.
if [ "${NVIDIA_VISIBLE_DEVICES:-}" = "void" ] || [ "${NVIDIA_VISIBLE_DEVICES:-}" = "none" ] || [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        export NVIDIA_VISIBLE_DEVICES=all
        export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
        echo "${LOG_PREFIX} Normalized NVIDIA_VISIBLE_DEVICES=all for RunPod GPU access."
    fi
fi
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"

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

# ── 2.5. Wait for RunPod GPU handoff ─────────────────────────────────────────
# RunPod can expose nvidia-smi before CUDA is usable from torch. Starting the
# backend too early can poison the long-lived process with a broken CUDA state.
echo "${LOG_PREFIX} Waiting for CUDA readiness (timeout ${CUDA_WAIT_SECONDS}s) ..."
cuda_wait_deadline=$((SECONDS + CUDA_WAIT_SECONDS))
while true; do
    if "${PYTHON}" - <<'PYCHECK' >/tmp/model_designer_cuda_check.log 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
count = torch.cuda.device_count()
if count <= 0:
    raise SystemExit("torch.cuda.device_count() is 0")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu_count={count}, gpu0={torch.cuda.get_device_name(0)}")
PYCHECK
    then
        sed "s/^/${LOG_PREFIX} CUDA ready: /" /tmp/model_designer_cuda_check.log
        break
    fi

    if [ "${SECONDS}" -ge "${cuda_wait_deadline}" ]; then
        echo "${LOG_PREFIX} ERROR: CUDA did not become ready within ${CUDA_WAIT_SECONDS}s."
        sed "s/^/${LOG_PREFIX} CUDA check: /" /tmp/model_designer_cuda_check.log || true
        echo "${LOG_PREFIX} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
        echo "${LOG_PREFIX} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L | sed "s/^/${LOG_PREFIX} nvidia-smi: /" || true
        fi
        echo "${LOG_PREFIX} Restart the RunPod pod or verify the pod was launched with a GPU attached."
        exit 1
    fi

    echo "${LOG_PREFIX} CUDA not ready yet; retrying in ${CUDA_WAIT_STEP_SECONDS}s ..."
    sleep "${CUDA_WAIT_STEP_SECONDS}"
done

start_gpu_keepalive() {
    if [ "${GPU_KEEPALIVE}" != "1" ]; then
        return
    fi

    echo "${LOG_PREFIX} Starting RunPod GPU keepalive every ${GPU_KEEPALIVE_INTERVAL}s."
    (
        while true; do
            # Do not touch CUDA while a training worker is active. The pulse is
            # short-lived so it releases its CUDA context before training starts.
            if find "${REPO_DIR}/backend/data/jobs" -name worker_process.pid -type f 2>/dev/null | grep -q .; then
                sleep "${GPU_KEEPALIVE_INTERVAL}"
                continue
            fi

            "${PYTHON}" - <<'PYCHECK' >/tmp/model_designer_gpu_keepalive.log 2>&1 || true
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    x = torch.empty((1,), device="cuda")
    x.fill_(1.0)
    torch.cuda.synchronize()
PYCHECK
            sleep "${GPU_KEEPALIVE_INTERVAL}"
        done
    ) &
    echo "$!" > /tmp/model_designer_gpu_keepalive.pid
}

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
start_gpu_keepalive
exec bash "${REPO_DIR}/run.sh"
