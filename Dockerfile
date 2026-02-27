# ── Stage 1: Build frontend ───────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Runtime (RunPod PyTorch base) ────────────────────────────────────
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System deps (OpenCV headless, build tools for pycocotools, Node skipped — frontend pre-built)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        gcc \
        g++ \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── App code ──────────────────────────────────────────────────────────────────
WORKDIR /workspace/Model_DESIGNER

# Copy project (venv/, node_modules/, frontend/dist/, backend/data/ excluded via .dockerignore)
COPY . .

# Copy pre-built frontend dist from Stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# ── Python venv + dependencies ────────────────────────────────────────────────
# Create venv inside the app so backend/app picks it up (run.py checks backend/venv/bin/python)
RUN python3 -m venv backend/venv \
    && backend/venv/bin/pip install --upgrade pip --quiet \
    && backend/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Data dir is on RunPod Network Volume → /workspace ─────────────────────────
# DATA_DIR env var points backend to the persistent volume path.
# Default: /workspace/data  (mounted as RunPod network volume)
ENV DATA_DIR=/workspace/data
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
