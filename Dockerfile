# ── Model DESIGNER — Self-contained Runtime Image ────────────────────────────
#
# All Python deps and frontend dist are baked into the image.
# On container start, entrypoint.sh:
#   1. Creates data directories on the mounted volume (idempotent)
#   2. Runs run.sh (which uses the baked-in venv) as PID 1
#
# Build:  docker build -t model-designer .
# Run:    docker compose up

# ══ Stage 1 — Frontend build ═════════════════════════════════════════════════
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline
COPY frontend/ ./
RUN npm run build

# ══ Stage 2 — Python deps ════════════════════════════════════════════════════
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 AS python-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        gcc g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip -q \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt

# ══ Stage 3 — Final runtime ══════════════════════════════════════════════════
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# ── System libs (no Node.js — frontend already built in stage 1) ──────────────
# This layer is stable and will be cached after the first build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        git curl tmux \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Copy baked venv from builder (stable — cached unless requirements.txt changes)
COPY --from=python-builder /opt/venv /opt/venv

# ── Copy built frontend dist (stable — cached unless frontend/ changes) ────────
ENV APP_DIR=/app
WORKDIR /app
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

# ── Copy project source last (changes every push — only this layer rebuilds) ───
COPY . /app/

# ── Point run.sh / run.py at the baked venv ───────────────────────────────────
# MODEL_DESIGNER_PYTHON lets run.py skip re-detection and use the baked venv.
ENV MODEL_DESIGNER_PYTHON=/opt/venv/bin/python3
ENV PATH=/opt/venv/bin:$PATH
ENV DATA_DIR=/workspace/data
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app/run.sh /app/server.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
