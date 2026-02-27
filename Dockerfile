# ── Model DESIGNER — Base Runtime Image ───────────────────────────────────────
#
# Project-from-volume mode:
#   - This image contains ONLY the OS runtime (CUDA + system libs + Node + git).
#   - The actual project code, venv, and data live on the RunPod Network Volume.
#   - entrypoint.sh handles:
#       1. git clone on first boot (if /workspace/Model_DESIGNER is missing)
#       2. python3 -m venv + pip install (if venv is missing)
#       3. npm run build (if frontend/dist is missing)
#       4. uvicorn startup
#
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1 + friends: OpenCV headless
# gcc/g++: pycocotools build
# nodejs/npm: frontend build (runs inside container on first boot)
# git: repo clone on first boot
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        gcc \
        g++ \
        git \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Default environment ────────────────────────────────────────────────────────
ENV APP_DIR=/workspace/Model_DESIGNER
ENV DATA_DIR=/workspace/data
ENV GIT_REPO=https://github.com/kittawat-thongpud/Model_DESIGNER
ENV GIT_BRANCH=main
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Entrypoint (the only file baked into the image) ───────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
