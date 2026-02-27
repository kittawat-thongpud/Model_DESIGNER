# Model DESIGNER — RunPod Docker Deployment

## Prerequisites
- RunPod account with a **Network Volume** (for persistent data)
- Docker installed locally (for building image)
- RunPod Pod using template: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

---

## Quick Start (RunPod)

### 1. Build and push image

```bash
# Build image
docker build -t your-dockerhub/model-designer:latest .

# Push to Docker Hub (or any registry RunPod can pull from)
docker push your-dockerhub/model-designer:latest
```

### 2. Deploy on RunPod

In the RunPod Pod configuration:
- **Docker Image**: `your-dockerhub/model-designer:latest`
- **Expose Port**: `8000`
- **Volume Mount**: Attach your Network Volume → mounted at `/workspace/data`
- **GPU**: Select desired GPU (RTX 5090, A100, etc.)

Or use **docker-compose** if RunPod supports it:

```bash
docker-compose up -d
```

### 3. Access the app

```
http://<pod-ip>:8000
```

---

## Volume Structure

RunPod Network Volume is mounted at `/workspace/data` inside the container.

```
/workspace/data/
├── datasets/       ← COCO, custom datasets (e.g. datasets/coco/)
├── models/         ← saved model YAML configs
├── weights/        ← trained .pt files
├── jobs/           ← training job logs and checkpoints
├── exports/        ← exported ONNX / TorchScript models
├── logs/           ← app logs
├── modules/        ← custom nn.Module blocks
└── splits/         ← partition txt file lists (cached)
```

> All data persists across container restarts via the network volume.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/workspace/data` | Root for all persistent data |
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

Override at runtime:
```bash
docker run -e DATA_DIR=/workspace/data -e LOG_LEVEL=INFO ...
```

---

## COCO Dataset Setup

Place COCO data on the network volume **before** starting training:

```
/workspace/data/datasets/coco/
├── images/
│   ├── train2017/   ← 118k images
│   └── val2017/     ← 5k images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

The backend auto-converts COCO JSON → YOLO `.txt` labels on first training run.

---

## Local Development with Docker

```bash
# Build
docker build -t model-designer:local .

# Run locally (no GPU)
docker run -p 8000:8000 \
  -v $(pwd)/backend/data:/workspace/data \
  model-designer:local

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/backend/data:/workspace/data \
  model-designer:local
```
