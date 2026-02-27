# Model DESIGNER — RunPod Docker Deployment

## Architecture (Project-from-Volume Mode)

```
Docker Image  ── CUDA runtime + system libs + git + node  (small, rarely rebuilt)
      │
      └── mounts ──▶  /workspace  (RunPod Network Volume)
                           ├── Model_DESIGNER/    ← git clone + venv  (auto on 1st boot)
                           │     ├── venv/         ← pip install       (auto on 1st boot)
                           │     └── frontend/dist ← npm build         (auto on 1st boot)
                           └── data/              ← datasets/weights/jobs (always persists)
```

**Benefit:** แก้โค้ด → `git pull` บน volume → restart container — ไม่ต้อง rebuild image

---

## Quick Start (RunPod)

### 1. สร้าง Network Volume
RunPod → **Storage** → **+ New Network Volume**
- Name: `model-designer-vol`
- Size: ≥ 50 GB (20 GB สำหรับ venv + project, ส่วนที่เหลือสำหรับ datasets)
- Region: เดียวกับ Pod

### 2. สร้าง Pod
- **Container Image**: `ghcr.io/kittawat-thongpud/model-designer:latest`
- **Expose HTTP Port**: `8000`
- **Volume**: attach `model-designer-vol` → **Mount Path**: `/workspace`
- **GPU**: เลือกตามต้องการ

### 3. Environment Variables (ใน Pod settings)

| Variable | ค่า | หมายเหตุ |
|---|---|---|
| `APP_DIR` | `/workspace/Model_DESIGNER` | path ของโปรเจค |
| `DATA_DIR` | `/workspace/data` | path ข้อมูล persistent |
| `GIT_REPO` | `https://github.com/kittawat-thongpud/Model_DESIGNER` | repo URL |
| `GIT_BRANCH` | `main` | branch |
| `PYTHONUNBUFFERED` | `1` | — |

### 4. First Boot (อัตโนมัติ)

entrypoint.sh จะทำสิ่งเหล่านี้อัตโนมัติถ้ายังไม่มีใน volume:
1. `git clone` โปรเจคลงใน `/workspace/Model_DESIGNER`
2. `python3 -m venv` + `pip install -r requirements.txt`
3. `npm install` + `npm run build` (frontend)
4. สร้าง data directories
5. เปิด uvicorn server

> **First boot ใช้เวลา ~5-10 นาที** (pip install torch + ultralytics)  
> **Boot ครั้งถัดไปเร็วทันที** เพราะ venv อยู่บน volume แล้ว

### 5. Access

```
https://<pod-id>-8000.proxy.runpod.net
```

---

## Volume Structure

```
/workspace/
├── Model_DESIGNER/          ← โปรเจค (git clone)
│   ├── venv/                ← Python virtual environment
│   ├── backend/
│   ├── frontend/dist/       ← built React app
│   └── ...
└── data/
    ├── datasets/            ← COCO, custom datasets
    ├── models/              ← model YAML configs
    ├── weights/             ← trained .pt files
    ├── jobs/                ← training logs + checkpoints
    ├── exports/             ← ONNX / TorchScript
    ├── logs/                ← app logs
    ├── modules/             ← custom nn.Module blocks
    └── splits/              ← partition txt lists (cached)
```

---

## Update โค้ด (ไม่ต้อง rebuild image)

```bash
# SSH เข้า Pod แล้วรัน:
cd /workspace/Model_DESIGNER
git pull

# Restart container ผ่าน RunPod UI
# หรือถ้าต้อง rebuild frontend:
cd frontend && npm run build
```

---

## COCO Dataset Setup

วาง COCO ลงบน network volume ก่อน training:

```
/workspace/data/datasets/coco/
├── images/
│   ├── train2017/   ← 118k images
│   └── val2017/     ← 5k images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Backend จะ auto-convert COCO JSON → YOLO `.txt` labels ใน training run แรก

---

## ถ้า Private Registry (ghcr.io)

RunPod → Pod settings → **Container Registry Credentials**:
- Registry: `ghcr.io`
- Username: `kittawat-thongpud`
- Password: GitHub PAT (scope: `read:packages`)
