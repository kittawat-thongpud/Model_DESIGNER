# Training Jobs Summary Report
**Dataset:** IDD (Indian Driving Dataset)  
**Generated:** 2026-04-06  
**Machines:** ku4070 · ku4070-aj · rase4090 · rase4090-2

---

## Data Sources

| Machine | MCP | Status |
|---------|-----|--------|
| ku4070 | model-designer-ku4070 | ✅ reachable |
| ku4070-aj | model-designer-ku4070-aj | ✅ reachable |
| rase4090 | model-designer-rase4090 | ✅ reachable |
| rase4090-2 | model-designer-rase4090-2 | ✅ reachable |

---

## Overview (reachable machines only)

| Machine | Total | 🔄 Running | ✅ Completed | ⏹️ Stopped | ❌ Failed |
|---------|-------|-----------|------------|-----------|---------|
| ku4070 | 14 | 1 | 11 | 2 | 0 |
| ku4070-aj | 14 | 0 | 9 | 4 | 1 |
| rase4090 | 11 | 0 | 10 | 1 | 0 |
| rase4090-2 | 14 | 1 | 13 | 0 | 0 |
| **Total** | **53** | **2** | **43** | **7** | **1** |

---

## ku4070

### Summary by imgsz / scale / status

| imgsz | scale | running | completed | stopped | failed | total |
|------:|:-----:|--------:|----------:|--------:|------:|------:|
| 640 | n | 1 | 3 | 1 | 0 | 5 |
| 640 | s | 0 | 3 | 1 | 0 | 4 |
| 1280 | n | 0 | 2 | 0 | 0 | 2 |
| 1280 | s | 0 | 2 | 0 | 0 | 2 |
| 640 | (all) | 1 | 6 | 2 | 0 | 9 |
| 1280 | (all) | 0 | 4 | 0 | 0 | 4 |

### Jobs (sorted by mAP0.5-0.95)

| # | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | HSG-DET [160,40] | s | 1280 | auto | completed | 184/500 | 0.44241 |
| 2 | HSG-DET [320,80] | s | 1280 | auto | completed | 185/500 | 0.44007 |
| 3 | HSG-DET [160,40] | n | 1280 | auto | completed | 447/500 | 0.40316 |
| 4 | HSG-DET V.2 [640,160,40] | n | 1280 | auto | completed | 370/500 | 0.40291 |
| 5 | HSG-DET [640,160] | s | 640 | auto | completed | 290/500 | 0.35078 |
| 6 | HSG-DET [320,80] | s | 640 | AdamW | completed | 196/500 | 0.34886 |
| 7 | HSG-DET [640,160] | s | 640 | AdamW | completed | 193/500 | 0.34736 |
| 8 | HSG-DET [640,160] | s | 640 | auto | stopped | 16/500 | 0.31509 |
| 9 | HSG-DET V.2 [640,160,40] | n | 640 | auto | completed | 476/500 | 0.30750 |
| 10 | HSG-DET [800,200] | n | 640 | auto | completed | 458/500 | 0.30587 |
| 11 | HSG-DET [640,160] | n | 640 | auto | completed | 473/500 | 0.30408 |
| 12 | YOLOV8 (yolov8n) | n | 640 | auto | completed | 433/500 | 0.30382 |
| 13 | HSG-DET V.2 [1920,480,120] | n | 640 | auto | running | 38/500 | 0.27919 |
| 14 | HSG-DET [640,160] | n | 640 | AdamW | stopped | 0/500 | — |

---

## ku4070-aj

### Summary by imgsz / scale / status

| imgsz | scale | running | completed | stopped | failed | total |
|------:|:-----:|--------:|----------:|--------:|------:|------:|
| 640 | n | 0 | 3 | 2 | 1 | 6 |
| 640 | s | 0 | 2 | 1 | 0 | 3 |
| 1280 | n | 0 | 2 | 0 | 0 | 2 |
| 1280 | s | 0 | 2 | 1 | 0 | 3 |
| 640 | (all) | 0 | 5 | 3 | 1 | 9 |
| 1280 | (all) | 0 | 4 | 1 | 0 | 5 |

### Jobs (sorted by mAP0.5-0.95)

| # | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | HSG-DET [800,200] | s | 1280 | auto | completed | 270/500 | 0.44260 |
| 2 | YOLOV8 (yolov8s) | s | 1280 | auto | completed | 237/500 | 0.44175 |
| 3 | HSG-DET V.2 [1280,320,80] | n | 1280 | auto | completed | 417/500 | 0.40460 |
| 4 | HSG-DET [320,80] | n | 1280 | auto | completed | 422/500 | 0.39445 |
| 5 | HSG-DET [320,80] | s | 640 | auto | completed | 276/500 | 0.34849 |
| 6 | HSG-DET [480,120] | s | 640 | AdamW | completed | 265/500 | 0.32794 |
| 7 | HSG-DET V.2 [1280,320,80] | n | 640 | SGD | completed | 447/500 | 0.30656 |
| 8 | HSG-DET [320,80] | n | 640 | auto | completed | 493/500 | 0.30562 |
| 9 | HSG-DET V.2 [1280,320,80] | n | 640 | auto | completed | 100/100 | 0.30468 |
| 10 | HSG-DET [320,80] | n | 640 | auto | stopped | 129/500 | 0.29789 |
| 11 | HSG-DET V.2 [2560,640,160] | n | 640 | auto | stopped | 0/500 | — |
| 12 | YOLOV8 (yolov8s) | s | 1280 | auto | stopped | 0/500 | — |
| 13 | HSG-DET [320,80] | s | 640 | auto | stopped | 0/500 | — |
| 14 | YOLOV8 (yolov8n) | n | 640 | SGD | failed | 0/500 | — |

## rase4090

### Summary by imgsz / scale / status

| imgsz | scale | running | completed | stopped | failed | total |
|------:|:-----:|--------:|----------:|--------:|------:|------:|
| 640 | n | 0 | 10 | 0 | 0 | 10 |
| 640 | t | 0 | 0 | 1 | 0 | 1 |

### Jobs (sorted by mAP0.5-0.95)

| # | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | HSG-DET [320,80] | n | 640 | auto | completed | 500/500 | 0.30363 |
| 2 | HSG-DET [480,120] | n | 640 | auto | completed | 500/500 | 0.30351 |
| 3 | HSG-DET [160,40] | n | 640 | auto | completed | 460/500 | 0.30342 |
| 4 | HSG-DET [640,160] | n | 640 | auto | completed | 300/300 | 0.30312 |
| 5 | HSG-DET [160,40] | n | 640 | auto | completed | 279/300 | 0.30194 |
| 6 | YOLOV8 (yolov8n) | n | 640 | AdamW | completed | 207/300 | 0.29964 |
| 7 | HSG-DET [320,80] | n | 640 | AdamW | completed | 120/300 | 0.29894 |
| 8 | HSG-DET [160,40] | n | 640 | AdamW | completed | 419/500 | 0.29851 |
| 9 | HSG-DET [160,40] | n | 640 | AdamW | completed | 234/500 | 0.29601 |
| 10 | HSG-DET [160,40] | n | 640 | AdamW | completed | 204/500 | 0.28851 |
| 11 | Mamba-YOLO T | t | 640 | AdamW | stopped | 5/500 | 0.15432 |

---

## rase4090-2

### Summary by imgsz / scale / status

| imgsz | scale | running | completed | stopped | failed | total |
|------:|:-----:|--------:|----------:|--------:|------:|------:|
| 640 | n | 0 | 3 | 0 | 0 | 3 |
| 640 | s | 0 | 7 | 0 | 0 | 7 |
| 640 | t | 0 | 1 | 0 | 0 | 1 |
| 640 | (all) | 0 | 11 | 0 | 0 | 11 |
| 1280 | n | 0 | 1 | 0 | 0 | 1 |
| 1280 | s | 1 | 1 | 0 | 0 | 2 |

### Jobs (sorted by mAP0.5-0.95)

| # | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | HSG-DET [640,160] | s | 1280 | auto | completed | 221/500 | 0.44084 |
| 2 | HSG-DET [480,120] | s | 1280 | auto | completed | 187/500 | 0.43660 |
| 3 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | completed | 274/500 | 0.43632 |
| 4 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | running | 18/500 | 0.42333 |
| 5 | YOLOV8 (yolov8n) | n | 1280 | auto | completed | 434/500 | 0.40229 |
| 6 | HSG-DET [800,200] | s | 640 | auto | completed | 240/500 | 0.34921 |
| 7 | YOLOV8 (yolov8s) | s | 640 | auto | completed | 279/500 | 0.34870 |
| 8 | HSG-DET [160,40] | s | 640 | auto | completed | 268/500 | 0.34855 |
| 9 | HSG-DET [640,160] | s | 640 | AdamW | completed | 205/500 | 0.34795 |
| 10 | HSG-DET [480,120] | s | 640 | auto | completed | 268/500 | 0.34664 |
| 11 | HSG-DET V.2 [640,160,40] | s | 640 | auto | completed | 326/500 | 0.34654 |
| 12 | Mamba-YOLO T | t | 640 | SGD | completed | 369/500 | 0.34000 |
| 13 | HSG-DET [800,200] | n | 640 | auto | completed | 427/500 | 0.30037 |
| 14 | HSG-DET [800,200] | n | 640 | AdamW | completed | 351/500 | 0.28646 |

---

## Combined Table (reachable machines)

Sorted by `mAP50-95` (desc). Rows with missing metrics are placed at the bottom.

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | ku4070-aj | HSG-DET [800,200] | s | 1280 | auto | completed | 270/500 | 0.44260 |
| 2 | ku4070 | HSG-DET [160,40] | s | 1280 | auto | completed | 184/500 | 0.44241 |
| 3 | ku4070-aj | YOLOV8 (yolov8s) | s | 1280 | auto | completed | 237/500 | 0.44175 |
| 4 | rase4090-2 | HSG-DET [640,160] | s | 1280 | auto | completed | 221/500 | 0.44084 |
| 5 | ku4070 | HSG-DET [320,80] | s | 1280 | auto | completed | 185/500 | 0.44007 |
| 6 | rase4090-2 | HSG-DET [480,120] | s | 1280 | auto | completed | 187/500 | 0.43660 |
| 7 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | completed | 274/500 | 0.43632 |
| 8 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | running | 18/500 | 0.42333 |
| 9 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 1280 | auto | completed | 417/500 | 0.40460 |
| 10 | ku4070 | HSG-DET [160,40] | n | 1280 | auto | completed | 447/500 | 0.40316 |
| 11 | ku4070 | HSG-DET V.2 [640,160,40] | n | 1280 | auto | completed | 370/500 | 0.40291 |
| 12 | rase4090-2 | YOLOV8 (yolov8n) | n | 1280 | auto | completed | 434/500 | 0.40229 |
| 13 | ku4070-aj | HSG-DET [320,80] | n | 1280 | auto | completed | 422/500 | 0.39445 |
| 14 | ku4070 | HSG-DET [640,160] | s | 640 | auto | completed | 290/500 | 0.35078 |
| 15 | rase4090-2 | HSG-DET [800,200] | s | 640 | auto | completed | 240/500 | 0.34921 |
| 16 | ku4070 | HSG-DET [320,80] | s | 640 | AdamW | completed | 196/500 | 0.34886 |
| 17 | rase4090-2 | YOLOV8 (yolov8s) | s | 640 | auto | completed | 279/500 | 0.34870 |
| 18 | rase4090-2 | HSG-DET [160,40] | s | 640 | auto | completed | 268/500 | 0.34855 |
| 19 | ku4070-aj | HSG-DET [320,80] | s | 640 | auto | completed | 276/500 | 0.34849 |
| 20 | rase4090-2 | HSG-DET [640,160] | s | 640 | AdamW | completed | 205/500 | 0.34795 |
| 21 | ku4070 | HSG-DET [640,160] | s | 640 | AdamW | completed | 193/500 | 0.34736 |
| 22 | rase4090-2 | HSG-DET [480,120] | s | 640 | auto | completed | 268/500 | 0.34664 |
| 23 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 640 | auto | completed | 326/500 | 0.34654 |
| 24 | rase4090-2 | Mamba-YOLO T | t | 640 | SGD | completed | 369/500 | 0.34000 |
| 25 | ku4070-aj | HSG-DET [480,120] | s | 640 | AdamW | completed | 265/500 | 0.32794 |
| 26 | ku4070 | HSG-DET [640,160] | s | 640 | auto | stopped | 16/500 | 0.31509 |
| 27 | ku4070 | HSG-DET V.2 [640,160,40] | n | 640 | auto | completed | 476/500 | 0.30750 |
| 28 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 640 | SGD | completed | 447/500 | 0.30656 |
| 29 | ku4070 | HSG-DET [800,200] | n | 640 | auto | completed | 458/500 | 0.30587 |
| 30 | ku4070-aj | HSG-DET [320,80] | n | 640 | auto | completed | 493/500 | 0.30562 |
| 31 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 640 | auto | completed | 100/100 | 0.30468 |
| 32 | ku4070 | HSG-DET [640,160] | n | 640 | auto | completed | 473/500 | 0.30408 |
| 33 | ku4070 | YOLOV8 (yolov8n) | n | 640 | auto | completed | 433/500 | 0.30382 |
| 34 | rase4090 | HSG-DET [320,80] | n | 640 | auto | completed | 500/500 | 0.30363 |
| 35 | rase4090 | HSG-DET [480,120] | n | 640 | auto | completed | 500/500 | 0.30351 |
| 36 | rase4090 | HSG-DET [160,40] | n | 640 | auto | completed | 460/500 | 0.30342 |
| 37 | rase4090 | HSG-DET [640,160] | n | 640 | auto | completed | 300/300 | 0.30312 |
| 38 | rase4090 | HSG-DET [160,40] | n | 640 | auto | completed | 279/300 | 0.30194 |
| 39 | rase4090-2 | HSG-DET [800,200] | n | 640 | auto | completed | 427/500 | 0.30037 |
| 40 | rase4090 | YOLOV8 (yolov8n) | n | 640 | AdamW | completed | 207/300 | 0.29964 |
| 41 | rase4090 | HSG-DET [320,80] | n | 640 | AdamW | completed | 120/300 | 0.29894 |
| 42 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 419/500 | 0.29851 |
| 43 | ku4070-aj | HSG-DET [320,80] | n | 640 | auto | stopped | 129/500 | 0.29789 |
| 44 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 234/500 | 0.29601 |
| 45 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 204/500 | 0.28851 |
| 46 | rase4090-2 | HSG-DET [800,200] | n | 640 | AdamW | completed | 351/500 | 0.28646 |
| 47 | ku4070 | HSG-DET V.2 [1920,480,120] | n | 640 | auto | running | 38/500 | 0.27919 |
| 48 | rase4090 | Mamba-YOLO T | t | 640 | AdamW | stopped | 5/500 | 0.15432 |
| 49 | ku4070 | HSG-DET [640,160] | n | 640 | AdamW | stopped | 0/500 | — |
| 50 | ku4070-aj | HSG-DET V.2 [2560,640,160] | n | 640 | auto | stopped | 0/500 | — |
| 51 | ku4070-aj | YOLOV8 (yolov8s) | s | 1280 | auto | stopped | 0/500 | — |
| 52 | ku4070-aj | HSG-DET [320,80] | s | 640 | auto | stopped | 0/500 | — |
| 53 | ku4070-aj | YOLOV8 (yolov8n) | n | 640 | SGD | failed | 0/500 | — |

---

## Jobs by scale / imgsz (reachable machines)

### scale `n` / imgsz `1280`

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 1280 | auto | completed | 417/500 | 0.40460 |
| 2 | ku4070 | HSG-DET [160,40] | n | 1280 | auto | completed | 447/500 | 0.40316 |
| 3 | ku4070 | HSG-DET V.2 [640,160,40] | n | 1280 | auto | completed | 370/500 | 0.40291 |
| 4 | rase4090-2 | YOLOV8 (yolov8n) | n | 1280 | auto | completed | 434/500 | 0.40229 |
| 5 | ku4070-aj | HSG-DET [320,80] | n | 1280 | auto | completed | 422/500 | 0.39445 |

### scale `n` / imgsz `640`

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | ku4070 | HSG-DET V.2 [640,160,40] | n | 640 | auto | completed | 476/500 | 0.30750 |
| 2 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 640 | SGD | completed | 447/500 | 0.30656 |
| 3 | ku4070 | HSG-DET [800,200] | n | 640 | auto | completed | 458/500 | 0.30587 |
| 4 | ku4070-aj | HSG-DET [320,80] | n | 640 | auto | completed | 493/500 | 0.30562 |
| 5 | ku4070-aj | HSG-DET V.2 [1280,320,80] | n | 640 | auto | completed | 100/100 | 0.30468 |
| 6 | ku4070 | HSG-DET [640,160] | n | 640 | auto | completed | 473/500 | 0.30408 |
| 7 | ku4070 | YOLOV8 (yolov8n) | n | 640 | auto | completed | 433/500 | 0.30382 |
| 8 | rase4090 | HSG-DET [320,80] | n | 640 | auto | completed | 500/500 | 0.30363 |
| 9 | rase4090 | HSG-DET [480,120] | n | 640 | auto | completed | 500/500 | 0.30351 |
| 10 | rase4090 | HSG-DET [160,40] | n | 640 | auto | completed | 460/500 | 0.30342 |
| 11 | rase4090 | HSG-DET [640,160] | n | 640 | auto | completed | 300/300 | 0.30312 |
| 12 | rase4090 | HSG-DET [160,40] | n | 640 | auto | completed | 279/300 | 0.30194 |
| 13 | rase4090-2 | HSG-DET [800,200] | n | 640 | auto | completed | 427/500 | 0.30037 |
| 14 | rase4090 | YOLOV8 (yolov8n) | n | 640 | AdamW | completed | 207/300 | 0.29964 |
| 15 | rase4090 | HSG-DET [320,80] | n | 640 | AdamW | completed | 120/300 | 0.29894 |
| 16 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 419/500 | 0.29851 |
| 17 | ku4070-aj | HSG-DET [320,80] | n | 640 | auto | stopped | 129/500 | 0.29789 |
| 18 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 234/500 | 0.29601 |
| 19 | rase4090 | HSG-DET [160,40] | n | 640 | AdamW | completed | 204/500 | 0.28851 |
| 20 | rase4090-2 | HSG-DET [800,200] | n | 640 | AdamW | completed | 351/500 | 0.28646 |
| 21 | ku4070 | HSG-DET V.2 [1920,480,120] | n | 640 | auto | running | 38/500 | 0.27919 |
| 22 | ku4070 | HSG-DET [640,160] | n | 640 | AdamW | stopped | 0/500 | — |
| 23 | ku4070-aj | HSG-DET V.2 [2560,640,160] | n | 640 | auto | stopped | 0/500 | — |
| 24 | ku4070-aj | YOLOV8 (yolov8n) | n | 640 | SGD | failed | 0/500 | — |

### scale `s` / imgsz `1280`

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | ku4070-aj | HSG-DET [800,200] | s | 1280 | auto | completed | 270/500 | 0.44260 |
| 2 | ku4070 | HSG-DET [160,40] | s | 1280 | auto | completed | 184/500 | 0.44241 |
| 3 | ku4070-aj | YOLOV8 (yolov8s) | s | 1280 | auto | completed | 237/500 | 0.44175 |
| 4 | rase4090-2 | HSG-DET [640,160] | s | 1280 | auto | completed | 221/500 | 0.44084 |
| 5 | ku4070 | HSG-DET [320,80] | s | 1280 | auto | completed | 185/500 | 0.44007 |
| 6 | rase4090-2 | HSG-DET [480,120] | s | 1280 | auto | completed | 187/500 | 0.43660 |
| 7 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | completed | 274/500 | 0.43632 |
| 8 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 1280 | auto | running | 18/500 | 0.42333 |
| 9 | ku4070-aj | YOLOV8 (yolov8s) | s | 1280 | auto | stopped | 0/500 | — |

### scale `s` / imgsz `640`

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | ku4070 | HSG-DET [640,160] | s | 640 | auto | completed | 290/500 | 0.35078 |
| 2 | rase4090-2 | HSG-DET [800,200] | s | 640 | auto | completed | 240/500 | 0.34921 |
| 3 | ku4070 | HSG-DET [320,80] | s | 640 | AdamW | completed | 196/500 | 0.34886 |
| 4 | rase4090-2 | YOLOV8 (yolov8s) | s | 640 | auto | completed | 279/500 | 0.34870 |
| 5 | rase4090-2 | HSG-DET [160,40] | s | 640 | auto | completed | 268/500 | 0.34855 |
| 6 | ku4070-aj | HSG-DET [320,80] | s | 640 | auto | completed | 276/500 | 0.34849 |
| 7 | rase4090-2 | HSG-DET [640,160] | s | 640 | AdamW | completed | 205/500 | 0.34795 |
| 8 | ku4070 | HSG-DET [640,160] | s | 640 | AdamW | completed | 193/500 | 0.34736 |
| 9 | rase4090-2 | HSG-DET [480,120] | s | 640 | auto | completed | 268/500 | 0.34664 |
| 10 | rase4090-2 | HSG-DET V.2 [640,160,40] | s | 640 | auto | completed | 326/500 | 0.34654 |
| 11 | ku4070-aj | HSG-DET [480,120] | s | 640 | AdamW | completed | 265/500 | 0.32794 |
| 12 | ku4070 | HSG-DET [640,160] | s | 640 | auto | stopped | 16/500 | 0.31509 |
| 13 | ku4070-aj | HSG-DET [320,80] | s | 640 | auto | stopped | 0/500 | — |

### scale `t` / imgsz `640`

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | rase4090-2 | Mamba-YOLO T | t | 640 | SGD | completed | 369/500 | 0.34000 |
| 2 | rase4090 | Mamba-YOLO T | t | 640 | AdamW | stopped | 5/500 | 0.15432 |

---

## Comparisons (HSG-DET vs YOLO)

Best-per-bucket is chosen by `mAP50-95` among reachable machines.

| scale | imgsz | best HSG-DET (mAP50-95) | best YOLO (mAP50-95) | winner |
|:-----:|------:|--------------------------|----------------------|--------|
| n | 640 | HSG-DET V.2 [640,160,40] (0.30750) | YOLOV8 (yolov8n) (0.30382) | HSG-DET |
| n | 1280 | HSG-DET V.2 [1280,320,80] (0.40460) | YOLOV8 (yolov8n) (0.40229) | HSG-DET |
| s | 640 | HSG-DET [640,160] (0.35078) | YOLOV8 (yolov8s) (0.34870) | HSG-DET |
| s | 1280 | HSG-DET [800,200] (0.44260) | YOLOV8 (yolov8s) (0.44175) | HSG-DET |
| t | 640 | — | — | — |

---

## Mamba-YOLO Baseline

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |
|---:|---------|-------|:-----:|------:|----------|--------|------|---------:|
| 1 | rase4090-2 | Mamba-YOLO T | t | 640 | SGD | completed | 369/500 | 0.34000 |
| 2 | rase4090 | Mamba-YOLO T | t | 640 | AdamW | stopped | 5/500 | 0.15432 |
