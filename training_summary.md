# Training Jobs Summary Report
**Dataset:** IDD (Indian Driving Dataset)  
**Date:** 2026-03-25  
**Machines:** ku4070 · ku4070-aj · rase4090 · rase4090-2

---

## Overview

| Machine | Total | 🔄 Running | ✅ Completed | ⏹️ Stopped | ❌ Failed |
|---------|-------|-----------|------------|-----------|---------|
| ku4070 | 10 | 0 | 8 | 2 | 0 |
| ku4070-aj | 9 | **1** | 4 | 3 | 1 |
| rase4090 | 10 | 0 | 10 | 0 | 0 |
| rase4090-2 | 9 | 0 | 9 | 0 | 0 |
| **Total** | **38** | **1** | **31** | **5** | **1** |

---

## Scale n

### imgsz 640

| # | Model | Optimizer | Epoch | mAP50 | **mAP50-95** | Machine |
|---|-------|-----------|-------|-------|-------------|---------|
| 1 | HSG-DET [800, 200] | auto | 458/500 | 0.480 | **0.3059** | ku4070 |
| 2 | HSG-DET [320, 80] | auto | 493/500 | 0.478 | 0.3056 | ku4070-aj |
| 3 | HSG-DET [640, 160] | auto | 473/500 | 0.475 | 0.3041 | ku4070 |
| 4 | YOLOv8n | auto | 433/500 | 0.477 | 0.3038 | ku4070 |
| 5 | HSG-DET [320, 80] | auto | 500/500 | 0.476 | 0.3036 | rase4090 |
| 6 | HSG-DET [480, 120] | auto | 500/500 | 0.476 | 0.3035 | rase4090 |
| 7 | HSG-DET [160, 40] | auto | 460/500 | 0.476 | 0.3034 | rase4090 |
| 8 | HSG-DET [640, 160] | auto | 300/300 | 0.473 | 0.3031 | rase4090 |
| 9 | HSG-DET [160, 40] | auto | 279/300 | 0.472 | 0.3019 | rase4090 |
| 10 | HSG-DET [800, 200] | auto | 427/500 | 0.472 | 0.3004 | rase4090-2 |
| 11 | YOLOv8n | AdamW | 207/300 | 0.468 | 0.2996 | rase4090 |
| 12 | HSG-DET [320, 80] | AdamW | 120/300 | 0.468 | 0.2989 | rase4090 |
| 13 | HSG-DET [160, 40] | AdamW | 419/500 | 0.467 | 0.2985 | rase4090 |
| 14 | HSG-DET [160, 40] | AdamW | 234/500 | 0.462 | 0.2960 | rase4090 |
| 15 | HSG-DET [160, 40] | AdamW | 204/500 | 0.449 | 0.2885 | rase4090 |
| 16 | HSG-DET [800, 200] | AdamW | 351/500 | 0.453 | 0.2865 | rase4090-2 |

### imgsz 1280

| # | Model | Optimizer | Epoch | mAP50 | **mAP50-95** | Machine |
|---|-------|-----------|-------|-------|-------------|---------|
| 1 | HSG-DET [160, 40] | auto | 447/500 | 0.618 | **0.4032** | ku4070 |
| 2 | YOLOv8n | auto | 434/500 | 0.619 | 0.4023 | rase4090-2 |
| 3 | HSG-DET [320, 80] | auto | 422/500 | 0.608 | 0.3945 | ku4070-aj |

---

## Scale s

### imgsz 640

| # | Model | Optimizer | Epoch | mAP50 | **mAP50-95** | Machine |
|---|-------|-----------|-------|-------|-------------|---------|
| 1 | HSG-DET [640, 160] | auto | 290/500 | 0.546 | **0.3508** | ku4070 |
| 2 | HSG-DET [800, 200] | auto | 240/500 | 0.542 | 0.3492 | rase4090-2 |
| 3 | HSG-DET [320, 80] | AdamW | 196/500 | 0.540 | 0.3489 | ku4070 |
| 4 | YOLOv8s | auto | 279/500 | 0.542 | 0.3487 | rase4090-2 |
| 5 | HSG-DET [160, 40] | auto | 268/500 | 0.541 | 0.3486 | rase4090-2 |
| 6 | HSG-DET [320, 80] | auto | 276/500 | 0.541 | 0.3485 | ku4070-aj |
| 7 | HSG-DET [640, 160] | AdamW | 205/500 | 0.539 | 0.3480 | rase4090-2 |
| 8 | HSG-DET [640, 160] | AdamW | 193/500 | 0.537 | 0.3474 | ku4070 |
| 9 | HSG-DET [480, 120] | auto | 268/500 | 0.538 | 0.3466 | rase4090-2 |
| 10 | HSG-DET [480, 120] | AdamW | 265/500 | 0.516 | 0.3279 | ku4070-aj |

### imgsz 1280

| # | Model | Optimizer | Epoch | mAP50 | **mAP50-95** | Machine |
|---|-------|-----------|-------|-------|-------------|---------|
| 1 | HSG-DET [160, 40] | auto | 184/500 | 0.675 | **0.4424** | ku4070 |
| 2 | YOLOv8s | auto | 234/500 🔄 | 0.676 | 0.4418 | ku4070-aj |
| 3 | HSG-DET [480, 120] | auto | 187/500 | 0.672 | 0.4366 | rase4090-2 |

---

## Best Results per Category

| Scale | imgsz | Best Model | mAP50 | mAP50-95 | Machine |
|-------|-------|------------|-------|----------|---------|
| n | 640 | HSG-DET [800, 200] | 0.480 | **0.3059** | ku4070 |
| n | 1280 | HSG-DET [160, 40] | 0.618 | **0.4032** | ku4070 |
| s | 640 | HSG-DET [640, 160] | 0.546 | **0.3508** | ku4070 |
| s | 1280 | HSG-DET [160, 40] | 0.675 | **0.4424** | ku4070 |

---

## Key Findings

### 1. imgsz 1280 vs 640
Higher resolution gives a significant boost across all scales:
- Scale n: +0.097 mAP50-95 (0.306 → 0.403)
- Scale s: +0.091 mAP50-95 (0.351 → 0.442)

### 2. Scale s vs Scale n
Larger model capacity consistently improves results:
- imgsz 640: +0.045 mAP50-95
- imgsz 1280: +0.039 mAP50-95

### 3. HSG-DET vs YOLOv8
- At **imgsz 640**: Performance is nearly identical (difference < 0.002 mAP50-95). HSG-DET [800,200] edges out YOLOv8n at scale n; YOLOv8s is competitive with HSG-DET at scale s.
- At **imgsz 1280**: HSG-DET [160,40]-n leads YOLOv8n by +0.001 mAP50-95 (effectively tied). For scale s, HSG-DET [160,40] leads the completed jobs; the running YOLOv8s (ku4070-aj, ep234/500) is very close.

### 4. Optimizer: auto vs AdamW
`optimizer=auto` (SGD) consistently outperforms explicit `AdamW` within the same architecture and imgsz:
- Typical gap: **+0.005 to +0.020 mAP50-95** in favour of `auto`
- Notable exception: none found — auto wins in every fair comparison

### 5. Token Size Effect (HSG-DET)
At imgsz 640, scale n — larger tokens do not clearly help:
- [160,40]: 0.303 mAP50-95 (auto)
- [320,80]: 0.304 mAP50-95 (auto)
- [480,120]: 0.304 mAP50-95 (auto)
- [640,160]: 0.304 mAP50-95 (auto)
- [800,200]: **0.306 mAP50-95** (auto) ← marginal best

At imgsz 1280, scale n — smaller tokens win:
- [160,40]: **0.403** mAP50-95
- [320,80]: 0.395 mAP50-95

---

## Currently Running

| Job ID | Model | Scale | imgsz | Epoch | mAP50 | mAP50-95 | Machine |
|--------|-------|-------|-------|-------|-------|----------|---------|
| `6c8af2af` | YOLOv8s | s | 1280 | 234/500 | 0.676 | 0.4418 | ku4070-aj |

This job is on track to become the best result overall if it continues to improve.
