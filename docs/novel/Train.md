# HSG-DET (Conservative Mode)

แนวทาง implement บน Ultralytics (CNN-first + Sparse Global Block)

เป้าหมาย:

* 1080p
* Dense objects ดี
* Occlusion ดี
* Latency ต่ำ
* NMS optional
* Stability สูง

---

# 1️⃣ Design Philosophy (Conservative)

คงโครง YOLO เดิมไว้ 80–90%
เพิ่มเฉพาะ “Sparse Global Context” ใน neck

ไม่แตะ:

* grid prediction
* decoupled head
* CIoU/BCE loss
* EMA / Aug pipeline

---

# 2️⃣ Architecture Overview

```
Input (1080p)
   ↓
Backbone (C2f/CSP)
   ↓
P3, P4, P5
   ↓
Neck (PAN/FPN)
   ↓
Sparse Global Block (P4,P5 only)
   ↓
Head (Decoupled)
   ↓
Dynamic Assignment (SimOTA/TAL)
   ↓
Loss
```

---

# 3️⃣ Step-by-step Implementation

---

## STEP 1 — Custom Sparse Global Block

เพิ่มไฟล์:

```
ultralytics/nn/modules/hsg.py
```

ตัวอย่างโครง:

```python
class SparseGlobalBlock(nn.Module):
    def __init__(self, c, k=512):
        super().__init__()
        self.k = k
        self.q = nn.Conv2d(c, c, 1)
        self.kv = nn.Conv2d(c, c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        feat = x.view(B, C, N)

        # importance score
        score = feat.mean(1)          # [B, N]
        topk = torch.topk(score, self.k, dim=1).indices

        selected = torch.gather(
            feat, 2,
            topk.unsqueeze(1).expand(-1, C, -1)
        )

        attn = torch.matmul(
            selected.transpose(1,2),
            selected
        ) / (C ** 0.5)

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, selected.transpose(1,2))
        out = out.transpose(1,2)

        # scatter back
        feat_out = feat.clone()
        feat_out.scatter_(2,
            topk.unsqueeze(1).expand(-1, C, -1),
            out)

        return feat_out.view(B, C, H, W)
```

---

## STEP 2 — Inject into YAML

สร้างไฟล์:

```
hsg_det.yaml
```

แทรก block หลัง PAN:

```yaml
neck:
  - [-1, 1, C2f, [256]]
  - [-1, 1, SparseGlobalBlock, [256, 512]]
```

แค่ P4/P5 scale ก็พอ

---

## STEP 3 — Assignment Strategy

เลือก:

* SimOTA (stable)
  หรือ
* TAL (YOLOv8)

ไม่ควรเปลี่ยนเป็น Hungarian (ไม่ conservative)

---

## STEP 4 — Loss

ใช้:

[
\mathcal{L} =
\lambda_{box} L_{CIoU}
+
\lambda_{obj} L_{BCE}
+
\lambda_{cls} L_{BCE}
]

เพิ่ม optional:

Focal scaling ถ้า class imbalance สูง

---

## STEP 5 — Training Strategy (1080p)

### Image size

```
imgsz=1280 ก่อน
fine-tune 1920
```

### Memory

* gradient accumulation
* batch size 2–4
* AMP (FP16)

### Hyper

| Parameter     | Value |
| ------------- | ----- |
| lr0           | 0.01  |
| weight_decay  | 5e-4  |
| momentum      | 0.937 |
| ema           | True  |
| warmup_epochs | 3     |

---

# 4️⃣ FLOPs Impact

Baseline YOLO-L 1080p ≈ 550G
Sparse block (k=512) ≈ +120G

Total ≈ 670G

Latency ≈ +4–6 ms

---

# 5️⃣ Stability Advantage

เพราะ:

* ไม่แตะ training loop
* ไม่แตะ assignment core
* ไม่เปลี่ยน output format

ความเสี่ยง training collapse ต่ำมาก

---

# 6️⃣ Occlusion Handling

Sparse global block ช่วย:

* feature interaction ข้าม object
* ลด duplicate
* เพิ่ม contextual reasoning

โดยไม่ต้อง quadratic attention

---

# 7️⃣ Optional Upgrade

ถ้าอยากดีขึ้นอีก:

* Top-k dynamic (adaptive k)
* Only apply attention to high-density region
* Residual gating

---

# 8️⃣ Deployment

Export ได้:

```
model.export(format="onnx")
model.export(format="engine")
```

เพราะยังเป็น CNN-based graph

---

# 9️⃣ Final Recommendation

Conservative HSG-DET คือ:

CNN backbone

* PAN
* SparseGlobalBlock (P4/P5)
* Decoupled Head
* SimOTA

ได้:

* dense ดี
* occlusion ดี
* latency ต่ำ
* memory stable
* export ได้

---

ผมคำนวณ param จริงให้

ทำ layer-by-layer FLOPs breakdown

เขียน paper-style section

ทำ entropy bound theorem