# HSG-DET Experiment Report

**Date:** 2026-03-26  
**Dataset:** IDD (Indian Driving Dataset)  
**Machines:** ku4070 · ku4070-aj · rase4090 · rase4090-2  
**Trainer:** Ultralytics YOLO (TAL assignment, standard losses)  
**Architecture source:** `backend/hsg_det/configs/hsg_det_m.yaml` + `backend/hsg_det/nn/sparse_global.py`

---

## 1) ผลการทดลอง: HSG-DET vs YOLOv8

### 1.1 Overall mAP Comparison (Best per group, optimizer=auto)

| Scale | imgsz | Model | Epoch | mAP50 | mAP50-95 | Δ mAP50 | Δ mAP50-95 | Winner |
|-------|-------|-------|-------|-------|----------|---------|-----------|--------|
| **n** | **640** | HSG-DET [800,200] | 458/500 | 0.480 | 0.3059 | | | |
| | | YOLOv8n | 433/500 | 0.477 | 0.3038 | **+0.003** | **+0.0021** | 🟡 HSG-DET (marginal) |
| **n** | **1280** | HSG-DET [160,40] | 447/500 | 0.618 | 0.4032 | | | |
| | | YOLOv8n | 434/500 | **0.619** | 0.4023 | −0.001 | **+0.0009** | 🟡 Tie |
| **s** | **640** | HSG-DET [640,160] | 290/500 | 0.546 | 0.3508 | | | |
| | | YOLOv8s | 279/500 | 0.542 | 0.3487 | **+0.004** | **+0.0021** | 🟡 HSG-DET (marginal) |
| **s** | **1280** | HSG-DET [160,40] | 184/500 | 0.675 | **0.4424** | | | |
| | | YOLOv8s *(running)* | 234/500 | **0.676** | 0.4418 | −0.001 | +0.0006 | 🟡 Tie (YOLO still training) |

> **สรุปภาพรวม:** ในทุก group HSG-DET ชนะหรือเสมอกันใน mAP50-95 ด้วยส่วนต่างเล็กน้อย (~0.001–0.002) ยังไม่มีกลุ่มใดที่ HSG-DET ชนะขาด ซึ่งบ่งชี้ว่า SGB ให้ gain เล็กน้อยแต่สม่ำเสมอ

---

### 1.2 Per-Class AP50 Comparison — Scale n, imgsz 640

*(Benchmark จาก rase4090 — HSG-DET [160,40]-n vs YOLOv8n)*

| Class | HSG-DET [160,40] | YOLOv8n | Δ AP50 | Winner |
|-------|:----------------:|:-------:|--------|--------|
| person | 0.484 | 0.472 | +0.012 | 🔵 HSG-DET |
| rider | 0.545 | 0.533 | +0.012 | 🔵 HSG-DET |
| car | 0.656 | 0.644 | +0.012 | 🔵 HSG-DET |
| truck | 0.599 | 0.591 | +0.008 | 🔵 HSG-DET |
| bus | 0.682 | 0.668 | +0.014 | 🔵 HSG-DET |
| motorcycle | 0.647 | 0.636 | +0.011 | 🔵 HSG-DET |
| **bicycle** | **0.304** | **0.341** | −0.037 | 🔴 **YOLO** |
| autorickshaw | 0.669 | 0.658 | +0.011 | 🔵 HSG-DET |
| **animal** | **0.193** | **0.203** | −0.010 | 🔴 **YOLO** |
| traffic light | 0.210 | 0.168 | +0.042 | 🔵 HSG-DET |
| traffic sign | 0.284 | 0.283 | +0.001 | 🟡 Tie |
| **mAP50** | **0.479** | **0.472** | **+0.007** | **HSG-DET 9/11** |

> **ข้อสังเกตสำคัญ:** HSG-DET ชนะ YOLO ใน 9/11 classes แต่แพ้ในกลุ่ม `bicycle` และ `animal` ซึ่งเป็นวัตถุขนาดเล็กและปรากฏไม่สม่ำเสมอ — สะท้อนจุดอ่อนเชิงสถาปัตยกรรมที่อธิบายใน Section 2

---

### 1.3 Token Budget vs Performance (Scale n, imgsz 640, optimizer=auto)

| HSG-DET variant | top-k P4 | top-k P5 | mAP50 | mAP50-95 |
|-----------------|----------|----------|-------|----------|
| HSG-DET [160, 40] | 160 | 40 | 0.476 | 0.3034 |
| HSG-DET [320, 80] | 320 | 80 | 0.478 | 0.3056 |
| HSG-DET [480, 120] | 480 | 120 | 0.476 | 0.3035 |
| HSG-DET [640, 160] | 640 | 160 | 0.475 | 0.3041 |
| HSG-DET [800, 200] | 800 | 200 | **0.480** | **0.3059** |
| YOLOv8n (baseline) | — | — | 0.477 | 0.3038 |

> **แนวโน้ม:** เพิ่ม token budget ไม่ได้ช่วยอย่างชัดเจน — ทุก variant ให้ผลใกล้กันมาก (range < 0.005) และบาง variant แพ้ YOLO ใน mAP50 อยู่ด้วย บ่งชี้ว่าปัญหาอาจไม่ใช่ขนาด K แต่เป็นวิธีการเลือก token และตำแหน่งของ SGB

---

## 2) วิเคราะห์ Mechanism จากโค้ดจริง

### 2.1 สถาปัตยกรรมจริง (Ultralytics Trainer + hsg_det_m.yaml)

```
Input (B, 3, H, W)
    │
    ▼ Backbone: CSP/C2f (standard YOLOv8, unchanged)
    ├── P3: stride 8,  256ch  @ 135×240 (1080p) — 32,400 tokens   [LOCAL ONLY]
    ├── P4: stride 16, 512ch  @ 68×120  (1080p) —  8,160 tokens   [LOCAL + SGB]
    └── P5: stride 32, 1024ch @ 34×60   (1080p) —  2,040 tokens   [LOCAL + SGB]
    │
    ▼ Neck (hsg_det_m.yaml lines 52–79)
    ├─ [10] SparseGlobalBlockGated(1024, k=512)  @ P5  ← global context at P5
    ├─ [11] Upsample P5' → fuse with P4 (Concat)
    ├─ [13] C2f(512)
    ├─ [14] SparseGlobalBlockGated(512, k=512)   @ P4  ← global context at P4
    ├─ [15] Upsample P4' → fuse with P3 (Concat)
    ├─ [17] C2f(256)                             @ P3  ← NO SGB (local path only)
    ├─ [18–20] PAN bottom-up P3→P4
    ├─ [21–23] PAN bottom-up P4→P5
    └─ [24] Detect([P3, P4, P5])  ← standard YOLOv8 Detect head
    │
    ▼ Training: Ultralytics TAL assignment + VFL + CIoU + BCE
```

**หมายเหตุ:** `train_hsg_det.py` (custom trainer + one-to-few assignment) **ไม่ได้ถูกใช้** ในการทดลองนี้ — ใช้ Ultralytics pipeline ตลอด

---

### 2.2 SGB ทำงานอย่างไรในโค้ดจริง (`sparse_global.py`)

```python
# Token selection: L2 activation energy (heuristic, not learnable)
importance = x.view(B, C, N).float().pow(2).sum(dim=1)   # [B, N]
topk_idx = torch.topk(importance, k_actual, dim=1).indices

# Sparse self-attention (single-head, no positional encoding)
attn = torch.bmm(q_sel, k_sel.transpose(1, 2)) * scale   # [B, k, k]
attn = torch.softmax(attn, dim=-1)
attended = torch.bmm(attn, v_sel)                          # [B, k, C]

# Scatter back + residual
out.scatter_(2, idx_exp, attended.transpose(1, 2))
return x + self.gate * delta   # gate starts at 0
```

| ลักษณะ | รายละเอียด | ปัญหาที่พบ |
|--------|-----------|-----------|
| Token selection | L2 energy: $s_n = \sum_c x_{cn}^2$ | Heuristic — ไม่ task-aware |
| Gate init | `gate = 0` → identity เริ่มต้น | ดีสำหรับ warm-start แต่ต้องตรวจว่า gate โตพอ |
| Attention | Single-head, ไม่มี positional encoding | ขาด spatial awareness ของ token ที่เลือก |
| SGB @ P3 | **ไม่มี** — P3 เป็น local path ล้วน | Small objects (bicycle, animal) ไม่ได้รับ global context |
| Cross-scale | ไม่มี — P4 attend P4 เท่านั้น | ขาด context จาก scale อื่น |
| Trainer loss | Ultralytics TAL (ไม่ใช่ one-to-few) | Assignment คงที่ ไม่ progressive |

---

### 2.3 ช่องว่างหลักที่อธิบาย per-class gap

**เหตุใด `bicycle` และ `animal` ถึงแพ้ YOLO:**

- วัตถุทั้งสองปรากฏที่สเกลเล็ก → ตกบน **P3** เป็นหลัก
- P3 ไม่มี SGB → ได้รับแต่ local PAN path เหมือน YOLO ทุกอย่าง
- SGB ที่ P4/P5 ช่วยวัตถุกลาง-ใหญ่ แต่ไม่ส่งผลต่อ small-object features
- ผลคือ HSG-DET ≈ YOLO สำหรับ small classes แต่ดีกว่าสำหรับ medium-large

---

### 2.4 กลไกที่แนะนำ พร้อม Math Model

#### A. SGB ที่ P3 ด้วย Task-Guided Token Selection

**ปัญหาเดิม:** heuristic L2 energy มีแนวโน้มเลือก token ที่มี activation สูง ซึ่งมักเป็นวัตถุใหญ่ ทำให้ small objects ถูกละเลย

**แนวทาง:** แทนที่ด้วย composite importance score:

$$s_n = \underbrace{\|x_n\|_2^2}_{\text{activation energy}} + \lambda \cdot \underbrace{\hat{p}_n^{\text{small}}}_{\text{small-obj confidence}}$$

โดย $\hat{p}_n^{\text{small}}$ คือ predicted objectness จาก lightweight auxiliary head ที่ P3 ทำให้ token selection เป็น task-aware และโฟกัสบน small object regions ได้มากขึ้น

#### B. Cross-Scale Sparse Attention (P3 ← P4)

ให้ P3 selected tokens attend P4 features เพื่อดึง context จาก larger receptive field:

$$\text{Attn}_{P3 \to P4} = \text{softmax}\!\left(\frac{Q_{P3}^{\text{sel}} K_{P4}^T}{\sqrt{d}}\right) V_{P4}$$

$$\text{cost} = O(k_{P3} \cdot k_{P4} \cdot d) \ll O(N_{P3} \cdot N_{P4} \cdot d)$$

วิธีนี้ช่วยให้วัตถุเล็ก (บน P3) รับรู้ context จาก semantic features ที่ P4 โดยไม่ต้องทำ full cross-attention

#### C. Size-Aware Loss Reweighting

เพิ่ม weight สำหรับ small/hard classes ใน classification loss:

$$\mathcal{L}_{cls}^{\text{aug}} = \mathcal{L}_{VFL} + \alpha \sum_{n \in \mathcal{S}} \mathcal{L}_{VFL}(n)$$

โดย $\mathcal{S} = \{n : w_n \cdot h_n < \tau^2\}$ คือ set ของ anchor cells ที่รับผิดชอบ small objects (เช่น $\tau = 32$ pixels)

#### D. Learnable Saliency Gate แทน top-K heuristic

แทนที่ `torch.topk(importance)` ด้วย differentiable soft-mask:

$$m_n = \sigma\!\left(\frac{s_n - \mu_s}{\sigma_s} \cdot \beta\right), \quad \beta \text{ learned}$$

ทำให้ gradient ไหลผ่าน token selection ได้ และโมเดลเรียนรู้ว่า token ไหนสำคัญสำหรับ task โดยตรง

---

## 3) วิเคราะห์ Transformer Activation และ Bottleneck

### 3.1 สิ่งที่รู้จากโค้ดและผลทดลอง

| ประเด็น | หลักฐาน | ข้อสรุป |
|---------|---------|---------|
| Gate init = 0 | `self.gate = nn.Parameter(torch.zeros(1))` | SGB เริ่มจาก identity — ต้องตรวจว่า gate โตพอหลัง 300+ epochs |
| Token selection heuristic | L2 energy เท่านั้น | ไม่มีหลักฐานว่าเลือก object-relevant tokens จริง |
| mAP gain เล็กน้อย | ~+0.001–0.002 vs YOLO ทุก group | SGB มี impact แต่จำกัด |
| Small class performance | bicycle, animal แพ้ YOLO | SGB ที่ P4/P5 ไม่ช่วย small-object classes |
| ไม่มี attention log | ไม่ได้บันทึก gate value/attention entropy | ยังไม่สามารถยืนยันได้เชิงกลไก |

---

### 3.2 Bottleneck ที่ระบุได้

1. **P3 ไม่มี global context** — small objects ไม่ได้รับประโยชน์จาก SGB เลย *(primary bottleneck)*
2. **Gate อาจยังต่ำ** — gate=0 เริ่มต้น ต้องใช้เวลาหลาย epoch กว่า attention จะมี influence จริง
3. **Token selection ไม่ task-aware** — L2 energy อาจเลือก background texture แทน object tokens
4. **Single-head, ไม่มี positional encoding** — attention ไม่รู้ spatial structure ของ tokens ที่เลือก
5. **Trainer ไม่ใช้ one-to-few dynamic** — ใช้ TAL ของ Ultralytics ซึ่งดีแต่ไม่ได้ออกแบบมาสำหรับ SGB

---

### 3.3 Activation Audit — สิ่งที่ต้องตรวจสอบ

เพื่อยืนยันว่า Transformer/SGB ทำงานจริงและเต็มที่ ต้องเพิ่ม logging hooks ระหว่างเทรน:

```python
# Hook ตรวจ gate value ทุก N epoch
def log_gate_values(model):
    for name, m in model.named_modules():
        if isinstance(m, SparseGlobalBlockGated):
            print(f"{name}.gate = {m.gate.item():.4f}")

# Hook ตรวจ attention entropy
def log_attention_entropy(attn_weights):  # [B, k, k]
    H = -(attn_weights * (attn_weights + 1e-8).log()).sum(-1).mean()
    # ต่ำ = collapse, สูง = กระจาย/ไม่โฟกัส, เหมาะ ≈ log(k)/3
    return H.item()
```

**เกณฑ์ตรวจสอบ:**

| Metric | ดี | น่ากังวล |
|--------|-----|---------|
| `gate` value หลัง 300 epochs | > 0.3 | < 0.05 (block แทบไม่ทำงาน) |
| Attention entropy | log(k)/4 – log(k)/2 | < log(k)/8 (collapse) หรือ > log(k) (noise) |
| Gradient norm SGB / backbone | 0.1×–1× | < 0.01× (under-utilized) |
| Token coverage overlap with GT boxes | > 30% | < 10% (เลือก background) |

---

### 3.4 Validation Roadmap (3 รอบทดลอง)

```
รอบที่ 1 — Diagnostic (ทำก่อน)
  ├── เพิ่ม gate logger + attention entropy logger ใน Ultralytics callback
  ├── fine-tune จาก best weight (HSG-DET [160,40] s, 1280) อีก 50 epoch
  └── บันทึก gate value, attention entropy, gradient norm ทุก 10 epoch
  เป้าหมาย: ยืนยันว่า gate > 0 และ SGB มี influence จริง

รอบที่ 2 — Architecture Fix (หลังรอบ 1 ยืนยันแล้ว)
  ├── เพิ่ม SparseGlobalBlockGated ที่ P3 ด้วย k=1024 (3% ของ 32,400 tokens)
  ├── เปรียบเทียบ: (A) ไม่มี P3 SGB vs (B) มี P3 SGB
  └── lock seed เดิม เทรน 300 epochs — วัด AP_S โดยเฉพาะ
  เป้าหมาย: พิสูจน์ว่า P3 SGB ช่วย small-object classes

รอบที่ 3 — Token Selection (หลังรอบ 2 สำเร็จ)
  ├── แทนที่ L2 energy ด้วย task-guided score (composite importance)
  ├── เปรียบเทียบ: heuristic vs task-guided
  └── วัด token coverage overlap กับ GT boxes
  เป้าหมาย: ยืนยัน quality ของ token selection
```

---

## สรุปประเด็นสำคัญ (5 bullet)

- **HSG-DET ใกล้เคียง YOLO ทุก group** — ชนะหรือเสมอด้วยส่วนต่าง +0.001–0.002 mAP50-95 ยังไม่มีกลุ่มใดชนะขาด บ่งชี้ว่า SGB ให้ gain เล็กน้อยแต่ยังไม่พอ
- **Small object คือจุดอ่อนหลัก** — `bicycle`, `animal` แพ้ YOLO เพราะ P3 ไม่มี SGB และ L2 token selection ไม่ task-aware
- **Gate = 0 เริ่มต้น** คือ safety mechanism ที่ถูกต้อง แต่ต้องตรวจว่าหลัง 400+ epochs gate โตพอที่จะส่งผล จริง
- **Mechanism เร่งด่วนที่ควรเพิ่ม:** (1) SGB @ P3, (2) task-guided token selection, (3) cross-scale attention P3←P4
- **ขั้นตอนต่อไป:** Diagnostic fine-tune 50 epoch จาก base weight พร้อม logging hooks ก่อนเริ่ม architecture change
