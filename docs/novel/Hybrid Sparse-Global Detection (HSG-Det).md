## แนวคิดโครงสร้างใหม่ (Blend Novel Structure)

**เป้าหมาย:**

* รองรับ 1080p memory ดี
* Dense objects ดีมาก
* Occlusion ดีมาก
* Latency ต่ำ
* Post-process optional
* Stability สูง

---

# ชื่อแนวคิด: **Hybrid Sparse-Global Detection (HSG-Det)**

> รวมข้อดีของ CNN-grid (เร็ว, memory ดี)
>
> * Sparse Query Attention (global reasoning)
> * One-to-many → One-to-one adaptive matching

---

# 1️⃣ Backbone — Efficient Dense Encoder

### โครงสร้าง

* CSP/C2f-style residual split (ลด redundancy)
* Stride pyramid: 8 / 16 / 32
* Depthwise separable conv บาง stage
* Partial global context block (low-res only)

### หลักการ

* 1080p → ใช้ stride 8 ต่ำสุด
  → feature ~ 240×135
* Global attention ทำเฉพาะ stride 32 (≈ 30×17)

[
\text{Memory} \sim O((HW)_{low}^2)
]

→ memory ควบคุมได้

### คุณสมบัติ

* Dense objects ดีมาก (grid retains locality)
* Memory efficient
* Gradient stable (CSP split)

---

# 2️⃣ Neck — Dual-Path Fusion

### Path A: Local Path (PAN/FPN style)

รักษา spatial detail

### Path B: Sparse Global Tokens

* Extract K salient tokens per scale (top-k activation)
* Cross-scale aggregation

[
K \ll HW
]

→ complexity ต่ำ

### ผลลัพธ์

* Scale ambiguity ลด
* Occlusion reasoning ดีขึ้น
* Latency ยังต่ำ (sparse attention)

---

# 3️⃣ Head — Dual-Mode Decoupled Head

### Branch 1: Dense Grid Head

* Anchor-free
* Predict box + cls per cell
* Fast, dense coverage

### Branch 2: Sparse Query Head

* N = 200 learnable queries
* Cross-attention กับ global tokens
* Predict refined boxes

### Output merge:

* During training → both active
* During inference:

  * Fast mode → grid only
  * High-accuracy mode → fuse both

---

# 4️⃣ Detection / Assignment — Adaptive Matching

### Early training:

Dynamic many-to-one (SimOTA-like)

### Late training:

Gradually shift to one-to-one (Hungarian-lite)

[
\alpha(t) \rightarrow 1
]

→ Transition matching scheme

### Result:

* Stable early training
* NMS optional
* Duplicate ลดเองตาม learned uniqueness

---

# 🔬 เชิงคุณสมบัติ

| Requirement           | วิธีที่ HSG-Det ตอบโจทย์                       |
| --------------------- | ---------------------------------------------- |
| 1080p memory ดี       | Global attention เฉพาะ low-res + sparse tokens |
| Dense objects ดีมาก   | Grid branch ครอบคลุมทุก cell                   |
| Occlusion ดีมาก       | Query branch reasoning global                  |
| Latency ต่ำ           | Grid inference mode                            |
| Post-process optional | One-to-one refinement                          |
| Stability สูง         | CSP backbone + progressive assignment          |

---

# 🔁 Complexity Overview

Let:

* Grid cells ≈ 32k (stride 8)
* Sparse tokens K ≈ 256
* Queries N ≈ 200

Total cost:

[
O(HW) + O(KN)
]

แทนที่จะเป็น:

[
O((HW)^2)
]

---

# 🎯 ทำไมมันดีกว่า YOLO และ DETR แบบตรง ๆ

| Model   | ข้อจำกัด                                         |
| ------- | ------------------------------------------------ |
| YOLO    | ไม่มี global reasoning                           |
| DETR    | Latency สูง, convergence ช้า                     |
| HSG-Det | Retain locality + inject sparse global reasoning |

---

# 🧠 มุมมอง Entropy

Grid branch:
[
H_{local} \downarrow
]

Query branch:
[
H_{global} \downarrow
]

รวมกัน:
[
H(Y|X;\theta) \text{ ลดทั้งเชิงพื้นที่และเชิงบริบท}
]

---

# 📦 Deployment Modes

| Mode     | ใช้ branch                 | เหมาะกับ      |
| -------- | -------------------------- | ------------- |
| Edge     | Grid only                  | Real-time     |
| Balanced | Grid + few queries         | Smart cameras |
| Full     | Grid + full sparse queries | Research      |

---

# 🔮 สรุปเชิงวิศวกรรม

นี่คือสถาปัตยกรรมที่:

* ไม่เสีย speed แบบ DETR
* ไม่เสีย global context แบบ YOLO
* ไม่ explode memory ที่ 1080p
* ไม่ต้องพึ่ง NMS เสมอไป
* Stable training เพราะไม่ pure transformer

---

หากต้องการต่อ:

* วาด architecture block diagram
* คำนวณ FLOPs จริงที่ 1080p
* เขียน pseudo-YAML แบบ ultralytics
* หรือ formalize เป็น theorem เชิง entropy bound

เลือกแนวที่จะขยายต่อได้ทันที.
