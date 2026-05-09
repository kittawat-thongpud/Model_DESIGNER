## Search Decomposition

**Query:** CNN-Transformer Hybrid Sparse detection papers for HSG-DETR improvement

### Facet 1: RT-DETR / RT-DETR2 — encoder & convergence math
### Facet 2: DINO denoising training + contrastive denoising convergence theory
### Facet 3: Sparse attention for vision detection — top-k selection math
### Facet 4: Occlusion-aware NMS-free detection + IDD dataset
### Facet 5: Mamba/SSM for vision detection — complexity vs attention

5 parallel research agents running. รอผลครับ — จะ synthesize เมื่อทุก agent เสร็จ

Sparse attention agent เสร็จแล้ว — ได้ข้อมูลดีมาก รอ 4 agents ที่เหลืออยู่ครับ

DINO agent เสร็จแล้ว รอ 3 agents ที่เหลือ (RT-DETR, Occlusion+IDD, Mamba) ครับ

RT-DETR agent เสร็จแล้ว รออีก 2 agents (Occlusion+IDD, Mamba) ครับ

รอ agent สุดท้าย (Occlusion+IDD) ครับ

ข้อมูลครบทั้ง 5 facets แล้ว สังเคราะห์เป็น report ได้เลยครับ

---

## External Context: CNN-Transformer Hybrid Sparse Detection — Reference Synthesis for HSG-DETR

---

### Key Finding Overview

| Facet | Core Paper | arXiv | ประโยชน์ต่อ HSG-DETR |
|---|---|---|---|
| RT-DETR encoder | Zhao et al. 2023 | 2304.08069 | AIFI+CCFF hybrid encoder, uncertainty query selection |
| RT-DETRv2 | Lv et al. 2024 | 2407.17140 | Scale-adaptive LR, aug schedule |
| DN-DETR | Li et al. 2022 | 2203.01305 | แก้ Hungarian instability → 2× เร็วขึ้น |
| DINO | Zhang et al. 2022 | 2203.03605 | CDN → ไม่มี duplicate detections |
| Sparse DETR | Roh et al. 2022 | 2111.14330 | DAM-supervised token selection |
| Conditional DETR | Meng et al. 2021 | 2108.06152 | Spatial query conditioning → 6.7-10× faster |
| Top-k Theory | Tzachristas et al. 2025 | 2512.07647 | Formal bound สำหรับ token budget |
| IDD Dataset | Varma et al. 2019 | 1811.10200 | 46,588 imgs, 40 classes, India-specific |
| Mamba | Gu & Dao 2023 | 2312.00752 | O(N) vs O(N²), SSM recurrence |
| VMamba | Liu et al. 2024 | 2401.10166 | SS2D scan, +3.8 box AP vs Swin-T |
| Mamba YOLO | Wang et al. 2024 | 2406.05835 | RG Block local+global, COCO 49.1 AP |

---

### 1. ทำไม HSG-DETR ถึง Converge ช้า — Root Cause จาก DN-DETR

**DN-DETR (arXiv:2203.01305, CVPR 2022 Oral)** นิยาม Hungarian Matching Instability อย่างเป็นทางการ:

```
IS^i = Σ_{n=0}^{N} 𝟙(V^i_n ≠ V^{i-1}_n)        [Eq. 2, DN-DETR]
```

`V^i_n` = ground truth ที่ query n ถูก match ในรอบ i. ค่า IS สูง = assignment พลิกไปมาระหว่าง epoch → gradient ขัดแย้งกันเอง → เรียนรู้ช้ามาก

**สาเหตุทางคณิตศาสตร์:** Hungarian matching optimize cost matrix ใหม่ทุก iteration โดยไม่มี "memory" ว่า query นี้เคย match GT ใดมาก่อน เมื่อ prediction ยังแย่ (ช่วงแรกของ training) cost matrix เกือบสุ่ม → IS พุ่งสูง → gradient ไม่สอดคล้องกัน

---

### 2. แก้ด้วย Denoising Training (DN-DETR + DINO CDN)

#### 2a. Denoising Groups (DN-DETR)

สร้าง P กลุ่ม parallel โดยแต่ละกลุ่มมี M queries = จำนวน GT objects:

```python
noise_center:  |Δx| < λ₁·w/2,  |Δy| < λ₁·h/2     (λ₁ = 0.4)
noise_size:    w' ∈ [(1−λ₂)w, (1+λ₂)w]              (λ₂ = 0.4)
```

**Attention mask (Eq. 7):**
```
a_{ij} = 1  ถ้า j ∈ denoising AND ⌊i/M⌋ ≠ ⌊j/M⌋   # คนละกลุ่มมองกันไม่เห็น
a_{ij} = 1  ถ้า j ∈ denoising AND i ∈ matching      # matching branch ห้ามเห็น DN
a_{ij} = 0  otherwise
```

เหตุผล: Denoising queries มี fixed target → gradient ชัดเจน, ไม่ขึ้นกับ matching instability → 2× faster convergence เทียบ DAB-DETR

#### 2b. Contrastive Denoising (DINO, arXiv:2203.03605, ICLR 2023)

DINO เพิ่ม **negative queries** ป้องกัน duplicate detections:

```
Positive queries : noise scale < λ₁          → predict GT class + box
Negative queries : noise scale ∈ (λ₁, λ₂)   → predict ∅ (background)
```

ผลลัพธ์เชิงปริมาณ: DETR ต้องการ 500 epochs → DINO ได้ AP เท่ากันด้วย 12-24 epochs (ResNet-50 on COCO)

**สำหรับ HSG-DETR:** `RTDETRDecoderSGB` ใช้ `get_cdn_group` จาก Ultralytics อยู่แล้ว แต่ควรตรวจสอบว่า λ₁=0.4, λ₂=0.5 และจำนวน CDN groups ถูกต้อง

---

### 3. RT-DETR Hybrid Encoder — สิ่งที่ HSG-DETR ยืมมาได้

**RT-DETR (arXiv:2304.08069, CVPR 2024):**

```
Q = K = V = Flatten(S₅)
F₅ = Reshape(AIFI(Q, K, V))        # Transformer บน S5 เท่านั้น
O  = CCFF({S₃, S₄, F₅})            # CNN cross-scale fusion
```

เหตุผลที่ apply Transformer บน S5 เท่านั้น:
> "applying the self-attention operation to high-level features with richer semantic concepts captures the connection between conceptual entities"
> "intra-scale interactions of lower-level features are unnecessary due to the lack of semantic concepts"

**AIFI ใช้ 2D sine-cosine PE** — inject ลงใน Q และ K ก่อน dot-product (**เหมือน V2 ของเราที่ถูก revert**)

**Uncertainty-minimal Query Selection (Eq. 2-3):**
```
U(X̂) = ‖P(X̂) − C(X̂)‖            # ความไม่สอดคล้องระหว่าง localization และ classification
L(X̂, Ŷ, Y) = L_box(b̂, b) + L_cls(U(X̂), ĉ, c)
```

เลือก top-K encoder features ที่ U ต่ำสุด → queries ที่มั่นใจทั้ง location และ class → decoder เริ่มจาก spatial priors ที่ดี

เทียบกับ HSG-DETR ปัจจุบัน: เราใช้ `alpha * energy_norm + cls_score_norm` ซึ่งใกล้เคียงแต่ไม่ได้วัด consistency ระหว่างทั้งสอง

---

### 4. Formal Bound สำหรับ Token Budget ของ SGTokenBlock

**"A Mathematical Theory of Top-k Sparse Attention" (arXiv:2512.07647, 2025)**

นี่คือ paper เดียวที่มี formal bound:

```
TV(P, P̂) = 1 − exp(−KL(P̂ ‖ P))          [Theorem 4.3 — exact equality]

‖Attn(q,K,V) − Attn_k(q,K,V)‖₂ = τ · ‖μ_tail − μ_head‖₂    [Theorem 5.2]
```

โดย τ = tail mass ที่ถูกตัดออก, μ_tail/μ_head = conditional mean ของ value vectors

**Design Rule — ต้องการ k เท่าไรเพื่อ TV error ≤ ε:**
```
k_ε/n ≈ Φ_c(σ + Φ⁻¹(ε))

เมื่อ σ = standard deviation ของ attention scores
     Φ_c = Gaussian survival function
```

**ตัวอย่าง:** ถ้า score มี σ=1.0 และยอม error ε=0.05 → k/n ≈ Φ_c(1.0 + Φ⁻¹(0.05)) = Φ_c(1.0 − 1.645) = Φ_c(−0.645) ≈ 0.74 → ต้องการ 74% tokens

**ความสำคัญต่อ HSG-DETR:** token ratios 12/24/48% ของ Legacy เป็นค่าจาก heuristic ไม่ใช่จาก score distribution — ควรวัด σ จริงของ attention scores ในระหว่าง training แล้วคำนวณ k_ε/n จาก formula นี้

---

### 5. Supervised Token Selection — Sparse DETR DAM Loss

**Sparse DETR (arXiv:2111.14330, ICLR 2022):**

L2 energy selector ของ SGTokenBlock เป็น unsupervised — ไม่รู้ว่า token ไหน "สำคัญ" ต่อ decoder จริงๆ

Sparse DETR แก้ด้วย **Decoder cross-Attention Map (DAM) supervision:**
```
DAM_x = Σ_{p,A,r} A · G(x, r+p)           # สะสม attention weights ของ decoder
L_dam = -(1/N) Σ BCE(g(x_feat)_i, DAM_i^bin)   # train scorer ให้ predict DAM
```

**คุณภาพ selector วัดด้วย:**
```
Corr = (Σ_{x ∈ Ω_D ∩ Ω_sp} DAM_x) / (Σ_{x ∈ Ω_D} DAM_x)
```

**ผล:** 38% FLOP reduction, 42% FPS gain ที่ ρ=10% โดย AP drop เพียง 0.1%

---

### 6. Occlusion — เหตุผลทางคณิตศาสตร์ว่า DETR ดีกว่า YOLO

**NMS failure mode:**
```
Keep b_i ถ้า ∀j≠i: score(j) > score(i) → IoU(b_i, b_j) < θ_NMS
```

เมื่อ IoU(b_A, b_B) > θ_NMS และทั้งคู่เป็น true positive → NMS suppresses ตัวใดตัวหนึ่งเสมอ (**structural failure**)

**Hungarian matching (DETR):**
```
σ* = argmin_{σ ∈ S_N} Σ L_match(y_i, ŷ_σ(i))
```

เป็น **globally optimal bijection** → แต่ละ GT object ได้ 1 query และแต่ละ query ได้ 1 GT → ไม่มี threshold-based suppression → Birkhoff–von Neumann theorem รับประกัน global optimum

**สำหรับ IDD:** CrowdHuman dataset (1811.10200) แสดงว่า 2.4 pairs/image มี IoU > 0.5 — NMS จะ fail ทุก pair นั้น DETR-family ไม่มีปัญหานี้

---

### 7. IDD Dataset — สถิติที่สำคัญ

**IDD (arXiv:1811.10200, WACV 2019)**:

| | IDD Detection | COCO |
|---|---|---|
| Images | 46,588 | 118,000 |
| Classes | 40 | 80 |
| Unique classes | Autorickshaw (32K inst), Animal (6K), Rider (97K) | ไม่มี |
| Traffic pattern | ไม่มี lane discipline, mixed road users | structured |

**Challenge หลักสำหรับ HSG-DETR:**
- **Autorickshaw 32,280 instances** — class ไม่มีใน COCO pretrained → saliency head ไม่รู้จัก → sparse token selection อาจ miss
- **Rider 97,626 instances** — person บน 2-wheeler มักถูก occlude โดย vehicle — ต้องการ NMS-free

---

### 8. Mamba — Complexity ทางคณิตศาสตร์

**Mamba (arXiv:2312.00752, ICLR 2024):**

```
h_t = Ā(x_t) · h_{t-1} + B̄(x_t) · x_t     # selective SSM recurrence
y_t = C(x_t) · h_t

Ā = exp(Δ(x_t) · A)                          # ZOH discretization, input-dependent
B̄ = (ΔA)⁻¹ · (exp(ΔA) − I) · Δ(x_t)·B
```

| | Complexity | Memory |
|---|---|---|
| Transformer | O(N²d) | O(N²) |
| Mamba | O(N·d·D_state) | O(N) |

**VMamba (arXiv:2401.10166, NeurIPS 2024 Spotlight):** Cross-Scan Module scan 4 directions → +3.8 box AP vs Swin-T บน COCO (46.5 vs 42.7) ด้วย Mask R-CNN

**Mamba YOLO (arXiv:2406.05835, AAAI 2025):** RG Block แก้ปัญหา local spatial ของ SSM:
```
f_RG = T_fusion(R_global ⊙ Y(R_local)) ⊕ f_A'
```
→ Mamba YOLO-B: 49.1 AP vs YOLOv8-S: 44.9 AP (+4.2 ที่ latency ใกล้กัน)

---

### สรุป: Improvement Roadmap สำหรับ HSG-DETR (เรียงตาม ROI)

| Priority | Improvement | Paper | Mathematical Basis | Expected Gain |
|---|---|---|---|---|
| **1** | Contrastive Denoising (CDN) | DINO 2203.03605 | IS metric → CDN removes Hungarian instability | ~2× fewer epochs |
| **2** | Uncertainty-minimal query selection | RT-DETR 2304.08069 | U(X̂) = ‖P−C‖ selects jointly confident features | +1-2 AP |
| **3** | DAM-supervised token selection | Sparse DETR 2111.14330 | L_dam = BCE(scorer, DAM^bin) | token Corr ↑, AP stable |
| **4** | Formal token budget sizing | Top-k Theory 2512.07647 | k_ε/n ≈ Φ_c(σ + Φ⁻¹(ε)) | justified ratios |
| **5** | MambaVision-T/S backbone | MambaVision 2407.08083 | O(N) SSM + hybrid attention stages | +3-5 AP base |

---

### Sources

- [RT-DETR arXiv:2304.08069](https://arxiv.org/abs/2304.08069) — DETRs Beat YOLOs, CVPR 2024
- [RT-DETRv2 arXiv:2407.17140](https://arxiv.org/abs/2407.17140) — Bag-of-Freebies, 2024
- [DINO arXiv:2203.03605](https://arxiv.org/abs/2203.03605) — ICLR 2023
- [DN-DETR arXiv:2203.01305](https://arxiv.org/abs/2203.01305) — CVPR 2022 Oral
- [Sparse DETR arXiv:2111.14330](https://arxiv.org/abs/2111.14330) — ICLR 2022
- [Conditional DETR arXiv:2108.06152](https://arxiv.org/abs/2108.06152) — ICCV 2021
- [Top-k Theory arXiv:2512.07647](https://arxiv.org/abs/2512.07647) — 2025
- [IDD arXiv:1811.10200](https://arxiv.org/abs/1811.10200) — WACV 2019
- [Mamba arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — ICLR 2024
- [VMamba arXiv:2401.10166](https://arxiv.org/abs/2401.10166) — NeurIPS 2024
- [Mamba YOLO arXiv:2406.05835](https://arxiv.org/abs/2406.05835) — AAAI 2025
- [MambaVision arXiv:2407.08083](https://arxiv.org/abs/2407.08083) — CVPR 2025
- [DETR arXiv:2005.12872](https://arxiv.org/abs/2005.12872) — ECCV 2020
- [PED arXiv:2012.06785](https://arxiv.org/abs/2012.06785) — DETR for crowd occlusion