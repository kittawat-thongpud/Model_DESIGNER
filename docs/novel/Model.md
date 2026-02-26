# HSG-DET — Layer Summary (1080p Input)

**Input:** 1920×1080
**Assume stride pyramid:** P3(1/8), P4(1/16), P5(1/32)
**Notation:** B=batch, C=channel, H,W=height,width

---

## 1️⃣ Layer Table

| id | layer               | input id | output id | shape (B, C, H, W) | math model               | description              |
| -- | ------------------- | -------- | --------- | ------------------ | ------------------------ | ------------------------ |
| 0  | Input               | —        | 1         | (B,3,1080,1920)    | —                        | RGB image                |
| 1  | Stem Conv           | 0        | 2         | (B,64,540,960)     | Conv3×3 s=2              | Initial compression      |
| 2  | CSP Block x3        | 1        | 3         | (B,128,270,480)    | Residual split-merge     | Low-level feature        |
| 3  | CSP Block x6        | 2        | 4         | (B,256,135,240)    | Deep residual mapping    | P3 candidate             |
| 4  | CSP Block x6        | 3        | 5         | (B,512,68,120)     | Hierarchical abstraction | P4 candidate             |
| 5  | CSP Block x3        | 4        | 6         | (B,1024,34,60)     | High semantic feature    | P5 candidate             |
| 6  | SPPF                | 5        | 7         | (B,1024,34,60)     | Multi-kernel pooling     | Context enrichment       |
| 7  | FPN Top-down        | 7,5,4    | 8         | multi-scale fused  | Upsample + concat        | Multi-scale transport    |
| 8  | PAN Bottom-up       | 8        | 9         | multi-scale fused  | Downsample + fuse        | Scale refinement         |
| 9  | Sparse Global Block | 9        | 10        | selected tokens k  | Top-k attention          | Global relation modeling |
| 10 | Box Branch          | 10       | 11        | (B,4,S,S)          | Conv→Linear              | Box regression           |
| 11 | Cls Branch          | 10       | 12        | (B,C,S,S)          | Conv→Linear              | Class logits             |
| 12 | Obj Branch          | 10       | 13        | (B,1,S,S)          | Conv→Sigmoid             | Objectness               |
| 13 | Matching            | 11–13    | 14        | N matched pairs    | Hungarian-lite / Top-k   | One-to-few dynamic       |
| 14 | Loss                | 14       | —         | scalar             | composite                | Risk minimization        |

---

# 2️⃣ Mathematical Model

## Detection Output Factorization

[
p(\mathbf{y}*{ij}|F*{ij})
=========================

p(\mathbf{b}*{ij}) \cdot p(o*{ij}) \cdot p(\mathbf{c}_{ij})
]

Sparse global modifies feature:

[
F' = F + \text{TopKAttention}(F)
]

Attention complexity:

[
O(k^2 d)
]

---

# 3️⃣ Recommended Loss Function

## Box Loss

* **CIoU + L1**
* For stability under occlusion

[
\mathcal{L}*{box} = \lambda_1 L*{CIoU} + \lambda_2 |b-\hat{b}|_1
]

---

## Classification Loss

* **Varifocal Loss** (better for dense imbalance)
  or
* **Focal BCE**

---

## Objectness

* BCE with positive reweight

---

## Total Loss

[
\mathcal{L}
===========

\lambda_{box} \mathcal{L}*{box}
+
\lambda*{cls} \mathcal{L}*{cls}
+
\lambda*{obj} \mathcal{L}_{obj}
]

---

# 4️⃣ Optimization Function

Recommended:

* **AdamW** (transformer component present)
* weight decay: 0.05
* cosine decay LR schedule

Alternative (pure CNN bias):

* SGD + momentum 0.937 (YOLO-style)

---

# 5️⃣ Gradient Update Strategy

### YOLO-style

* Dense one-to-many
* NMS post-process
* Stable gradient early

### RF-DETR-style

* One-to-one Hungarian
* Global attention
* Slower but structured

---

### Recommended Hybrid (HSG-DET)

* One-to-few matching (Top-k cost)
* Sparse global attention
* No full Hungarian
* Optional NMS fallback

Gradient path:

[
\nabla_\theta
=============

\nabla_{local}
+
\nabla_{global}
]

Stable like YOLO
Structured like DETR
But computationally bounded

---

# 6️⃣ Training Strategy

### Stage 1 — Warmup (YOLO-style)

* 10 epochs
* Dense assignment
* No sparse selection

### Stage 2 — Hybrid Training

* Activate sparse global
* Dynamic matching
* Mosaic + MixUp

### Stage 3 — Fine-tune

* Disable heavy augmentation
* EMA enabled
* Reduce LR 10×

---

# 7️⃣ Stability Comparison

| Aspect               | YOLO     | RF-DETR | HSG-DET |
| -------------------- | -------- | ------- | ------- |
| Convergence speed    | Fast     | Slow    | Fast    |
| Assignment stability | Medium   | High    | High    |
| Memory @1080p        | Low      | High    | Medium  |
| Occlusion handling   | Moderate | Strong  | Strong  |

---

# 8️⃣ Final Recommendation for 1080p Dense IDD

Use:

* CSP-based backbone
* Sparse Global at P4/P5 only
* Decoupled head
* One-to-few dynamic assignment
* CIoU + Varifocal loss
* AdamW + cosine schedule
* EMA enabled

---

If needed next:

* Full FLOPs derivation
* Parameter count exact
* Theorem: Entropy bound under sparse attention
* Ultralytics-style YAML prototype
