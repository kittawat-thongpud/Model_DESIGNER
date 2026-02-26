# สรุปโครงสร้างเชิง **Ultralytics YAML View**

## เปรียบเทียบ **YOLO11** vs **RF-DETR**

---

# 1) โครงสร้างเชิง YAML: YOLO11 (Ultralytics style)

> แนวคิด: CNN backbone + FPN/PAN neck + decoupled head (one-to-many)

```yaml
# yolo11.yaml (conceptual)
nc: 80
depth_multiple: 1.0
width_multiple: 1.0

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]        # 0 stem
  - [-1, 3, C2f, [128, True]]       # 1
  - [-1, 1, Conv, [256, 3, 2]]      # 2
  - [-1, 6, C2f, [256, True]]       # 3
  - [-1, 1, Conv, [512, 3, 2]]      # 4
  - [-1, 6, C2f, [512, True]]       # 5
  - [-1, 1, SPPF, [512, 5]]         # 6

neck:
  - [[6, 3], 1, Concat, [1]]        # FPN up
  - [-1, 3, C2f, [256]]             # P4
  - [[-1, 1], 1, Concat, [1]]
  - [-1, 3, C2f, [128]]             # P3
  - [[-1, 5], 1, Concat, [1]]       # PAN down
  - [-1, 3, C2f, [256]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

head:
  - [[P3, P4, P5], 1, Detect, [nc]]  # anchor-free decoupled head
```

---

## โครงสร้างเชิงแนวคิด

| ส่วน         | บทบาท                    | โครงสร้าง                |
| ------------ | ------------------------ | ------------------------ |
| Backbone     | Local feature extraction | CNN (C2f, residual-like) |
| Neck         | Multi-scale fusion       | FPN + PAN                |
| Head         | Decoupled prediction     | one-to-many              |
| Assignment   | Dynamic (e.g. TAL)       | many positives           |
| Post-process | NMS                      | จำเป็น                   |

---

# 2) โครงสร้างเชิง YAML: RF-DETR (Ultralytics-style abstraction)

> แนวคิด: CNN/Hybrid backbone + Transformer encoder/decoder + one-to-one set prediction

```yaml
# rf_detr.yaml (conceptual)
nc: 80
num_queries: 300
hidden_dim: 256

backbone:
  - [-1, 1, ConvStem, [64]]
  - [-1, 3, C2f, [128]]
  - [-1, 6, C2f, [256]]
  - [-1, 6, C2f, [512]]

encoder:
  - [-1, 6, TransformerEncoderLayer, [hidden_dim, 8]]

decoder:
  - [-1, 6, TransformerDecoderLayer, [hidden_dim, 8, num_queries]]

head:
  - [-1, 1, Linear, [4]]      # box regression
  - [-1, 1, Linear, [nc]]     # class logits

criterion:
  - HungarianMatcher
  - SetLoss
```

---

## โครงสร้างเชิงแนวคิด

| ส่วน         | บทบาท               | โครงสร้าง       |
| ------------ | ------------------- | --------------- |
| Backbone     | Spatial compression | CNN             |
| Encoder      | Global interaction  | Self-attention  |
| Decoder      | Object queries      | Cross-attention |
| Head         | 1 query → 1 object  | one-to-one      |
| Assignment   | Hungarian           | unique          |
| Post-process | none                | ไม่ต้อง NMS     |

---

# 3) โครงสร้างเชิงคณิตศาสตร์

## YOLO11

[
p(Y|X)
======

\prod_{i,j}
p(y_{ij} | F_{ij})
]

* Grid-wise factorization
* Local receptive field
* Output redundancy → NMS

---

## RF-DETR

[
p(Y|X)
======

\prod_{k=1}^{M}
p(y_k | X)
]

* Set prediction
* Global receptive field (attention)
* One-to-one matching

---

# 4) เปรียบเทียบเชิงระบบ

| Dimension            | YOLO11              | RF-DETR              |
| -------------------- | ------------------- | -------------------- |
| Paradigm             | Dense prediction    | Set prediction       |
| Complexity           | O(HW)               | O((HW)^2) attention  |
| Matching             | Dynamic many-to-one | Hungarian one-to-one |
| NMS                  | Required            | Not required         |
| Gradient noise       | Higher (duplicates) | Lower                |
| Real-time            | Excellent           | Moderate             |
| Long-range reasoning | Limited             | Strong               |

---

# 5) ความต่างเชิง Entropy Reduction

## YOLO11

Entropy reduction แบบ local

* Backbone → ลด spatial redundancy
* Assignment → ลด label ambiguity
* NMS → ลด output redundancy

---

## RF-DETR

Entropy reduction แบบ global

* Attention → ลด joint entropy
* Hungarian → eliminate duplicate hypothesis
* No NMS → entropy minimized inside model

---

# 6) มุมมองสรุปเชิงสถาปัตยกรรม

YOLO11 คือ

> CNN + Multi-scale + Dense Regression + Post-hoc Filtering

RF-DETR คือ

> CNN + Global Attention + Set Matching + End-to-End Structured Prediction

---

# 7) เชิง Practical สำหรับ 1080p

| ถ้าเป้าหมายคือ                   | ควรเลือก                     |
| -------------------------------- | ---------------------------- |
| Real-time edge                   | YOLO11                       |
| Scene complexity สูง / occlusion | RF-DETR                      |
| Small objects density สูง        | YOLO (multi-scale advantage) |
| Clean end-to-end pipeline        | RF-DETR                      |

---


# ขยายโครงสร้าง **Encoder–Decoder ของ RF-DETR**

(เทียบกรอบคิดกับ YOLO แบบเป็นระบบ)

---

# 1) ภาพรวมเชิง Pipeline

## YOLO (Dense Grid)

```
Image → CNN → [B, C, H, W]
              ↓
          Grid cells (H×W)
              ↓
      Per-cell prediction
```

* หน่วยพื้นฐาน = **grid cell**
* โครงสร้างข้อมูล = tensor 4D
* แต่ละตำแหน่งทำนายอิสระ (local factorization)

---

## RF-DETR (Set Prediction via Transformer)

```
Image → CNN → Feature map
              ↓ flatten
        Tokens [B, HW, C]
              ↓
         Transformer Encoder
              ↓
        Object Queries (N)
              ↓
         Transformer Decoder
              ↓
      N predicted objects
```

* หน่วยพื้นฐาน = **token**
* โครงสร้างข้อมูล = sequence
* ทำนายเป็น **set** (no grid)

---

# 2) Encoder คืออะไร?

## Input ของ Encoder

จาก backbone ได้ feature map:

[
F \in \mathbb{R}^{B \times C \times H \times W}
]

flatten เป็น sequence:

[
F_{seq} \in \mathbb{R}^{B \times (HW) \times C}
]

เพิ่ม positional encoding:

[
F_{enc_in} = F_{seq} + PE
]

---

## Encoder ทำอะไร?

### Self-Attention

[
Q = F W_Q,\quad
K = F W_K,\quad
V = F W_V
]

[
\text{Attention}(Q,K,V)
=======================

\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
]

### Output shape

[
F_{enc_out} \in \mathbb{R}^{B \times (HW) \times C}
]

ความหมาย:

* ทุก pixel token เห็น pixel อื่นทั้งหมด
* สร้าง **global context representation**

---

# 3) Decoder คืออะไร?

## Concept: Object Queries

ไม่ใช้ grid

สร้าง learnable queries:

[
Q_{obj} \in \mathbb{R}^{B \times N \times C}
]

* N = จำนวน object hypothesis (เช่น 300)
* ไม่ผูกกับตำแหน่งใด ๆ

---

## Decoder ทำอะไร?

### Step 1: Self-attention ระหว่าง queries

[
Q_{obj} \rightarrow
\text{Self-Attn}
]

ให้ object hypotheses คุยกันเอง

---

### Step 2: Cross-attention กับ encoder output

[
\text{CrossAttn}(Q_{obj}, F_{enc})
]

Query = object query
Key/Value = encoded feature tokens

---

## Output shape

[
F_{dec_out} \in \mathbb{R}^{B \times N \times C}
]

จากนั้น:

* Linear → box (4)
* Linear → class (nc)

สุดท้าย:

[
Y \in \mathbb{R}^{B \times N \times (4 + nc)}
]

---

# 4) Queue คืออะไรใน Transformer?

ใน YOLO:

* Grid = implicit queue
* แต่ละ cell = 1 hypothesis
* Indexing แบบ spatial

ใน RF-DETR:

* Sequence of tokens = queue
* Shape: [HW tokens]
* Decoder queue = [N object queries]

สองระดับ queue:

| ระดับ               | ขนาด | ความหมาย          |
| ------------------- | ---- | ----------------- |
| Spatial token queue | HW   | ทุก pixel feature |
| Object query queue  | N    | candidate objects |

---

# 5) เทียบ Grid vs Query

## YOLO

[
H \times W \times A
]

เช่น 80×80×3 = 19,200 hypotheses

* anchored to location
* many-to-one assignment

---

## RF-DETR

[
N = 300
]

* location-free
* Hungarian matching (one-to-one)

---

# 6) Attention ต่างจาก Conv ยังไง?

## Convolution (YOLO)

* Local receptive field
* O(HW)
* Weight sharing spatially

---

## Self-Attention (RF-DETR)

* Global receptive field
* O((HW)^2)
* Dynamic weighting

---

# 7) Input / Output Shape เปรียบเทียบ

| Stage        | YOLO             | RF-DETR      |
| ------------ | ---------------- | ------------ |
| Backbone out | [B,C,H,W]        | [B,C,H,W]    |
| Flatten      | ❌                | [B,HW,C]     |
| Encoder      | ❌                | [B,HW,C]     |
| Queries      | grid implicit    | [B,N,C]      |
| Decoder out  | ❌                | [B,N,C]      |
| Final pred   | [B,H,W,A*(4+nc)] | [B,N,(4+nc)] |

---

# 8) การจัดเรียงข้อมูล (Data Layout)

YOLO:

```
Tensor
[B, C, H, W]
→ reshape
[B, H, W, A, 4+nc]
```

RF-DETR:

```
Tensor
[B, C, H, W]
→ flatten
[B, HW, C]
→ attention
→ [B, N, C]
```

---

# 9) Intuition แบบง่าย

YOLO:

> ทุกตำแหน่งในภาพ พยายามทำนายวัตถุ

RF-DETR:

> สร้าง N คนไปค้นหาวัตถุทั่วทั้งภาพ

---

# 10) ความแตกต่างเชิง Optimization

YOLO:

* Assignment external
* NMS external
* Redundant hypothesis

RF-DETR:

* Assignment internal (Hungarian)
* No NMS
* Unique hypothesis

---

# 11) ถ้าภาพ 1080p (1920×1080)

สมมติ stride 32:

Feature map ≈ 60×34 ≈ 2040 tokens

Self-attention complexity:

[
2040^2 ≈ 4.1M
]

ยัง manageable

แต่ถ้าใช้ stride 16:

[
(120×68)^2 ≈ 66M
]

ต้องใช้:

* multi-scale encoder
* deformable attention
* sparse attention

---

# 12) สรุปเชิงโครงสร้าง

YOLO:

* Spatial grid factorization
* Local prediction
* Post-hoc filtering

RF-DETR:

* Token sequence modeling
* Global reasoning
* Set prediction

---