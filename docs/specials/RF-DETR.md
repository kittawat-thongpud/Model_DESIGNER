# บทพิเศษ

# พิสูจน์โครงสร้างของ **RF-DETR** เทียบกับ **YOLO**

---

## 1. ปัญหาพื้นฐาน: Detection คืออะไรในเชิงคณิตศาสตร์

จาก Ch.3 (Probabilistic Formalization)

ให้

$$
Y = {(b_k, c_k)}_{k=1}^{N}
$$

เป็น **set ของวัตถุ**
จำนวน (N) ไม่คงที่

โจทย์ detection คือประมาณค่า

$$
p(Y \mid X; \theta)
$$

ซึ่งเป็น **distribution บน structured set**

---

# 2. โครงสร้าง YOLO

## 2.1 Factorization แบบ Grid-wise

YOLO ใช้ conditional independence assumption

$$
p(Y \mid X; \theta) =
\prod_{i,j}
p(y_{ij} \mid F_{ij}; \theta_{\text{head}})
$$

(Ch.3.2)

โดย:

* แปลงภาพเป็น grid
* แต่ละ grid cell ทำนายอิสระ
* assignment ทำภายหลัง

---

## 2.2 ข้อจำกัดเชิงโครงสร้าง

### Lemma 1 — Independence Bias

ถ้า object span หลาย cell

$$
y_{ij} \not!\perp y_{kl}
$$

แต่โมเดลบังคับให้

$$
p(y_{ij}, y_{kl}) = 
p(y_{ij})p(y_{kl})
$$

⇒ เกิด **structural bias**

---

### Lemma 2 — Post-hoc De-duplication

YOLO ต้องใช้ NMS

เพราะ

$$
\exists i \neq j : \hat b_i \approx \hat b_j
$$

แปลว่า output space มี redundancy

---

# 3. โครงสร้าง RF-DETR

RF-DETR = Receptive-Field aware DETR (แนวคิด transformer-set prediction)

---

## 3.1 Set Prediction Formulation

แทนที่จะ factorize แบบ grid

RF-DETR ใช้ fixed set queries:

$$
Q = {q_k}_{k=1}^{M}
$$

แล้วให้

$$
p(Y \mid X)=
\prod_{k=1}^{M}
p(y_k \mid X)
$$

พร้อม constraint:

* one-to-one assignment
* Hungarian matching

(Ch.10 – One-to-One Assignment)

---

## 3.2 Structural Difference

| Property            | YOLO        | RF-DETR            |
| ------------------- | ----------- | ------------------ |
| Factorization       | Grid-wise   | Query-wise         |
| Assignment          | Many-to-one | One-to-one         |
| NMS                 | Required    | Not required       |
| Dependency modeling | Local       | Global (attention) |

---

# 4. พิสูจน์เชิงโครงสร้าง

---

## Theorem 1 — YOLO = Structured Local Approximation

ให้ $ F_{ij}$ เป็น local receptive field

YOLO ประมาณค่า:

$$
p(Y \mid X)
\approx
\prod_{i,j}
p(y_{ij} \mid F_{ij})
$$

ซึ่งเป็น:

$$
\text{Local Markov approximation}
$$

ดังนั้น dependency ข้ามพื้นที่ถูกตัดออก

---

## Theorem 2 — RF-DETR = Permutation-invariant Set Estimator

Transformer decoder ให้:

$$
h_k = \text{Attention}(q_k, F)
$$

โดย attention เชื่อมทุก spatial location

ดังนั้น

$$
p(y_k \mid X)
p(y_k \mid F_{\text{global}})
$$

จึง model cross-object interaction ได้

---

# 5. Entropy Perspective (Ch.21)

เป้าหมายคือ minimize

$$
H(Y \mid X; \theta)
$$

---

## YOLO Entropy Reduction

ลด entropy แบบแยก cell:

* Backbone → ลด spatial redundancy
* Assignment → ลด label ambiguity
* NMS → ลด output redundancy

แต่ entropy ข้าม cell ไม่ถูก model

---

## RF-DETR Entropy Reduction

Attention layer:

$$
I(F_i ; F_j) > 0
$$

จึงสามารถลด joint entropy ได้โดยตรง

---

# 6. ความต่างด้าน Gradient

## YOLO

Gradient:

$$
\nabla_\theta
\sum_{i,j}
\nabla_\theta \mathcal{L}_{ij}
$$

หลาย positive per object

→ gradient competition
→ assignment noise

---

## RF-DETR

Hungarian matching:

หนึ่ง object → หนึ่ง query

$$
\nabla_\theta =
\sum_{k=1}^{N}
\nabla_\theta \mathcal{L}_{k}
$$

ไม่มี duplicate gradients

(Ch.18 – Gradient Behavior)

---

# 7. Receptive Field Formal Proof Sketch

---

## Definition

ให้ receptive field ของ output node เป็น

$$
\mathcal{R}(y)
\subseteq X
$$

---

### YOLO

$$
|\mathcal{R}(y_{ij})|
\approx
\text{limited by backbone depth}
$$

จำกัด spatial locality

---

### RF-DETR

Self-attention:

$$
\mathcal{R}(y_k)
X
$$

ทุก token เห็นภาพทั้งหมด

⇒ global receptive field

---

## Corollary

สำหรับ scene ที่มี long-range dependency (เช่น occlusion)

$$
\text{RF-DETR expressivity} > \text{YOLO}
$$

---

# 8. Computational Trade-off

| Aspect        | YOLO             | RF-DETR                   |
| ------------- | ---------------- | ------------------------- |
| Complexity    | O(HW)            | O((HW)^2) attention       |
| Latency       | ต่ำ              | สูงกว่า                   |
| Small objects | ดี (multi-scale) | ดีถ้า multi-scale encoder |
| Real-time     | เหมาะ            | ต้อง optimize             |

---

# 9. บทสรุปเชิงโครงสร้าง

YOLO คือ:

> Local factorized structured regression

RF-DETR คือ:

> Global permutation-invariant set estimator

---

## Structural Hierarchy

$$
\text{Grid-based models}
\subset
\text{Set-based models}
$$

ในแง่ expressivity

แต่ไม่จำเป็นว่า accuracy > เสมอ
เพราะ bias–variance tradeoff (Ch.4)

---

# 10. Insight ระดับงานวิจัย

ถ้าต้องพิสูจน์เชิง formal เต็มรูปแบบ ต้องแสดงว่า:

1. RF-DETR hypothesis class strictly larger
2. YOLO เป็น constrained version ภายใต้ local independence
3. One-to-one matching ลด output entropy โดยไม่ต้อง NMS

---

หากต้องการ ผมสามารถ:

* เขียนบทนี้เป็น section หนังสือระดับ PhD พร้อม formal theorem/proof
* ใส่ diagram mermaid เปรียบเทียบ computation graph
* เพิ่ม information-theoretic bound เทียบสองโมเดล
* เพิ่ม experimental hypothesis สำหรับ IDD 1080p

เลือกทิศทางต่อได้เลย.
