# บทที่ 3 --- Dense Prediction as [Structured Regression](#บทท-3-dense-prediction-as-structured-regression)

## 3.1 Detection เป็น Structured Output Problem

[Object detection](#11-นิยามปัญหา-object-detection) ไม่ใช่ปัญหา [classification](#a.3-score-classification) หรือ regression ทั่วไป --- แต่เป็น **structured prediction** ที่ output space มีโครงสร้างซับซ้อน

### Output Space Decomposition

ผลลัพธ์ของ detection ประกอบด้วย 3 ส่วนย่อย:

$$
[ \mathcal{Y} = \mathcal{B} \times \mathcal{O} \times \mathcal{C} ]
$$

Component Space ความหมาย

---

$\mathcal{B} \subseteq \mathbb{R}^4$ [Bounding box](#คำสำคัญ-bounding-box) $(x, y, w, h)$ coordinates

$\mathcal{O} \subseteq [0,1]$ [Objectness](#24-head--prediction-module) ความน่าจะเป็นว่ามีวัตถุ

$\mathcal{C} \subseteq [0,1]^C$ [Classification](#คำสำคัญ-classification) probability vector over $C$ classes

### ความแตกต่างจาก Standard Regression

| ด้าน                 | Standard Regression | Detection                                        |
| -------------------- | ------------------- | ------------------------------------------------ |
| **Output dimension** | คงที่               | **ผันแปร** (จำนวนวัตถุไม่คงที่)                  |
| **Output structure** | Vector              | **Set of structured tuples** (box, score, class) |
| **Evaluation**       | MSE / MAE           | **mAP**, IoU-based metrics                       |
| **Training target**  | 1:1 mapping         | **Assignment problem** (grid / anchor → GT)      |

YOLO แก้ปัญหา variable-length output โดยการ **discretize** ลง grid คงที่ --- ทำให้ output มีขนาดคงที่ $S \times S \times D$ แม้จำนวนวัตถุจริงจะต่างกัน

## 3.2 Probabilistic Formalization

### Joint Probability Model

ให้ $\mathbf{x}$ เป็นภาพอินพุตและ $\mathbf{y}$ เป็น structured output เราสามารถเขียน detection เป็น **conditional probability**:

$$
[ p(\mathbf{y} \| \mathbf{x}; \theta) ]
$$

โดยที่ $\theta$ เป็นพารามิเตอร์ของเครือข่าย

### Factorization ข้าม Grid Cells

ภายใต้ **conditional independence assumption** — grid cell แต่ละตัวทำนายอิสระจากกัน เมื่อกำหนด feature แล้ว:

$$
p(\mathbf{y} \mid \mathbf{x}; \theta) \approx \prod_{i=1}^{S} \prod_{j=1}^{S} p(\mathbf{y}_{ij} \mid F_{ij}; \theta_{\text{head}})
$$

โดยที่ $F_{ij}$ คือ feature vector ที่ grid cell ตำแหน่ง $(i,j)$
และ $\theta_{\text{head}}$ คือพารามิเตอร์ของ head

**Assumption ที่ต้องระบุ:**

> **Assumption 3.1** (Conditional Independence of Grid Predictions)
>
> เมื่อกำหนด feature map $\mathbf{F}$ จาก backbone + neck แล้ว predictions ที่ grid cells ต่าง ๆ เป็นอิสระต่อกัน:
>
> $$
> p(\mathbf{y}_{ij} \mid \mathbf{F}; \theta)
> =
> p(\mathbf{y}_{ij} \mid F_{ij}; \theta_{\text{head}})
> $$
>
> Assumption นี้ไม่จริงทั้งหมด (วัตถุใกล้กันมี spatial correlation) แต่ทำให้ training tractable ระบบจัดการปัญหา correlation ผ่าน **NMS** ในขั้น post-processing

### Per-Cell Factorization

แต่ละ Grid cell $\mathbf{y}\_{ij}$ สามารถ factorize ต่อ:

$$
[ p(\mathbf{y}\_{ij} \| F\_{ij}) = p(\mathbf{b}\_{ij} \| F\_{ij}) \cdot p(o\_{ij} \| F\_{ij}) \cdot p(\mathbf{c}\_{ij} \| F\_{ij}) ]
$$

| Component              | Distribution Assumption                | Loss ที่เชื่อมโยง                              |
| ---------------------- | -------------------------------------- | ---------------------------------------------- |
| $p(\mathbf{b} \mid F)$ | Regression (implicit Gaussian)         | IoU-based Loss (เช่น IoU / GIoU / DIoU / CIoU) |
| $p(o \mid F)$          | Bernoulli                              | BCE Loss                                       |
| $p(\mathbf{c} \mid F)$ | Categorical หรือ Independent Bernoulli | Cross-Entropy (CE) หรือ BCE Loss               |

## 3.3 Grid Discretization

### จากพื้นที่ต่อเนื่องสู่ Lattice

YOLO แปลง continuous image space เป็น discrete **prediction lattice**:

$$
[ \Lambda = \{(i, j) \mid i \in \{0, \ldots, S-1\}, \, j \in \{0, \ldots, S-1\}\} ]
$$

แต่ละ lattice point $(i,j)$ เป็น reference coordinate ที่:

- **[Anchor-based](#ปญหาของ-anchor-based)**: มี $A$ **anchor slots** ($A$ [bounding box](#a.2-bounding-box) hypotheses)
- **[Anchor-free](#anchor-free-distance-regression)**: มี 1 prediction point (regression จาก center)

### Multi-Scale Grid

เมื่อใช้ [multi-scale prediction](#22-backbone--feature-extractor) จำนวน prediction ทั้งหมด:

$$
[ N\_{total} = \sum\_{l} S_l^2 \cdot A_l ]
$$

ตัวอย่างสำหรับ input $640 \times 640$:

| Scale     | Grid $S_l$ | Anchors $A_l$ | Predictions (anchor-based / anchor-free) |
| --------- | ---------- | ------------- | ---------------------------------------- |
| P3        | 80 × 80    | 3 / 1         | 19,200 / 6,400                           |
| P4        | 40 × 40    | 3 / 1         | 4,800 / 1,600                            |
| P5        | 20 × 20    | 3 / 1         | 1,200 / 400                              |
| **Total** | ---        | ---           | **25,200 / 8,400**                       |

### Grid Responsibility

> **Definition 3.1** (Grid Responsibility)
>
> Grid cell $G\_{ij}$ ที่ scale $l$ มี "responsibility" ต่อ **ground truth** $g_k$ เมื่อ center ของ $g_k$ ตกอยู่ใน spatial region ของ $G\_{ij}$ ใน [anchor-based](#ปญหาของ-anchor-based) paradigm ส่วนใน [dynamic assignment](#simota-dynamic-assignment) ใช้เกณฑ์ที่ยืดหยุ่นกว่า

## 3.4 Connection ระหว่าง Likelihood กับ [Loss Functions](#a.5-loss-functions)

### [BCE](#g.1-bce-gradient) เป็น Negative Log-Likelihood

Objectness score $o$ ถูก model เป็น Bernoulli:

$$
[ p(y \| o) = o^y (1-o)^{1-y} ]
$$

Negative log-likelihood:

$$
[ -\log p(y\|o) = -[y \log o + (1-y) \log(1-o)] ]
$$

นี่คือ **Binary Cross-Entropy (BCE)** --- loss ที่ใช้สำหรับ objectness ทุกเวอร์ชัน YOLO

> **Important**: BCE ไม่ใช่ heuristic --- เป็น MLE ภายใต้ Bernoulli assumption

### CE เป็น Categorical NLL

**Classification** score $\mathbf{c} = (c_1, \ldots, c_C)$ ถ้า model เป็น categorical (softmax):

$$
[ p(y=k \| \mathbf{c}) = \frac{e^{c_k}}{\sum\_{j} e^{c_j}} ]
$$

$$
[ -\log p(y=k \| \mathbf{c}) = -c_k + \log \sum_j e^{c_j} ]
$$

นี่คือ **Cross-Entropy (CE)** loss

> **หมายเหตุ**: YOLOv3+ ใช้ **independent sigmoid per class** (multi-label BCE) แทน softmax เพื่อรองรับ multi-label scenarios

### Total Likelihood → Total Loss

$$
[ L(\theta) = -\sum\_{i} \log p(\mathbf{y}\_i \| \mathbf{x}\_i; \theta) ]
$$

ซึ่งแยกออกเป็น:

$$
\mathcal{L}
=
\underbrace{\lambda_{box} \mathcal{L}_{box}}_{\text{localization}}
+
\underbrace{\lambda_{obj} \mathcal{L}_{obj}}_{\text{objectness}}
+
\underbrace{\lambda_{cls} \mathcal{L}_{cls}}_{\text{classification}}
$$

| Term                | Likelihood Connection                                        | Loss Function                    |
| ------------------- | ------------------------------------------------------------ | -------------------------------- |
| $\mathcal{L}_{box}$ | Implicit Gaussian on box coordinates → geometric consistency | IoU / CIoU                       |
| $\mathcal{L}_{obj}$ | Bernoulli negative log-likelihood                            | BCE                              |
| $\mathcal{L}_{cls}$ | Categorical / multi-label negative log-likelihood            | CE / BCE                         |
| $\lambda_{*}$       | Prior weights over task components                           | Hyperparameters (task balancing) |

> รายละเอียด loss design อยู่ใน บทที่ 17

## 3.5 Proposition 3.1: Grid Coverage Sufficiency

> **Proposition 3.1** (Grid Coverage Sufficiency)
>
> ให้วัตถุมีขนาดเล็กสุด $w\_{min} \times h\_{min}$ ในภาพขนาด $W \times H$ ถ้าใช้ grid ขนาด $S \times S$ ที่ scale ละเอียดที่สุด (stride $s = W/S$) แล้ว Grid cell สามารถ "รับผิดชอบ" วัตถุทุกตัวได้เมื่อ:

$$
[s \leq \min(w\_{min}, h\_{min})]
$$

> กล่าวคือ stride ของ grid ต้องไม่เกินขนาดเล็กสุดของวัตถุ
>
> **Proof sketch:**
>
> ถ้า center ของวัตถุตกที่ตำแหน่ง $(x_c, y_c)$ ใน continuous space จะถูก map ไปที่ Grid cell:

$$
[G\_{ij} : i = \lfloor x_c / s \rfloor, \quad j = \lfloor y_c / s \rfloor]
$$

> Grid cell นี้ครอบคลุมพื้นที่ $[is, (i+1)s) \times [js, (j+1)s)$
>
> เมื่อ $s \leq w\_{min}$ วัตถุจะมีขนาดอย่างน้อยเท่ากับ 1 Grid cell ในแต่ละมิติ --- ทำให้ network มี feature ที่เพียงพอสำหรับ prediction
>
> ในทางปฏิบัติ YOLOv3+ ใช้ 3 scales (stride 8, 16, 32) ทำให้วัตถุขนาด $\geq 8 \times 8$ pixels สามารถ detect ได้จาก $P_3$
>
> **ข้อจำกัด**: Proposition นี้เป็น necessary condition ไม่ใช่ sufficient --- ในทางปฏิบัติ detection ต้องการ receptive field ที่ใหญ่กว่า grid cell เพื่อเข้าใจ context $\square$

## เอกสารอ้างอิง

1.  Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." _CVPR 2016_. arXiv:1506.02640

2.  Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer. --- Ch. 14 (Structured Output)

3.  LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. A., & Huang, F. J. (2006). "A Tutorial on Energy-Based Learning." _Predicting Structured Data_, MIT Press.

4.  Lin, T.-Y., et al. (2014). "Microsoft COCO: Common Objects in Context." _ECCV 2014_. arXiv:1405.0312

5.  Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." arXiv:1804.02767
