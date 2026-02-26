---
marp: true
theme: default
paginate: true
header: 'YOLO vs RF-DETR: Architecture Comparison'
footer: 'Object Detection Paradigms | 2026'
size: 16:9
style: |
  section {
    background-color: #ffffff;
    font-size: 18px;
    padding: 40px 60px;
  }
  h1 {
    color: #1a237e;
    font-size: 38px;
    margin-bottom: 0.3em;
  }
  h2 {
    color: #283593;
    font-size: 28px;
    margin-bottom: 0.3em;
  }
  h3 {
    font-size: 22px;
    margin-bottom: 0.3em;
    margin-top: 0.5em;
  }
  p, li, td {
    font-size: 16px;
    line-height: 1.4;
  }
  table {
    font-size: 14px;
    margin: 0.5em 0;
  }
  code {
    font-size: 14px;
  }
  ul, ol {
    margin: 0.3em 0;
    padding-left: 1.5em;
  }
  li {
    margin: 0.2em 0;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .yolo-color {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
  }
  .rfdetr-color {
    background-color: #e8f5e9;
    padding: 1rem;
    border-radius: 8px;
  }
  pre.mermaid {
    background-color: transparent;
  }
---

<!-- _class: lead -->

# YOLO vs RF-DETR
## Architecture Comparison

**Two Paradigms:**
- Grid-based (YOLO)
- Set-based (RF-DETR)

---

<!-- _class: lead -->

# The Problem
## Object Detection

<div class="columns">

<div>

### Goal
> **"What & where?"**

### Formulation
$$Y = \{(b_k, c_k)\}_{k=1}^{N}$$

$b_k$ = box, $c_k$ = class, $N$ = count

### Objective
$$p(Y \mid X; \theta)$$

</div>

<div>

![w:400](diagrams/images/yolo-grid-detection.png)

</div>

</div>

---

# YOLO Philosophy
## Grid-based Predictions

<div class="columns">

<div>

![w:450](diagrams/images/yolo-grid-detection.png)

</div>

<div>

### Concept
- Grid cells (p3: 80×80, p4: 40x40, p5: 20x20)
- Independent predictions

### Formula
$$p(Y \mid X) = \prod_{i,j} p(y_{ij} \mid F_{ij})$$

Grid factorization, Local $F_{ij}$

### Result
Multiple detections → **NMS needed**

</div>

</div>

---

# YOLO Architecture Pipeline

![w:1100](diagrams/images/yolo-pipeline.png)

| Step | Component | Function |
|------|-----------|----------|
| 1 | **Backbone** | CNN features |
| 2 | **Grid** | 80×80 cells |
| 3 | **Head** | Box + Class |
| 4 | **NMS** | Remove duplicates |

---

# YOLO Characteristics

<div class="columns">

<div class="yolo-color">

### Strengths ✅
- **Speed**: 30-60 FPS
- **Efficiency**: O(HW)
- **Memory**: Low VRAM
- **Params**: ~43M (YOLOv8l)
- **FLOPs**: ~165G @ 640×1080

### Math
$$\text{Cost} = O(HW)$$

</div>

<div>

### Limitations ❌
- Multiple predictions
- NMS required
- Limited receptive field
- Dense scenes struggle

### Duplication

![w:350](diagrams/images/yolo-duplication.png)

</div>

</div>

---

# RF-DETR Philosophy
## Set-based Prediction

<div class="columns">

<div>

![w:450](diagrams/images/rfdetr-query-system.png)

</div>

<div>

### Concept
- Fixed queries (M=100)
- See entire image

### Formula
$$p(Y \mid X) = \prod_{k=1}^{M} p(y_k \mid X)$$

Set prediction, Global attention

### Result
One-to-one → **No NMS**

</div>

</div>

---

# RF-DETR Architecture Pipeline

![w:1100](diagrams/images/rfdetr-pipeline.png)

| Step | Component | Function |
|------|-----------|----------|
| 1 | **Backbone** | CNN features |
| 2 | **Encoder** | Self-attention |
| 3 | **Queries** | M embeddings |
| 4 | **Decoder** | Cross-attention |
| 5 | **Heads** | Box + Class |
| 6 | **Hungarian** | 1-to-1 match |

---

# Hungarian Matching
## One-to-One Assignment

<div class="columns">

<div>

![w:500](diagrams/images/hungarian-matching.png)

</div>

<div>

### Algorithm
$$\min_{\sigma} \sum_{i} \mathcal{L}(\hat{y}_{\sigma(i)}, y_i)$$

$\sigma$ = assignment, $\mathcal{L}$ = cost

### Properties
✅ No duplicates ✅ No NMS ✅ End-to-end

</div>

</div>

---

# RF-DETR Characteristics

<div class="columns">

<div class="rfdetr-color">

### Strengths ✅
- Global attention
- No NMS
- Dense scenes OK
- Set prediction
- **Params**: ~40M (DETR-R50)
- **FLOPs**: ~86G @ 640×1080

### Math
$$\mathcal{R}(y_k) = X$$
$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

</div>

<div>

### Limitations ❌
- 15-30 FPS
- O((HW)²)
- High VRAM
- GPU required

### Cost
$$\text{Cost} = O(N^2 \cdot d)$$


</div>

</div>

---

# Architecture Comparison

| Dimension | YOLO | RF-DETR |
|-----------|------|---------|
| **Paradigm** | Grid-based | Set-based |
| **Context** | Local (CNN) | Global (Transformer) |
| **Assignment** | Many-to-one | One-to-one |
| **NMS** | Required | Not needed |
| **Complexity** | O(HW) | O((HW)²) |
| **Speed** | 30-60 FPS | 15-30 FPS |
| **Memory** | ~0.6 GB | ~1.2 GB |
| **Field** | Limited | Global |

---

# Trade-offs Analysis

<div class="columns">

<div>

### Performance

| Scenario | YOLO | RF-DETR |
|----------|------|---------|
| **Complex scenes** | 😐 | 😊 🟢 |
| **Speed** | 😊 🔵 | 😐 |
| **Low hardware** | 😊 🔵 | 😐 |
| **Accuracy** | 😐 | 😊 🟢 |

</div>

<div>

### Computational Trade-off

![w:450](diagrams/images/tradeoff-decision.png)

</div>

</div>

---

## Specifications Comparison

<div class="columns">

<div>

| Metric | YOLO (YOLOv8l) | RF-DETR (DETR-R50) |
|--------|----------------|---------------------|
| **Parameters** | ~43M | ~40M |
| **FLOPs @ 640×1080** | ~165G | ~86G |
| **Speed** | 30-60 FPS | 15-30 FPS |
| **Memory** | ~0.6 GB | ~1.2 GB |
| **Complexity** | O(HW) | O((HW)²) |

</div>

<div>

---
## Benchmark Results

![w:550](images/benchmark-comparison.png)

**COCO Segmentation Results**
- YOLO: Fast, efficient
- RF-DETR: Higher accuracy

</div>

</div>

---
# Use Cases

<div class="columns">

<div class="yolo-color">

### YOLO ✅

**When:**
Real-time, Edge devices, Limited hardware

**Apps:**
Vehicles, Surveillance, Mobile, Drones

</div>

<div class="rfdetr-color">

### RF-DETR ✅

**When:**
Accuracy priority, Complex scenes, GPU

**Apps:**
Medical, Satellite, Crowds, Research

</div>

</div>

---

<!-- _class: lead -->

# Thank You

## YOLO vs RF-DETR: Architecture Comparison

**Key Insight:**  
Both paradigms solve object detection differently - choose based on your constraints and priorities.

**Further Reading:**
- YOLO: Grid-based detection with local context
- RF-DETR: Set-based detection with global attention
- Trade-off: Speed ↔ Accuracy

