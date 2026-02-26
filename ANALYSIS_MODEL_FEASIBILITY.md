# Model DESIGNER — Model Feasibility Analysis

> วิเคราะห์ว่าระบบปัจจุบันสามารถสร้าง Model อะไรได้บ้าง  
> วันที่: 2026-02-12

---

## สรุปรวม (Executive Summary)

| Task | สถานะ | ทำได้เลย? | ขาดอะไร |
|------|--------|-----------|---------|
| **1. Classification** | ✅ พร้อมใช้งาน | ใช่ | — |
| **2. Detection** | ⚠️ ทำได้บางส่วน | ต้องแก้ไข | ขาด node สำคัญหลายตัว |
| **3. Scene Graph** | ❌ ยังทำไม่ได้ | ไม่ | ขาดเกือบทั้งหมด |

---

## 1. Classification ✅ ทำได้เลย

### สิ่งที่มีพร้อม
- **Node ครบ**: Input → Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear → Softmax → Output
- **Dataset**: MNIST (1×28×28, 10 classes), CIFAR-10 (3×32×32, 10 classes), Fashion MNIST
- **Training**: CrossEntropyLoss, optimizer (Adam/AdamW/SGD), LR scheduling, early stopping, AMP
- **Data Augmentation**: RandomFlip, RandomAffine, ColorJitter, RandAugment, AutoAugment, RandomErasing
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, Per-class metrics
- **Export**: Python module, Full pipeline script, ONNX
- **Inference**: Image upload → prediction with class name + confidence
- **Codegen**: Topological sort, auto shape inference, global variable injection

### ตัวอย่าง Model ที่สร้างได้

```
Input (1,28,28)
  → Conv2d (1→32, k=3, p=1)
  → BatchNorm2d (32)
  → ReLU
  → MaxPool2d (2)
  → Conv2d (32→64, k=3, p=1)
  → BatchNorm2d (64)
  → ReLU
  → MaxPool2d (2)
  → Flatten
  → Linear (3136→128)
  → ReLU
  → Dropout (0.5)
  → Linear (128→10)
  → Output (10 classes)
```

### สิ่งที่จะทำให้ดีขึ้น (Nice-to-have)
- ❑ เพิ่ม dataset: CIFAR-100, SVHN, ImageNet subset
- ❑ เพิ่ม node: AvgPool2d, AdaptiveAvgPool2d, LeakyReLU, GELU, Sigmoid
- ❑ เพิ่ม pretrained backbone: ResNet, VGG, MobileNet (transfer learning)

---

## 2. Object Detection ⚠️ ทำได้บางส่วน — ต้องเพิ่ม Node

### สิ่งที่มีแล้ว
- ✅ **Dataset**: COCO 2017 (80 classes) — มี loader ใน trainer
- ✅ **Detection Loss**: `DetectionLoss` (YOLO-style) — coord + obj + class loss
- ✅ **Detection Utils**: `encode_target()`, `decode_prediction()`, `nms()`, `box_iou()`
- ✅ **Inference**: detection task type recognized, bounding box output with NMS
- ✅ **Trainer**: detection collate_fn, detection branch in training loop
- ✅ **Conv2d, MaxPool2d, BatchNorm2d, ReLU** — พื้นฐานของ backbone

### สิ่งที่ขาด (ต้องเพิ่ม)

| สิ่งที่ขาด | ความสำคัญ | รายละเอียด |
|-----------|-----------|-----------|
| **Anchor Box / Detection Head node** | 🔴 Critical | ไม่มี node ที่แปลง feature map → (B, 5+C, S, S) detection output ปัจจุบัน user ต้อง design Conv2d output ให้ตรงกับ (5+num_classes) เอง — error-prone มาก |
| **Upsample + Concatenate pipeline** | 🟡 High | มี node แล้ว แต่ codegen ยังไม่รองรับ multi-scale feature fusion (FPN neck) อย่างเต็มที่ |
| **Anchor-free head template** | 🔴 Critical | ไม่มี template สำหรับ YOLO/SSD/RetinaNet detection head |
| **COCO data transform** | 🟡 High | `_build_transforms` ใช้ classification transforms กับ COCO — ไม่มี bbox transform (resize + flip ต้อง transform bbox ด้วย) |
| **mAP metric** | 🟡 High | trainer validation ใช้ `predicted.max(1)` ซึ่งเป็น classification metric — ไม่ compute mAP@0.5 สำหรับ detection |
| **Multi-scale output** | 🟡 High | Detection model ต้องการ output หลาย scale (P3/P4/P5) — ปัจจุบัน graph มี single output path เท่านั้น |
| **DepthwiseSeparableConv node** | 🟢 Nice | สำหรับ lightweight detection (MobileNet-SSD style) |
| **Residual / Skip connection** | 🟡 High | Concatenate node มีแล้ว แต่ไม่มี Add/Residual node สำหรับ ResNet backbone |

### สิ่งที่ต้องทำเพื่อให้ Detection ใช้งานได้จริง

1. **เพิ่ม `DetectionHead` node** — auto-compute output shape (5+C)×S×S จาก input feature map
2. **เพิ่ม `Add` (Element-wise Add) node** — สำหรับ residual connections
3. **แก้ COCO transforms** — ต้อง transform bbox ตาม image augmentation
4. **เพิ่ม mAP metric** — compute AP@0.5 และ AP@0.5:0.95 ใน validation loop
5. **เพิ่ม multi-output support ใน codegen** — ให้ forward() return dict หรือ tuple สำหรับ multi-scale

---

## 3. Scene Graph Generation ❌ ยังทำไม่ได้

### Scene Graph คืออะไร
Scene Graph Generation (SGG) คือการสร้าง graph ของ relationships ระหว่าง objects ในรูปภาพ:
- **Input**: รูปภาพ
- **Output**: set of triplets `(subject, predicate, object)` เช่น "person riding horse", "dog on grass"
- **ต้องการ**: Object Detection + Relationship Classification

### สิ่งที่ขาดทั้งหมด

| สิ่งที่ขาด | ความสำคัญ | รายละเอียด |
|-----------|-----------|-----------|
| **Object Detection pipeline** | 🔴 Critical | SGG ต้องการ detected objects ก่อน — ดูรายการ Detection ข้างบน |
| **ROI Pooling / ROI Align node** | 🔴 Critical | ต้อง extract feature จาก bounding box ของแต่ละ object |
| **Relationship Classifier node** | 🔴 Critical | ต้อง classify relationship ระหว่าง object pairs (subject, object) |
| **Pair Proposal module** | 🔴 Critical | ต้อง generate candidate pairs จาก detected objects |
| **Graph Neural Network node** | 🟡 High | Message passing ระหว่าง nodes ใน scene graph (GCN, GAT) |
| **Visual Genome dataset** | 🔴 Critical | Dataset หลักสำหรับ SGG — ไม่มีใน dataset_registry |
| **Predicate vocabulary** | 🔴 Critical | ต้องมี predicate classes (e.g. "on", "riding", "wearing", "near") |
| **Triplet Loss / Relationship Loss** | 🔴 Critical | Loss function เฉพาะสำหรับ SGG |
| **SGG metrics** | 🔴 Critical | Recall@K, mean Recall@K, SGGen, SGCls, PredCls evaluation |
| **Multi-head output codegen** | 🔴 Critical | Model ต้อง output: boxes, object labels, relationship labels พร้อมกัน |
| **Attention mechanism node** | 🟡 High | Self-attention / Cross-attention สำหรับ context modeling |
| **Embedding layer node** | 🟡 High | Word/label embedding สำหรับ predicate classification |

### Architecture ที่ต้องการ (แบบง่ายที่สุด)

```
Image
  → Backbone (Conv2d stack)         ← มีแล้ว (บางส่วน)
  → Detection Head                  ← ขาด
  → ROI Pooling per object          ← ขาด
  → Object Feature Extraction       ← ขาด
  → Pair Proposal (N×N pairs)       ← ขาด
  → Union Feature (subject ∪ object) ← ขาด
  → Relationship Classifier          ← ขาด
  → Output: [(subj, pred, obj), ...]  ← ขาด
```

### ประมาณการ effort
SGG ต้องการ **node types ใหม่อย่างน้อย 8-10 ตัว** + **dataset ใหม่** + **metrics ใหม่** + **codegen overhaul สำหรับ multi-task output** — เป็น feature set ขนาดใหญ่ที่ต้องใช้เวลาพัฒนาหลาย phase

---

## สรุป: Roadmap แนะนำ

### Phase A: ทำ Classification ให้สมบูรณ์ (Low effort — 1-2 days)
- [ ] เพิ่ม dataset: CIFAR-100, SVHN
- [ ] เพิ่ม node: AvgPool2d, AdaptiveAvgPool2d, LeakyReLU, GELU
- [ ] เพิ่ม Add (residual) node

### Phase B: ทำ Detection ให้ใช้ได้จริง (Medium effort — 3-5 days)
- [ ] เพิ่ม `DetectionHead` node (auto output shape)
- [ ] เพิ่ม `Add` node (element-wise addition for skip connections)
- [ ] แก้ COCO transform ให้ transform bbox ด้วย
- [ ] เพิ่ม mAP metric ใน trainer
- [ ] เพิ่ม multi-scale output support ใน codegen
- [ ] ทดสอบ end-to-end: design → train → inference บน COCO

### Phase C: Scene Graph (Large effort — 2-4 weeks)
- [ ] สร้าง Detection pipeline ให้เสถียรก่อน (Phase B)
- [ ] เพิ่ม ROIPooling, ROIAlign nodes
- [ ] เพิ่ม Visual Genome dataset loader
- [ ] เพิ่ม Relationship Classifier node
- [ ] เพิ่ม Pair Proposal module
- [ ] เพิ่ม GNN / Attention nodes
- [ ] เพิ่ม SGG-specific losses and metrics
- [ ] Overhaul codegen for multi-task output

---

## Current Node Inventory (15 types)

| Node | Category | Has Codegen | Shape Rule |
|------|----------|-------------|------------|
| Input | I/O | — | none_to_2d |
| Output | I/O | — | terminal |
| Conv2d | Processing | ✅ | conv2d |
| MaxPool2d | Processing | ✅ | pool2d |
| Linear | Processing | ✅ | linear |
| BatchNorm2d | Regularization | ✅ | passthrough |
| ReLU | Activation | ✅ | passthrough |
| Softmax | Activation | ✅ | passthrough |
| Dropout | Regularization | ✅ | passthrough |
| Flatten | Reshape | ✅ | flatten |
| Upsample | Reshape | ✅ | upsample |
| Concatenate | Functional | ✅ (functional) | passthrough |
| Package | Package | — | passthrough |
| IfElse | Logic | — | passthrough |
| Switch | Logic | — | passthrough |
