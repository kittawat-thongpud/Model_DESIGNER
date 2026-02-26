# บทที่ 3 — Computational Constraints

## 6.1 ทำไม Constraints สำคัญ?

จากบทที่ 2 detection คือ risk minimization:

$$
\theta^* = \arg\min_\theta \hat{R}(\theta)
$$

แต่ในทางปฏิบัติ YOLO มี **latency budget** — ไม่สามารถเพิ่ม capacity ได้ไม่จำกัด:

$$
\boxed{
\theta^* = \arg\min_\theta \hat{R}(\theta) \quad \text{subject to} \quad T(\theta) \leq T_{budget}
}
$$

นี่คือ **constrained optimization** ที่เป็นแกนกลางของวิวัฒนาการ YOLO — ทุก generation พยายาม **ลด risk โดยไม่เกิน budget**

---

## 6.2 FLOPs Calculation

### Standard Convolution FLOPs

สำหรับ convolution layer ที่มี kernel ขนาด $K \times K$:

$$
\text{FLOPs}_{conv} = 2 \cdot C_{in} \cdot C_{out} \cdot K^2 \cdot H_{out} \cdot W_{out}
$$

| สัญลักษณ์          | ความหมาย                    |
| :----------------- | :-------------------------- |
| $C_{in}$           | input channels              |
| $C_{out}$          | output channels             |
| $K$                | kernel size                 |
| $H_{out}, W_{out}$ | output spatial dimensions   |
| factor 2           | multiply + accumulate (MAC) |

### Total Network FLOPs

$$
\text{FLOPs}_{total} = \sum_{l=1}^{L} \text{FLOPs}_l
$$

### Special Layers FLOPs

| Layer Type           | FLOPs Formula               | หมายเหตุ                            |
| :------------------- | :-------------------------- | :---------------------------------- |
| Conv $K \times K$    | $2 C_{in} C_{out} K^2 H W$  | Standard                            |
| **Depthwise Conv**   | $2 C K^2 H W$               | $C_{in} = C_{out} = C$, per-channel |
| Pointwise Conv (1×1) | $2 C_{in} C_{out} H W$      | $K=1$                               |
| **BatchNorm**        | $4 C H W$                   | ≈ negligible                        |
| **SPP/SPPF**         | MaxPool (no FLOPs) + concat | Compute-free                        |

### YOLO FLOPs ตัวอย่าง

| Model   | Parameters | FLOPs (640×640) | Source             |
| :------ | :--------- | :-------------- | :----------------- |
| YOLOv5n | 1.9M       | 4.5G            | Ultralytics (2020) |
| YOLOv5s | 7.2M       | 16.5G           | Ultralytics (2020) |
| YOLOv5m | 21.2M      | 49.0G           | Ultralytics (2020) |
| YOLOv8n | 6.2M       | 8.7G            | Ultralytics (2023) |
| YOLOv8s | 11.2M      | 28.6G           | Ultralytics (2023) |

> **Empirical Observation 6.1**: FLOPs ใน YOLO เพิ่มขึ้นราว $3\times$–$4\times$ ต่อ model scale tier (n→s→m) เนื่องจาก width multiplier affects $C^2$ term quadratically
>
> _Citation_: Ultralytics YOLOv5 README, Table: Model Comparison; Ultralytics YOLOv8 docs, Performance section

---

## 6.3 Latency Decomposition

### End-to-End Latency

Inference latency ทั้งหมดของ detection pipeline:

$$
\boxed{
T_{total} = \underbrace{T_{pre}}_{\text{preprocess}} + \underbrace{T_{backbone}}_{\text{feature}} + \underbrace{T_{neck}}_{\text{fusion}} + \underbrace{T_{head}}_{\text{predict}} + \underbrace{T_{NMS}}_{\text{post-process}}
}
$$

| Component      | ขึ้นกับ                           | สัดส่วน (typical) |
| :------------- | :-------------------------------- | :---------------- |
| $T_{pre}$      | Image resize, normalization       | 5–10%             |
| $T_{backbone}$ | FLOPs, memory bandwidth           | 50–65%            |
| $T_{neck}$     | Multi-scale fusion, upsample      | 15–25%            |
| $T_{head}$     | 1×1 conv, sigmoid                 | 5–10%             |
| $T_{NMS}$      | Number of predictions, IoU thresh | 5–15%             |

### Compute-Bound vs Memory-Bound

Latency ไม่ได้แปรผันตาม FLOPs เสมอ — operation แต่ละตัวมี bottleneck ต่างกัน:

$$
T_{layer} = \max\left( \frac{\text{FLOPs}}{\text{FLOPS capacity}}, \frac{\text{Memory access}}{\text{Bandwidth}} \right)
$$

| Regime            | Bottleneck                     | ตัวอย่าง Layer                             |
| :---------------- | :----------------------------- | :----------------------------------------- |
| **Compute-bound** | FLOPs > bandwidth can supply   | Conv3×3 กับ channel สูง                    |
| **Memory-bound**  | Data transfer > compute needed | Concat, Upsample, 1×1 Conv ที่ channel ต่ำ |

### Arithmetic Intensity (Roofline Model)

$$
I = \frac{\text{FLOPs}}{\text{Bytes accessed}}
$$

| Layer             | $I$ (approx.) | Regime        |
| :---------------- | :------------ | :------------ |
| Conv 3×3, $C=256$ | ~150          | Compute-bound |
| Conv 1×1, $C=64$  | ~30           | Borderline    |
| Concat            | 0             | Memory-bound  |
| Upsample          | 0             | Memory-bound  |

> **Empirical Observation 6.2**: EfficientRep (YOLOv6) ลด latency ได้มากแม้ FLOPs ไม่ลดลง เพราะ re-parameterization ลด memory-bound operations ตอน inference
>
> _Citation_: Li et al. (2022), "YOLOv6", Section 3.1, Table 2

---

## 6.4 NMS Cost Analysis

### Non-Maximum Suppression Latency

NMS เป็น post-processing ที่ **ไม่มี FLOPs ใน model graph** แต่ใช้เวลาจริง:

$$
T_{NMS} = \mathcal{O}(N_{pred} \cdot N_{pred}) \approx \mathcal{O}(N_{pred}^2)
$$

ใน worst case (ทุก box เทียบกับทุก box) แต่ในทางปฏิบัติใช้ per-class NMS:

$$
T_{NMS} \approx \sum_{c=1}^{C} \mathcal{O}(N_c^2)
$$

| ปัจจัย                   | ผลต่อ NMS latency | กลไก                                |
| :----------------------- | :---------------- | :---------------------------------- |
| Confidence threshold สูง | ลดลงมาก           | Filter predictions ก่อน NMS         |
| IoU threshold ต่ำ        | ลดลง              | Suppress มากขึ้น → fewer iterations |
| จำนวน classes มาก        | เพิ่มขึ้น         | NMS ทำแยกแต่ละ class                |

### NMS-Free Implication

YOLOv10 เสนอ **NMS-free** detection ด้วย dual assignment — ทำให้:

$$
T_{total}^{v10} = T_{pre} + T_{backbone} + T_{neck} + T_{head}
$$

ลด $T_{NMS}$ ออกทั้งหมด — มี latency benefit โดยเฉพาะบน edge devices ที่ NMS ทำบน CPU

---

## 6.5 Throughput Model

### FPS (Frames Per Second)

$$
\text{FPS} = \frac{1}{T_{total}} = \frac{1}{T_{pre} + T_{compute} + T_{NMS}}
$$

### Throughput on Batch Processing

$$
\text{Throughput} = \frac{B}{T_{compute}(B)}
$$

โดยที่ $T_{compute}(B)$ ไม่ scale linearly กับ batch size $B$ เพราะ GPU parallelism:

| Batch Size | Throughput (relative) | หมายเหตุ                     |
| :--------- | :-------------------- | :--------------------------- |
| 1          | ×1                    | Baseline                     |
| 4          | ×3.2                  | Sub-linear because of memory |
| 8          | ×5.5                  | Diminishing returns          |
| 16         | ×8                    | Near saturation              |

> **Key Takeaway**: Single-image latency ($B=1$) คือ metric ที่สำคัญสำหรับ real-time deployment ในขณะที่ throughput ($B>1$) สำคัญสำหรับ batch processing (video analytics)

---

## 6.6 Memory Complexity

### Training Memory

$$
\text{Memory}_{train} = \underbrace{M_{params}}_{\text{weights}} + \underbrace{M_{grad}}_{\text{gradients}} + \underbrace{M_{act}}_{\text{activations}} + \underbrace{M_{opt}}_{\text{optimizer state}}
$$

| Component    | ขนาด (FP32)                            | หมายเหตุ                 |
| :----------- | :------------------------------------- | :----------------------- |
| $M_{params}$ | 4 bytes per parameter                  | weight storage           |
| $M_{grad}$   | 4 bytes per parameter                  | gradient storage         |
| $M_{act}$    | $\sum_l 4 \times C_l H_l W_l \times B$ | activations for backprop |
| $M_{opt}$    | 8 bytes per parameter (Adam)           | momentum + variance      |

### Inference Memory

$$
\text{Memory}_{infer} = M_{params} + \max_l (M_{act,l})
$$

เฉพาะ **peak activation** — ไม่ต้องเก็บทุก layer (reuse ได้)

### ผลต่อ YOLO Design

| Design Choice           | ผลต่อ Memory             | ตัวอย่าง        |
| :---------------------- | :----------------------- | :-------------- |
| CSP                     | ลด activations (partial) | v4, v5          |
| C2f                     | ลด memory movement       | v8              |
| **Depthwise separable** | ลด parameters            | YOLO26          |
| FP16 training           | ลด $M_{act}$ ≈ 50%       | ทุกรุ่นปัจจุบัน |

---

## 6.7 Scaling Laws

### Width–Depth–Resolution Scaling

YOLO models scale ผ่าน 3 มิติ (ตาม compound scaling, Tan & Le, 2019):

| มิติ                        | Symbol | ผลต่อ FLOPs               |
| :-------------------------- | :----- | :------------------------ |
| **Width** (channels)        | $w$    | $\propto w^2$ (quadratic) |
| **Depth** (layers)          | $d$    | $\propto d$ (linear)      |
| **Resolution** (image size) | $r$    | $\propto r^2$ (quadratic) |

### Compound Scaling Rule

$$
\text{FLOPs}_{scaled} \approx \text{FLOPs}_{base} \cdot w^2 \cdot d \cdot r^2
$$

### YOLOv5 Scaling ตัวอย่าง

| Scale | Width mult. | Depth mult. | FLOPs ratio |
| :---- | :---------- | :---------- | :---------- |
| n     | 0.25        | 0.33        | ×1 (base)   |
| s     | 0.50        | 0.33        | ×3.7        |
| m     | 0.75        | 0.67        | ×10.9       |
| l     | 1.00        | 1.00        | ×24.4       |
| x     | 1.25        | 1.33        | ×41.8       |

---

## 6.8 Proposition 6.1: Optimal Compute Allocation

> **Proposition 6.1** (Optimal Compute Allocation Under Fixed Budget)
>
> ให้ compute budget คงที่ $\text{FLOPs}_{max}$ สำหรับ detection network ที่มี backbone + neck + head การจัดสรรที่ optimal อยู่ที่:

$$
\text{FLOPs}_{backbone} \approx 0.6\text{-}0.7 \cdot \text{FLOPs}_{total}
$$

$$
\text{FLOPs}_{neck} \approx 0.2\text{-}0.3 \cdot \text{FLOPs}_{total}
$$

$$
\text{FLOPs}_{head} \approx 0.05\text{-}0.1 \cdot \text{FLOPs}_{total}
$$

> **Proof sketch:**
>
> 1. **Backbone** ต้องการ FLOPs สูงสุดเพราะทำงานบน high-resolution feature maps (FLOPs ∝ $H \times W$)
> 2. **Neck** ใช้ FLOPs ปานกลาง — ทำงานบน reduced-resolution features แต่ต้อง fuse หลาย scales
> 3. **Head** ใช้ FLOPs น้อยที่สุด — 1×1 conv กับ output channels $A \times (5+C)$ ซึ่งเล็ก
>
> **Evidence:**
>
> | Model   | Backbone % | Neck % | Head % |
> | :------ | :--------- | :----- | :----- |
> | YOLOv5s | 68%        | 26%    | 6%     |
> | YOLOv8s | 65%        | 28%    | 7%     |
>
> จาก Ultralytics model profiling (NVIDIA T4, FP16)
>
> **นัยเชิงออกแบบ**: ถ้าต้องลด FLOPs ควรเริ่มจาก backbone (เช่น depthwise conv, CSP) เพราะมี impact สูงสุด $\square$

## เอกสารอ้างอิง

1. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." _ICML 2019_. arXiv:1905.11946

2. Li, C., et al. (2022). "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications." arXiv:2209.02976

3. Ultralytics. (2020). "YOLOv5." _GitHub_. <https://github.com/ultralytics/yolov5>

4. Ultralytics. (2023). "YOLOv8." _GitHub_. <https://github.com/ultralytics/ultralytics>

5. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." _CACM_ 52(4).Roofline Model

6. Wang, A., et al. (2024). "YOLOv10: Real-Time End-to-End Object Detection." arXiv:2405.14458
