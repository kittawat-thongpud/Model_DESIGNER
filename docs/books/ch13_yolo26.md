# บทที่ 13 --- YOLO26: Edge-Optimized Detection

> **Warning**  
> YOLO26 เป็น **community-driven evolution** (non-peer-reviewed) —  
> จัดเป็น industry development ไม่ใช่ academic contribution

## 13.1 Architectural Profile

### Architectural Profile

| ด้าน         | รายละเอียด                             |
| :----------- | :------------------------------------- |
| **Year**     | 2026                                   |
| **Origin**   | Community / Industry                   |
| **Status**   | Non-peer-reviewed                      |
| **Target**   | **Edge deployment** (mobile, embedded) |
| **Backbone** | Ultra-light (Depthwise + Pointwise)    |
| **Head**     | Decoupled, NMS-free                    |

## 13.2 Edge Optimization Strategies

### Operator-Level

| เทคนิค                           | ผลต่อ FLOPs                                           |
| :------------------------------- | :---------------------------------------------------- |
| **Depthwise Separable Conv**     | $\frac{1}{C_{out}} + \frac{1}{K^2}$ ของ standard conv |
| **Quantization-friendly design** | INT8 compatible → ×2--4 speedup                       |
| NMS-free                         | ลด post-processing latency                            |

### FLOPs Reduction: Depthwise Separable

Standard conv: $2 C_{in} C_{out} K^2 HW$

Depthwise separable:

$$
 \text{DW}: 2 C_{in} K^2 HW + \text{PW}: 2 C_{in} C_{out} HW
$$

Ratio:

$$
 \frac{\text{DW+PW}}{\text{Standard}} = \frac{1}{C_{out}} + \frac{1}{K^2}
$$

สำหรับ $K=3, C_{out}=256$: ratio $\approx 0.115$ (ลด **88.5%**)

## 13.3 ข้อจำกัดและ Status

| ด้าน            | สถานะ                                                  |
| :-------------- | :----------------------------------------------------- |
| Peer review     | ❌ ไม่ผ่าน                                             |
| Reproducibility | จำกัด                                                  |
| COCO benchmarks | ยังไม่มี standardized                                  |
| Novelty         | Engineering optimization มากกว่า architectural novelty |

> ดูเปรียบเทียบเชิง normalized ใน บทที่ 20 (under "community models" section)

## เอกสารอ้างอิง

1.  Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861

2.  Ultralytics community discussion. (2026). YOLO26 implementation notes.
