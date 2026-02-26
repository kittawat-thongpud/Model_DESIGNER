# ภาคผนวก H — ดัชนีคำสำคัญ

> ดัชนีสำหรับค้นหาคำศัพท์เทคนิคและแนวคิดหลักตลอดเล่ม

---

## สถาปัตยกรรม YOLO

| คำสำคัญ     | บทที่กล่าวถึง | คำอธิบายย่อ                             |
| :---------- | :------------ | :-------------------------------------- |
| **YOLOv1**  | 1, 7, B.1     | Single-stage detection proof-of-concept |
| **YOLOv2**  | 7, B.2        | Anchor-based + Darknet-19               |
| **YOLOv3**  | 7, B.2        | Multi-scale FPN + Darknet-53            |
| **YOLOv4**  | 8, B.3        | CSPDarknet + Bag of Freebies            |
| **YOLOv5**  | 8, B.3        | Production-grade, auto-anchor           |
| **YOLOv6**  | 11, B.4       | EfficientRep, industrial                |
| **YOLOv7**  | 11, B.5       | E-ELAN, planned re-parameterization     |
| **YOLOv8**  | 11, B.6       | Anchor-free, C2f, TAL                   |
| **YOLOv9**  | 12, B.7       | GELAN + PGI                             |
| **YOLOv10** | 12            | NMS-free dual head                      |
| **YOLOv11** | 12, B.8       | C3k2 + C2PSA attention                  |
| **YOLOv12** | 12, B.8       | Area attention                          |
| **YOLO26**  | 13, B.9       | Edge-first, ultra-light                 |

---

## Backbone Components

| คำสำคัญ          | ภาคผนวก | คำอธิบาย                               |
| :--------------- | :------ | :------------------------------------- |
| **CSP**          | B.3     | Split-merge gradient efficiency        |
| **CSPDarknet53** | B.3     | v4/v5 backbone                         |
| **Darknet-19**   | B.2     | v2 backbone                            |
| **Darknet-53**   | B.2     | v3 backbone with residuals             |
| **C2f**          | B.6     | Hardware-aware CSP variant (v8)        |
| **C3k2**         | B.8     | Efficient CSP bottleneck (v11)         |
| **C2PSA**        | B.8     | CSP + Spatial Attention (v11)          |
| **E-ELAN**       | B.5     | Extended efficient aggregation (v7)    |
| **EfficientRep** | B.4     | Re-parameterized backbone (v6)         |
| **GELAN**        | B.7     | Generalized efficient aggregation (v9) |
| **RepConv**      | B.4, 11 | Re-parameterizable convolution         |

---

## Neck Components

| คำสำคัญ         | ภาคผนวก | คำอธิบาย                           |
| :-------------- | :------ | :--------------------------------- |
| **FPN**         | C.2     | Feature Pyramid Network (top-down) |
| **PAN / PANet** | C.3     | Path Aggregation Network           |
| **BiFPN**       | C.4     | Bidirectional weighted FPN         |
| **SPP / SPPF**  | C.1     | Spatial Pyramid Pooling            |

---

## Detection Head

| คำสำคัญ            | บท    | คำอธิบาย                       |
| :----------------- | :---- | :----------------------------- |
| **Coupled Head**   | 7, 8  | Shared conv สำหรับ cls+box+obj |
| **Decoupled Head** | 9, 11 | แยก branch cls กับ box         |
| **Anchor-Based**   | 7, 8  | ใช้ prior boxes                |
| **Anchor-Free**    | 9, 11 | ทำนาย distances                |
| **NMS-Free**       | 12    | Dual assignment                |

---

## Loss Functions

| คำสำคัญ        | ภาคผนวก  | คำอธิบาย                  |
| :------------- | :------- | :------------------------ |
| **IoU**        | D.1      | Intersection over Union   |
| **GIoU**       | D.1      | Generalized IoU           |
| **DIoU**       | D.1      | Distance IoU              |
| **CIoU**       | D.1, G.3 | Complete IoU              |
| **BCE**        | D.2, G.1 | Binary Cross-Entropy      |
| **Focal Loss** | D.2, G.2 | Down-weight easy examples |
| **VFL**        | D.2      | VariFocal Loss            |
| **DFL**        | D.3, G.4 | Distribution Focal Loss   |

---

## Label Assignment

| คำสำคัญ                | บท       | คำอธิบาย                                    |
| :--------------------- | :------- | :------------------------------------------ |
| **Static Assignment**  | 7, 8, 10 | IoU threshold กับ anchors                   |
| **SimOTA**             | 10, 11   | Simplified Optimal Transport                |
| **TAL (Task-Aligned)** | 10, 11   | Alignment metric ( s^\alpha \cdot u^\beta ) |
| **Dynamic Assignment** | 10       | ปรับ positive samples ระหว่าง training      |

---

## คณิตศาสตร์พื้นฐาน

| คำสำคัญ                    | บท      | คำอธิบาย                          |
| :------------------------- | :------ | :-------------------------------- |
| **Bayes Risk**             | 4       | $ R^\* = \inf_f R(f) $            |
| **Convolution**            | 5       | $ y = x \ast w + b $              |
| **Dense Prediction**       | 3       | Pixel-level prediction            |
| **ERM**                    | 4       | Empirical Risk Minimization       |
| **FLOPs**                  | 6       | Floating Point Operations         |
| **Gradient Flow**          | 18, G.5 | Backpropagation path analysis     |
| **Information Bottleneck** | 21      | Mutual information compression    |
| **Receptive Field**        | 5       | Input region affecting one output |

---

## ตัวอักษรย่อ

| ย่อ  | คำเต็ม                            |
| :--- | :-------------------------------- |
| AP   | Average Precision                 |
| BN   | Batch Normalization               |
| COCO | Common Objects in Context         |
| FC   | Fully Connected                   |
| FPN  | Feature Pyramid Network           |
| mAP  | mean Average Precision            |
| NMS  | Non-Maximum Suppression           |
| PAN  | Path Aggregation Network          |
| PGI  | Programmable Gradient Information |
| ReLU | Rectified Linear Unit             |
| SGD  | Stochastic Gradient Descent       |
| SiLU | Sigmoid Linear Unit               |
| SPP  | Spatial Pyramid Pooling           |
| TAL  | Task-Aligned Learning             |

---
