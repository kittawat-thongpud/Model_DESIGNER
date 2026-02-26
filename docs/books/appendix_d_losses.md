# ภาคผนวก D --- Loss Function Reference

> _Loss function คือเข็มทิศที่นำทางการเรียนรู้ --- ใน detection ต้องบาลานซ์ระหว่าง classification, localization, และ objectness_

## D.1 IoU Loss Family

### IoU (Intersection over Union)

$$
 IoU = \frac{\|B_p \cap B_{gt}\|}{\|B_p \cup B_{gt}\|}
$$

**ปัญหา:** gradient เป็น 0 เมื่อ $IoU = 0$ (boxes ไม่ overlap)

### GIoU (Generalized IoU)

$$
 GIoU = IoU - \frac{\|C \setminus (B_p \cup B_{gt})\|}{\|C\|}
$$

- $C$: smallest enclosing box
- **แก้ปัญหา:** ให้ gradient เมื่อ $IoU = 0$

**อ้างอิง:** Rezatofighi, H., et al. "Generalized Intersection over Union." _CVPR_, 2019. arXiv:1902.09630

### DIoU (Distance IoU)

$$
 DIoU = IoU - \frac{\rho^2(\mathbf{b}, \mathbf{b}_{gt})}{c^2}
$$

- $\rho$: Euclidean distance ระหว่าง centers
- $c$: diagonal ของ smallest enclosing box
- **convergence เร็วกว่า GIoU**

### CIoU (Complete IoU)

$$
 CIoU = IoU - \frac{\rho^2(\mathbf{b}, \mathbf{b}_{gt})}{c^2} - \alpha v
$$

โดยที่:

$$
 v = \frac{4}{\pi^2}\left(\arctan\frac{w_{gt}}{h_{gt}} - \arctan\frac{w}{h}\right)^2
$$

$$
 \alpha = \frac{v}{(1-IoU)+v}
$$

**ใช้ใน:** YOLOv5, YOLOv7, YOLOv8

**อ้างอิง:** Zheng, Z., et al. "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression." _AAAI_, 2020. arXiv:1911.08287

## D.2 Classification Losses

### Binary Cross-Entropy (BCE)

$$
 L_{BCE} = -y\log(p) + (1-y)\log(1-p)
$$

**ใช้ใน:** objectness + per-class classification (YOLOv3--v5)

### Focal Loss

$$
 L_{FL} = -\alpha_t(1-p_t)^\gamma \log(p_t)
$$

- $\alpha_t$: class balancing weight
- $\gamma$: focusing parameter (ลด loss ของ easy examples)

**ใช้ใน:** YOLOX, YOLOv8 (cls loss)

**อ้างอิง:** Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." _ICCV_, 2017. arXiv:1708.02002

### VariFocal Loss (VFL)

$$
 VFL = \begin{cases} -q(q\log(p) + (1-q)\log(1-p)) & q > 0 \\ -\alpha p^\gamma \log(1-p) & q = 0 \end{cases}
$$

- $q$: target IoU score (soft label)
- **ใช้ IoU-aware classification** แทน hard labels

**ใช้ใน:** YOLOv8

**อ้างอิง:** Zhang, H., et al. "VarifocalNet: An IoU-aware Dense Object Detector." _CVPR_, 2021. arXiv:2008.13367

## D.3 Distribution Focal Loss (DFL)

$$
 DFL(S_i, y) = -\left((y_{i+1}-y)\log S_i + (y-y_i)\log S_{i+1}\right)
$$

- $y$: continuous target value
- $y_i, y_{i+1}$: nearest discrete bins
- $S_i, S_{i+1}$: predicted probabilities

**แนวคิด:** จัด box regression เป็น discrete probability distribution

**ใช้ใน:** YOLOv8

**อ้างอิง:** Li, X., et al. "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes." _NeurIPS_, 2020. arXiv:2006.04388

## D.4 Combined YOLO Loss

### YOLOv5 / v7

$$
 L_{total} = \lambda_{box} L_{CIoU} + \lambda_{obj} L_{BCE}^{obj} + \lambda_{cls} L_{BCE}^{cls}
$$

### YOLOv8

$$
 L_{total} = \lambda_{box} (L_{CIoU} + L_{DFL}) + \lambda_{cls} L_{VFL}
$$

> **Key difference:** YOLOv8 ลบ objectness branch ออก --- ใช้ task-aligned assignment แทน

## ตารางสรุป Loss Evolution

ยุค Box Loss Cls Loss Objectness Assignment

---

v1 MSE MSE ✓ Grid
v2--v3 MSE BCE ✓ IoU threshold
v4--v5 CIoU BCE ✓ IoU threshold
YOLOX IoU BCE + Focal ✓ SimOTA
v8 CIoU + DFL VFL ✗ TAL
