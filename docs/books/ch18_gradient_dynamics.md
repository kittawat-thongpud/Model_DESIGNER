# บทที่ 18 --- Gradient Dynamics ข้ามเวอร์ชัน

## 18.1 Gradient Scale Formula

จาก Ch.5:

## 18.1 Gradient Scale Formula

จาก Ch.5:

$$
 \left\|\frac{\partial \mathcal{L}}{\partial F_0}\right\| = \left\|\frac{\partial \mathcal{L}}{\partial F_L}\right\| \cdot \prod_{l=1}^{L} \left\|\frac{\partial F_l}{\partial F_{l-1}}\right\|
$$

ปัจจัยที่มีผล:

| ปัจจัย                 | ผล                                         | ดูบท         |
| :--------------------- | :----------------------------------------- | :----------- |
| Network depth $L$      | Product ยาวขึ้น → vanishing/exploding risk | Ch.5         |
| Residual connections   | $I + J$ → gradient bounded                 | Prop 5.1     |
| CSP split              | Bypass path → clean gradient               | Prop 8.1     |
| BatchNorm              | Normalize activation → stable $\|J_l\|$    |              |
| Activation function    | ReLU: $J = 0$ or $1$                       | dead neurons |
| Loss weights $\lambda$ | Scale upstream gradient                    | Ch.17        |

## 18.2 Stability Table v1--v12

| Version | Architecture          | Gradient Stability   | Key Reason                  |
| :------ | :-------------------- | :------------------- | :-------------------------- |
| v1      | 24 Conv + FC          | ❌ Unstable          | No residual, FC layers      |
| v2      | Darknet-19 + BN       | ⚠️ Moderate          | BN helps, still no shortcut |
| v3      | Darknet-53 + residual | ✅ Stable            | Residual bounds             |
| v4      | CSPDarknet53 + SPP    | ✅ Stable            | CSP + residual              |
| v5      | CSP + SPPF + EMA      | ✅ Very stable       | EMA smoothing               |
| YOLOX   | CSP + decoupled       | ✅ Stable            | Gradient isolation          |
| v6      | RepVGG                | ✅ Stable            | Re-param normalizes         |
| v7      | **E-ELAN**            | ✅✅ **Very stable** | Gradient expansion          |
| v8      | C2f                   | ✅ Stable            | Fine-grained concat         |
| v9      | GELAN + **PGI**       | ✅✅ Very stable     | Explicit gradient injection |
| v10     | NMS-free dual         | ✅ Stable            | Standard backbone           |
| v11--12 | C3k2 + C2PSA          | ✅ Stable            | Attention smoothing         |

## 18.3 Discussion: Optimization Landscape

> **Important**  
> ส่วนนี้เป็น **discussion section** — ไม่ใช่ formal claim  
> จัดเป็น exploratory analysis

### Curvature / Hessian Intuition

สำหรับ loss function $\mathcal{L}(\theta)$ curvature ที่ minimum $\theta^*$ วัดจาก Hessian:

$$
 H = \nabla^2 \mathcal{L}(\theta^*)
$$

| ลักษณะ Minimum | Hessian Eigenvalues  | Generalization       |
| :------------- | :------------------- | :------------------- |
| **Sharp**      | ค่า eigenvalues ใหญ่ | อาจ generalize ไม่ดี |
| **Flat**       | ค่า eigenvalues เล็ก | มักจะ generalize ดี  |

### Connection กับ YOLO Architecture (Discussion)

| Architecture   | Predicted Landscape | Reasoning (Qualitative)         |
| :------------- | :------------------ | :------------------------------ |
| Plain (v1--v2) | Sharp               | Limited gradient paths          |
| Residual (v3)  | Flatter             | Multiple gradient paths         |
| CSP (v4--v5)   | Flatter             | Bypass reduces constraint       |
| E-ELAN (v7)    | **Flattest**        | Maximum gradient diversity      |
| PGI (v9)       | Controlled          | Programmable gradient injection |

> **ข้อจำกัดสำคัญ**: ยังไม่มีการวัด Hessian spectrum โดยตรงในงาน YOLO --- ข้อสรุปข้างต้นอิงจาก:
>
> - Analogy กับ general deep learning literature (Li et al., 2018, "Visualizing the Loss Landscape")
> - Indirect evidence จาก generalization performance
> - Theoretical intuition จาก gradient path counting

## 18.4 Training Recommendations

จาก gradient analysis ข้ามเวอร์ชัน:

| คำแนะนำ                       | เหตุผล                         | Applicable versions |
| :---------------------------- | :----------------------------- | :------------------ |
| ใช้ learning rate warmup      | ป้องกัน gradient spike ช่วงแรก | ทุกรุ่น             |
| Cosine annealing scheduler    | Smooth decay ดีกว่า step decay | v4+                 |
| EMA                           | ลด gradient noise in updates   | v5+                 |
| Mixed precision (FP16)        | ลด memory + gradient สมดุล     | ทุกรุ่น             |
| Gradient clipping (if needed) | ป้องกัน exploding ใน v1--v2    | v1--v2              |

## เอกสารอ้างอิง

1.  Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). "Visualizing the Loss Landscape of Neural Nets." _NeurIPS 2018_. arXiv:1712.09913
2.  Foret, P., et al. (2021). "Sharpness-Aware Minimization." _ICLR 2021_. arXiv:2010.01412
